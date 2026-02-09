
#include <torch/extension.h>

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace huffman_cpp {

static constexpr int kFixedL = 7;
static constexpr int kFixedTableSize = 1 << kFixedL;              // 128
static constexpr int kFixedPackedLenSize = kFixedTableSize >> 1;  // 64
static constexpr int kMaxTablesPerBucket = 256;
static constexpr int kFixedShift = 64 - kFixedL;                  // 57

// AWQ INT4 bucket decoder (specialized L=11).
static constexpr int kAwqFixedL = 11;
static constexpr int kAwqTableSize = 1 << kAwqFixedL;             // 2048
static constexpr int kAwqPackedLenSize = kAwqTableSize >> 1;      // 1024
static constexpr int kAwqMaxTablesPerBucket = 16;                 // qv in [0,15]
static constexpr int kAwqShift = 64 - kAwqFixedL;                 // 53

#ifndef HUFFMAN_L7_READER_MODE
// 0: bitpos + cached_w (recommended)
// 1: incremental (w,o,a,b) with refill
#define HUFFMAN_L7_READER_MODE 1
#endif

#ifndef HUFFMAN_ROW_SHARED_CAP
// Max number of row indices to cache in shared per block.
// Keeping this bounded avoids large dynamic shared allocations that can tank occupancy.
#define HUFFMAN_ROW_SHARED_CAP 2048
#endif

__device__ __forceinline__ uint64_t shf_l_wrap_u64(uint64_t a, uint64_t b, int o) {
	// Return (a<<o) | (b>>(64-o)) for o in [0,63] using 32-bit funnel shifts.
	// This avoids shift-by-64 hazards and avoids branches on o.
	const uint32_t a0 = static_cast<uint32_t>(a);
	const uint32_t a1 = static_cast<uint32_t>(a >> 32);
	const uint32_t b0 = static_cast<uint32_t>(b);
	const uint32_t b1 = static_cast<uint32_t>(b >> 32);
	const uint32_t s = static_cast<uint32_t>(o) & 31u;
	const uint32_t k = static_cast<uint32_t>(o) >> 5;  // 0 or 1
	// NOTE: On CUDA, __funnelshift_l(x, y, s) matches the PTX shf.l behavior and acts like:
	//   (y << s) | (x >> (32 - s))
	// So we pass (low, high) in that order to get the desired 64-bit behavior.
	const uint32_t u0 = __funnelshift_l(a0, a1, s);
	const uint32_t u1 = __funnelshift_l(b1, a0, s);
	const uint32_t u2 = __funnelshift_l(b0, b1, s);
	const uint32_t hi = (k == 0u) ? u0 : u1;
	const uint32_t lo = (k == 0u) ? u1 : u2;
	return (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
}

static inline void check_cuda_contig(const torch::Tensor& t, const char* name) {
	if (!t.defined()) throw std::invalid_argument(std::string(name) + " must be defined");
	if (!t.is_cuda()) throw std::invalid_argument(std::string(name) + " must be a CUDA tensor");
	if (!t.is_contiguous()) throw std::invalid_argument(std::string(name) + " must be contiguous");
}

__device__ __forceinline__ uint32_t read_bits_msb_u64(
	const uint64_t* __restrict__ bs,
	int64_t nwords,
	int64_t bitpos,
	int L) {
	const int64_t w = bitpos >> 6;
	const int o = static_cast<int>(bitpos & 63);
	uint64_t a = 0ull;
	uint64_t b = 0ull;
	if (w >= 0 && w < nwords) a = bs[w];
	if ((w + 1) >= 0 && (w + 1) < nwords) b = bs[w + 1];
	unsigned __int128 window = (static_cast<unsigned __int128>(a) << 64) | static_cast<unsigned __int128>(b);
	const int shift = 128 - (o + L);
	const uint32_t mask = (L == 32) ? 0xFFFFFFFFu : ((1u << L) - 1u);
	return static_cast<uint32_t>((window >> shift) & mask);
}

__device__ __forceinline__ uint64_t load_u64_or_zero(
	const uint64_t* __restrict__ bs,
	int64_t nwords,
	int64_t idx) {
	if (idx >= 0 && idx < nwords) return bs[idx];
	return 0ull;
}

// Register-window MSB reader specialized for L=7.
// Keeps a 128-bit window (a,b) cached per thread and reloads only when the 64-bit word index changes.
__device__ __forceinline__ uint32_t read_bits_msb_window_L7(
	const uint64_t* __restrict__ bs,
	int64_t nwords,
	int64_t bitpos,
	int64_t& cached_w,
	uint64_t& a,
	uint64_t& b) {
	const int64_t w = bitpos >> 6;
	if (w != cached_w) {
		cached_w = w;
		a = load_u64_or_zero(bs, nwords, w);
		b = load_u64_or_zero(bs, nwords, w + 1);
	}
	const int o = static_cast<int>(bitpos & 63);
	const uint64_t hi = shf_l_wrap_u64(a, b, o);
	return static_cast<uint32_t>((hi >> kFixedShift) & 0x7Fu);
}

// Incremental MSB bit-reader for the L=7 table lookup.
// Similar spirit to load.cu: keep (word_idx, bit_off) as state and refill only when crossing word boundary.
__device__ __forceinline__ uint32_t peek7_msb_inc_L7(
	const uint64_t* __restrict__ bs,
	int64_t nwords,
	int64_t& w,
	int& o,
	uint64_t& a,
	uint64_t& b) {
	(void)bs;
	(void)nwords;
	(void)w;
	const uint64_t hi = shf_l_wrap_u64(a, b, o);
	return static_cast<uint32_t>((hi >> kFixedShift) & 0x7Fu);
}

__device__ __forceinline__ uint32_t peek16_msb_inc(
	const uint64_t* __restrict__ bs,
	int64_t nwords,
	int64_t& w,
	int& o,
	uint64_t& a,
	uint64_t& b) {
	(void)bs;
	(void)nwords;
	(void)w;
	const uint64_t hi = shf_l_wrap_u64(a, b, o);
	return static_cast<uint32_t>((hi >> (64 - 16)) & 0xFFFFu);
}

__device__ __forceinline__ uint32_t peek11_msb_inc_L11(
	const uint64_t* __restrict__ bs,
	int64_t nwords,
	int64_t& w,
	int& o,
	uint64_t& a,
	uint64_t& b) {
	(void)bs;
	(void)nwords;
	(void)w;
	const uint64_t hi = shf_l_wrap_u64(a, b, o);
	return static_cast<uint32_t>((hi >> kAwqShift) & ((1u << kAwqFixedL) - 1u));
}

__device__ __forceinline__ void consume_msb_inc(
	const uint64_t* __restrict__ bs,
	int64_t nwords,
	int64_t& w,
	int& o,
	uint64_t& a,
	uint64_t& b,
	int nbits) {
	o += nbits;
	if (o >= 64) {
		o -= 64;
		w += 1;
		a = b;
		b = load_u64_or_zero(bs, nwords, w + 1);
	}
}

__device__ __forceinline__ uint8_t unpack_len_4bit(
	const uint8_t* __restrict__ packed,
	int64_t packed_len_size,
	int64_t table_id,
	uint32_t t) {
	const int64_t byte_idx = static_cast<int64_t>(t) >> 1;
	const uint8_t v = packed[table_id * packed_len_size + byte_idx];
	return (t & 1u) ? (v >> 4) : (v & 0x0Fu);
}

__device__ __forceinline__ size_t align_up_size_t(size_t x, size_t a) {
	return (x + (a - 1)) & ~(a - 1);
}

// One block handles one bucket.
// - Shared memory caches the bucket's fp8->table mapping and the bucket-used decode tables.
// - This kernel is specialized for L=7 (table size 128).
__global__ void huffman_decode_fp8_bucket_kernel(
	const uint8_t* __restrict__ fp8_u8,
	int64_t n,
	int64_t m,
	const uint64_t* __restrict__ bitstream_u64,
	int64_t bitstream_words,
	const uint16_t* __restrict__ decode_syms_u16,
	const uint8_t* __restrict__ decode_lens_u8_packed,
	const int32_t* __restrict__ bucket_table_i32,
	const int32_t* __restrict__ bucket_col_start_i32,
	const int32_t* __restrict__ bucket_col_len_i32,
	const int32_t* __restrict__ bucket_row_offsets_i32,
	const int32_t* __restrict__ row_indices_i32,
	int32_t row_shared_cap,
	const uint32_t* __restrict__ bucket_thread_startbit_u32,
	const int64_t* __restrict__ bucket_bit_base_i64,
	uint16_t* __restrict__ out_bf16_u16) {
	const int64_t bi = static_cast<int64_t>(blockIdx.x);
	const int tid = static_cast<int>(threadIdx.x);
	extern __shared__ uint8_t smem_raw[];
	uint8_t* sm = smem_raw;
	size_t off = 0;

	off = align_up_size_t(off, alignof(int32_t));
	int32_t* sm_fp8_to_gid = reinterpret_cast<int32_t*>(sm + off);
	off += 256 * sizeof(int32_t);

	off = align_up_size_t(off, alignof(uint16_t));
	uint16_t* sm_decode_syms = reinterpret_cast<uint16_t*>(sm + off);
	off += static_cast<size_t>(kMaxTablesPerBucket) * kFixedTableSize * sizeof(uint16_t);

	uint8_t* sm_decode_lens = reinterpret_cast<uint8_t*>(sm + off);
	off += static_cast<size_t>(kMaxTablesPerBucket) * kFixedPackedLenSize * sizeof(uint8_t);

	off = align_up_size_t(off, alignof(int32_t));
	int32_t* sm_row_indices = reinterpret_cast<int32_t*>(sm + off);
	off += static_cast<size_t>(row_shared_cap) * sizeof(int32_t);

    
    uint64_t* sm_bitstream = reinterpret_cast<uint64_t*>(sm + off);
    bool flag = 0;
    if (bitstream_words < 10 * 1024) {
        flag = 1;
        off = align_up_size_t(off, alignof(int64_t));
        sm_bitstream = reinterpret_cast<uint64_t*>(sm + off);
        off += static_cast<size_t>(bitstream_words) * sizeof(int64_t);
        for (int64_t i = static_cast<int64_t>(tid); i < bitstream_words; i += static_cast<int64_t>(blockDim.x)) {
            sm_bitstream[i] = bitstream_u64[i];
        }
    }
    // printf("Shared memory used: %u bytes\n", off);
	// Load fp8->gid mapping into shared.
	for (int i = tid; i < 256; i += static_cast<int>(blockDim.x)) {
		sm_fp8_to_gid[i] = bucket_table_i32[bi * 256 + i];
	}
	__syncthreads();

	// Load ALL 256 fp8 decode tables into shared memory.
	// Layout: sm_decode_syms[fp8, t] where t in [0,128)
	for (int32_t j = tid; j < kMaxTablesPerBucket * kFixedTableSize; j += static_cast<int32_t>(blockDim.x)) {
		const int32_t fp8 = j / kFixedTableSize;
		const int32_t t = j - fp8 * kFixedTableSize;
		const int32_t gid = sm_fp8_to_gid[fp8];
		sm_decode_syms[static_cast<size_t>(fp8) * kFixedTableSize + static_cast<size_t>(t)] =
			(gid >= 0)
				? decode_syms_u16[static_cast<int64_t>(gid) * kFixedTableSize + static_cast<int64_t>(t)]
				: static_cast<uint16_t>(0);
	}
	// Layout: sm_decode_lens[fp8, byte] where byte in [0,64)
	for (int32_t j = tid; j < kMaxTablesPerBucket * kFixedPackedLenSize; j += static_cast<int32_t>(blockDim.x)) {
		const int32_t fp8 = j / kFixedPackedLenSize;
		const int32_t byte_idx = j - fp8 * kFixedPackedLenSize;
		const int32_t gid = sm_fp8_to_gid[fp8];
		sm_decode_lens[static_cast<size_t>(fp8) * kFixedPackedLenSize + static_cast<size_t>(byte_idx)] =
			(gid >= 0)
				? decode_lens_u8_packed[static_cast<int64_t>(gid) * kFixedPackedLenSize + static_cast<int64_t>(byte_idx)]
				: static_cast<uint8_t>(0);
	}
	__syncthreads();

	const int32_t row_a = bucket_row_offsets_i32[bi];
	const int32_t row_b = bucket_row_offsets_i32[bi + 1];
	const int32_t row_count = row_b - row_a;
	if (row_count <= 0) return;

	const int32_t col_start = bucket_col_start_i32[bi];
	const int32_t col_len = bucket_col_len_i32[bi];
	if (col_len <= 0) return;
	// Basic bounds check (avoid OOB on mispacked buckets).
	if (col_start < 0 || static_cast<int64_t>(col_start) + static_cast<int64_t>(col_len) > m) return;

	// Prefetch row indices for this bucket into shared memory when it fits entirely.
	const bool rows_all_shared = (row_shared_cap > 0) && (row_count <= row_shared_cap);
	if (rows_all_shared) {
		for (int32_t i = static_cast<int32_t>(tid); i < row_count; i += static_cast<int32_t>(blockDim.x)) {
			sm_row_indices[i] = __ldg(row_indices_i32 + static_cast<int64_t>(row_a + i));
		}
		__syncthreads();
	}

    // if(tid == 0) printf("block: %lld, size: %d\n", bi, row_count);
	const int64_t total_elems = static_cast<int64_t>(row_count) * static_cast<int64_t>(col_len);
	const int32_t k_threads = static_cast<int32_t>(blockDim.x);
	int32_t k_eff = k_threads;
	if (total_elems < static_cast<int64_t>(k_eff)) k_eff = static_cast<int32_t>(total_elems);
	if (k_eff < 1) k_eff = 1;
	if (tid >= k_eff) return;

	// Element assignment must match the encoder:
	// each thread decodes strided elements e = tid + i*k_eff.

	const int64_t bucket_base = bucket_bit_base_i64[bi];
	const uint32_t rel_start = bucket_thread_startbit_u32[bi * (static_cast<int64_t>(k_threads) + 1) + static_cast<int64_t>(tid)];
    const uint32_t next_start = bucket_thread_startbit_u32[bi * (static_cast<int64_t>(k_threads) + 1) + static_cast<int64_t>(tid + 1)];
	const int64_t bitpos0 = bucket_base + static_cast<int64_t>(rel_start);
    const int64_t bitpos1 = bucket_base + static_cast<int64_t>(next_start);
    const uint32_t bitlen = next_start - rel_start;
    // assert(bitlen <= 256 || printf("Huffman decode: %d bits assigned to thread exceeds 256", bitlen));
#if HUFFMAN_L7_READER_MODE == 0
	int64_t bitpos = bitpos0;
	int64_t cached_w = -1;
	uint64_t a = 0ull;
	uint64_t b = 0ull;
#else
	int64_t w = bitpos0 >> 6;
	int o = static_cast<int>(bitpos0 & 63);
	uint64_t a = load_u64_or_zero(flag ? sm_bitstream : bitstream_u64, bitstream_words, w);
	uint64_t b = load_u64_or_zero(flag ? sm_bitstream : bitstream_u64, bitstream_words, w + 1);
#endif

	// Use 32-bit rr/cc and update incrementally to avoid per-iter 64-bit div/mod.
	const int32_t m32 = static_cast<int32_t>(col_len);
	int32_t rr32 = 0;
	int32_t cc32 = tid;
	int64_t e = static_cast<int64_t>(tid);
	int32_t last_rr32 = -1;
	int32_t row = 0;
	int64_t row_base = 0;

	while (e < total_elems) {
		// Cache row index and row_base; rr changes only when cc wraps.
		if (rr32 != last_rr32) {
			row = rows_all_shared
				? sm_row_indices[rr32]
				: __ldg(row_indices_i32 + static_cast<int64_t>(row_a + rr32));
            // row = __ldg(row_indices_i32 + static_cast<int64_t>(row_a + rr32));
			row_base = static_cast<int64_t>(row) * m;
			last_rr32 = rr32;
		}
		const int64_t idx = row_base + static_cast<int64_t>(col_start + cc32);
		const uint32_t fp8 = static_cast<uint32_t>(__ldg(fp8_u8 + idx));

		uint16_t t;
#if HUFFMAN_L7_READER_MODE == 0
		t = static_cast<uint16_t>(read_bits_msb_window_L7(bitstream_u64, bitstream_words, bitpos, cached_w, a, b));
#else
		t = static_cast<uint16_t>(peek7_msb_inc_L7(bitstream_u64, bitstream_words, w, o, a, b));
#endif
		const uint8_t* lens = sm_decode_lens + (static_cast<size_t>(fp8) << 6);  // *64
		const uint16_t* syms = sm_decode_syms + (static_cast<size_t>(fp8) << 7); // *128
		const uint8_t v = lens[t >> 1];
		const uint8_t nbits_raw = static_cast<uint8_t>((v >> ((t & 1u) << 2)) & 0x0Fu);
		// const bool is_esc = ((nbits_raw & 0x8u) != 0u);
		const uint8_t nbits = static_cast<uint8_t>(nbits_raw & 0x7u);
        // assert(is_esc==false);
        out_bf16_u16[idx] = syms[t];
        
        consume_msb_inc(flag ? sm_bitstream : bitstream_u64, bitstream_words, w, o, a, b, static_cast<int>(nbits));


		e += static_cast<int64_t>(k_eff);
		int32_t cc_next = cc32 + k_eff;
		if (cc_next >= m32) {
			cc_next -= m32;
			rr32 += 1;
			// Handle rare case when m < k_eff (may wrap multiple rows).
			// if (cc_next >= m32) {
			// 	const int32_t q = cc_next / m32;
			// 	rr32 += q;
			// 	cc_next -= q * m32;
			// }
		}
		cc32 = cc_next;
	}
}

// One block handles one bucket for AWQ int4.
// - Shared memory caches qv->table mapping (16 entries) and the 16 decode tables.
// - Specialized for L=11 (table size 2048) with packed 4-bit lengths.
// - ESC is indicated by len_nibble==15, and always consumes L bits then reads raw16.
__global__ void huffman_decode_awq_int4_bucket_kernel(
	const uint8_t* __restrict__ int4_u8,
	int64_t n,
	int64_t m,
	const uint64_t* __restrict__ bitstream_u64,
	int64_t bitstream_words,
	const uint16_t* __restrict__ decode_syms_u16,
	const uint8_t* __restrict__ decode_lens_u8_packed,
	const int32_t* __restrict__ bucket_table_i32,      // [B,16]
	const int32_t* __restrict__ bucket_col_start_i32,
	const int32_t* __restrict__ bucket_col_len_i32,
	const int32_t* __restrict__ bucket_row_offsets_i32,
	const int32_t* __restrict__ row_indices_i32,
	int32_t row_shared_cap,
	const uint32_t* __restrict__ bucket_thread_startbit_u32,
	const int64_t* __restrict__ bucket_bit_base_i64,
	uint16_t* __restrict__ out_bf16_u16) {
	const int64_t bi = static_cast<int64_t>(blockIdx.x);
	const int tid = static_cast<int>(threadIdx.x);
	extern __shared__ uint8_t smem_raw[];
	uint8_t* sm = smem_raw;
	size_t off = 0;

	off = align_up_size_t(off, alignof(int32_t));
	int32_t* sm_qv_to_gid = reinterpret_cast<int32_t*>(sm + off);
	off += 16 * sizeof(int32_t);

	off = align_up_size_t(off, alignof(int32_t));
	int32_t* sm_row_indices = reinterpret_cast<int32_t*>(sm + off);
	off += static_cast<size_t>(row_shared_cap) * sizeof(int32_t);

	// Load qv->gid mapping into shared.
	for (int i = tid; i < 16; i += static_cast<int>(blockDim.x)) {
		sm_qv_to_gid[i] = bucket_table_i32[bi * 16 + i];
	}
	__syncthreads();

	const int32_t row_a = bucket_row_offsets_i32[bi];
	const int32_t row_b = bucket_row_offsets_i32[bi + 1];
	const int32_t row_count = row_b - row_a;
	if (row_count <= 0) return;

	const int32_t col_start = bucket_col_start_i32[bi];
	const int32_t col_len = bucket_col_len_i32[bi];
	if (col_len <= 0) return;
	if (col_start < 0 || static_cast<int64_t>(col_start) + static_cast<int64_t>(col_len) > m) return;

	const bool rows_all_shared = (row_shared_cap > 0) && (row_count <= row_shared_cap);
	if (rows_all_shared) {
		for (int32_t i = static_cast<int32_t>(tid); i < row_count; i += static_cast<int32_t>(blockDim.x)) {
			sm_row_indices[i] = __ldg(row_indices_i32 + static_cast<int64_t>(row_a + i));
		}
		__syncthreads();
	}

	const int64_t total_elems = static_cast<int64_t>(row_count) * static_cast<int64_t>(col_len);
	const int32_t k_threads = static_cast<int32_t>(blockDim.x);
	int32_t k_eff = k_threads;
	if (total_elems < static_cast<int64_t>(k_eff)) k_eff = static_cast<int32_t>(total_elems);
	if (k_eff < 1) k_eff = 1;
	if (tid >= k_eff) return;

	const int64_t bucket_base = bucket_bit_base_i64[bi];
	const uint32_t rel_start = bucket_thread_startbit_u32[bi * (static_cast<int64_t>(k_threads) + 1) + static_cast<int64_t>(tid)];
	const int64_t bitpos0 = bucket_base + static_cast<int64_t>(rel_start);

	int64_t w = bitpos0 >> 6;
	int o = static_cast<int>(bitpos0 & 63);
	uint64_t a = load_u64_or_zero(bitstream_u64, bitstream_words, w);
	uint64_t b = load_u64_or_zero(bitstream_u64, bitstream_words, w + 1);

	const int32_t m32 = static_cast<int32_t>(col_len);
	int32_t rr32 = 0;
	int32_t cc32 = tid;
	int64_t e = static_cast<int64_t>(tid);
	int32_t last_rr32 = -1;
	int32_t row = 0;
	int64_t row_base = 0;

	while (e < total_elems) {
		if (rr32 != last_rr32) {
			row = __ldg(row_indices_i32 + static_cast<int64_t>(row_a + rr32));
			row_base = static_cast<int64_t>(row) * m;
			last_rr32 = rr32;
		}
		const int64_t idx = row_base + static_cast<int64_t>(col_start + cc32);
		const uint32_t qv = static_cast<uint32_t>(__ldg(int4_u8 + idx)) & 0x0Fu;
		const int32_t gid = sm_qv_to_gid[static_cast<int>(qv)];

		const uint32_t t = peek11_msb_inc_L11(bitstream_u64, bitstream_words, w, o, a, b);
		const uint8_t* lens = (gid >= 0)
			? (decode_lens_u8_packed + static_cast<int64_t>(gid) * kAwqPackedLenSize)
			: nullptr;
		const uint16_t* syms = (gid >= 0)
			? (decode_syms_u16 + static_cast<int64_t>(gid) * kAwqTableSize)
			: nullptr;
		const uint8_t v = (gid >= 0) ? __ldg(lens + (t >> 1)) : static_cast<uint8_t>(0);
		const uint8_t nbits_raw = static_cast<uint8_t>((v >> ((t & 1u) << 2)) & 0x0Fu);
		if (gid >= 0 && nbits_raw != 15u) {
			out_bf16_u16[idx] = __ldg(syms + t);
			consume_msb_inc(bitstream_u64, bitstream_words, w, o, a, b, static_cast<int>(nbits_raw));
		} else {
			// ESC: consume L bits (fixed 11), then read raw 16-bit bf16 symbol bits.
			consume_msb_inc(bitstream_u64, bitstream_words, w, o, a, b, kAwqFixedL);
			const uint16_t raw = static_cast<uint16_t>(peek16_msb_inc(bitstream_u64, bitstream_words, w, o, a, b));
			consume_msb_inc(bitstream_u64, bitstream_words, w, o, a, b, 16);
			out_bf16_u16[idx] = raw;
		}

		e += static_cast<int64_t>(k_eff);
		int32_t cc_next = cc32 + k_eff;
		if (cc_next >= m32) {
			cc_next -= m32;
			rr32 += 1;
		}
		cc32 = cc_next;
	}
}

// One block handles one ragged bucket for AWQ int4 (bucketed by exact scale).
// - Bucket contains a list of quantization groups (row, col_start).
// - Each group expands to group_size consecutive elements.
// - Shared memory caches qv->gid mapping (16 entries).
// - Specialized for L=11 (table size 2048) with packed 4-bit lengths.
// - ESC is indicated by len_nibble==15, and always consumes L bits then reads raw16.
__global__ void huffman_decode_awq_int4_bucket_by_scale_kernel(
	const uint8_t* __restrict__ int4_u8,
	int64_t n,
	int64_t m,
	int32_t group_size,
	const uint64_t* __restrict__ bitstream_u64,
	int64_t bitstream_words,
	const uint16_t* __restrict__ decode_syms_u16,
	const uint8_t* __restrict__ decode_lens_u8_packed,
	const int32_t* __restrict__ bucket_table_i32,            // [B,16]
	const int32_t* __restrict__ bucket_group_offsets_i32,    // [B+1]
	const int32_t* __restrict__ group_rows_i32,              // [NG]
	const int32_t* __restrict__ group_col_start_i32,         // [NG]
	const uint32_t* __restrict__ bucket_thread_startbit_u32, // [B,k+1]
	const int64_t* __restrict__ bucket_bit_base_i64,         // [B+1]
	uint16_t* __restrict__ out_bf16_u16) {
	const int64_t bi = static_cast<int64_t>(blockIdx.x);
	const int tid = static_cast<int>(threadIdx.x);
	extern __shared__ uint8_t smem_raw[];
	uint8_t* sm = smem_raw;
	size_t off = 0;

	off = align_up_size_t(off, alignof(int32_t));
	int32_t* sm_qv_to_gid = reinterpret_cast<int32_t*>(sm + off);
	off += 16 * sizeof(int32_t);

	for (int i = tid; i < 16; i += static_cast<int>(blockDim.x)) {
		sm_qv_to_gid[i] = bucket_table_i32[bi * 16 + i];
	}
	__syncthreads();

	const int32_t ga = bucket_group_offsets_i32[bi];
	const int32_t gb = bucket_group_offsets_i32[bi + 1];
	const int32_t group_count = gb - ga;
	if (group_count <= 0) return;
	if (group_size <= 0) return;

	const int64_t total_elems = static_cast<int64_t>(group_count) * static_cast<int64_t>(group_size);
	const int32_t k_threads = static_cast<int32_t>(blockDim.x);
	int32_t k_eff = k_threads;
	if (total_elems < static_cast<int64_t>(k_eff)) k_eff = static_cast<int32_t>(total_elems);
	if (k_eff < 1) k_eff = 1;
	if (tid >= k_eff) return;

	const int64_t bucket_base = bucket_bit_base_i64[bi];
	const uint32_t rel_start = bucket_thread_startbit_u32[bi * (static_cast<int64_t>(k_threads) + 1) + static_cast<int64_t>(tid)];
	const int64_t bitpos0 = bucket_base + static_cast<int64_t>(rel_start);

	int64_t w = bitpos0 >> 6;
	int o = static_cast<int>(bitpos0 & 63);
	uint64_t a = load_u64_or_zero(bitstream_u64, bitstream_words, w);
	uint64_t b = load_u64_or_zero(bitstream_u64, bitstream_words, w + 1);

	int64_t e = static_cast<int64_t>(tid);
	while (e < total_elems) {
		const int64_t gi = e / static_cast<int64_t>(group_size);
		const int64_t j = e - gi * static_cast<int64_t>(group_size);
		const int32_t row = __ldg(group_rows_i32 + static_cast<int64_t>(ga) + gi);
		const int32_t c0 = __ldg(group_col_start_i32 + static_cast<int64_t>(ga) + gi);
		const int64_t idx = static_cast<int64_t>(row) * m + static_cast<int64_t>(c0) + j;

		const uint32_t qv = static_cast<uint32_t>(__ldg(int4_u8 + idx)) & 0x0Fu;
		const int32_t gid = sm_qv_to_gid[static_cast<int>(qv)];

		const uint32_t t = peek11_msb_inc_L11(bitstream_u64, bitstream_words, w, o, a, b);
		const uint8_t* lens = (gid >= 0)
			? (decode_lens_u8_packed + static_cast<int64_t>(gid) * kAwqPackedLenSize)
			: nullptr;
		const uint16_t* syms = (gid >= 0)
			? (decode_syms_u16 + static_cast<int64_t>(gid) * kAwqTableSize)
			: nullptr;
		const uint8_t v = (gid >= 0) ? __ldg(lens + (t >> 1)) : static_cast<uint8_t>(0);
		const uint8_t nbits_raw = static_cast<uint8_t>((v >> ((t & 1u) << 2)) & 0x0Fu);

		if (gid >= 0 && nbits_raw != 15u) {
			out_bf16_u16[idx] = __ldg(syms + t);
			consume_msb_inc(bitstream_u64, bitstream_words, w, o, a, b, static_cast<int>(nbits_raw));
		} else {
			consume_msb_inc(bitstream_u64, bitstream_words, w, o, a, b, kAwqFixedL);
			const uint16_t raw = static_cast<uint16_t>(peek16_msb_inc(bitstream_u64, bitstream_words, w, o, a, b));
			consume_msb_inc(bitstream_u64, bitstream_words, w, o, a, b, 16);
			out_bf16_u16[idx] = raw;
		}

		e += static_cast<int64_t>(k_eff);
	}
}

torch::Tensor huffman_decode_fp8_entry(
	torch::Tensor fp8_params_u8,
	torch::Tensor bitstream_u64,
	torch::Tensor decode_syms_u16,
	torch::Tensor decode_lens_u8,
	torch::Tensor bucket_table_i32,
	torch::Tensor bucket_col_start_i32,
	torch::Tensor bucket_col_len_i32,
	torch::Tensor bucket_row_offsets_i32,
	torch::Tensor row_indices_i32,
	torch::Tensor bucket_thread_startbit_u32,
	torch::Tensor bucket_bit_base_i64,
	int64_t n,
	int64_t m,
	int64_t max_len) {
	check_cuda_contig(fp8_params_u8, "fp8_params_u8");
	check_cuda_contig(bitstream_u64, "bitstream_u64");
	check_cuda_contig(decode_syms_u16, "decode_syms_u16");
	check_cuda_contig(decode_lens_u8, "decode_lens_u8");
	check_cuda_contig(bucket_table_i32, "bucket_table_i32");
	check_cuda_contig(bucket_col_start_i32, "bucket_col_start_i32");
	check_cuda_contig(bucket_col_len_i32, "bucket_col_len_i32");
	check_cuda_contig(bucket_row_offsets_i32, "bucket_row_offsets_i32");
	check_cuda_contig(row_indices_i32, "row_indices_i32");
	check_cuda_contig(bucket_thread_startbit_u32, "bucket_thread_startbit_u32");
	check_cuda_contig(bucket_bit_base_i64, "bucket_bit_base_i64");

	if (fp8_params_u8.scalar_type() != torch::kUInt8) {
		throw std::invalid_argument("fp8_params_u8 must be uint8 CUDA");
	}
	if (bitstream_u64.scalar_type() != torch::kUInt64 || bitstream_u64.dim() != 1) {
		throw std::invalid_argument("bitstream_u64 must be uint64 1D CUDA");
	}
	if (decode_syms_u16.scalar_type() != torch::kUInt16 || decode_syms_u16.dim() != 2) {
		throw std::invalid_argument("decode_syms_u16 must be uint16 2D CUDA [G, 2^L]");
	}
	if (decode_lens_u8.scalar_type() != torch::kUInt8 || decode_lens_u8.dim() != 2) {
		throw std::invalid_argument("decode_lens_u8 must be uint8 2D CUDA [G, 2^(L-1)]");
	}
	if (bucket_table_i32.scalar_type() != torch::kInt32 || bucket_table_i32.dim() != 2 || bucket_table_i32.size(1) != 256) {
		throw std::invalid_argument("bucket_table_i32 must be int32 CUDA [B,256]");
	}
	if (bucket_col_start_i32.scalar_type() != torch::kInt32 || bucket_col_start_i32.dim() != 1) {
		throw std::invalid_argument("bucket_col_start_i32 must be int32 1D CUDA [B]");
	}
	if (bucket_col_len_i32.scalar_type() != torch::kInt32 || bucket_col_len_i32.dim() != 1) {
		throw std::invalid_argument("bucket_col_len_i32 must be int32 1D CUDA [B]");
	}
	if (bucket_row_offsets_i32.scalar_type() != torch::kInt32 || bucket_row_offsets_i32.dim() != 1) {
		throw std::invalid_argument("bucket_row_offsets_i32 must be int32 1D CUDA [B+1]");
	}
	if (row_indices_i32.scalar_type() != torch::kInt32 || row_indices_i32.dim() != 1) {
		throw std::invalid_argument("row_indices_i32 must be int32 1D CUDA");
	}
	if (bucket_thread_startbit_u32.scalar_type() != torch::kUInt32 || bucket_thread_startbit_u32.dim() != 2) {
		throw std::invalid_argument("bucket_thread_startbit_u32 must be uint32 2D CUDA [B,k+1]");
	}
	if (bucket_bit_base_i64.scalar_type() != torch::kInt64 || bucket_bit_base_i64.dim() != 1) {
		throw std::invalid_argument("bucket_bit_base_i64 must be int64 1D CUDA [B+1]");
	}

	if (max_len <= 0 || max_len > 15) {
		throw std::invalid_argument("max_len must be in [1,15]");
	}
	if (max_len != kFixedL) {
		throw std::invalid_argument("decode kernel is specialized: max_len must be 7");
	}
	if (n <= 0 || m <= 0) {
		throw std::invalid_argument("n,m must be > 0");
	}
	if (fp8_params_u8.numel() != n * m) {
		throw std::invalid_argument("fp8_params_u8 numel must equal n*m");
	}

	const int64_t B = bucket_table_i32.size(0);
	if (bucket_col_start_i32.numel() != B) {
		throw std::invalid_argument("bucket_col_start_i32 length must be B");
	}
	if (bucket_col_len_i32.numel() != B) {
		throw std::invalid_argument("bucket_col_len_i32 length must be B");
	}
	if (bucket_row_offsets_i32.numel() != B + 1) {
		throw std::invalid_argument("bucket_row_offsets_i32 length must be B+1");
	}
	if (bucket_thread_startbit_u32.size(0) != B) {
		throw std::invalid_argument("bucket_thread_startbit_u32 first dim must be B");
	}
	if (bucket_bit_base_i64.numel() != B + 1) {
		throw std::invalid_argument("bucket_bit_base_i64 length must be B+1");
	}

	const int64_t table_size = decode_syms_u16.size(1);
	const int64_t packed_len_size = decode_lens_u8.size(1);
	if (table_size != kFixedTableSize) {
		throw std::invalid_argument("decode_syms_u16 second dim must be 2^max_len");
	}
	if (packed_len_size != kFixedPackedLenSize) {
		throw std::invalid_argument("decode_lens_u8 second dim must be 2^(max_len-1)");
	}
	if (decode_syms_u16.size(0) != decode_lens_u8.size(0)) {
		throw std::invalid_argument("decode_syms_u16 and decode_lens_u8 first dim must match");
	}

	const int threads = static_cast<int>(bucket_thread_startbit_u32.size(1) - 1);
	if (threads <= 0 || threads > 1024) {
		throw std::invalid_argument("bucket_thread_startbit_u32 has invalid k (must be 1..1024)");
	}

	auto out = torch::empty({n, m}, torch::TensorOptions().dtype(torch::kBFloat16).device(fp8_params_u8.device()));

	// Worst-case shared memory:
	// - fp8->gid (256 * int32)
	// - decode_syms (256 * 128 * uint16)
	// - decode_lens packed (256 * 64 * uint8)
	const int base_shared_bytes =
		256 * static_cast<int>(sizeof(int32_t)) +
		kMaxTablesPerBucket * kFixedTableSize * static_cast<int>(sizeof(uint16_t)) +
		kMaxTablesPerBucket * kFixedPackedLenSize * static_cast<int>(sizeof(uint8_t));
	const int base_shared_aligned = (base_shared_bytes + 3) & ~3;

	int dev = 0;
	(void)cudaGetDevice(&dev);
	int max_shared_optin = 0;
	cudaError_t attr_err = cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	if (attr_err != cudaSuccess) {
		max_shared_optin = 0;
	}

	// Try to place per-bucket row_indices into shared memory.
	// Cap by what the device allows for opt-in dynamic shared.
	int32_t row_shared_cap = 0;
	if (max_shared_optin > base_shared_aligned) {
		const int64_t avail_rows = static_cast<int64_t>(max_shared_optin - base_shared_aligned) / static_cast<int64_t>(sizeof(int32_t));
		const int64_t want_rows = (n < static_cast<int64_t>(HUFFMAN_ROW_SHARED_CAP)) ? n : static_cast<int64_t>(HUFFMAN_ROW_SHARED_CAP);
		int64_t cap_rows = (want_rows < avail_rows) ? want_rows : avail_rows;
		if (cap_rows < 0) cap_rows = 0;
		if (cap_rows > static_cast<int64_t>(INT32_MAX)) cap_rows = static_cast<int64_t>(INT32_MAX);
		row_shared_cap = static_cast<int32_t>(cap_rows);
	}

	const int shared_bytes = base_shared_aligned + static_cast<int>(static_cast<int64_t>(row_shared_cap) * static_cast<int64_t>(sizeof(int32_t))) 
                             + 3 * 1024 * 8; // extra buffer for L7 reader
	cudaError_t err = cudaFuncSetAttribute(
		huffman_decode_fp8_bucket_kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		shared_bytes);
	(void)err;  // best-effort

	huffman_decode_fp8_bucket_kernel<<<static_cast<int>(B), threads, shared_bytes>>>(
		fp8_params_u8.data_ptr<uint8_t>(),
		n,
		m,
		bitstream_u64.data_ptr<uint64_t>(),
		bitstream_u64.numel(),
		decode_syms_u16.data_ptr<uint16_t>(),
		decode_lens_u8.data_ptr<uint8_t>(),
		bucket_table_i32.data_ptr<int32_t>(),
		bucket_col_start_i32.data_ptr<int32_t>(),
		bucket_col_len_i32.data_ptr<int32_t>(),
		bucket_row_offsets_i32.data_ptr<int32_t>(),
		row_indices_i32.data_ptr<int32_t>(),
		row_shared_cap,
		bucket_thread_startbit_u32.data_ptr<uint32_t>(),
		bucket_bit_base_i64.data_ptr<int64_t>(),
		reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()));

	return out;
}

torch::Tensor huffman_decode_awq_int4_bucket_by_scale_entry(
	torch::Tensor int4_values_u8,
	torch::Tensor bitstream_u64,
	torch::Tensor decode_syms_u16,
	torch::Tensor decode_lens_u8,
	torch::Tensor bucket_table_i32,
	torch::Tensor bucket_group_offsets_i32,
	torch::Tensor group_rows_i32,
	torch::Tensor group_col_start_i32,
	torch::Tensor bucket_thread_startbit_u32,
	torch::Tensor bucket_bit_base_i64,
	int64_t n,
	int64_t m,
	int64_t max_len,
	int64_t group_size) {
	check_cuda_contig(int4_values_u8, "int4_values_u8");
	check_cuda_contig(bitstream_u64, "bitstream_u64");
	check_cuda_contig(decode_syms_u16, "decode_syms_u16");
	check_cuda_contig(decode_lens_u8, "decode_lens_u8");
	check_cuda_contig(bucket_table_i32, "bucket_table_i32");
	check_cuda_contig(bucket_group_offsets_i32, "bucket_group_offsets_i32");
	check_cuda_contig(group_rows_i32, "group_rows_i32");
	check_cuda_contig(group_col_start_i32, "group_col_start_i32");
	check_cuda_contig(bucket_thread_startbit_u32, "bucket_thread_startbit_u32");
	check_cuda_contig(bucket_bit_base_i64, "bucket_bit_base_i64");

	if (int4_values_u8.scalar_type() != torch::kUInt8 || int4_values_u8.dim() != 2) {
		throw std::invalid_argument("int4_values_u8 must be uint8 2D CUDA [n,m]");
	}
	if (bitstream_u64.scalar_type() != torch::kUInt64 || bitstream_u64.dim() != 1) {
		throw std::invalid_argument("bitstream_u64 must be uint64 1D CUDA");
	}
	if (decode_syms_u16.scalar_type() != torch::kUInt16 || decode_syms_u16.dim() != 2) {
		throw std::invalid_argument("decode_syms_u16 must be uint16 2D CUDA [G,2^L]");
	}
	if (decode_lens_u8.scalar_type() != torch::kUInt8 || decode_lens_u8.dim() != 2) {
		throw std::invalid_argument("decode_lens_u8 must be uint8 2D CUDA [G,2^(L-1)]");
	}
	if (bucket_table_i32.scalar_type() != torch::kInt32 || bucket_table_i32.dim() != 2 || bucket_table_i32.size(1) != 16) {
		throw std::invalid_argument("bucket_table_i32 must be int32 2D CUDA [B,16]");
	}
	if (bucket_group_offsets_i32.scalar_type() != torch::kInt32 || bucket_group_offsets_i32.dim() != 1) {
		throw std::invalid_argument("bucket_group_offsets_i32 must be int32 1D CUDA [B+1]");
	}
	if (group_rows_i32.scalar_type() != torch::kInt32 || group_rows_i32.dim() != 1) {
		throw std::invalid_argument("group_rows_i32 must be int32 1D CUDA");
	}
	if (group_col_start_i32.scalar_type() != torch::kInt32 || group_col_start_i32.dim() != 1) {
		throw std::invalid_argument("group_col_start_i32 must be int32 1D CUDA");
	}
	if (bucket_thread_startbit_u32.scalar_type() != torch::kUInt32 || bucket_thread_startbit_u32.dim() != 2) {
		throw std::invalid_argument("bucket_thread_startbit_u32 must be uint32 2D CUDA [B,k+1]");
	}
	if (bucket_bit_base_i64.scalar_type() != torch::kInt64 || bucket_bit_base_i64.dim() != 1) {
		throw std::invalid_argument("bucket_bit_base_i64 must be int64 1D CUDA [B+1]");
	}
	if (max_len <= 0 || max_len > 15) {
		throw std::invalid_argument("max_len must be in [1,15]");
	}
	if (max_len != kAwqFixedL) {
		throw std::invalid_argument("AWQ ragged bucket decode kernel is specialized: max_len must be 11");
	}
	if (n <= 0 || m <= 0) {
		throw std::invalid_argument("n,m must be > 0");
	}
	if (group_size <= 0 || group_size > m) {
		throw std::invalid_argument("group_size must be in (0,m]");
	}
	if (int4_values_u8.numel() != n * m) {
		throw std::invalid_argument("int4_values_u8 numel must equal n*m");
	}

	const int64_t B = bucket_table_i32.size(0);
	if (bucket_group_offsets_i32.numel() != B + 1) throw std::invalid_argument("bucket_group_offsets_i32 length must be B+1");
	if (bucket_thread_startbit_u32.size(0) != B) throw std::invalid_argument("bucket_thread_startbit_u32 first dim must be B");
	if (bucket_bit_base_i64.numel() != B + 1) throw std::invalid_argument("bucket_bit_base_i64 length must be B+1");
	if (group_rows_i32.numel() != group_col_start_i32.numel()) throw std::invalid_argument("group_rows_i32 and group_col_start_i32 length must match");

	const int64_t table_size = decode_syms_u16.size(1);
	const int64_t packed_len_size = decode_lens_u8.size(1);
	if (table_size != kAwqTableSize) throw std::invalid_argument("decode_syms_u16 second dim must be 2^max_len");
	if (packed_len_size != kAwqPackedLenSize) throw std::invalid_argument("decode_lens_u8 second dim must be 2^(max_len-1)");
	if (decode_syms_u16.size(0) != decode_lens_u8.size(0)) throw std::invalid_argument("decode_syms_u16 and decode_lens_u8 first dim must match");

	const int threads = static_cast<int>(bucket_thread_startbit_u32.size(1) - 1);
	if (threads <= 0 || threads > 1024) throw std::invalid_argument("bucket_thread_startbit_u32 has invalid k (must be 1..1024)");

	auto out = torch::empty({n, m}, torch::TensorOptions().dtype(torch::kBFloat16).device(int4_values_u8.device()));

	const int base_shared_bytes = 16 * static_cast<int>(sizeof(int32_t));
	const int shared_bytes = (base_shared_bytes + 3) & ~3;
	cudaError_t err = cudaFuncSetAttribute(
		huffman_decode_awq_int4_bucket_by_scale_kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		shared_bytes);
	(void)err;

	huffman_decode_awq_int4_bucket_by_scale_kernel<<<static_cast<int>(B), threads, shared_bytes>>>(
		int4_values_u8.data_ptr<uint8_t>(),
		n,
		m,
		static_cast<int32_t>(group_size),
		bitstream_u64.data_ptr<uint64_t>(),
		bitstream_u64.numel(),
		decode_syms_u16.data_ptr<uint16_t>(),
		decode_lens_u8.data_ptr<uint8_t>(),
		bucket_table_i32.data_ptr<int32_t>(),
		bucket_group_offsets_i32.data_ptr<int32_t>(),
		group_rows_i32.data_ptr<int32_t>(),
		group_col_start_i32.data_ptr<int32_t>(),
		bucket_thread_startbit_u32.data_ptr<uint32_t>(),
		bucket_bit_base_i64.data_ptr<int64_t>(),
		reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()));

	return out;
}

torch::Tensor huffman_decode_awq_int4_bucket_entry(
	torch::Tensor int4_values_u8,
	torch::Tensor bitstream_u64,
	torch::Tensor decode_syms_u16,
	torch::Tensor decode_lens_u8,
	torch::Tensor bucket_table_i32,
	torch::Tensor bucket_col_start_i32,
	torch::Tensor bucket_col_len_i32,
	torch::Tensor bucket_row_offsets_i32,
	torch::Tensor row_indices_i32,
	torch::Tensor bucket_thread_startbit_u32,
	torch::Tensor bucket_bit_base_i64,
	int64_t n,
	int64_t m,
	int64_t max_len) {
	check_cuda_contig(int4_values_u8, "int4_values_u8");
	check_cuda_contig(bitstream_u64, "bitstream_u64");
	check_cuda_contig(decode_syms_u16, "decode_syms_u16");
	check_cuda_contig(decode_lens_u8, "decode_lens_u8");
	check_cuda_contig(bucket_table_i32, "bucket_table_i32");
	check_cuda_contig(bucket_col_start_i32, "bucket_col_start_i32");
	check_cuda_contig(bucket_col_len_i32, "bucket_col_len_i32");
	check_cuda_contig(bucket_row_offsets_i32, "bucket_row_offsets_i32");
	check_cuda_contig(row_indices_i32, "row_indices_i32");
	check_cuda_contig(bucket_thread_startbit_u32, "bucket_thread_startbit_u32");
	check_cuda_contig(bucket_bit_base_i64, "bucket_bit_base_i64");

	if (int4_values_u8.scalar_type() != torch::kUInt8) {
		throw std::invalid_argument("int4_values_u8 must be uint8 CUDA");
	}
	if (bitstream_u64.scalar_type() != torch::kUInt64 || bitstream_u64.dim() != 1) {
		throw std::invalid_argument("bitstream_u64 must be uint64 1D CUDA");
	}
	if (decode_syms_u16.scalar_type() != torch::kUInt16 || decode_syms_u16.dim() != 2) {
		throw std::invalid_argument("decode_syms_u16 must be uint16 2D CUDA [G, 2^L]");
	}
	if (decode_lens_u8.scalar_type() != torch::kUInt8 || decode_lens_u8.dim() != 2) {
		throw std::invalid_argument("decode_lens_u8 must be uint8 2D CUDA [G, 2^(L-1)]");
	}
	if (bucket_table_i32.scalar_type() != torch::kInt32 || bucket_table_i32.dim() != 2 || bucket_table_i32.size(1) != 16) {
		throw std::invalid_argument("bucket_table_i32 must be int32 CUDA [B,16]");
	}
	if (bucket_col_start_i32.scalar_type() != torch::kInt32 || bucket_col_start_i32.dim() != 1) {
		throw std::invalid_argument("bucket_col_start_i32 must be int32 1D CUDA [B]");
	}
	if (bucket_col_len_i32.scalar_type() != torch::kInt32 || bucket_col_len_i32.dim() != 1) {
		throw std::invalid_argument("bucket_col_len_i32 must be int32 1D CUDA [B]");
	}
	if (bucket_row_offsets_i32.scalar_type() != torch::kInt32 || bucket_row_offsets_i32.dim() != 1) {
		throw std::invalid_argument("bucket_row_offsets_i32 must be int32 1D CUDA [B+1]");
	}
	if (row_indices_i32.scalar_type() != torch::kInt32 || row_indices_i32.dim() != 1) {
		throw std::invalid_argument("row_indices_i32 must be int32 1D CUDA");
	}
	if (bucket_thread_startbit_u32.scalar_type() != torch::kUInt32 || bucket_thread_startbit_u32.dim() != 2) {
		throw std::invalid_argument("bucket_thread_startbit_u32 must be uint32 2D CUDA [B,k+1]");
	}
	if (bucket_bit_base_i64.scalar_type() != torch::kInt64 || bucket_bit_base_i64.dim() != 1) {
		throw std::invalid_argument("bucket_bit_base_i64 must be int64 1D CUDA [B+1]");
	}
	if (max_len <= 0 || max_len > 15) {
		throw std::invalid_argument("max_len must be in [1,15]");
	}
	if (max_len != kAwqFixedL) {
		throw std::invalid_argument("AWQ bucket decode kernel is specialized: max_len must be 11");
	}
	if (n <= 0 || m <= 0) {
		throw std::invalid_argument("n,m must be > 0");
	}
	if (int4_values_u8.numel() != n * m) {
		throw std::invalid_argument("int4_values_u8 numel must equal n*m");
	}

	const int64_t B = bucket_table_i32.size(0);
	if (bucket_col_start_i32.numel() != B) throw std::invalid_argument("bucket_col_start_i32 length must be B");
	if (bucket_col_len_i32.numel() != B) throw std::invalid_argument("bucket_col_len_i32 length must be B");
	if (bucket_row_offsets_i32.numel() != B + 1) throw std::invalid_argument("bucket_row_offsets_i32 length must be B+1");
	if (bucket_thread_startbit_u32.size(0) != B) throw std::invalid_argument("bucket_thread_startbit_u32 first dim must be B");
	if (bucket_bit_base_i64.numel() != B + 1) throw std::invalid_argument("bucket_bit_base_i64 length must be B+1");

	const int64_t table_size = decode_syms_u16.size(1);
	const int64_t packed_len_size = decode_lens_u8.size(1);
	if (table_size != kAwqTableSize) throw std::invalid_argument("decode_syms_u16 second dim must be 2^max_len");
	if (packed_len_size != kAwqPackedLenSize) throw std::invalid_argument("decode_lens_u8 second dim must be 2^(max_len-1)");
	if (decode_syms_u16.size(0) != decode_lens_u8.size(0)) throw std::invalid_argument("decode_syms_u16 and decode_lens_u8 first dim must match");

	const int threads = static_cast<int>(bucket_thread_startbit_u32.size(1) - 1);
	if (threads <= 0 || threads > 1024) throw std::invalid_argument("bucket_thread_startbit_u32 has invalid k (must be 1..1024)");

	auto out = torch::empty({n, m}, torch::TensorOptions().dtype(torch::kBFloat16).device(int4_values_u8.device()));

	// Worst-case shared memory:
	// - qv->gid (16 * int32)
	const int base_shared_bytes = 16 * static_cast<int>(sizeof(int32_t));
	const int base_shared_aligned = (base_shared_bytes + 3) & ~3;

	int dev = 0;
	(void)cudaGetDevice(&dev);
	int max_shared_optin = 0;
	cudaError_t attr_err = cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	if (attr_err != cudaSuccess) {
		max_shared_optin = 0;
	}

	int32_t row_shared_cap = 0;
	if (max_shared_optin > base_shared_aligned) {
		const int64_t avail_rows = static_cast<int64_t>(max_shared_optin - base_shared_aligned) / static_cast<int64_t>(sizeof(int32_t));
		const int64_t want_rows = (n < static_cast<int64_t>(HUFFMAN_ROW_SHARED_CAP)) ? n : static_cast<int64_t>(HUFFMAN_ROW_SHARED_CAP);
		int64_t cap_rows = (want_rows < avail_rows) ? want_rows : avail_rows;
		if (cap_rows < 0) cap_rows = 0;
		if (cap_rows > static_cast<int64_t>(INT32_MAX)) cap_rows = static_cast<int64_t>(INT32_MAX);
		row_shared_cap = static_cast<int32_t>(cap_rows);
	}

	const int shared_bytes = base_shared_aligned + static_cast<int>(static_cast<int64_t>(row_shared_cap) * static_cast<int64_t>(sizeof(int32_t)));
	cudaError_t err = cudaFuncSetAttribute(
		huffman_decode_awq_int4_bucket_kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		shared_bytes);
	(void)err;

	huffman_decode_awq_int4_bucket_kernel<<<static_cast<int>(B), threads, shared_bytes>>>(
		int4_values_u8.data_ptr<uint8_t>(),
		n,
		m,
		bitstream_u64.data_ptr<uint64_t>(),
		bitstream_u64.numel(),
		decode_syms_u16.data_ptr<uint16_t>(),
		decode_lens_u8.data_ptr<uint8_t>(),
		bucket_table_i32.data_ptr<int32_t>(),
		bucket_col_start_i32.data_ptr<int32_t>(),
		bucket_col_len_i32.data_ptr<int32_t>(),
		bucket_row_offsets_i32.data_ptr<int32_t>(),
		row_indices_i32.data_ptr<int32_t>(),
		row_shared_cap,
		bucket_thread_startbit_u32.data_ptr<uint32_t>(),
		bucket_bit_base_i64.data_ptr<int64_t>(),
		reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>()));

	return out;
}

}  // namespace huffman_cpp

