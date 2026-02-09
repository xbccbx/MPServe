#include <torch/extension.h>

#include <cuda_runtime.h>

#include <ATen/Parallel.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace huffman_decode_cpp {

using DecodeTablesRet = std::tuple<torch::Tensor, torch::Tensor>;  // decode_syms_i16, decode_nbits_u8
using StreamChunkIndexRet = std::tuple<torch::Tensor, torch::Tensor>;  // stream_chunk_ofs_i64, stream_chunk_ids_i32

static inline void check_cpu_contig(const torch::Tensor& t, const char* name) {
	if (!t.defined()) throw std::invalid_argument(std::string(name) + " must be defined");
	if (!t.is_cpu()) throw std::invalid_argument(std::string(name) + " must be a CPU tensor");
	if (!t.is_contiguous()) throw std::invalid_argument(std::string(name) + " must be contiguous");
}

static StreamChunkIndexRet huffman_build_stream_chunk_index(
	torch::Tensor chunk_meta_u32,
	torch::Tensor chunk_scale_id_i32,
	int64_t num_scales) {
	check_cpu_contig(chunk_meta_u32, "chunk_meta_u32");
	check_cpu_contig(chunk_scale_id_i32, "chunk_scale_id_i32");
	if (chunk_meta_u32.scalar_type() != torch::kUInt32 || chunk_meta_u32.dim() != 1) {
		throw std::invalid_argument("chunk_meta_u32 must be uint32 1D (CPU)");
	}
	if (chunk_scale_id_i32.scalar_type() != torch::kInt32 || chunk_scale_id_i32.dim() != 1) {
		throw std::invalid_argument("chunk_scale_id_i32 must be int32 1D (CPU)");
	}
	if (chunk_meta_u32.numel() != chunk_scale_id_i32.numel()) {
		throw std::invalid_argument("chunk_meta_u32 and chunk_scale_id_i32 must have same length");
	}
	if (num_scales <= 0) {
		throw std::invalid_argument("num_scales must be > 0");
	}
	const int64_t num_chunks = chunk_meta_u32.numel();
	const int64_t num_streams = num_scales * 16;
	if (num_streams > (1LL << 30)) {
		throw std::invalid_argument("num_streams too large");
	}

	auto cpu = torch::TensorOptions().device(torch::kCPU);
	torch::Tensor stream_chunk_ofs_i64 = torch::zeros({num_streams + 1}, cpu.dtype(torch::kInt64));
	torch::Tensor stream_chunk_ids_i32 = torch::empty({num_chunks}, cpu.dtype(torch::kInt32));

	const uint32_t* meta = chunk_meta_u32.data_ptr<uint32_t>();
	const int32_t* sid = chunk_scale_id_i32.data_ptr<int32_t>();
	int64_t* ofs = stream_chunk_ofs_i64.data_ptr<int64_t>();
	int32_t* ids = stream_chunk_ids_i32.data_ptr<int32_t>();

	// Count
	for (int64_t c = 0; c < num_chunks; ++c) {
		const int32_t s = sid[c];
		const uint32_t qv = meta[c] & 0xFFu;
		if (s < 0 || s >= num_scales) {
			throw std::invalid_argument("chunk_scale_id_i32 out of range for num_scales");
		}
		const int64_t stream = static_cast<int64_t>(s) * 16 + static_cast<int64_t>(qv);
		++ofs[stream + 1];
	}
	// Prefix sum
	for (int64_t i = 0; i < num_streams; ++i) {
		ofs[i + 1] += ofs[i];
	}
	// Fill (stable-ish): use a cursor array
	std::vector<int64_t> cursor(num_streams);
	for (int64_t i = 0; i < num_streams; ++i) cursor[i] = ofs[i];
	for (int64_t c = 0; c < num_chunks; ++c) {
		const int32_t s = sid[c];
		const uint32_t qv = meta[c] & 0xFFu;
		const int64_t stream = static_cast<int64_t>(s) * 16 + static_cast<int64_t>(qv);
		const int64_t pos = cursor[stream]++;
		ids[pos] = static_cast<int32_t>(c);
	}

	return std::make_tuple(stream_chunk_ofs_i64, stream_chunk_ids_i32);
}

static inline void check_cuda_contig(const torch::Tensor& t, const char* name) {
	if (!t.defined()) throw std::invalid_argument(std::string(name) + " must be defined");
	if (!t.is_cuda()) throw std::invalid_argument(std::string(name) + " must be a CUDA tensor");
	if (!t.is_contiguous()) throw std::invalid_argument(std::string(name) + " must be contiguous");
}

static DecodeTablesRet huffman_build_decode_tables(
	torch::Tensor enc_ofs_i64,
	torch::Tensor enc_syms_i16,
	torch::Tensor enc_codes_u32,
	torch::Tensor enc_lens_u8,
	int64_t max_len) {
	check_cpu_contig(enc_ofs_i64, "enc_ofs_i64");
	check_cpu_contig(enc_syms_i16, "enc_syms_i16");
	check_cpu_contig(enc_codes_u32, "enc_codes_u32");
	check_cpu_contig(enc_lens_u8, "enc_lens_u8");

	if (enc_ofs_i64.scalar_type() != torch::kInt64 || enc_ofs_i64.dim() != 1) {
		throw std::invalid_argument("enc_ofs_i64 must be int64 1D");
	}
	if (enc_syms_i16.scalar_type() != torch::kInt16 || enc_syms_i16.dim() != 1) {
		throw std::invalid_argument("enc_syms_i16 must be int16 1D");
	}
	if (enc_codes_u32.scalar_type() != torch::kUInt32 || enc_codes_u32.dim() != 1) {
		throw std::invalid_argument("enc_codes_u32 must be uint32 1D");
	}
	if (enc_lens_u8.scalar_type() != torch::kUInt8 || enc_lens_u8.dim() != 1) {
		throw std::invalid_argument("enc_lens_u8 must be uint8 1D");
	}
	if (enc_syms_i16.numel() != enc_codes_u32.numel() || enc_syms_i16.numel() != enc_lens_u8.numel()) {
		throw std::invalid_argument("enc_* tensors must have same length");
	}
	const int L = static_cast<int>(max_len);
	if (L <= 0 || L > 20) {
		throw std::invalid_argument("max_len must be in (0,20]");
	}
	const int64_t num_streams = enc_ofs_i64.numel() - 1;
	if (num_streams <= 0) {
		throw std::invalid_argument("enc_ofs_i64 must have length >= 2");
	}
	const int64_t table_size = 1LL << L;

	auto cpu = torch::TensorOptions().device(torch::kCPU);
	torch::Tensor decode_syms_i16 = torch::zeros({num_streams, table_size}, cpu.dtype(torch::kInt16));
	torch::Tensor decode_nbits_u8 = torch::zeros({num_streams, table_size}, cpu.dtype(torch::kUInt8));

	const int64_t* ofs = enc_ofs_i64.data_ptr<int64_t>();
	const int16_t* syms = enc_syms_i16.data_ptr<int16_t>();
	const uint32_t* codes = enc_codes_u32.data_ptr<uint32_t>();
	const uint8_t* lens = enc_lens_u8.data_ptr<uint8_t>();

	int16_t* out_syms = decode_syms_i16.data_ptr<int16_t>();
	uint8_t* out_lens = decode_nbits_u8.data_ptr<uint8_t>();

	for (int64_t s = 0; s < num_streams; ++s) {
		const int64_t a = ofs[s];
		const int64_t b = ofs[s + 1];
		if (a < 0 || b < a || b > enc_syms_i16.numel()) {
			throw std::invalid_argument("enc_ofs_i64 out of range");
		}
	}
	for (int64_t i = 0; i < enc_lens_u8.numel(); ++i) {
		const uint8_t len = lens[i];
		if (len == 0 || len > L) {
			throw std::invalid_argument("enc_lens contains invalid length");
		}
	}

	// Parallelize across streams (each stream writes to its own table slice).
	at::parallel_for(0, num_streams, 1, [&](int64_t begin, int64_t end) {
		for (int64_t s = begin; s < end; ++s) {
			const int64_t a = ofs[s];
			const int64_t b = ofs[s + 1];
			const int64_t base = s * table_size;
			for (int64_t i = a; i < b; ++i) {
				const uint8_t len = lens[i];
				const uint32_t code = codes[i];
				const int fill_bits = L - static_cast<int>(len);
				const int64_t fill = 1LL << fill_bits;
				const int64_t start = static_cast<int64_t>(code) << fill_bits;
				for (int64_t suf = 0; suf < fill; ++suf) {
					const int64_t idx = start | suf;
					out_syms[base + idx] = syms[i];
					out_lens[base + idx] = len;
				}
			}
		}
	});

	return std::make_tuple(decode_syms_i16, decode_nbits_u8);
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

// One block decodes chunks for exactly one stream. The stream's dense decode table is cached in shared memory.
// Parallelism: block has multiple warps; each warp grabs a batch of 32 chunks; each lane decodes one chunk.
__global__ void huffman_decode_by_stream_shared_table_kernel(
	const uint64_t* __restrict__ bitstream_u64,
	int64_t bitstream_words,
	const int16_t* __restrict__ decode_syms_i16,
	const uint8_t* __restrict__ decode_nbits_u8,
	int64_t table_size,
	const int64_t* __restrict__ scale_bit_base_i64,
	const uint32_t* __restrict__ chunk_startbit_rel_u32,
	const uint32_t* __restrict__ chunk_out_base_u32,
	const uint32_t* __restrict__ chunk_meta_u32,
	const int32_t* __restrict__ chunk_scale_id_i32,
	const int32_t* __restrict__ out_idx_i32,
	int64_t out_len,
	int L,
	const int64_t* __restrict__ stream_chunk_ofs_i64,
	const int32_t* __restrict__ stream_chunk_ids_i32,
	int16_t* __restrict__ out_i16) {
	const int64_t stream = static_cast<int64_t>(blockIdx.x);
	const int lane = static_cast<int>(threadIdx.x) & 31;

	extern __shared__ uint8_t smem_raw[];
	int16_t* sm_syms = reinterpret_cast<int16_t*>(smem_raw);
	uint8_t* sm_nbits = reinterpret_cast<uint8_t*>(sm_syms + table_size);

	// Load table into shared memory
	for (int64_t i = static_cast<int64_t>(threadIdx.x); i < table_size; i += static_cast<int64_t>(blockDim.x)) {
		sm_syms[i] = decode_syms_i16[stream * table_size + i];
		sm_nbits[i] = decode_nbits_u8[stream * table_size + i];
	}
	__syncthreads();

	const int64_t a = stream_chunk_ofs_i64[stream];
	const int64_t b = stream_chunk_ofs_i64[stream + 1];
	if (a >= b) return;

	__shared__ int32_t counter;
	if (threadIdx.x == 0) counter = 0;
	__syncthreads();

	for (;;) {
		int32_t base_local = 0;
		if (lane == 0) {
			// One atomic per warp on shared counter
			base_local = atomicAdd(&counter, 32);
		}
		base_local = __shfl_sync(0xFFFFFFFF, base_local, 0);
		const int64_t idx0 = a + static_cast<int64_t>(base_local);
		if (idx0 >= b) return;

		const int64_t idx = idx0 + static_cast<int64_t>(lane);
		if (idx >= b) continue;

		const int64_t chunk = static_cast<int64_t>(stream_chunk_ids_i32[idx]);
		const int32_t sid = chunk_scale_id_i32[chunk];
		const uint32_t meta = chunk_meta_u32[chunk];
		const uint32_t take = meta >> 8;

		int64_t bitpos = scale_bit_base_i64[sid] + static_cast<int64_t>(chunk_startbit_rel_u32[chunk]);
		const int64_t flat = static_cast<int64_t>(chunk_out_base_u32[chunk]);

		for (uint32_t k = 0; k < take; ++k) {
			const uint32_t t = read_bits_msb_u64(bitstream_u64, bitstream_words, bitpos, L);
			const uint8_t nb_raw = sm_nbits[t];
			const uint8_t is_esc = static_cast<uint8_t>(nb_raw & 0x80u);
			const uint8_t nbits = static_cast<uint8_t>(nb_raw & 0x7Fu);
			int16_t sym = sm_syms[t];
			bitpos += static_cast<int64_t>(nbits);
			if (is_esc) {
				const uint32_t raw16 = read_bits_msb_u64(bitstream_u64, bitstream_words, bitpos, 16);
				bitpos += 16;
				sym = static_cast<int16_t>(raw16);
			}

			const int64_t p = flat + static_cast<int64_t>(k);
			if (p >= 0 && p < out_len) {
				const int32_t dst = out_idx_i32[p];
				out_i16[static_cast<int64_t>(dst)] = sym;
			}
		}
	}
}

static torch::Tensor huffman_decode_awq_shared_table_cuda(
	torch::Tensor bitstream_u64,
	torch::Tensor decode_syms_i16,
	torch::Tensor decode_nbits_u8,
	torch::Tensor scale_bit_base_i64,
	torch::Tensor chunk_startbit_rel_u32,
	torch::Tensor chunk_out_base_u32,
	torch::Tensor chunk_meta_u32,
	torch::Tensor chunk_scale_id_i32,
	torch::Tensor out_idx_i32,
	torch::Tensor stream_chunk_ofs_i64,
	torch::Tensor stream_chunk_ids_i32,
	int64_t out_len,
	int64_t max_len) {
	// Basic checks (reuse existing ones where possible)
	check_cuda_contig(bitstream_u64, "bitstream_u64");
	check_cuda_contig(decode_syms_i16, "decode_syms_i16");
	check_cuda_contig(decode_nbits_u8, "decode_nbits_u8");
	check_cuda_contig(scale_bit_base_i64, "scale_bit_base_i64");
	check_cuda_contig(chunk_startbit_rel_u32, "chunk_startbit_rel_u32");
	check_cuda_contig(chunk_out_base_u32, "chunk_out_base_u32");
	check_cuda_contig(chunk_meta_u32, "chunk_meta_u32");
	check_cuda_contig(chunk_scale_id_i32, "chunk_scale_id_i32");
	check_cuda_contig(out_idx_i32, "out_idx_i32");
	check_cuda_contig(stream_chunk_ofs_i64, "stream_chunk_ofs_i64");
	check_cuda_contig(stream_chunk_ids_i32, "stream_chunk_ids_i32");

	if (stream_chunk_ofs_i64.scalar_type() != torch::kInt64 || stream_chunk_ofs_i64.dim() != 1) {
		throw std::invalid_argument("stream_chunk_ofs_i64 must be int64 1D (CUDA)");
	}
	if (stream_chunk_ids_i32.scalar_type() != torch::kInt32 || stream_chunk_ids_i32.dim() != 1) {
		throw std::invalid_argument("stream_chunk_ids_i32 must be int32 1D (CUDA)");
	}

	const int L = static_cast<int>(max_len);
	if (L <= 0 || L > 20) throw std::invalid_argument("max_len must be in (0,20]");
	const int64_t table_size = decode_syms_i16.size(1);
	if (table_size != (1LL << L)) throw std::invalid_argument("decode table second dim must be 2^max_len");
	const int64_t num_streams = decode_syms_i16.size(0);
	if (stream_chunk_ofs_i64.numel() != num_streams + 1) {
		throw std::invalid_argument("stream_chunk_ofs_i64 must have length num_streams+1");
	}
	if (stream_chunk_ids_i32.numel() != chunk_meta_u32.numel()) {
		throw std::invalid_argument("stream_chunk_ids_i32 must have length num_chunks");
	}

	// Shared memory size: (int16 table + uint8 table)
	const size_t shmem_bytes = static_cast<size_t>(table_size) * (sizeof(int16_t) + sizeof(uint8_t));
	// Opt-in dynamic shared memory when supported (Hopper can be much larger than 48KB).
	int dev = 0;
	cudaError_t derr = cudaGetDevice(&dev);
	if (derr != cudaSuccess) {
		throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(derr));
	}
	int max_default = 0;
	int max_optin = 0;
	cudaDeviceGetAttribute(&max_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
	// cudaDevAttrMaxSharedMemoryPerBlockOptin may be 0 on older drivers.
	cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
	int limit = max_default;
	if (max_optin > limit) limit = max_optin;
	if (static_cast<int>(shmem_bytes) > limit) {
		throw std::invalid_argument("decode table too large for shared memory; reduce max_len");
	}
	if (static_cast<int>(shmem_bytes) > max_default && max_optin > max_default) {
		cudaError_t aerr = cudaFuncSetAttribute(
			huffman_decode_by_stream_shared_table_kernel,
			cudaFuncAttributeMaxDynamicSharedMemorySize,
			static_cast<int>(shmem_bytes));
		if (aerr != cudaSuccess) {
			throw std::runtime_error(std::string("cudaFuncSetAttribute failed: ") + cudaGetErrorString(aerr));
		}
	}

	auto opts = torch::TensorOptions().device(bitstream_u64.device()).dtype(torch::kInt16);
	torch::Tensor out_i16 = torch::empty({out_len}, opts);

	const int threads = 256;  // 8 warps
	int blocks = static_cast<int>(std::min<int64_t>(num_streams, 65535));
	if (blocks <= 0) blocks = 1;

	huffman_decode_by_stream_shared_table_kernel<<<blocks, threads, shmem_bytes>>>(
		bitstream_u64.data_ptr<uint64_t>(),
		bitstream_u64.numel(),
		decode_syms_i16.data_ptr<int16_t>(),
		decode_nbits_u8.data_ptr<uint8_t>(),
		table_size,
		scale_bit_base_i64.data_ptr<int64_t>(),
		chunk_startbit_rel_u32.data_ptr<uint32_t>(),
		chunk_out_base_u32.data_ptr<uint32_t>(),
		chunk_meta_u32.data_ptr<uint32_t>(),
		chunk_scale_id_i32.data_ptr<int32_t>(),
		out_idx_i32.data_ptr<int32_t>(),
		out_len,
		L,
		stream_chunk_ofs_i64.data_ptr<int64_t>(),
		stream_chunk_ids_i32.data_ptr<int32_t>(),
		out_i16.data_ptr<int16_t>());

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		throw std::runtime_error(std::string("CUDA shared-table kernel launch failed: ") + cudaGetErrorString(err));
	}
	return out_i16;
}

DecodeTablesRet huffman_build_decode_tables_entry(
	torch::Tensor enc_ofs_i64,
	torch::Tensor enc_syms_i16,
	torch::Tensor enc_codes_u32,
	torch::Tensor enc_lens_u8,
	int64_t max_len) {
	return huffman_build_decode_tables(enc_ofs_i64, enc_syms_i16, enc_codes_u32, enc_lens_u8, max_len);
}

StreamChunkIndexRet huffman_build_stream_chunk_index_entry(
	torch::Tensor chunk_meta_u32,
	torch::Tensor chunk_scale_id_i32,
	int64_t num_scales) {
	return huffman_build_stream_chunk_index(chunk_meta_u32, chunk_scale_id_i32, num_scales);
}

torch::Tensor huffman_decode_awq_entry(
	torch::Tensor bitstream_u64,
	torch::Tensor decode_syms_i16,
	torch::Tensor decode_nbits_u8,
	torch::Tensor scale_bit_base_i64,
	torch::Tensor chunk_startbit_rel_u32,
	torch::Tensor chunk_out_base_u32,
	torch::Tensor chunk_meta_u32,
	torch::Tensor chunk_scale_id_i32,
	torch::Tensor out_idx_i32,
	int64_t out_len,
	int64_t max_len) {
	throw std::invalid_argument("huffman_decode_awq_entry removed; use huffman_decode_awq_shared_table_entry");
}

torch::Tensor huffman_decode_awq_shared_table_entry(
	torch::Tensor bitstream_u64,
	torch::Tensor decode_syms_i16,
	torch::Tensor decode_nbits_u8,
	torch::Tensor scale_bit_base_i64,
	torch::Tensor chunk_startbit_rel_u32,
	torch::Tensor chunk_out_base_u32,
	torch::Tensor chunk_meta_u32,
	torch::Tensor chunk_scale_id_i32,
	torch::Tensor out_idx_i32,
	torch::Tensor stream_chunk_ofs_i64,
	torch::Tensor stream_chunk_ids_i32,
	int64_t out_len,
	int64_t max_len) {
	return huffman_decode_awq_shared_table_cuda(
		bitstream_u64,
		decode_syms_i16,
		decode_nbits_u8,
		scale_bit_base_i64,
		chunk_startbit_rel_u32,
		chunk_out_base_u32,
		chunk_meta_u32,
		chunk_scale_id_i32,
		out_idx_i32,
		stream_chunk_ofs_i64,
		stream_chunk_ids_i32,
		out_len,
		max_len);
}

}  // namespace huffman_decode_cpp
