#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace huffman_cpp {

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static inline uint32_t ceil_div_u32(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}

static inline uint16_t sym_u16(int16_t s) {
    return static_cast<uint16_t>(s);
}

struct CanonicalSymCode {
    int16_t sym;
    uint8_t len;
    uint32_t code;
};

struct CanonicalSymCodeU16 {
    uint16_t sym;
    uint8_t len;
    uint32_t code;
};

// -----------------------------------------------------------------------------
// Package-merge (length-limited Huffman) from weights
// -----------------------------------------------------------------------------

static std::vector<int> package_merge_code_lengths_from_weights(std::vector<int> weights, int max_len) {
    const int L = max_len;
    if (L <= 0 || L > 30) throw std::invalid_argument("max_len must be in (0, 30]");
    const int n = static_cast<int>(weights.size());
    if (n == 0) return {};
    if (n == 1) return {1};
    if (static_cast<int64_t>(n) > (1LL << L)) throw std::invalid_argument("Too many symbols for max_len");

    std::stable_sort(weights.begin(), weights.end());

    const int K = 2 * (n - 1);
    const int keep = 2 * K;

    std::vector<std::pair<int, int>> leaf_items;
    leaf_items.reserve(n);
    for (int i = 0; i < n; ++i) leaf_items.emplace_back(weights[i], i);

    std::vector<int> pkg_left;
    std::vector<int> pkg_right;
    pkg_left.reserve(L * keep);
    pkg_right.reserve(L * keep);

    auto new_pkg = [&](int a_ref, int b_ref) -> int {
        pkg_left.push_back(a_ref);
        pkg_right.push_back(b_ref);
        return n + static_cast<int>(pkg_left.size()) - 1;
    };

    auto merge_sorted = [&](const std::vector<std::pair<int, int>>& a,
                           const std::vector<std::pair<int, int>>& b,
                           int limit) {
        std::vector<std::pair<int, int>> out;
        out.reserve(std::min<int>(limit, (int)(a.size() + b.size())));
        int i = 0, j = 0;
        while ((int)out.size() < limit && i < (int)a.size() && j < (int)b.size()) {
            if (a[i].first <= b[j].first) out.push_back(a[i++]);
            else out.push_back(b[j++]);
        }
        while ((int)out.size() < limit && i < (int)a.size()) out.push_back(a[i++]);
        while ((int)out.size() < limit && j < (int)b.size()) out.push_back(b[j++]);
        return out;
    };

    std::vector<std::pair<int, int>> cur = leaf_items;
    if ((int)cur.size() > keep) cur.resize(keep);

    for (int p = 0; p < L - 1; ++p) {
        const int trunc_len = std::min<int>(keep, (int)cur.size());
        std::vector<std::pair<int, int>> packages;
        packages.reserve(trunc_len / 2);
        for (int i = 0; i + 1 < trunc_len; i += 2) {
            packages.emplace_back(cur[i].first + cur[i + 1].first, new_pkg(cur[i].second, cur[i + 1].second));
        }
        cur = merge_sorted(leaf_items, packages, keep);
    }

    if ((int)cur.size() < K) throw std::runtime_error("Internal error: not enough packages");

    std::vector<int> lens(n, 0);
    auto bump_leaf_depths = [&](int ref) {
        std::vector<int> st = {ref};
        while (!st.empty()) {
            int r = st.back(); st.pop_back();
            if (r < n) { lens[r]++; continue; }
            int pi = r - n;
            st.push_back(pkg_left[pi]);
            st.push_back(pkg_right[pi]);
        }
    };

    for (int i = 0; i < K; ++i) bump_leaf_depths(cur[i].second);
    return lens;
}

// -----------------------------------------------------------------------------
// Canonical code builder from code lengths
// -----------------------------------------------------------------------------

static std::vector<CanonicalSymCode> build_canonical_codes(const std::vector<uint8_t>& lengths) {
    std::vector<CanonicalSymCode> out;
    out.reserve(lengths.size());
    std::vector<int> bl_count(33, 0);
    for (auto len : lengths) {
        if (len > 0) bl_count[len]++;
    }
    std::vector<uint32_t> next_code(33, 0);
    uint32_t code = 0;
    for (int bits = 1; bits <= 32; ++bits) {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    for (int sym = 0; sym < static_cast<int>(lengths.size()); ++sym) {
        const uint8_t len = lengths[sym];
        if (len == 0) continue;
        out.push_back({static_cast<int16_t>(sym), len, next_code[len]++});
    }

    std::stable_sort(out.begin(), out.end(), [](const CanonicalSymCode& a, const CanonicalSymCode& b) {
        if (a.len != b.len) return a.len < b.len;
        return a.sym < b.sym;
    });
    return out;
}

// Fill fast decode tables of size 2^L (symbols and lengths)
static void fill_decode_tables(const std::vector<CanonicalSymCode>& codes,
                               int max_len,
                               std::vector<uint16_t>* decode_sym,
                               std::vector<uint8_t>* decode_len) {
    const uint32_t table_size = 1u << max_len;
    decode_sym->assign(table_size, 0);
    decode_len->assign(table_size, 0);
    for (const auto& c : codes) {
        const uint32_t fill = 1u << (max_len - c.len);
        const uint32_t base = c.code << (max_len - c.len);
        for (uint32_t i = 0; i < fill; ++i) {
            const uint32_t idx = base + i;
            (*decode_sym)[idx] = sym_u16(c.sym);
            (*decode_len)[idx] = c.len;
        }
    }
}


static std::vector<CanonicalSymCodeU16> build_canonical_codes_u16(const std::vector<std::pair<uint16_t, uint8_t>>& sym_lens) {
    std::vector<CanonicalSymCodeU16> out;
    out.reserve(sym_lens.size());

    std::vector<int> bl_count(16, 0);
    for (const auto& p : sym_lens) {
        const uint8_t len = p.second;
        if (len == 0) continue;
        if (len >= bl_count.size()) throw std::invalid_argument("code length too large");
        bl_count[len]++;
    }

    std::vector<uint32_t> next_code(16, 0);
    uint32_t code = 0;
    for (int bits = 1; bits < static_cast<int>(next_code.size()); ++bits) {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    out.reserve(sym_lens.size());
    for (const auto& p : sym_lens) {
        const uint16_t sym = p.first;
        const uint8_t len = p.second;
        if (len == 0) continue;
        out.push_back({sym, len, next_code[len]++});
    }

    std::stable_sort(out.begin(), out.end(), [](const CanonicalSymCodeU16& a, const CanonicalSymCodeU16& b) {
        if (a.len != b.len) return a.len < b.len;
        return a.sym < b.sym;
    });
    return out;
}

static void fill_decode_tables_u16(const std::vector<CanonicalSymCodeU16>& codes,
                                   int max_len,
                                   std::vector<uint16_t>* decode_sym,
                                   std::vector<uint8_t>* decode_len_unpacked,
                                   bool mark_esc = false,
                                   uint16_t esc_sym = 0) {
    const uint32_t table_size = 1u << max_len;
    decode_sym->assign(table_size, 0);
    decode_len_unpacked->assign(table_size, 0);
    for (const auto& c : codes) {
        const uint32_t fill = 1u << (max_len - c.len);
        const uint32_t base = c.code << (max_len - c.len);
        for (uint32_t i = 0; i < fill; ++i) {
            const uint32_t idx = base + i;
            (*decode_sym)[idx] = c.sym;
            // For FP8 L=7 decode, we can store 4-bit lengths. Reserve the high bit (0x8)
            // as an ESC marker: when set, the decoder will additionally read 16 raw bits
            // (bf16 bits) after consuming the Huffman code bits.
            (*decode_len_unpacked)[idx] = (mark_esc && (c.sym == esc_sym))
                ? static_cast<uint8_t>(c.len | 0x8u)
                : c.len;
        }
    }
}

static void pack_decode_lengths_4bit(const std::vector<uint8_t>& decode_len_unpacked,
                                     std::vector<uint8_t>* decode_len_packed) {
    if (decode_len_unpacked.size() % 2 != 0) throw std::invalid_argument("decode_len size must be even");
    decode_len_packed->assign(decode_len_unpacked.size() / 2, 0);
    for (size_t i = 0; i < decode_len_packed->size(); ++i) {
        const uint8_t lo = static_cast<uint8_t>(decode_len_unpacked[2 * i] & 0x0F);
        const uint8_t hi = static_cast<uint8_t>(decode_len_unpacked[2 * i + 1] & 0x0F);
        (*decode_len_packed)[i] = static_cast<uint8_t>(lo | (hi << 4));
    }
}

static inline uint16_t bf16_bits_at(const torch::Tensor& t_bf16, int64_t idx) {
    const auto* p = reinterpret_cast<const uint16_t*>(t_bf16.data_ptr<at::BFloat16>());
    return p[idx];
}

static inline uint16_t bf16_bits_at_u16ptr(const uint16_t* p_u16, int64_t idx) {
    return p_u16[idx];
}

static inline uint16_t u16_at(const torch::Tensor& t_u16, int64_t idx) {
    return t_u16.data_ptr<uint16_t>()[idx];
}

static inline uint8_t u8_at(const torch::Tensor& t_u8, int64_t idx) {
    return t_u8.data_ptr<uint8_t>()[idx];
}

// Entry: build length-limited Huffman decode tables per (scale, fp8) group.
// - bf16_params: torch.bfloat16, shape [n, m] (or any shape), CPU
// - fp8_params:  torch.uint8, same numel as bf16_params, CPU
// - scales:      torch.uint16 or torch.bfloat16
//               allowed shapes:
//                 - scalar
//                 - per-row [n]
//                 - per-element (same numel as bf16_params)
//                 - 2D block-wise [n/bs0, m/bs1] (requires bf16_params 2D and exact divisibility)
// Returns:
// - decode_syms_u16: [G, 2^L] uint16 (bf16 raw bits)
// - decode_lens_u8:  [G, 2^(L-1)] uint8 (packed 2x4bit lengths)
// - bucket_scale_u16: [G] uint16
// - bucket_fp8_u16:   [G] uint16
using BuildFP8TablesRet = std::tuple<
    torch::Tensor,  // decode_syms_u16
    torch::Tensor,  // decode_lens_u8
    torch::Tensor,  // table_scale_u16
    torch::Tensor,  // table_fp8_u16
    torch::Tensor,  // bucket_scale_u16
    torch::Tensor,  // bucket_row_offsets_i32
    torch::Tensor,  // row_indices_i32
    torch::Tensor,  // bucket_table_i32
    torch::Tensor,  // bucket_col_start_i32
    torch::Tensor,  // bucket_col_len_i32
    torch::Tensor,  // bucket_thread_startbit_u32
    torch::Tensor,  // bucket_bit_base_i64
    torch::Tensor,  // bitstream_u64
    torch::Tensor,  // shape_nmk_i64
    torch::Tensor,  // bucket_U_i32
    torch::Tensor,  // bucket_unique_gid_i32
    torch::Tensor   // bucket_fp8_uidx_i16
>;

BuildFP8TablesRet
huffman_build_fp8_decode_tables_entry(torch::Tensor bf16_params,
                                      torch::Tensor fp8_params,
                                      torch::Tensor scales,
                                      int64_t max_len,
                                      int64_t threads_per_block) {
    if (!bf16_params.defined() || !fp8_params.defined() || !scales.defined()) {
        throw std::invalid_argument("inputs must be defined");
    }
    if (bf16_params.device().is_cuda() || fp8_params.device().is_cuda() || scales.device().is_cuda()) {
        throw std::invalid_argument("CPU tensors only for now");
    }
    if (bf16_params.scalar_type() != torch::kBFloat16) {
        throw std::invalid_argument("bf16_params must be torch.bfloat16");
    }
    if (fp8_params.scalar_type() != torch::kUInt8) {
        throw std::invalid_argument("fp8_params must be torch.uint8");
    }
    if (max_len <= 0 || max_len > 15) {
        throw std::invalid_argument("max_len must be in [1, 15]");
    }
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        throw std::invalid_argument("threads_per_block must be in [1, 1024]");
    }

    bf16_params = bf16_params.contiguous();
    fp8_params = fp8_params.contiguous();
    scales = scales.contiguous();

    const int64_t numel = bf16_params.numel();
    if (fp8_params.numel() != numel) {
        throw std::invalid_argument("bf16_params and fp8_params must have same numel");
    }
    const bool scales_is_bf16 = (scales.scalar_type() == torch::kBFloat16);
    const bool scales_is_u16 = (scales.scalar_type() == torch::kUInt16);
    if (!scales_is_bf16 && !scales_is_u16) {
        throw std::invalid_argument("scales must be torch.uint16 or torch.bfloat16");
    }

    // Scale broadcasting modes.
    const int64_t scale_numel = scales.numel();
    const bool scale_scalar = (scale_numel == 1);
    const bool scale_per_elem = (scale_numel == numel);
    bool scale_per_row = false;
    bool scale_block2d = false;
    int64_t n_rows = -1;
    int64_t row_stride = -1;
    int64_t n_cols = -1;
    int64_t sb0 = -1;
    int64_t sb1 = -1;
    int64_t nb0 = -1;
    int64_t nb1 = -1;
    if (!scale_scalar && !scale_per_elem && bf16_params.dim() == 2) {
        n_rows = bf16_params.size(0);
        row_stride = bf16_params.size(1);
        n_cols = row_stride;
        scale_per_row = (scale_numel == n_rows);

        // 2D block-wise: scales shaped [nb0, nb1] with n=nb0*bs0 and m=nb1*bs1.
        if (!scale_per_row && scales.dim() == 2) {
            nb0 = scales.size(0);
            nb1 = scales.size(1);
            if (nb0 > 0 && nb1 > 0 && (n_rows % nb0 == 0) && (n_cols % nb1 == 0)) {
                sb0 = n_rows / nb0;
                sb1 = n_cols / nb1;
                if (sb0 > 0 && sb1 > 0) {
                    scale_block2d = true;
                }
            }
        }
    }
    if (!scale_scalar && !scale_per_elem && !scale_per_row && !scale_block2d) {
        throw std::invalid_argument(
            "unsupported scales shape: use scalar, per-row (n,), per-element (numel), or 2D block-wise [n/bs0,m/bs1]");
    }

    auto get_scale_u16 = [&](int64_t flat_idx) -> uint16_t {
        int64_t sidx = 0;
        if (scale_scalar) {
            sidx = 0;
        } else if (scale_per_elem) {
            sidx = flat_idx;
        } else if (scale_block2d) {
            // bf16_params is 2D [n_rows, n_cols]
            const int64_t r = flat_idx / n_cols;
            const int64_t c = flat_idx - r * n_cols;
            const int64_t br = r / sb0;
            const int64_t bc = c / sb1;
            sidx = br * nb1 + bc;
        } else {
            // per-row
            const int64_t r = flat_idx / row_stride;
            sidx = r;
        }
        if (scales_is_u16) return u16_at(scales, sidx);
        // bfloat16 bits
        const auto* p = reinterpret_cast<const uint16_t*>(scales.data_ptr<at::BFloat16>());
        return p[sidx];
    };

    std::unordered_map<uint32_t, std::unordered_map<uint16_t, uint32_t>> freq_by_key;
    freq_by_key.reserve(1024);

    for (int64_t i = 0; i < numel; ++i) {
        const uint16_t scale_u16 = get_scale_u16(i);
        const uint8_t fp8_u8 = u8_at(fp8_params, i);
        const uint16_t sym_u16 = bf16_bits_at(bf16_params, i);
        const uint32_t key = (static_cast<uint32_t>(scale_u16) << 16) | static_cast<uint32_t>(fp8_u8);
        freq_by_key[key][sym_u16] += 1;
    }

    std::vector<uint32_t> keys;
    keys.reserve(freq_by_key.size());
    for (const auto& kv : freq_by_key) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    std::unordered_map<uint32_t, int32_t> key_to_gid;
    key_to_gid.reserve(keys.size() * 2 + 1);
    for (int32_t gi = 0; gi < static_cast<int32_t>(keys.size()); ++gi) {
        key_to_gid.emplace(keys[gi], gi);
    }

    const int64_t G = static_cast<int64_t>(keys.size());
    const int64_t table_size = 1LL << max_len;
    const int64_t packed_len_size = table_size / 2;

    auto opts_u16 = torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCPU);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);

    torch::Tensor decode_syms_u16 = torch::empty({G, table_size}, opts_u16);
    torch::Tensor decode_lens_u8 = torch::empty({G, packed_len_size}, opts_u8);
    torch::Tensor table_scale_u16 = torch::empty({G}, opts_u16);
    torch::Tensor table_fp8_u16 = torch::empty({G}, opts_u16);

    auto* out_syms = decode_syms_u16.data_ptr<uint16_t>();
    auto* out_lens = decode_lens_u8.data_ptr<uint8_t>();
    auto* out_scale = table_scale_u16.data_ptr<uint16_t>();
    auto* out_fp8 = table_fp8_u16.data_ptr<uint16_t>();

    std::vector<uint16_t> decode_syms_vec;
    std::vector<uint8_t> decode_len_unpacked;
    std::vector<uint8_t> decode_len_packed;

    struct EncCode {
        uint16_t sym;
        uint8_t len;
        uint32_t code;
    };
    std::unordered_map<uint32_t, std::unordered_map<uint16_t, uint32_t>> enc_by_key;
    enc_by_key.reserve(freq_by_key.size() * 2 + 1);
    std::unordered_map<uint32_t, uint16_t> esc_sym_by_key;
    esc_sym_by_key.reserve(freq_by_key.size() / 8 + 1);

    for (int64_t gi = 0; gi < G; ++gi) {
        const uint32_t key = keys[gi];
        const uint16_t scale_u16 = static_cast<uint16_t>(key >> 16);
        const uint16_t fp8_u16 = static_cast<uint16_t>(key & 0xFFFFu);
        out_scale[gi] = scale_u16;
        out_fp8[gi] = fp8_u16;

        const auto& freq = freq_by_key.at(key);
        std::vector<std::pair<int, uint16_t>> items;
        items.reserve(freq.size());
        for (const auto& sv : freq) {
            items.emplace_back(static_cast<int>(sv.second), sv.first);
        }

        if (items.empty()) {
            // Should not happen.
            decode_syms_vec.assign(table_size, 0);
            decode_len_unpacked.assign(table_size, 0);
            enc_by_key[key] = {};
        } else if (items.size() == 1) {
            // Single symbol: define len=1, code=0.
            const uint16_t sym = items[0].second;
            decode_syms_vec.assign(table_size, sym);
            decode_len_unpacked.assign(table_size, 1);

            auto& enc = enc_by_key[key];
            enc.reserve(1);
            enc[sym] = (static_cast<uint32_t>(1u) << 16) | 0u;
        } else {
            std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });

            // Fixed-L decode kernel (L=7) implies decode table size 128.
            // Some (scale,fp8) groups can have >128 unique bf16 symbols. To keep lossless
            // while respecting the fixed table size, we use an ESC fallback:
            // - Keep top-(2^L-1) symbols by frequency
            // - Add a synthetic ESC symbol whose code indicates "read raw 16-bit bf16 bits"
            // - For all other symbols, encode: ESC_CODE + raw16(sym)
            bool use_esc = false;
            uint16_t esc_sym = 0;
            int esc_freq = 0;

            if (items.size() > static_cast<size_t>(table_size)) {
                use_esc = true;

                std::vector<std::pair<int, uint16_t>> items_desc = items;
                std::stable_sort(items_desc.begin(), items_desc.end(), [](const auto& a, const auto& b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second < b.second;
                });

                const size_t keep_k = static_cast<size_t>(table_size - 1);
                std::unordered_set<uint16_t> keep;
                keep.reserve(keep_k * 2 + 1);
                for (size_t i = 0; i < keep_k; ++i) {
                    keep.insert(items_desc[i].second);
                }
                for (const auto& it : items_desc) {
                    if (keep.find(it.second) == keep.end()) {
                        esc_freq += it.first;
                    }
                }

                // Pick an esc_sym that does not collide with kept symbols.
                for (uint32_t cand = 0; cand <= 0xFFFFu; ++cand) {
                    const uint16_t c16 = static_cast<uint16_t>(cand);
                    if (keep.find(c16) == keep.end()) {
                        esc_sym = c16;
                        break;
                    }
                }
                // Rebuild items with kept + ESC.
                std::vector<std::pair<int, uint16_t>> rebuilt;
                rebuilt.reserve(keep_k + 1);
                for (const auto& it : items_desc) {
                    if (keep.find(it.second) != keep.end()) {
                        rebuilt.push_back(it);
                        if (rebuilt.size() == keep_k) break;
                    }
                }
                rebuilt.push_back({esc_freq, esc_sym});
                items.swap(rebuilt);

                // Re-sort ascending for package-merge.
                std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second;
                });
            }

            std::vector<int> weights;
            weights.reserve(items.size());
            for (const auto& it : items) weights.push_back(it.first);

            std::vector<int> lens_i = package_merge_code_lengths_from_weights(weights, static_cast<int>(max_len));
            if (lens_i.size() != items.size()) throw std::runtime_error("length vector size mismatch");

            std::vector<std::pair<uint16_t, uint8_t>> sym_lens;
            sym_lens.reserve(items.size());
            for (size_t si = 0; si < items.size(); ++si) {
                const int li = lens_i[si];
                if (li <= 0 || li > max_len) throw std::runtime_error("invalid code length");
                sym_lens.emplace_back(items[si].second, static_cast<uint8_t>(li));
            }

            auto codes = build_canonical_codes_u16(sym_lens);
            fill_decode_tables_u16(
                codes,
                static_cast<int>(max_len),
                &decode_syms_vec,
                &decode_len_unpacked,
                use_esc,
                esc_sym);

            auto& enc = enc_by_key[key];
            enc.reserve(codes.size() * 2 + 1);
            for (const auto& c : codes) {
                if (c.len == 0 || c.len > max_len) throw std::runtime_error("invalid code length in canonical codes");
                enc[c.sym] = (static_cast<uint32_t>(c.len) << 16) | static_cast<uint32_t>(c.code);
            }

            if (use_esc) {
                esc_sym_by_key[key] = esc_sym;
            }
        }

        pack_decode_lengths_4bit(decode_len_unpacked, &decode_len_packed);

        // Copy to output tensors.
        std::memcpy(out_syms + gi * table_size, decode_syms_vec.data(), table_size * sizeof(uint16_t));
        std::memcpy(out_lens + gi * packed_len_size, decode_len_packed.data(), packed_len_size * sizeof(uint8_t));
    }

    // ---------------------------------------------------------------------
    // Build per-scale buckets (<=256 rows each), fp8->table mapping, bitstream,
    // and per-thread start bit offsets.
    // ---------------------------------------------------------------------

    if (bf16_params.dim() != 2) {
        throw std::invalid_argument("bucket packing requires bf16_params to be 2D [n,m]");
    }
    const int64_t n = bf16_params.size(0);
    const int64_t m = bf16_params.size(1);
    if (n <= 0 || m <= 0) {
        throw std::invalid_argument("bf16_params shape must be non-empty");
    }

    struct Bucket {
        uint16_t scale;
        int32_t row_start;  // offset into row_indices
        int32_t row_count;
        int32_t col_start;  // starting column in [0,m)
        int32_t col_len;    // number of columns this bucket covers
    };

    std::vector<int32_t> row_indices;
    std::vector<int32_t> bucket_row_offsets;
    std::vector<uint16_t> bucket_scales;
    std::vector<int32_t> bucket_col_start;
    std::vector<int32_t> bucket_col_len;
    std::vector<Bucket> buckets;

    bucket_row_offsets.push_back(0);

    if (!scale_block2d) {
        // Bucketing-by-row (legacy): group rows by (scalar/per-row) scale.
        // For bucketing-by-row, we only support scalar scale or per-row scale.
        const bool scale_ok_for_bucket = scale_scalar || (!scale_per_elem && scale_per_row);
        if (!scale_ok_for_bucket) {
            throw std::invalid_argument("bucket packing requires scales to be scalar, per-row (n,), or 2D block-wise");
        }

        auto get_row_scale_u16 = [&](int64_t row) -> uint16_t {
            if (scale_scalar) return get_scale_u16(0);
            // per-row
            const int64_t sidx = row;
            if (scales_is_u16) return u16_at(scales, sidx);
            const auto* p = reinterpret_cast<const uint16_t*>(scales.data_ptr<at::BFloat16>());
            return p[sidx];
        };

        // scale -> rows
        std::unordered_map<uint16_t, std::vector<int32_t>> rows_by_scale;
        rows_by_scale.reserve(static_cast<size_t>(n));
        for (int64_t r = 0; r < n; ++r) {
            const uint16_t sc = get_row_scale_u16(r);
            rows_by_scale[sc].push_back(static_cast<int32_t>(r));
        }

        std::vector<uint16_t> unique_scales;
        unique_scales.reserve(rows_by_scale.size());
        for (const auto& kv : rows_by_scale) unique_scales.push_back(kv.first);
        std::sort(unique_scales.begin(), unique_scales.end());

        for (uint16_t sc : unique_scales) {
            const auto& rows = rows_by_scale.at(sc);
            for (size_t off = 0; off < rows.size(); off += 128) {
                const size_t take = std::min<size_t>(128, rows.size() - off);
                const int32_t start = static_cast<int32_t>(row_indices.size());
                row_indices.insert(
                    row_indices.end(),
                    rows.begin() + static_cast<int64_t>(off),
                    rows.begin() + static_cast<int64_t>(off + take));
                buckets.push_back({sc, start, static_cast<int32_t>(take), 0, static_cast<int32_t>(m)});
                bucket_scales.push_back(sc);
                bucket_col_start.push_back(0);
                bucket_col_len.push_back(static_cast<int32_t>(m));
                bucket_row_offsets.push_back(static_cast<int32_t>(row_indices.size()));
            }
        }
    } else {
        // 2D block-wise bucketing: create buckets over (row-block, col-block), so each bucket has a single scale.
        // Here scales is [nb0, nb1], with block sizes sb0=n/nb0 and sb1=m/nb1.
        auto get_block_scale_u16 = [&](int64_t br, int64_t bc) -> uint16_t {
            const int64_t sidx = br * nb1 + bc;
            if (scales_is_u16) return u16_at(scales, sidx);
            const auto* p = reinterpret_cast<const uint16_t*>(scales.data_ptr<at::BFloat16>());
            return p[sidx];
        };

        for (int64_t br = 0; br < nb0; ++br) {
            const int64_t r0 = br * sb0;
            const int64_t r1 = (br + 1) * sb0;

            // Build the row list for this row-block.
            std::vector<int32_t> rows;
            rows.reserve(static_cast<size_t>(sb0));
            for (int64_t r = r0; r < r1; ++r) rows.push_back(static_cast<int32_t>(r));

            for (int64_t bc = 0; bc < nb1; ++bc) {
                const uint16_t sc = get_block_scale_u16(br, bc);
                const int32_t c0 = static_cast<int32_t>(bc * sb1);
                const int32_t clen = static_cast<int32_t>(sb1);

                for (size_t off = 0; off < rows.size(); off += 128) {
                    const size_t take = std::min<size_t>(128, rows.size() - off);
                    const int32_t start = static_cast<int32_t>(row_indices.size());
                    row_indices.insert(
                        row_indices.end(),
                        rows.begin() + static_cast<int64_t>(off),
                        rows.begin() + static_cast<int64_t>(off + take));
                    buckets.push_back({sc, start, static_cast<int32_t>(take), c0, clen});
                    bucket_scales.push_back(sc);
                    bucket_col_start.push_back(c0);
                    bucket_col_len.push_back(clen);
                    bucket_row_offsets.push_back(static_cast<int32_t>(row_indices.size()));
                }
            }
        }
    }

    const int64_t B = static_cast<int64_t>(buckets.size());

    // Choose threads-per-block k (variable). For now, use a heuristic: k=min(256, m) but >=1.
    // TODO: expose k as an argument when integrating with decode kernel.
    int32_t k_threads = static_cast<int32_t>(threads_per_block);
    if (k_threads <= 0) k_threads = 1;

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto opts_i16 = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU);
    auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU);

    torch::Tensor bucket_scale_u16 = torch::empty({B}, opts_u16);
    torch::Tensor bucket_row_offsets_i32 = torch::empty({B + 1}, opts_i32);
    torch::Tensor row_indices_i32 = torch::empty({static_cast<int64_t>(row_indices.size())}, opts_i32);
    torch::Tensor bucket_table_i32 = torch::empty({B, 256}, opts_i32);
    torch::Tensor bucket_col_start_i32 = torch::empty({B}, opts_i32);
    torch::Tensor bucket_col_len_i32 = torch::empty({B}, opts_i32);
    torch::Tensor bucket_U_i32 = torch::empty({B}, opts_i32);
    torch::Tensor bucket_unique_gid_i32 = torch::empty({B, 256}, opts_i32);
    torch::Tensor bucket_fp8_uidx_i16 = torch::empty({B, 256}, opts_i16);
    torch::Tensor bucket_thread_startbit_u32 = torch::empty({B, static_cast<int64_t>(k_threads) + 1}, opts_u32);
    torch::Tensor bucket_bit_base_i64 = torch::empty({B + 1}, opts_i64);
    torch::Tensor shape_nmk_i64 = torch::empty({3}, opts_i64);

    shape_nmk_i64.data_ptr<int64_t>()[0] = n;
    shape_nmk_i64.data_ptr<int64_t>()[1] = m;
    shape_nmk_i64.data_ptr<int64_t>()[2] = static_cast<int64_t>(k_threads);

    // Fill row offsets and row indices.
    std::memcpy(bucket_row_offsets_i32.data_ptr<int32_t>(), bucket_row_offsets.data(), bucket_row_offsets.size() * sizeof(int32_t));
    std::memcpy(row_indices_i32.data_ptr<int32_t>(), row_indices.data(), row_indices.size() * sizeof(int32_t));
    for (int64_t bi = 0; bi < B; ++bi) {
        bucket_scale_u16.data_ptr<uint16_t>()[bi] = bucket_scales[static_cast<size_t>(bi)];
        bucket_col_start_i32.data_ptr<int32_t>()[bi] = bucket_col_start[static_cast<size_t>(bi)];
        bucket_col_len_i32.data_ptr<int32_t>()[bi] = bucket_col_len[static_cast<size_t>(bi)];
    }

    // Fill bucket_table_i32 (fp8 -> global table id).
    int32_t* bt = bucket_table_i32.data_ptr<int32_t>();
    for (int64_t bi = 0; bi < B; ++bi) {
        const uint16_t sc = bucket_scales[static_cast<size_t>(bi)];
        for (int fp8 = 0; fp8 < 256; ++fp8) {
            const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(fp8);
            auto it = key_to_gid.find(key);
            bt[bi * 256 + fp8] = (it == key_to_gid.end()) ? -1 : it->second;
        }
    }

    // Precompute per-bucket:
    // - U: number of unique gids used by this bucket
    // - unique_gid[U]: first-appearance order list of gids
    // - fp8_uidx[256]: fp8 -> uidx into unique_gid
    // This removes the need to build these on GPU during decode.
    int32_t* bucket_U = bucket_U_i32.data_ptr<int32_t>();
    int32_t* uniq = bucket_unique_gid_i32.data_ptr<int32_t>();
    int16_t* uidx = bucket_fp8_uidx_i16.data_ptr<int16_t>();
    for (int64_t bi = 0; bi < B; ++bi) {
        // init
        for (int i = 0; i < 256; ++i) {
            uniq[bi * 256 + i] = -1;
            uidx[bi * 256 + i] = static_cast<int16_t>(-1);
        }
        int32_t U = 0;
        for (int i = 0; i < 256; ++i) {
            const int32_t gid = bt[bi * 256 + i];
            if (gid < 0) continue;
            int32_t found = -1;
            for (int32_t j = 0; j < U; ++j) {
                if (uniq[bi * 256 + j] == gid) {
                    found = j;
                    break;
                }
            }
            if (found < 0) {
                if (U < 256) {
                    uniq[bi * 256 + U] = gid;
                    found = U;
                    ++U;
                }
            }
            uidx[bi * 256 + i] = static_cast<int16_t>(found);
        }
        bucket_U[bi] = U;
    }

    // Encode buckets into a single MSB-first bitstream_u64.
    const uint16_t* bf16_u16 = reinterpret_cast<const uint16_t*>(bf16_params.data_ptr<at::BFloat16>());
    const uint8_t* fp8_u8 = fp8_params.data_ptr<uint8_t>();

    std::vector<uint64_t> bitstream;
    bitstream.reserve(1024);
    int64_t global_bitpos = 0;

    auto append_bits_msb = [&](uint32_t code, uint8_t len) {
        if (len == 0) return;
        const int64_t w = global_bitpos >> 6;
        const int o = static_cast<int>(global_bitpos & 63);
        while (static_cast<int64_t>(bitstream.size()) < w + 2) bitstream.push_back(0ull);
        unsigned __int128 window = (static_cast<unsigned __int128>(bitstream[static_cast<size_t>(w)]) << 64) |
                                   static_cast<unsigned __int128>(bitstream[static_cast<size_t>(w + 1)]);
        const unsigned __int128 v = static_cast<unsigned __int128>(code) & ((static_cast<unsigned __int128>(1u) << len) - 1u);
        const int shift = 128 - (o + static_cast<int>(len));
        window |= (v << shift);
        bitstream[static_cast<size_t>(w)] = static_cast<uint64_t>(window >> 64);
        bitstream[static_cast<size_t>(w + 1)] = static_cast<uint64_t>(window);
        global_bitpos += static_cast<int64_t>(len);
    };

    int64_t* bucket_base_out = bucket_bit_base_i64.data_ptr<int64_t>();
    uint32_t* thread_start_out = bucket_thread_startbit_u32.data_ptr<uint32_t>();

    for (int64_t bi = 0; bi < B; ++bi) {
        // 64-bit align each bucket for simplicity.
        if (global_bitpos & 63) {
            global_bitpos = (global_bitpos + 63) & ~63LL;
        }
        const int64_t bucket_base = global_bitpos;
        bucket_base_out[bi] = bucket_base;

        const Bucket& buck = buckets[static_cast<size_t>(bi)];
        const int32_t row_count = buck.row_count;
        const int64_t total_elems = static_cast<int64_t>(row_count) * static_cast<int64_t>(buck.col_len);
        const int32_t k_eff = static_cast<int32_t>(std::min<int64_t>(static_cast<int64_t>(k_threads), std::max<int64_t>(1, total_elems)));

        // Emit thread start bits. We lay out the bucket bitstream as a concatenation of per-thread substreams.
        // Each thread t encodes elements e = t + i*k_eff (flattened bucket row-major index).
        // This keeps each thread's Huffman bits contiguous while improving warp memory coalescing in decode.
        uint32_t* starts = thread_start_out + bi * (static_cast<int64_t>(k_threads) + 1);
        for (int32_t t = 0; t <= k_threads; ++t) starts[t] = 0u;

        // First pass: compute number of bits produced by each active thread.
        std::vector<int64_t> bits_per_thread(static_cast<size_t>(k_eff), 0);
        const uint16_t sc = bucket_scales[static_cast<size_t>(bi)];
        const int32_t col_start = buck.col_start;
        const int32_t col_len = buck.col_len;
        for (int32_t t = 0; t < k_eff; ++t) {
            int64_t bits = 0;
            for (int64_t e = t; e < total_elems; e += static_cast<int64_t>(k_eff)) {
            const int64_t rr = e / static_cast<int64_t>(col_len);
            const int64_t cc = e - rr * static_cast<int64_t>(col_len);
                const int32_t row = row_indices[static_cast<size_t>(buck.row_start) + static_cast<size_t>(rr)];
            const int64_t idx = static_cast<int64_t>(row) * m + static_cast<int64_t>(col_start) + cc;
                const uint8_t q = fp8_u8[idx];
                const uint16_t sym = bf16_bits_at_u16ptr(bf16_u16, idx);
                const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(q);
                auto it_key = enc_by_key.find(key);
                if (it_key == enc_by_key.end()) {
                    throw std::runtime_error("missing enc table for (scale,fp8) key");
                }
                const auto& enc = it_key->second;
                auto it_sym = enc.find(sym);
                if (it_sym == enc.end()) {
                    auto it_esc = esc_sym_by_key.find(key);
                    if (it_esc == esc_sym_by_key.end()) {
                        throw std::runtime_error("symbol not found and no ESC for (scale,fp8) key");
                    }
                    const uint16_t esc_sym = it_esc->second;
                    const uint32_t packed = enc.at(esc_sym);
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    bits += static_cast<int64_t>(len) + 16;
                } else {
                    const uint32_t packed = it_sym->second;
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    bits += static_cast<int64_t>(len);
                }
            }
            bits_per_thread[static_cast<size_t>(t)] = bits;
        }

        // Prefix sum -> starts[t]
        int64_t rel = 0;
        starts[0] = 0u;
        for (int32_t t = 0; t < k_eff; ++t) {
            rel += bits_per_thread[static_cast<size_t>(t)];
            starts[t + 1] = static_cast<uint32_t>(rel);
        }
        // For inactive threads, point them to the end.
        for (int32_t t = k_eff + 1; t <= k_threads; ++t) {
            starts[t] = starts[k_eff];
        }

        // Second pass: actually append codes in the same per-thread substream order.
        for (int32_t t = 0; t < k_eff; ++t) {
            for (int64_t e = t; e < total_elems; e += static_cast<int64_t>(k_eff)) {
                const int64_t rr = e / static_cast<int64_t>(col_len);
                const int64_t cc = e - rr * static_cast<int64_t>(col_len);
                const int32_t row = row_indices[static_cast<size_t>(buck.row_start) + static_cast<size_t>(rr)];
                const int64_t idx = static_cast<int64_t>(row) * m + static_cast<int64_t>(col_start) + cc;
                const uint8_t q = fp8_u8[idx];
                const uint16_t sym = bf16_bits_at_u16ptr(bf16_u16, idx);
                const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(q);
                const auto& enc = enc_by_key.at(key);
                auto it_sym = enc.find(sym);
                if (it_sym == enc.end()) {
                    auto it_esc = esc_sym_by_key.find(key);
                    if (it_esc == esc_sym_by_key.end()) {
                        throw std::runtime_error("symbol not found and no ESC for (scale,fp8) key");
                    }
                    const uint16_t esc_sym = it_esc->second;
                    const uint32_t packed = enc.at(esc_sym);
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    const uint32_t code = packed & 0xFFFFu;
                    append_bits_msb(code, len);
                    append_bits_msb(static_cast<uint32_t>(sym), 16);
                } else {
                    const uint32_t packed = it_sym->second;
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    const uint32_t code = packed & 0xFFFFu;
                    append_bits_msb(code, len);
                }
            }
        }

        // Sanity: the relative end should match the produced bits.
        if (static_cast<uint32_t>(global_bitpos - bucket_base) != starts[k_eff]) {
            throw std::runtime_error("internal error: bucket bit length mismatch");
        }
    }
    bucket_base_out[B] = global_bitpos;

    const int64_t nwords = (global_bitpos + 63) >> 6;
    bitstream.resize(static_cast<size_t>(nwords));
    torch::Tensor bitstream_u64 = torch::empty({nwords}, opts_u64);
    if (nwords > 0) {
        std::memcpy(bitstream_u64.data_ptr<uint64_t>(), bitstream.data(), static_cast<size_t>(nwords) * sizeof(uint64_t));
    }

    // modify
    torch::Tensor bitstream_tmp = torch::zeros_like(bitstream_u64);
    std::memcpy(bitstream_tmp.data_ptr<uint64_t>(), bitstream.data(), static_cast<size_t>(nwords) * sizeof(uint64_t));
    int ids = 0;
    int tot_len = 0;
    for (int bi = 0; bi < B; ++bi) {
        int mx_len = 0;
        for (int tid = 0; tid < threads_per_block; ++tid) {
            const uint32_t rel_start = (bucket_thread_startbit_u32.data_ptr<uint32_t>()[bi * (threads_per_block + 1) + tid] + 63) >> 6;
            const uint32_t next_start = (bucket_thread_startbit_u32.data_ptr<uint32_t>()[bi * (threads_per_block + 1) + (tid + 1)] + 63) >> 6;
            if (tid == 0 && next_start > rel_start && next_start - rel_start < 5) break;
            int len = static_cast<int>(next_start - rel_start + 1);
            mx_len = max(mx_len, len);
        }
        if (mx_len > 0) {
            const uint32_t start_next_block = (bucket_base_out[bi + 1] + 63) >> 6;
            const uint32_t start_block = (bucket_base_out[bi] + 63) >> 6;
            tot_len += mx_len * threads_per_block - static_cast<int>(start_next_block - start_block);
        }
        printf("added len: %d, last len: %d, mx_len: %d, start: %llu\n", tot_len, nwords, mx_len, bucket_base_out[bi]);
    }

    // NOTE: returning a larger tuple; wrapper will expose a dict.

    return std::make_tuple(
        decode_syms_u16,
        decode_lens_u8,
        table_scale_u16,
        table_fp8_u16,
        bucket_scale_u16,
        bucket_row_offsets_i32,
        row_indices_i32,
        bucket_table_i32,
        bucket_col_start_i32,
        bucket_col_len_i32,
        bucket_thread_startbit_u32,
        bucket_bit_base_i64,
        bitstream_u64,
        shape_nmk_i64,
        bucket_U_i32,
        bucket_unique_gid_i32,
        bucket_fp8_uidx_i16);
}

// -----------------------------------------------------------------------------
// AWQ int4 + scale shared-table Huffman build (CPU-side)
//
// Design notes (repo-local):
// - We build independent Huffman codebooks per stream, where a stream is defined by
//   (scale_id, qv) with qv in [0,15]. This matches the shared-table decoder, which
//   groups chunks by stream and caches each stream's dense decode table in shared memory.
// - Input tensors:
//     bf16_params: bfloat16 [n,m] CPU
//     int4_values_u8: uint8 [n,m] CPU, values in [0,15]
//     scales_u16: uint16 CPU, allowed shapes:
//         - scalar
//         - per-row [n]
//         - 2D block-wise [n/bs0, m/bs1] (requires exact divisibility)
//   For AWQ W4A16 common case (group_size=128 along input dim), callers can pass
//   2D scales as [n, m/128] (i.e. nb0=n, nb1=m/128) so bs0=1, bs1=128.
// - Output artifacts:
//     decode_syms_i16 / decode_nbits_u8 : [num_streams, 2^L] CPU
//     scale_values_u16: [num_scales] CPU
//     scale_bit_base_i64: [num_scales+1] CPU, bit offset base for each scale's bitstream
//     chunk_*: chunked layout metadata (CPU)
//     stream_chunk_*: stream index for decoding order (CPU)
//     bitstream_u64: concatenated MSB-first bitstream (CPU)
//
// Chunk format:
//   - We partition flattened indices into fixed-size chunks (take<=255).
//   - chunk_meta_u32: low 8 bits = qv, high 24 bits = take (#symbols)
//   - chunk_scale_id_i32: scale_id for this chunk
//   - chunk_out_base_u32: base position in out_idx_i32 array
//   - chunk_startbit_rel_u32: start bit relative to scale_bit_base_i64[scale_id]
//   - out_idx_i32: mapping from flat position -> destination index in output tensor
//
using BuildAWQTablesRet = std::tuple<
    torch::Tensor,  // decode_syms_i16 [S*16, 2^L]
    torch::Tensor,  // decode_nbits_u8 [S*16, 2^L]
    torch::Tensor,  // scale_values_u16 [S]
    torch::Tensor,  // scale_bit_base_i64 [S+1]
    torch::Tensor,  // chunk_startbit_rel_u32 [C]
    torch::Tensor,  // chunk_out_base_u32 [C]
    torch::Tensor,  // chunk_meta_u32 [C]
    torch::Tensor,  // chunk_scale_id_i32 [C]
    torch::Tensor,  // out_idx_i32 [n*m]
    torch::Tensor,  // stream_chunk_ofs_i64 [S*16+1]
    torch::Tensor,  // stream_chunk_ids_i32 [C]
    torch::Tensor   // bitstream_u64 [W]
>;

// -----------------------------------------------------------------------------
// AWQ int4 + scale bucket-based Huffman build (CPU-side)
//
// This mirrors the FP8 bucket build layout so the decode kernel can use the same
// scheduling strategy (one block per bucket, per-thread contiguous bit ranges).
//
// Key differences from FP8:
// - Symbol selector is qv in [0,15] (int4 value), not fp8 in [0,255].
// - max_len is typically 11.
// - We keep packed 4-bit code lengths, but represent ESC via len_nibble==15.
//   To make that decodable without storing the real Huffman length elsewhere,
//   we force the ESC symbol code length to exactly max_len.
//
using BuildAWQInt4BucketTablesRet = std::tuple<
    torch::Tensor,  // decode_syms_u16 [G, 2^L] uint16 (bf16 raw bits)
    torch::Tensor,  // decode_lens_u8  [G, 2^(L-1)] uint8 (packed 2x4bit lengths)
    torch::Tensor,  // table_scale_u16 [G] uint16
    torch::Tensor,  // table_qv_u16    [G] uint16 (qv in low bits)
    torch::Tensor,  // bucket_scale_u16 [B] uint16
    torch::Tensor,  // bucket_row_offsets_i32 [B+1]
    torch::Tensor,  // row_indices_i32 [R]
    torch::Tensor,  // bucket_table_i32 [B,16] (qv->gid)
    torch::Tensor,  // bucket_col_start_i32 [B]
    torch::Tensor,  // bucket_col_len_i32 [B]
    torch::Tensor,  // bucket_thread_startbit_u32 [B,k+1]
    torch::Tensor,  // bucket_bit_base_i64 [B+1]
    torch::Tensor,  // bitstream_u64 [W]
    torch::Tensor   // shape_nmk_i64 [3] (n,m,k)
>;

// -----------------------------------------------------------------------------
// AWQ int4 + scale bucket-based Huffman build (CPU-side), bucketing by exact scale.
//
// This is a "ragged" bucket layout: each bucket contains up to `group_cap` quantization
// groups that share the same scale_u16. Each group is identified by (row, col_start),
// and expands to `group_size` consecutive elements.
//
// This reduces the number of buckets from ~n*(m/group_size) down to O(#unique_scales),
// at the cost of more irregular memory access.
//
using BuildAWQInt4BucketByScaleRet = std::tuple<
    torch::Tensor,  // decode_syms_u16 [G, 2^L] uint16
    torch::Tensor,  // decode_lens_u8  [G, 2^(L-1)] uint8 (packed)
    torch::Tensor,  // table_scale_u16 [G] uint16
    torch::Tensor,  // table_qv_u16    [G] uint16
    torch::Tensor,  // bucket_scale_u16 [B] uint16
    torch::Tensor,  // bucket_group_offsets_i32 [B+1]
    torch::Tensor,  // group_rows_i32 [NG]
    torch::Tensor,  // group_col_start_i32 [NG]
    torch::Tensor,  // bucket_table_i32 [B,16] (qv->gid)
    torch::Tensor,  // bucket_thread_startbit_u32 [B,k+1]
    torch::Tensor,  // bucket_bit_base_i64 [B+1]
    torch::Tensor,  // bitstream_u64 [W]
    torch::Tensor   // shape_nmkg_i64 [4] (n,m,k,group_size)
>;

BuildAWQInt4BucketByScaleRet
huffman_build_awq_int4_bucket_decode_tables_by_scale_entry(
    torch::Tensor bf16_params,
    torch::Tensor int4_values_u8,
    torch::Tensor scales_u16,
    int64_t max_len,
    int64_t threads_per_block,
    int64_t group_cap) {
    if (!bf16_params.defined() || !int4_values_u8.defined() || !scales_u16.defined()) {
        throw std::invalid_argument("inputs must be defined");
    }
    if (bf16_params.device().is_cuda() || int4_values_u8.device().is_cuda() || scales_u16.device().is_cuda()) {
        throw std::invalid_argument("CPU tensors only");
    }
    if (bf16_params.scalar_type() != torch::kBFloat16) {
        throw std::invalid_argument("bf16_params must be torch.bfloat16");
    }
    if (int4_values_u8.scalar_type() != torch::kUInt8) {
        throw std::invalid_argument("int4_values_u8 must be torch.uint8");
    }
    if (scales_u16.scalar_type() != torch::kUInt16) {
        throw std::invalid_argument("scales_u16 must be torch.uint16 (bf16 raw bits)");
    }
    const int L = static_cast<int>(max_len);
    if (L <= 0 || L > 15) {
        throw std::invalid_argument("max_len must be in (0,15]");
    }
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        throw std::invalid_argument("threads_per_block must be in [1,1024]");
    }
    if (group_cap <= 0) {
        throw std::invalid_argument("group_cap must be > 0");
    }
    if (bf16_params.dim() != 2) {
        throw std::invalid_argument("bf16_params must be 2D [n,m]");
    }
    if (int4_values_u8.sizes() != bf16_params.sizes()) {
        throw std::invalid_argument("int4_values_u8 shape must match bf16_params");
    }

    bf16_params = bf16_params.contiguous();
    int4_values_u8 = int4_values_u8.contiguous();
    scales_u16 = scales_u16.contiguous();

    const int64_t n = bf16_params.size(0);
    const int64_t m = bf16_params.size(1);
    if (n <= 0 || m <= 0) {
        throw std::invalid_argument("bf16_params shape must be non-empty");
    }

    // Validate / interpret scale broadcasting.
    const int64_t scale_numel = scales_u16.numel();
    const bool scale_scalar = (scale_numel == 1);
    const bool scale_per_row = (!scale_scalar && scales_u16.dim() == 1 && scale_numel == n);
    bool scale_block2d = false;
    int64_t nb0 = -1, nb1 = -1, sb0 = -1, sb1 = -1;
    if (!scale_scalar && !scale_per_row && scales_u16.dim() == 2) {
        nb0 = scales_u16.size(0);
        nb1 = scales_u16.size(1);
        if (nb0 > 0 && nb1 > 0 && (n % nb0 == 0) && (m % nb1 == 0)) {
            sb0 = n / nb0;
            sb1 = m / nb1;
            if (sb0 > 0 && sb1 > 0) {
                scale_block2d = true;
            }
        }
    }
    if (!scale_scalar && !scale_per_row && !scale_block2d) {
        throw std::invalid_argument(
            "unsupported scales_u16 shape: use scalar, per-row (n,), or 2D block-wise [n/bs0,m/bs1]");
    }

    const int32_t group_size = static_cast<int32_t>(scale_block2d ? sb1 : m);
    const int32_t groups_per_row = static_cast<int32_t>(scale_block2d ? nb1 : 1);
    if (group_size <= 0 || groups_per_row <= 0) {
        throw std::invalid_argument("invalid derived group_size/groups_per_row");
    }

    auto get_scale_u16_rc = [&](int64_t r, int64_t c) -> uint16_t {
        if (scale_scalar) return scales_u16.data_ptr<uint16_t>()[0];
        if (scale_per_row) return scales_u16.data_ptr<uint16_t>()[r];
        const int64_t br = r / sb0;
        const int64_t bc = c / sb1;
        const int64_t sidx = br * nb1 + bc;
        return scales_u16.data_ptr<uint16_t>()[sidx];
    };

    // ---------------------------------------------------------------------
    // Build per-(scale_u16, qv) Huffman decode tables (same as rectangular build).
    // ---------------------------------------------------------------------
    const uint16_t* bf16_u16 = reinterpret_cast<const uint16_t*>(bf16_params.data_ptr<at::BFloat16>());
    const uint8_t* qv_u8 = int4_values_u8.data_ptr<uint8_t>();

    std::unordered_map<uint32_t, std::unordered_map<uint16_t, uint32_t>> freq_by_key;
    freq_by_key.reserve(1024);

    for (int64_t r = 0; r < n; ++r) {
        for (int64_t c = 0; c < m; ++c) {
            const int64_t idx = r * m + c;
            const uint16_t sc = get_scale_u16_rc(r, c);
            const uint8_t qv = static_cast<uint8_t>(qv_u8[idx] & 0x0Fu);
            const uint16_t sym = bf16_u16[idx];
            const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(qv);
            freq_by_key[key][sym] += 1u;
        }
    }

    std::vector<uint32_t> keys;
    keys.reserve(freq_by_key.size());
    for (const auto& kv : freq_by_key) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    std::unordered_map<uint32_t, int32_t> key_to_gid;
    key_to_gid.reserve(keys.size() * 2 + 1);
    for (int32_t gi = 0; gi < static_cast<int32_t>(keys.size()); ++gi) {
        key_to_gid.emplace(keys[gi], gi);
    }

    const int64_t G = static_cast<int64_t>(keys.size());
    const int64_t table_size = 1LL << L;
    const int64_t packed_len_size = table_size / 2;

    auto opts_u16 = torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCPU);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);

    torch::Tensor decode_syms_u16 = torch::empty({G, table_size}, opts_u16);
    torch::Tensor decode_lens_u8 = torch::empty({G, packed_len_size}, opts_u8);
    torch::Tensor table_scale_u16 = torch::empty({G}, opts_u16);
    torch::Tensor table_qv_u16 = torch::empty({G}, opts_u16);

    auto* out_syms = decode_syms_u16.data_ptr<uint16_t>();
    auto* out_lens = decode_lens_u8.data_ptr<uint8_t>();
    auto* out_scale = table_scale_u16.data_ptr<uint16_t>();
    auto* out_qv = table_qv_u16.data_ptr<uint16_t>();

    std::unordered_map<uint32_t, std::unordered_map<uint16_t, uint32_t>> enc_by_key;
    enc_by_key.reserve(freq_by_key.size() * 2 + 1);
    std::unordered_map<uint32_t, uint16_t> esc_sym_by_key;
    esc_sym_by_key.reserve(freq_by_key.size() / 8 + 1);

    std::vector<uint16_t> decode_syms_vec;
    std::vector<uint8_t> decode_len_unpacked;
    std::vector<uint8_t> decode_len_packed;

    for (int64_t gi = 0; gi < G; ++gi) {
        const uint32_t key = keys[gi];
        const uint16_t sc = static_cast<uint16_t>(key >> 16);
        const uint16_t qv = static_cast<uint16_t>(key & 0xFFFFu);
        out_scale[gi] = sc;
        out_qv[gi] = qv;

        const auto& freq = freq_by_key.at(key);
        std::vector<std::pair<int, uint16_t>> items;
        items.reserve(freq.size());
        for (const auto& sv : freq) {
            items.emplace_back(static_cast<int>(sv.second), sv.first);
        }

        if (items.empty()) {
            decode_syms_vec.assign(static_cast<size_t>(table_size), 0);
            decode_len_unpacked.assign(static_cast<size_t>(table_size), 0);
            enc_by_key[key] = {};
        } else if (items.size() == 1) {
            const uint16_t sym = items[0].second;
            decode_syms_vec.assign(static_cast<size_t>(table_size), sym);
            decode_len_unpacked.assign(static_cast<size_t>(table_size), 1);
            auto& enc = enc_by_key[key];
            enc.reserve(1);
            enc[sym] = (static_cast<uint32_t>(1u) << 16) | 0u;
        } else {
            std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });

            bool use_esc = false;
            uint16_t esc_sym = 0;
            int esc_freq = 0;

            if (items.size() > static_cast<size_t>(table_size)) {
                use_esc = true;
                std::vector<std::pair<int, uint16_t>> items_desc = items;
                std::stable_sort(items_desc.begin(), items_desc.end(), [](const auto& a, const auto& b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second < b.second;
                });
                const size_t keep_k = static_cast<size_t>(table_size - 1);
                std::unordered_set<uint16_t> keep;
                keep.reserve(keep_k * 2 + 1);
                for (size_t i = 0; i < keep_k; ++i) {
                    keep.insert(items_desc[i].second);
                }
                for (const auto& it : items_desc) {
                    if (keep.find(it.second) == keep.end()) {
                        esc_freq += it.first;
                    }
                }
                for (uint32_t cand = 0; cand <= 0xFFFFu; ++cand) {
                    const uint16_t c16 = static_cast<uint16_t>(cand);
                    if (freq.find(c16) == freq.end()) {
                        esc_sym = c16;
                        break;
                    }
                }
                if (esc_sym == 0 && freq.find(static_cast<uint16_t>(0)) != freq.end()) {
                    // Fallback (should be extremely unlikely).
                    esc_sym = static_cast<uint16_t>(0xFFFFu);
                }

                std::vector<std::pair<int, uint16_t>> filtered;
                filtered.reserve(keep_k + 1);
                for (const auto& it : items_desc) {
                    if (keep.find(it.second) != keep.end()) {
                        filtered.emplace_back(it.first, it.second);
                    }
                }
                filtered.emplace_back(esc_freq, esc_sym);
                items.swap(filtered);
                esc_sym_by_key[key] = esc_sym;
            }

            // Build length-limited Huffman codes and decode tables.
            // Reuse existing helpers in this TU (package-merge -> canonical codes).
            // NOTE: we force ESC code length to exactly L and encode ESC len nibble as 15.
            std::vector<int> weights;
            weights.reserve(items.size());
            for (const auto& it : items) weights.push_back(it.first);
            std::vector<int> lens_i = package_merge_code_lengths_from_weights(weights, L);
            if (lens_i.size() != items.size()) throw std::runtime_error("length vector size mismatch");

            std::vector<std::pair<uint16_t, uint8_t>> sym_lens;
            sym_lens.reserve(items.size());
            for (size_t si = 0; si < items.size(); ++si) {
                int li = lens_i[si];
                if (li <= 0 || li > L) throw std::runtime_error("invalid code length");
                const uint16_t sym = items[si].second;
                if (use_esc && sym == esc_sym) {
                    li = L;
                }
                sym_lens.emplace_back(sym, static_cast<uint8_t>(li));
            }

            auto codes = build_canonical_codes_u16(sym_lens);
            fill_decode_tables_u16(codes, L, &decode_syms_vec, &decode_len_unpacked);
            if (use_esc) {
                for (size_t i = 0; i < decode_syms_vec.size(); ++i) {
                    if (decode_syms_vec[i] == esc_sym) {
                        decode_len_unpacked[i] = static_cast<uint8_t>(15u);
                    }
                }
                esc_sym_by_key[key] = esc_sym;
            }

            // Build encoder map.
            auto& enc = enc_by_key[key];
            enc.clear();
            enc.reserve(codes.size() * 2 + 1);
            for (const auto& c : codes) {
                if (c.len == 0 || c.len > L) throw std::runtime_error("invalid code length in canonical codes");
                enc[c.sym] = (static_cast<uint32_t>(c.len) << 16) | static_cast<uint32_t>(c.code);
            }
        }

        // Pack lengths (2x4bit) and write to tensors.
        decode_len_packed.assign(static_cast<size_t>(packed_len_size), 0);
        for (int64_t i = 0; i < packed_len_size; ++i) {
            const uint8_t lo = decode_len_unpacked[static_cast<size_t>(2 * i + 0)] & 0x0Fu;
            const uint8_t hi = decode_len_unpacked[static_cast<size_t>(2 * i + 1)] & 0x0Fu;
            decode_len_packed[static_cast<size_t>(i)] = static_cast<uint8_t>(lo | (hi << 4));
        }

        std::memcpy(out_syms + gi * table_size, decode_syms_vec.data(), static_cast<size_t>(table_size) * sizeof(uint16_t));
        std::memcpy(out_lens + gi * packed_len_size, decode_len_packed.data(), static_cast<size_t>(packed_len_size) * sizeof(uint8_t));
    }

    // ---------------------------------------------------------------------
    // Build ragged buckets by exact scale_u16 over quantization groups.
    // Each group corresponds to (row, col_start) and expands to group_size elems.
    // ---------------------------------------------------------------------
    std::unordered_map<uint16_t, std::vector<std::pair<int32_t, int32_t>>> groups_by_scale;
    groups_by_scale.reserve(1024);

    for (int64_t r = 0; r < n; ++r) {
        for (int32_t bc = 0; bc < groups_per_row; ++bc) {
            const int32_t c0 = bc * group_size;
            const uint16_t sc = get_scale_u16_rc(r, static_cast<int64_t>(c0));
            groups_by_scale[sc].push_back({static_cast<int32_t>(r), c0});
        }
    }

    std::vector<uint16_t> unique_scales;
    unique_scales.reserve(groups_by_scale.size());
    for (const auto& kv : groups_by_scale) unique_scales.push_back(kv.first);
    std::sort(unique_scales.begin(), unique_scales.end());

    std::vector<uint16_t> bucket_scales;
    std::vector<int32_t> bucket_group_offsets;
    std::vector<int32_t> group_rows;
    std::vector<int32_t> group_col_start;
    bucket_group_offsets.push_back(0);

    for (uint16_t sc : unique_scales) {
        auto& v = groups_by_scale.at(sc);
        // deterministic order
        std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });

        for (size_t off = 0; off < v.size(); off += static_cast<size_t>(group_cap)) {
            const size_t take = std::min<size_t>(static_cast<size_t>(group_cap), v.size() - off);
            bucket_scales.push_back(sc);
            for (size_t i = 0; i < take; ++i) {
                group_rows.push_back(v[off + i].first);
                group_col_start.push_back(v[off + i].second);
            }
            bucket_group_offsets.push_back(static_cast<int32_t>(group_rows.size()));
        }
    }

    const int64_t B = static_cast<int64_t>(bucket_scales.size());
    int32_t k_threads = static_cast<int32_t>(threads_per_block);
    if (k_threads <= 0) k_threads = 1;

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU);
    auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU);

    torch::Tensor bucket_scale_u16 = torch::empty({B}, opts_u16);
    torch::Tensor bucket_group_offsets_i32 = torch::empty({B + 1}, opts_i32);
    torch::Tensor group_rows_i32 = torch::empty({static_cast<int64_t>(group_rows.size())}, opts_i32);
    torch::Tensor group_col_start_i32 = torch::empty({static_cast<int64_t>(group_col_start.size())}, opts_i32);
    torch::Tensor bucket_table_i32 = torch::empty({B, 16}, opts_i32);
    torch::Tensor bucket_thread_startbit_u32 = torch::empty({B, static_cast<int64_t>(k_threads) + 1}, opts_u32);
    torch::Tensor bucket_bit_base_i64 = torch::empty({B + 1}, opts_i64);
    torch::Tensor shape_nmkg_i64 = torch::empty({4}, opts_i64);

    shape_nmkg_i64.data_ptr<int64_t>()[0] = n;
    shape_nmkg_i64.data_ptr<int64_t>()[1] = m;
    shape_nmkg_i64.data_ptr<int64_t>()[2] = static_cast<int64_t>(k_threads);
    shape_nmkg_i64.data_ptr<int64_t>()[3] = static_cast<int64_t>(group_size);

    std::memcpy(bucket_scale_u16.data_ptr<uint16_t>(), bucket_scales.data(), bucket_scales.size() * sizeof(uint16_t));
    std::memcpy(bucket_group_offsets_i32.data_ptr<int32_t>(), bucket_group_offsets.data(), bucket_group_offsets.size() * sizeof(int32_t));
    std::memcpy(group_rows_i32.data_ptr<int32_t>(), group_rows.data(), group_rows.size() * sizeof(int32_t));
    std::memcpy(group_col_start_i32.data_ptr<int32_t>(), group_col_start.data(), group_col_start.size() * sizeof(int32_t));

    // Fill bucket_table_i32 (qv -> global table id) per bucket scale.
    int32_t* bt = bucket_table_i32.data_ptr<int32_t>();
    for (int64_t bi = 0; bi < B; ++bi) {
        const uint16_t sc = bucket_scales[static_cast<size_t>(bi)];
        for (int qv = 0; qv < 16; ++qv) {
            const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(qv);
            auto it = key_to_gid.find(key);
            bt[bi * 16 + qv] = (it == key_to_gid.end()) ? -1 : it->second;
        }
    }

    // Encode buckets into a single MSB-first bitstream_u64.
    std::vector<uint64_t> bitstream;
    bitstream.reserve(1024);
    int64_t global_bitpos = 0;
    auto append_bits_msb = [&](uint32_t code, uint8_t len) {
        if (len == 0) return;
        const int64_t w = global_bitpos >> 6;
        const int o = static_cast<int>(global_bitpos & 63);
        while (static_cast<int64_t>(bitstream.size()) < w + 2) bitstream.push_back(0ull);
        unsigned __int128 window = (static_cast<unsigned __int128>(bitstream[static_cast<size_t>(w)]) << 64) |
                                   static_cast<unsigned __int128>(bitstream[static_cast<size_t>(w + 1)]);
        const unsigned __int128 v = static_cast<unsigned __int128>(code) & ((static_cast<unsigned __int128>(1u) << len) - 1u);
        const int shift = 128 - (o + static_cast<int>(len));
        window |= (v << shift);
        bitstream[static_cast<size_t>(w)] = static_cast<uint64_t>(window >> 64);
        bitstream[static_cast<size_t>(w + 1)] = static_cast<uint64_t>(window);
        global_bitpos += static_cast<int64_t>(len);
    };

    int64_t* bucket_base_out = bucket_bit_base_i64.data_ptr<int64_t>();
    uint32_t* thread_start_out = bucket_thread_startbit_u32.data_ptr<uint32_t>();
    const int32_t* gro = group_rows.data();
    const int32_t* gcs = group_col_start.data();
    const int32_t* gofs = bucket_group_offsets.data();

    for (int64_t bi = 0; bi < B; ++bi) {
        if (global_bitpos & 63) {
            global_bitpos = (global_bitpos + 63) & ~63LL;
        }
        const int64_t bucket_base = global_bitpos;
        bucket_base_out[bi] = bucket_base;

        const uint16_t sc = bucket_scales[static_cast<size_t>(bi)];
        const int32_t ga = gofs[bi];
        const int32_t gb = gofs[bi + 1];
        const int32_t group_count = gb - ga;
        const int64_t total_elems = static_cast<int64_t>(group_count) * static_cast<int64_t>(group_size);
        const int32_t k_eff = static_cast<int32_t>(std::min<int64_t>(static_cast<int64_t>(k_threads), std::max<int64_t>(1, total_elems)));

        uint32_t* starts = thread_start_out + bi * (static_cast<int64_t>(k_threads) + 1);
        for (int32_t t = 0; t <= k_threads; ++t) starts[t] = 0u;

        std::vector<int64_t> bits_per_thread(static_cast<size_t>(k_eff), 0);
        for (int32_t t = 0; t < k_eff; ++t) {
            int64_t bits = 0;
            for (int64_t e = t; e < total_elems; e += static_cast<int64_t>(k_eff)) {
                const int64_t gi = e / static_cast<int64_t>(group_size);
                const int64_t j = e - gi * static_cast<int64_t>(group_size);
                const int32_t row = gro[static_cast<size_t>(ga + static_cast<int32_t>(gi))];
                const int32_t c0 = gcs[static_cast<size_t>(ga + static_cast<int32_t>(gi))];
                const int64_t idx = static_cast<int64_t>(row) * m + static_cast<int64_t>(c0) + j;
                const uint8_t qv = static_cast<uint8_t>(qv_u8[idx] & 0x0Fu);
                const uint16_t sym = bf16_u16[idx];
                const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(qv);
                auto it_key = enc_by_key.find(key);
                if (it_key == enc_by_key.end()) {
                    throw std::runtime_error("missing enc table for (scale,qv) key");
                }
                const auto& enc = it_key->second;
                auto it_sym = enc.find(sym);
                if (it_sym == enc.end()) {
                    auto it_esc = esc_sym_by_key.find(key);
                    if (it_esc == esc_sym_by_key.end()) {
                        throw std::runtime_error("symbol not found and no ESC for (scale,qv) key");
                    }
                    const uint16_t esc_sym = it_esc->second;
                    const uint32_t packed = enc.at(esc_sym);
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    bits += static_cast<int64_t>(len) + 16;
                } else {
                    const uint32_t packed = it_sym->second;
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    bits += static_cast<int64_t>(len);
                }
            }
            bits_per_thread[static_cast<size_t>(t)] = bits;
        }

        int64_t rel = 0;
        starts[0] = 0u;
        for (int32_t t = 0; t < k_eff; ++t) {
            rel += bits_per_thread[static_cast<size_t>(t)];
            starts[t + 1] = static_cast<uint32_t>(rel);
        }
        for (int32_t t = k_eff + 1; t <= k_threads; ++t) {
            starts[t] = starts[k_eff];
        }

        for (int32_t t = 0; t < k_eff; ++t) {
            for (int64_t e = t; e < total_elems; e += static_cast<int64_t>(k_eff)) {
                const int64_t gi = e / static_cast<int64_t>(group_size);
                const int64_t j = e - gi * static_cast<int64_t>(group_size);
                const int32_t row = gro[static_cast<size_t>(ga + static_cast<int32_t>(gi))];
                const int32_t c0 = gcs[static_cast<size_t>(ga + static_cast<int32_t>(gi))];
                const int64_t idx = static_cast<int64_t>(row) * m + static_cast<int64_t>(c0) + j;
                const uint8_t qv = static_cast<uint8_t>(qv_u8[idx] & 0x0Fu);
                const uint16_t sym = bf16_u16[idx];
                const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(qv);
                const auto& enc = enc_by_key.at(key);
                auto it_sym = enc.find(sym);
                if (it_sym == enc.end()) {
                    auto it_esc = esc_sym_by_key.find(key);
                    if (it_esc == esc_sym_by_key.end()) {
                        throw std::runtime_error("symbol not found and no ESC for (scale,qv) key");
                    }
                    const uint16_t esc_sym = it_esc->second;
                    const uint32_t packed = enc.at(esc_sym);
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    const uint32_t code = packed & 0xFFFFu;
                    append_bits_msb(code, len);
                    append_bits_msb(static_cast<uint32_t>(sym), 16);
                } else {
                    const uint32_t packed = it_sym->second;
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    const uint32_t code = packed & 0xFFFFu;
                    append_bits_msb(code, len);
                }
            }
        }

        if (static_cast<uint32_t>(global_bitpos - bucket_base) != starts[k_eff]) {
            throw std::runtime_error("internal error: bucket bit length mismatch");
        }
    }
    bucket_base_out[B] = global_bitpos;

    const int64_t nwords = (global_bitpos + 63) >> 6;
    bitstream.resize(static_cast<size_t>(nwords));
    torch::Tensor bitstream_u64 = torch::empty({nwords}, opts_u64);
    if (nwords > 0) {
        std::memcpy(bitstream_u64.data_ptr<uint64_t>(), bitstream.data(), static_cast<size_t>(nwords) * sizeof(uint64_t));
    }

    return std::make_tuple(
        decode_syms_u16,
        decode_lens_u8,
        table_scale_u16,
        table_qv_u16,
        bucket_scale_u16,
        bucket_group_offsets_i32,
        group_rows_i32,
        group_col_start_i32,
        bucket_table_i32,
        bucket_thread_startbit_u32,
        bucket_bit_base_i64,
        bitstream_u64,
        shape_nmkg_i64);
}

BuildAWQInt4BucketTablesRet
huffman_build_awq_int4_bucket_decode_tables_entry(
    torch::Tensor bf16_params,
    torch::Tensor int4_values_u8,
    torch::Tensor scales_u16,
    int64_t max_len,
    int64_t threads_per_block) {
    if (!bf16_params.defined() || !int4_values_u8.defined() || !scales_u16.defined()) {
        throw std::invalid_argument("inputs must be defined");
    }
    if (bf16_params.device().is_cuda() || int4_values_u8.device().is_cuda() || scales_u16.device().is_cuda()) {
        throw std::invalid_argument("CPU tensors only");
    }
    if (bf16_params.scalar_type() != torch::kBFloat16) {
        throw std::invalid_argument("bf16_params must be torch.bfloat16");
    }
    if (int4_values_u8.scalar_type() != torch::kUInt8) {
        throw std::invalid_argument("int4_values_u8 must be torch.uint8");
    }
    if (scales_u16.scalar_type() != torch::kUInt16) {
        throw std::invalid_argument("scales_u16 must be torch.uint16 (bf16 raw bits)");
    }
    const int L = static_cast<int>(max_len);
    if (L <= 0 || L > 15) {
        throw std::invalid_argument("max_len must be in (0,15]");
    }
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        throw std::invalid_argument("threads_per_block must be in [1,1024]");
    }
    if (bf16_params.dim() != 2) {
        throw std::invalid_argument("bf16_params must be 2D [n,m]");
    }
    if (int4_values_u8.sizes() != bf16_params.sizes()) {
        throw std::invalid_argument("int4_values_u8 shape must match bf16_params");
    }

    bf16_params = bf16_params.contiguous();
    int4_values_u8 = int4_values_u8.contiguous();
    scales_u16 = scales_u16.contiguous();

    const int64_t n = bf16_params.size(0);
    const int64_t m = bf16_params.size(1);
    if (n <= 0 || m <= 0) {
        throw std::invalid_argument("bf16_params shape must be non-empty");
    }

    // Validate / interpret scale broadcasting.
    const int64_t scale_numel = scales_u16.numel();
    const bool scale_scalar = (scale_numel == 1);
    const bool scale_per_row = (!scale_scalar && scales_u16.dim() == 1 && scale_numel == n);
    bool scale_block2d = false;
    int64_t nb0 = -1, nb1 = -1, sb0 = -1, sb1 = -1;
    if (!scale_scalar && !scale_per_row && scales_u16.dim() == 2) {
        nb0 = scales_u16.size(0);
        nb1 = scales_u16.size(1);
        if (nb0 > 0 && nb1 > 0 && (n % nb0 == 0) && (m % nb1 == 0)) {
            sb0 = n / nb0;
            sb1 = m / nb1;
            if (sb0 > 0 && sb1 > 0) {
                scale_block2d = true;
            }
        }
    }
    if (!scale_scalar && !scale_per_row && !scale_block2d) {
        throw std::invalid_argument(
            "unsupported scales_u16 shape: use scalar, per-row (n,), or 2D block-wise [n/bs0,m/bs1]");
    }

    auto get_scale_u16_rc = [&](int64_t r, int64_t c) -> uint16_t {
        if (scale_scalar) return scales_u16.data_ptr<uint16_t>()[0];
        if (scale_per_row) return scales_u16.data_ptr<uint16_t>()[r];
        // block2d
        const int64_t br = r / sb0;
        const int64_t bc = c / sb1;
        const int64_t sidx = br * nb1 + bc;
        return scales_u16.data_ptr<uint16_t>()[sidx];
    };

    // ---------------------------------------------------------------------
    // Build per-(scale_u16, qv) Huffman decode tables.
    // ---------------------------------------------------------------------
    const int64_t numel = n * m;
    const uint16_t* bf16_u16 = reinterpret_cast<const uint16_t*>(bf16_params.data_ptr<at::BFloat16>());
    const uint8_t* qv_u8 = int4_values_u8.data_ptr<uint8_t>();

    std::unordered_map<uint32_t, std::unordered_map<uint16_t, uint32_t>> freq_by_key;
    freq_by_key.reserve(1024);

    for (int64_t r = 0; r < n; ++r) {
        for (int64_t c = 0; c < m; ++c) {
            const int64_t idx = r * m + c;
            const uint16_t sc = get_scale_u16_rc(r, c);
            const uint8_t qv = static_cast<uint8_t>(qv_u8[idx] & 0x0Fu);
            const uint16_t sym = bf16_u16[idx];
            const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(qv);
            freq_by_key[key][sym] += 1u;
        }
    }

    std::vector<uint32_t> keys;
    keys.reserve(freq_by_key.size());
    for (const auto& kv : freq_by_key) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end());

    std::unordered_map<uint32_t, int32_t> key_to_gid;
    key_to_gid.reserve(keys.size() * 2 + 1);
    for (int32_t gi = 0; gi < static_cast<int32_t>(keys.size()); ++gi) {
        key_to_gid.emplace(keys[gi], gi);
    }

    const int64_t G = static_cast<int64_t>(keys.size());
    const int64_t table_size = 1LL << L;
    const int64_t packed_len_size = table_size / 2;

    auto opts_u16 = torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCPU);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);

    torch::Tensor decode_syms_u16 = torch::empty({G, table_size}, opts_u16);
    torch::Tensor decode_lens_u8 = torch::empty({G, packed_len_size}, opts_u8);
    torch::Tensor table_scale_u16 = torch::empty({G}, opts_u16);
    torch::Tensor table_qv_u16 = torch::empty({G}, opts_u16);

    auto* out_syms = decode_syms_u16.data_ptr<uint16_t>();
    auto* out_lens = decode_lens_u8.data_ptr<uint8_t>();
    auto* out_scale = table_scale_u16.data_ptr<uint16_t>();
    auto* out_qv = table_qv_u16.data_ptr<uint16_t>();

    struct EncCode {
        uint16_t sym;
        uint8_t len;
        uint32_t code;
    };
    std::unordered_map<uint32_t, std::unordered_map<uint16_t, uint32_t>> enc_by_key;
    enc_by_key.reserve(freq_by_key.size() * 2 + 1);
    std::unordered_map<uint32_t, uint16_t> esc_sym_by_key;
    esc_sym_by_key.reserve(freq_by_key.size() / 8 + 1);

    std::vector<uint16_t> decode_syms_vec;
    std::vector<uint8_t> decode_len_unpacked;
    std::vector<uint8_t> decode_len_packed;

    for (int64_t gi = 0; gi < G; ++gi) {
        const uint32_t key = keys[gi];
        const uint16_t sc = static_cast<uint16_t>(key >> 16);
        const uint16_t qv = static_cast<uint16_t>(key & 0xFFFFu);
        out_scale[gi] = sc;
        out_qv[gi] = qv;

        const auto& freq = freq_by_key.at(key);
        std::vector<std::pair<int, uint16_t>> items;
        items.reserve(freq.size());
        for (const auto& sv : freq) {
            items.emplace_back(static_cast<int>(sv.second), sv.first);
        }

        bool use_esc = false;
        uint16_t esc_sym = 0;
        int esc_freq = 0;

        if (items.empty()) {
            decode_syms_vec.assign(static_cast<size_t>(table_size), 0);
            decode_len_unpacked.assign(static_cast<size_t>(table_size), 0);
            enc_by_key[key] = {};
        } else if (items.size() == 1) {
            const uint16_t sym = items[0].second;
            decode_syms_vec.assign(static_cast<size_t>(table_size), sym);
            decode_len_unpacked.assign(static_cast<size_t>(table_size), 1);
            auto& enc = enc_by_key[key];
            enc.reserve(1);
            enc[sym] = (static_cast<uint32_t>(1u) << 16) | 0u;
        } else {
            std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
            });

            if (items.size() > static_cast<size_t>(table_size)) {
                use_esc = true;
                std::vector<std::pair<int, uint16_t>> items_desc = items;
                std::stable_sort(items_desc.begin(), items_desc.end(), [](const auto& a, const auto& b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second < b.second;
                });
                const size_t keep_k = static_cast<size_t>(table_size - 1);
                std::unordered_set<uint16_t> keep;
                keep.reserve(keep_k * 2 + 1);
                for (size_t i = 0; i < keep_k; ++i) {
                    keep.insert(items_desc[i].second);
                }
                for (const auto& it : items_desc) {
                    if (keep.find(it.second) == keep.end()) {
                        esc_freq += it.first;
                    }
                }
                for (uint32_t cand = 0; cand <= 0xFFFFu; ++cand) {
                    const uint16_t c16 = static_cast<uint16_t>(cand);
                    if (keep.find(c16) == keep.end()) {
                        esc_sym = c16;
                        break;
                    }
                }
                std::vector<std::pair<int, uint16_t>> rebuilt;
                rebuilt.reserve(keep_k + 1);
                for (const auto& it : items_desc) {
                    if (keep.find(it.second) != keep.end()) {
                        rebuilt.push_back(it);
                        if (rebuilt.size() == keep_k) break;
                    }
                }
                rebuilt.push_back({esc_freq > 0 ? esc_freq : 1, esc_sym});
                items.swap(rebuilt);

                std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
                    if (a.first != b.first) return a.first < b.first;
                    return a.second < b.second;
                });
            }

            std::vector<int> weights;
            weights.reserve(items.size());
            for (const auto& it : items) weights.push_back(it.first);
            std::vector<int> lens_i = package_merge_code_lengths_from_weights(weights, L);
            if (lens_i.size() != items.size()) throw std::runtime_error("length vector size mismatch");

            std::vector<std::pair<uint16_t, uint8_t>> sym_lens;
            sym_lens.reserve(items.size());
            for (size_t si = 0; si < items.size(); ++si) {
                int li = lens_i[si];
                if (li <= 0 || li > L) throw std::runtime_error("invalid code length");
                const uint16_t sym = items[si].second;
                if (use_esc && sym == esc_sym) {
                    // Force ESC symbol to have length == max_len, so we can represent
                    // ESC purely as len_nibble==15 in the packed length table.
                    li = L;
                }
                sym_lens.emplace_back(sym, static_cast<uint8_t>(li));
            }

            auto codes = build_canonical_codes_u16(sym_lens);
            fill_decode_tables_u16(codes, L, &decode_syms_vec, &decode_len_unpacked);
            if (use_esc) {
                // Mark ESC entries with len_nibble=15 (0xF).
                for (size_t i = 0; i < decode_syms_vec.size(); ++i) {
                    if (decode_syms_vec[i] == esc_sym) {
                        decode_len_unpacked[i] = static_cast<uint8_t>(15u);
                    }
                }
            }

            auto& enc = enc_by_key[key];
            enc.reserve(codes.size() * 2 + 1);
            for (const auto& c : codes) {
                if (c.len == 0 || c.len > L) throw std::runtime_error("invalid code length in canonical codes");
                enc[c.sym] = (static_cast<uint32_t>(c.len) << 16) | static_cast<uint32_t>(c.code);
            }
            if (use_esc) {
                esc_sym_by_key[key] = esc_sym;
            }
        }

        pack_decode_lengths_4bit(decode_len_unpacked, &decode_len_packed);
        std::memcpy(out_syms + gi * table_size, decode_syms_vec.data(), static_cast<size_t>(table_size) * sizeof(uint16_t));
        std::memcpy(out_lens + gi * packed_len_size, decode_len_packed.data(), static_cast<size_t>(packed_len_size) * sizeof(uint8_t));
    }

    // ---------------------------------------------------------------------
    // Build per-scale buckets (<=128 rows each), qv->table mapping, bitstream,
    // and per-thread start bit offsets (same layout as FP8).
    // ---------------------------------------------------------------------
    struct Bucket {
        uint16_t scale;
        int32_t row_start;
        int32_t row_count;
        int32_t col_start;
        int32_t col_len;
    };

    std::vector<int32_t> row_indices;
    std::vector<int32_t> bucket_row_offsets;
    std::vector<uint16_t> bucket_scales;
    std::vector<int32_t> bucket_col_start;
    std::vector<int32_t> bucket_col_len;
    std::vector<Bucket> buckets;

    bucket_row_offsets.push_back(0);

    if (!scale_block2d) {
        // Bucketing-by-row: group rows by scale (scalar/per-row).
        const bool scale_ok_for_bucket = scale_scalar || scale_per_row;
        if (!scale_ok_for_bucket) {
            throw std::invalid_argument("bucket packing requires scales_u16 to be scalar, per-row (n,), or 2D block-wise");
        }
        auto get_row_scale_u16 = [&](int64_t row) -> uint16_t {
            if (scale_scalar) return scales_u16.data_ptr<uint16_t>()[0];
            return scales_u16.data_ptr<uint16_t>()[row];
        };

        std::unordered_map<uint16_t, std::vector<int32_t>> rows_by_scale;
        rows_by_scale.reserve(static_cast<size_t>(n));
        for (int64_t r = 0; r < n; ++r) {
            const uint16_t sc = get_row_scale_u16(r);
            rows_by_scale[sc].push_back(static_cast<int32_t>(r));
        }
        std::vector<uint16_t> unique_scales;
        unique_scales.reserve(rows_by_scale.size());
        for (const auto& kv : rows_by_scale) unique_scales.push_back(kv.first);
        std::sort(unique_scales.begin(), unique_scales.end());

        for (uint16_t sc : unique_scales) {
            const auto& rows = rows_by_scale.at(sc);
            for (size_t off = 0; off < rows.size(); off += 128) {
                const size_t take = std::min<size_t>(128, rows.size() - off);
                const int32_t start = static_cast<int32_t>(row_indices.size());
                row_indices.insert(
                    row_indices.end(),
                    rows.begin() + static_cast<int64_t>(off),
                    rows.begin() + static_cast<int64_t>(off + take));
                buckets.push_back({sc, start, static_cast<int32_t>(take), 0, static_cast<int32_t>(m)});
                bucket_scales.push_back(sc);
                bucket_col_start.push_back(0);
                bucket_col_len.push_back(static_cast<int32_t>(m));
                bucket_row_offsets.push_back(static_cast<int32_t>(row_indices.size()));
            }
        }
    } else {
        // 2D block-wise bucketing: create buckets over (row-block, col-block).
        auto get_block_scale_u16 = [&](int64_t br, int64_t bc) -> uint16_t {
            const int64_t sidx = br * nb1 + bc;
            return scales_u16.data_ptr<uint16_t>()[sidx];
        };
        for (int64_t br = 0; br < nb0; ++br) {
            const int64_t r0 = br * sb0;
            const int64_t r1 = (br + 1) * sb0;
            std::vector<int32_t> rows;
            rows.reserve(static_cast<size_t>(sb0));
            for (int64_t r = r0; r < r1; ++r) rows.push_back(static_cast<int32_t>(r));

            for (int64_t bc = 0; bc < nb1; ++bc) {
                const uint16_t sc = get_block_scale_u16(br, bc);
                const int32_t c0 = static_cast<int32_t>(bc * sb1);
                const int32_t clen = static_cast<int32_t>(sb1);
                for (size_t off = 0; off < rows.size(); off += 128) {
                    const size_t take = std::min<size_t>(128, rows.size() - off);
                    const int32_t start = static_cast<int32_t>(row_indices.size());
                    row_indices.insert(
                        row_indices.end(),
                        rows.begin() + static_cast<int64_t>(off),
                        rows.begin() + static_cast<int64_t>(off + take));
                    buckets.push_back({sc, start, static_cast<int32_t>(take), c0, clen});
                    bucket_scales.push_back(sc);
                    bucket_col_start.push_back(c0);
                    bucket_col_len.push_back(clen);
                    bucket_row_offsets.push_back(static_cast<int32_t>(row_indices.size()));
                }
            }
        }
    }

    const int64_t B = static_cast<int64_t>(buckets.size());
    int32_t k_threads = static_cast<int32_t>(threads_per_block);
    if (k_threads <= 0) k_threads = 1;

    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU);
    auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU);

    torch::Tensor bucket_scale_u16 = torch::empty({B}, opts_u16);
    torch::Tensor bucket_row_offsets_i32 = torch::empty({B + 1}, opts_i32);
    torch::Tensor row_indices_i32 = torch::empty({static_cast<int64_t>(row_indices.size())}, opts_i32);
    torch::Tensor bucket_table_i32 = torch::empty({B, 16}, opts_i32);
    torch::Tensor bucket_col_start_i32 = torch::empty({B}, opts_i32);
    torch::Tensor bucket_col_len_i32 = torch::empty({B}, opts_i32);
    torch::Tensor bucket_thread_startbit_u32 = torch::empty({B, static_cast<int64_t>(k_threads) + 1}, opts_u32);
    torch::Tensor bucket_bit_base_i64 = torch::empty({B + 1}, opts_i64);
    torch::Tensor shape_nmk_i64 = torch::empty({3}, opts_i64);

    shape_nmk_i64.data_ptr<int64_t>()[0] = n;
    shape_nmk_i64.data_ptr<int64_t>()[1] = m;
    shape_nmk_i64.data_ptr<int64_t>()[2] = static_cast<int64_t>(k_threads);

    std::memcpy(bucket_row_offsets_i32.data_ptr<int32_t>(), bucket_row_offsets.data(), bucket_row_offsets.size() * sizeof(int32_t));
    std::memcpy(row_indices_i32.data_ptr<int32_t>(), row_indices.data(), row_indices.size() * sizeof(int32_t));
    for (int64_t bi = 0; bi < B; ++bi) {
        bucket_scale_u16.data_ptr<uint16_t>()[bi] = bucket_scales[static_cast<size_t>(bi)];
        bucket_col_start_i32.data_ptr<int32_t>()[bi] = bucket_col_start[static_cast<size_t>(bi)];
        bucket_col_len_i32.data_ptr<int32_t>()[bi] = bucket_col_len[static_cast<size_t>(bi)];
    }

    // Fill bucket_table_i32 (qv -> global table id).
    int32_t* bt = bucket_table_i32.data_ptr<int32_t>();
    for (int64_t bi = 0; bi < B; ++bi) {
        const uint16_t sc = bucket_scales[static_cast<size_t>(bi)];
        for (int qv = 0; qv < 16; ++qv) {
            const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(qv);
            auto it = key_to_gid.find(key);
            bt[bi * 16 + qv] = (it == key_to_gid.end()) ? -1 : it->second;
        }
    }

    // Encode buckets into a single MSB-first bitstream_u64.
    std::vector<uint64_t> bitstream;
    bitstream.reserve(1024);
    int64_t global_bitpos = 0;
    auto append_bits_msb = [&](uint32_t code, uint8_t len) {
        if (len == 0) return;
        const int64_t w = global_bitpos >> 6;
        const int o = static_cast<int>(global_bitpos & 63);
        while (static_cast<int64_t>(bitstream.size()) < w + 2) bitstream.push_back(0ull);
        unsigned __int128 window = (static_cast<unsigned __int128>(bitstream[static_cast<size_t>(w)]) << 64) |
                                   static_cast<unsigned __int128>(bitstream[static_cast<size_t>(w + 1)]);
        const unsigned __int128 v = static_cast<unsigned __int128>(code) & ((static_cast<unsigned __int128>(1u) << len) - 1u);
        const int shift = 128 - (o + static_cast<int>(len));
        window |= (v << shift);
        bitstream[static_cast<size_t>(w)] = static_cast<uint64_t>(window >> 64);
        bitstream[static_cast<size_t>(w + 1)] = static_cast<uint64_t>(window);
        global_bitpos += static_cast<int64_t>(len);
    };

    int64_t* bucket_base_out = bucket_bit_base_i64.data_ptr<int64_t>();
    uint32_t* thread_start_out = bucket_thread_startbit_u32.data_ptr<uint32_t>();

    for (int64_t bi = 0; bi < B; ++bi) {
        if (global_bitpos & 63) {
            global_bitpos = (global_bitpos + 63) & ~63LL;
        }
        const int64_t bucket_base = global_bitpos;
        bucket_base_out[bi] = bucket_base;

        const Bucket& buck = buckets[static_cast<size_t>(bi)];
        const int32_t row_count = buck.row_count;
        const int64_t total_elems = static_cast<int64_t>(row_count) * static_cast<int64_t>(buck.col_len);
        const int32_t k_eff = static_cast<int32_t>(std::min<int64_t>(static_cast<int64_t>(k_threads), std::max<int64_t>(1, total_elems)));

        uint32_t* starts = thread_start_out + bi * (static_cast<int64_t>(k_threads) + 1);
        for (int32_t t = 0; t <= k_threads; ++t) starts[t] = 0u;

        std::vector<int64_t> bits_per_thread(static_cast<size_t>(k_eff), 0);
        const uint16_t sc = bucket_scales[static_cast<size_t>(bi)];
        const int32_t col_start = buck.col_start;
        const int32_t col_len = buck.col_len;
        for (int32_t t = 0; t < k_eff; ++t) {
            int64_t bits = 0;
            for (int64_t e = t; e < total_elems; e += static_cast<int64_t>(k_eff)) {
                const int64_t rr = e / static_cast<int64_t>(col_len);
                const int64_t cc = e - rr * static_cast<int64_t>(col_len);
                const int32_t row = row_indices[static_cast<size_t>(buck.row_start) + static_cast<size_t>(rr)];
                const int64_t idx = static_cast<int64_t>(row) * m + static_cast<int64_t>(col_start) + cc;
                const uint8_t qv = static_cast<uint8_t>(qv_u8[idx] & 0x0Fu);
                const uint16_t sym = bf16_u16[idx];
                const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(qv);
                auto it_key = enc_by_key.find(key);
                if (it_key == enc_by_key.end()) {
                    throw std::runtime_error("missing enc table for (scale,qv) key");
                }
                const auto& enc = it_key->second;
                auto it_sym = enc.find(sym);
                if (it_sym == enc.end()) {
                    auto it_esc = esc_sym_by_key.find(key);
                    if (it_esc == esc_sym_by_key.end()) {
                        throw std::runtime_error("symbol not found and no ESC for (scale,qv) key");
                    }
                    const uint16_t esc_sym = it_esc->second;
                    const uint32_t packed = enc.at(esc_sym);
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    bits += static_cast<int64_t>(len) + 16;
                } else {
                    const uint32_t packed = it_sym->second;
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    bits += static_cast<int64_t>(len);
                }
            }
            bits_per_thread[static_cast<size_t>(t)] = bits;
        }

        int64_t rel = 0;
        starts[0] = 0u;
        for (int32_t t = 0; t < k_eff; ++t) {
            rel += bits_per_thread[static_cast<size_t>(t)];
            starts[t + 1] = static_cast<uint32_t>(rel);
        }
        for (int32_t t = k_eff + 1; t <= k_threads; ++t) {
            starts[t] = starts[k_eff];
        }

        for (int32_t t = 0; t < k_eff; ++t) {
            for (int64_t e = t; e < total_elems; e += static_cast<int64_t>(k_eff)) {
                const int64_t rr = e / static_cast<int64_t>(col_len);
                const int64_t cc = e - rr * static_cast<int64_t>(col_len);
                const int32_t row = row_indices[static_cast<size_t>(buck.row_start) + static_cast<size_t>(rr)];
                const int64_t idx = static_cast<int64_t>(row) * m + static_cast<int64_t>(col_start) + cc;
                const uint8_t qv = static_cast<uint8_t>(qv_u8[idx] & 0x0Fu);
                const uint16_t sym = bf16_u16[idx];
                const uint32_t key = (static_cast<uint32_t>(sc) << 16) | static_cast<uint32_t>(qv);
                const auto& enc = enc_by_key.at(key);
                auto it_sym = enc.find(sym);
                if (it_sym == enc.end()) {
                    auto it_esc = esc_sym_by_key.find(key);
                    if (it_esc == esc_sym_by_key.end()) {
                        throw std::runtime_error("symbol not found and no ESC for (scale,qv) key");
                    }
                    const uint16_t esc_sym = it_esc->second;
                    const uint32_t packed = enc.at(esc_sym);
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    const uint32_t code = packed & 0xFFFFu;
                    append_bits_msb(code, len);
                    append_bits_msb(static_cast<uint32_t>(sym), 16);
                } else {
                    const uint32_t packed = it_sym->second;
                    const uint8_t len = static_cast<uint8_t>(packed >> 16);
                    const uint32_t code = packed & 0xFFFFu;
                    append_bits_msb(code, len);
                }
            }
        }

        if (static_cast<uint32_t>(global_bitpos - bucket_base) != starts[k_eff]) {
            throw std::runtime_error("internal error: bucket bit length mismatch");
        }
    }
    bucket_base_out[B] = global_bitpos;

    const int64_t nwords = (global_bitpos + 63) >> 6;
    bitstream.resize(static_cast<size_t>(nwords));
    torch::Tensor bitstream_u64 = torch::empty({nwords}, opts_u64);
    if (nwords > 0) {
        std::memcpy(bitstream_u64.data_ptr<uint64_t>(), bitstream.data(), static_cast<size_t>(nwords) * sizeof(uint64_t));
    }

    return std::make_tuple(
        decode_syms_u16,
        decode_lens_u8,
        table_scale_u16,
        table_qv_u16,
        bucket_scale_u16,
        bucket_row_offsets_i32,
        row_indices_i32,
        bucket_table_i32,
        bucket_col_start_i32,
        bucket_col_len_i32,
        bucket_thread_startbit_u32,
        bucket_bit_base_i64,
        bitstream_u64,
        shape_nmk_i64);
}

static inline uint16_t scale_u16_at(
    const torch::Tensor& scales_u16,
    bool scale_scalar,
    bool scale_per_row,
    bool scale_block2d,
    int64_t n_rows,
    int64_t n_cols,
    int64_t nb0,
    int64_t nb1,
    int64_t sb0,
    int64_t sb1,
    int64_t r,
    int64_t c) {
    if (scale_scalar) {
        return scales_u16.data_ptr<uint16_t>()[0];
    }
    if (scale_per_row) {
        return scales_u16.data_ptr<uint16_t>()[r];
    }
    if (scale_block2d) {
        const int64_t br = r / sb0;
        const int64_t bc = c / sb1;
        const int64_t sidx = br * nb1 + bc;
        return scales_u16.data_ptr<uint16_t>()[sidx];
    }
    // Should be unreachable when validated.
    return scales_u16.data_ptr<uint16_t>()[0];
}

BuildAWQTablesRet
huffman_build_awq_decode_tables_entry(
    torch::Tensor bf16_params,
    torch::Tensor int4_values_u8,
    torch::Tensor scales_u16,
    int64_t max_len) {
    if (!bf16_params.defined() || !int4_values_u8.defined() || !scales_u16.defined()) {
        throw std::invalid_argument("inputs must be defined");
    }
    if (bf16_params.device().is_cuda() || int4_values_u8.device().is_cuda() || scales_u16.device().is_cuda()) {
        throw std::invalid_argument("CPU tensors only");
    }
    if (bf16_params.scalar_type() != torch::kBFloat16) {
        throw std::invalid_argument("bf16_params must be torch.bfloat16");
    }
    if (int4_values_u8.scalar_type() != torch::kUInt8) {
        throw std::invalid_argument("int4_values_u8 must be torch.uint8");
    }
    if (scales_u16.scalar_type() != torch::kUInt16) {
        throw std::invalid_argument("scales_u16 must be torch.uint16");
    }
    const int L = static_cast<int>(max_len);
    if (L <= 0 || L > 20) {
        throw std::invalid_argument("max_len must be in (0,20]");
    }
    if (bf16_params.dim() != 2) {
        throw std::invalid_argument("bf16_params must be 2D [n,m]");
    }
    if (int4_values_u8.sizes() != bf16_params.sizes()) {
        throw std::invalid_argument("int4_values_u8 shape must match bf16_params");
    }

    bf16_params = bf16_params.contiguous();
    int4_values_u8 = int4_values_u8.contiguous();
    scales_u16 = scales_u16.contiguous();

    const int64_t n = bf16_params.size(0);
    const int64_t m = bf16_params.size(1);
    if (n <= 0 || m <= 0) {
        throw std::invalid_argument("bf16_params shape must be non-empty");
    }

    // Validate / interpret scale broadcasting.
    const int64_t scale_numel = scales_u16.numel();
    const bool scale_scalar = (scale_numel == 1);
    const bool scale_per_row = (!scale_scalar && scales_u16.dim() == 1 && scale_numel == n);
    bool scale_block2d = false;
    int64_t nb0 = -1, nb1 = -1, sb0 = -1, sb1 = -1;
    if (!scale_scalar && !scale_per_row && scales_u16.dim() == 2) {
        nb0 = scales_u16.size(0);
        nb1 = scales_u16.size(1);
        if (nb0 > 0 && nb1 > 0 && (n % nb0 == 0) && (m % nb1 == 0)) {
            sb0 = n / nb0;
            sb1 = m / nb1;
            if (sb0 > 0 && sb1 > 0) {
                scale_block2d = true;
            }
        }
    }
    if (!scale_scalar && !scale_per_row && !scale_block2d) {
        throw std::invalid_argument(
            "unsupported scales_u16 shape: use scalar, per-row (n,), or 2D block-wise [n/bs0,m/bs1]");
    }

    // Map scale_u16 values to dense scale_id [0,S).
    // We keep scale_values_u16 sorted for determinism.
    std::vector<uint16_t> scale_values;
    if (scale_scalar) {
        scale_values.push_back(scales_u16.data_ptr<uint16_t>()[0]);
    } else {
        // Collect unique scale values by scanning the matrix broadcast pattern.
        std::unordered_map<uint16_t, int32_t> seen;
        seen.reserve(static_cast<size_t>(std::min<int64_t>(n * m, 1LL << 20)));
        for (int64_t r = 0; r < n; ++r) {
            // For block2d, scale can vary across columns; scan by blocks for speed.
            if (scale_block2d) {
                for (int64_t bc = 0; bc < nb1; ++bc) {
                    const int64_t c = bc * sb1;
                    const uint16_t sc = scale_u16_at(scales_u16, scale_scalar, scale_per_row, scale_block2d,
                                                    n, m, nb0, nb1, sb0, sb1, r, c);
                    if (seen.emplace(sc, 1).second) scale_values.push_back(sc);
                }
            } else {
                const uint16_t sc = scale_u16_at(scales_u16, scale_scalar, scale_per_row, scale_block2d,
                                                n, m, nb0, nb1, sb0, sb1, r, 0);
                if (seen.emplace(sc, 1).second) scale_values.push_back(sc);
            }
        }
    }
    std::sort(scale_values.begin(), scale_values.end());
    scale_values.erase(std::unique(scale_values.begin(), scale_values.end()), scale_values.end());
    const int64_t S = static_cast<int64_t>(scale_values.size());
    if (S <= 0) {
        throw std::runtime_error("no scales found");
    }
    std::unordered_map<uint16_t, int32_t> scale_to_id;
    scale_to_id.reserve(scale_values.size() * 2 + 1);
    for (int32_t i = 0; i < static_cast<int32_t>(scale_values.size()); ++i) {
        scale_to_id.emplace(scale_values[static_cast<size_t>(i)], i);
    }

    // ---------------------------------------------------------------------
    // Build Huffman codes per stream (scale_id, qv).
    // ---------------------------------------------------------------------
    // Frequency table: for each stream, count bf16 symbol bits.
    const int64_t num_streams = S * 16;
    std::vector<std::unordered_map<uint16_t, uint32_t>> freq(num_streams);
    for (auto& mref : freq) mref.reserve(256);

    const uint16_t* bf16_u16 = reinterpret_cast<const uint16_t*>(bf16_params.data_ptr<at::BFloat16>());
    const uint8_t* qv_u8 = int4_values_u8.data_ptr<uint8_t>();

    for (int64_t r = 0; r < n; ++r) {
        for (int64_t c = 0; c < m; ++c) {
            const int64_t idx = r * m + c;
            const uint8_t qv = static_cast<uint8_t>(qv_u8[idx] & 0x0Fu);
            const uint16_t sc = scale_u16_at(scales_u16, scale_scalar, scale_per_row, scale_block2d,
                                            n, m, nb0, nb1, sb0, sb1, r, c);
            const auto it = scale_to_id.find(sc);
            if (it == scale_to_id.end()) {
                throw std::runtime_error("scale_to_id lookup failed");
            }
            const int32_t sid = it->second;
            const int64_t stream = static_cast<int64_t>(sid) * 16 + static_cast<int64_t>(qv);
            const uint16_t sym = bf16_u16[idx];
            freq[static_cast<size_t>(stream)][sym] += 1u;
        }
    }

    const int64_t table_size = 1LL << L;
    auto opts_i16 = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
    auto opts_u16 = torch::TensorOptions().dtype(torch::kUInt16).device(torch::kCPU);
    auto opts_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    auto opts_i64 = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto opts_u32 = torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCPU);
    auto opts_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU);

    torch::Tensor decode_syms_i16 = torch::empty({num_streams, table_size}, opts_i16);
    torch::Tensor decode_nbits_u8 = torch::empty({num_streams, table_size}, opts_u8);

    // Encoder map per stream: sym -> packed(len<<16|code)
    std::vector<std::unordered_map<uint16_t, uint32_t>> enc(num_streams);
    for (auto& e : enc) e.reserve(512);

    // Escape support per stream when unique symbols exceed 2^L.
    std::vector<uint16_t> esc_sym_by_stream(static_cast<size_t>(num_streams), 0);
    std::vector<uint8_t> has_esc_by_stream(static_cast<size_t>(num_streams), 0);

    std::vector<uint16_t> decode_syms_vec;
    std::vector<uint8_t> decode_len_unpacked;
    decode_syms_vec.reserve(static_cast<size_t>(table_size));
    decode_len_unpacked.reserve(static_cast<size_t>(table_size));

    for (int64_t s = 0; s < num_streams; ++s) {
        const auto& f = freq[static_cast<size_t>(s)];
        std::vector<std::pair<int, uint16_t>> items;
        items.reserve(f.size());
        for (const auto& kv : f) {
            items.emplace_back(static_cast<int>(kv.second), kv.first);
        }

        // If unique symbols exceed 2^L, fall back to an escape scheme:
        // - Keep the top-(2^L-1) most frequent symbols.
        // - Aggregate remaining frequency into one ESC symbol.
        // - Encode rare symbols as ESC + raw16bits.
        bool has_escape = false;
        uint16_t esc_sym = 0;
        if (items.size() > static_cast<size_t>(table_size)) {
            const size_t keep = static_cast<size_t>(table_size) - 1;
            std::vector<std::pair<int, uint16_t>> items_desc = items;
            std::stable_sort(items_desc.begin(), items_desc.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });

            std::unordered_map<uint16_t, uint8_t> kept;
            kept.reserve(keep * 2 + 1);

            std::vector<std::pair<int, uint16_t>> reduced;
            reduced.reserve(keep + 1);
            uint64_t esc_count_u64 = 0;
            for (size_t i = 0; i < items_desc.size(); ++i) {
                if (reduced.size() < keep) {
                    reduced.push_back(items_desc[i]);
                    kept.emplace(items_desc[i].second, 1);
                } else {
                    esc_count_u64 += static_cast<uint64_t>(std::max<int>(items_desc[i].first, 0));
                }
            }

            // Pick an ESC symbol not used by the kept set.
            uint16_t cand = 0;
            while (kept.find(cand) != kept.end()) {
                ++cand;
            }
            esc_sym = cand;
            const int esc_count = (esc_count_u64 > static_cast<uint64_t>(INT32_MAX))
                                    ? INT32_MAX
                                    : static_cast<int>(esc_count_u64);
            reduced.emplace_back(esc_count > 0 ? esc_count : 1, esc_sym);

            items.swap(reduced);
            has_escape = true;
        }

        // Deterministic ordering before package-merge.
        std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });

        auto* out_syms = decode_syms_i16.data_ptr<int16_t>() + s * table_size;
        auto* out_lens = decode_nbits_u8.data_ptr<uint8_t>() + s * table_size;

        if (items.empty()) {
            // No symbols in this stream: define all-zero table.
            std::memset(out_syms, 0, static_cast<size_t>(table_size) * sizeof(int16_t));
            std::memset(out_lens, 0, static_cast<size_t>(table_size) * sizeof(uint8_t));
            enc[static_cast<size_t>(s)].clear();
            continue;
        }
        if (items.size() == 1) {
            const uint16_t sym = items[0].second;
            for (int64_t i = 0; i < table_size; ++i) {
                out_syms[i] = static_cast<int16_t>(sym);
                out_lens[i] = 1;
            }
            enc[static_cast<size_t>(s)][sym] = (static_cast<uint32_t>(1u) << 16) | 0u;
            continue;
        }

        std::vector<int> weights;
        weights.reserve(items.size());
        for (const auto& it : items) weights.push_back(it.first);
        std::vector<int> lens_i = package_merge_code_lengths_from_weights(weights, L);
        if (lens_i.size() != items.size()) throw std::runtime_error("length vector size mismatch");

        std::vector<std::pair<uint16_t, uint8_t>> sym_lens;
        sym_lens.reserve(items.size());
        for (size_t i = 0; i < items.size(); ++i) {
            const int li = lens_i[i];
            if (li <= 0 || li > L) throw std::runtime_error("invalid code length");
            sym_lens.emplace_back(items[i].second, static_cast<uint8_t>(li));
        }
        auto codes = build_canonical_codes_u16(sym_lens);
        std::vector<uint16_t> decode_sym_u16;
        fill_decode_tables_u16(codes, L, &decode_sym_u16, &decode_len_unpacked);
        // Write
        for (int64_t i = 0; i < table_size; ++i) {
            const uint16_t sym_u16 = decode_sym_u16[static_cast<size_t>(i)];
            const uint8_t len_u8 = decode_len_unpacked[static_cast<size_t>(i)];
            out_syms[i] = static_cast<int16_t>(sym_u16);
            if (has_escape && sym_u16 == esc_sym) {
                // Mark ESC entries by setting the top bit; actual length is in low 7 bits.
                out_lens[i] = static_cast<uint8_t>(len_u8 | 0x80u);
            } else {
                out_lens[i] = len_u8;
            }
        }
        auto& enc_map = enc[static_cast<size_t>(s)];
        enc_map.reserve(codes.size() * 2 + 1);
        for (const auto& c : codes) {
            const uint16_t sym = c.sym;
            enc_map[sym] = (static_cast<uint32_t>(c.len) << 16) | static_cast<uint32_t>(c.code);
        }

        if (has_escape) {
            has_esc_by_stream[static_cast<size_t>(s)] = 1;
            esc_sym_by_stream[static_cast<size_t>(s)] = esc_sym;
        }
    }

    // ---------------------------------------------------------------------
    // Build chunks, out_idx, and per-scale bitstream.
    // ---------------------------------------------------------------------
    // We encode in row-major order but chunk by constant take for better GPU work distribution.
    const uint32_t kChunk = 256;  // we store take in 24 bits, keep <=255
    const uint32_t kMaxTake = 255;

    std::vector<int32_t> out_idx;
    out_idx.reserve(static_cast<size_t>(n * m));
    for (int64_t i = 0; i < n * m; ++i) out_idx.push_back(static_cast<int32_t>(i));

    // chunk arrays
    std::vector<uint32_t> chunk_startbit_rel;
    std::vector<uint32_t> chunk_out_base;
    std::vector<uint32_t> chunk_meta;
    std::vector<int32_t> chunk_scale_id;

    chunk_startbit_rel.reserve(static_cast<size_t>((n * m + kMaxTake - 1) / kMaxTake));
    chunk_out_base.reserve(chunk_startbit_rel.capacity());
    chunk_meta.reserve(chunk_startbit_rel.capacity());
    chunk_scale_id.reserve(chunk_startbit_rel.capacity());

    // Per-scale bit encoding state
    std::vector<std::vector<uint64_t>> bitstream_by_scale(static_cast<size_t>(S));
    std::vector<int64_t> bitpos_by_scale(static_cast<size_t>(S), 0);

    auto append_bits_msb_scale = [&](int32_t sid, uint32_t code, uint8_t len) {
        if (len == 0) return;
        int64_t& bitpos = bitpos_by_scale[static_cast<size_t>(sid)];
        std::vector<uint64_t>& bs = bitstream_by_scale[static_cast<size_t>(sid)];
        const int64_t w = bitpos >> 6;
        const int o = static_cast<int>(bitpos & 63);
        while (static_cast<int64_t>(bs.size()) < w + 2) bs.push_back(0ull);
        unsigned __int128 window = (static_cast<unsigned __int128>(bs[static_cast<size_t>(w)]) << 64) |
                                   static_cast<unsigned __int128>(bs[static_cast<size_t>(w + 1)]);
        const unsigned __int128 v = static_cast<unsigned __int128>(code) & ((static_cast<unsigned __int128>(1u) << len) - 1u);
        const int shift = 128 - (o + static_cast<int>(len));
        window |= (v << shift);
        bs[static_cast<size_t>(w)] = static_cast<uint64_t>(window >> 64);
        bs[static_cast<size_t>(w + 1)] = static_cast<uint64_t>(window);
        bitpos += static_cast<int64_t>(len);
    };

    // Encode and create chunk metadata.
    // For each chunk, we enforce constant (sid,qv) by splitting on changes.
    int64_t flat = 0;
    while (flat < n * m) {
        const int64_t r0 = flat / m;
        const int64_t c0 = flat - r0 * m;
        const uint8_t qv0 = static_cast<uint8_t>(qv_u8[flat] & 0x0F);
        const uint16_t sc0 = scale_u16_at(scales_u16, scale_scalar, scale_per_row, scale_block2d,
                                         n, m, nb0, nb1, sb0, sb1, r0, c0);
        const int32_t sid0 = scale_to_id.at(sc0);

        const int64_t stream0 = static_cast<int64_t>(sid0) * 16 + static_cast<int64_t>(qv0);
        const auto& enc_map = enc[static_cast<size_t>(stream0)];

        const uint32_t out_base = static_cast<uint32_t>(flat);
        const uint32_t startbit = static_cast<uint32_t>(bitpos_by_scale[static_cast<size_t>(sid0)]);

        uint32_t take = 0;
        for (; take < kMaxTake && (flat + static_cast<int64_t>(take)) < n * m; ++take) {
            const int64_t idx = flat + static_cast<int64_t>(take);
            const int64_t r = idx / m;
            const int64_t c = idx - r * m;
            const uint8_t qv = static_cast<uint8_t>(qv_u8[idx] & 0x0F);
            const uint16_t sc = scale_u16_at(scales_u16, scale_scalar, scale_per_row, scale_block2d,
                                            n, m, nb0, nb1, sb0, sb1, r, c);
            const int32_t sid = scale_to_id.at(sc);
            if (sid != sid0 || qv != qv0) {
                break;
            }
            const uint16_t sym = bf16_u16[idx];
            auto it = enc_map.find(sym);
            if (it == enc_map.end()) {
                // Escape path: emit ESC code then raw 16-bit symbol.
                if (has_esc_by_stream[static_cast<size_t>(stream0)] == 0) {
                    throw std::runtime_error("symbol not found in enc table for AWQ stream");
                }
                const uint16_t esc_sym_local = esc_sym_by_stream[static_cast<size_t>(stream0)];
                auto it_esc = enc_map.find(esc_sym_local);
                if (it_esc == enc_map.end()) {
                    throw std::runtime_error("ESC symbol missing in enc table for AWQ stream");
                }
                const uint32_t packed_esc = it_esc->second;
                const uint8_t esc_len = static_cast<uint8_t>(packed_esc >> 16);
                const uint32_t esc_code = packed_esc & 0xFFFFu;
                append_bits_msb_scale(sid0, esc_code, esc_len);
                append_bits_msb_scale(sid0, static_cast<uint32_t>(sym), 16);
                continue;
            }
            const uint32_t packed = it->second;
            const uint8_t len = static_cast<uint8_t>(packed >> 16);
            const uint32_t code = packed & 0xFFFFu;
            append_bits_msb_scale(sid0, code, len);
        }
        if (take == 0) {
            throw std::runtime_error("internal error: zero-length chunk");
        }

        chunk_out_base.push_back(out_base);
        chunk_startbit_rel.push_back(startbit);
        chunk_scale_id.push_back(sid0);
        const uint32_t meta = (static_cast<uint32_t>(take) << 8) | static_cast<uint32_t>(qv0);
        chunk_meta.push_back(meta);

        flat += static_cast<int64_t>(take);
    }

    // Finalize per-scale bitstream into a single concatenated bitstream.
    torch::Tensor scale_values_u16 = torch::empty({S}, opts_u16);
    std::memcpy(scale_values_u16.data_ptr<uint16_t>(), scale_values.data(), static_cast<size_t>(S) * sizeof(uint16_t));

    torch::Tensor scale_bit_base_i64 = torch::zeros({S + 1}, opts_i64);
    int64_t* base = scale_bit_base_i64.data_ptr<int64_t>();
    base[0] = 0;
    int64_t total_bits = 0;
    for (int64_t sid = 0; sid < S; ++sid) {
        // 64-bit align each scale stream.
        if (total_bits & 63) total_bits = (total_bits + 63) & ~63LL;
        base[sid] = total_bits;
        total_bits += bitpos_by_scale[static_cast<size_t>(sid)];
    }
    if (total_bits & 63) total_bits = (total_bits + 63) & ~63LL;
    base[S] = total_bits;

    const int64_t total_words = (total_bits + 63) >> 6;
    torch::Tensor bitstream_u64 = torch::zeros({total_words}, opts_u64);
    uint64_t* out_bs = bitstream_u64.data_ptr<uint64_t>();

    for (int64_t sid = 0; sid < S; ++sid) {
        const int64_t bit_base = base[sid];
        const int64_t w_base = bit_base >> 6;
        std::vector<uint64_t>& bs = bitstream_by_scale[static_cast<size_t>(sid)];
        const int64_t bits = bitpos_by_scale[static_cast<size_t>(sid)];
        const int64_t words_needed = (bits + 63) >> 6;
        if (words_needed > 0) {
            if (static_cast<int64_t>(bs.size()) < words_needed) {
                throw std::runtime_error("internal error: bitstream_by_scale shorter than words_needed");
            }
            std::memcpy(out_bs + w_base, bs.data(), static_cast<size_t>(words_needed) * sizeof(uint64_t));
        }
    }

    // Build stream->chunk index (CPU):
    // stream = scale_id * 16 + qv, where qv is chunk_meta & 0xFF.
    const int64_t C = static_cast<int64_t>(chunk_meta.size());
    torch::Tensor chunk_meta_u32 = torch::empty({C}, opts_u32);
    torch::Tensor chunk_scale_id_i32 = torch::empty({C}, opts_i32);
    torch::Tensor chunk_startbit_rel_u32 = torch::empty({C}, opts_u32);
    torch::Tensor chunk_out_base_u32 = torch::empty({C}, opts_u32);
    std::memcpy(chunk_meta_u32.data_ptr<uint32_t>(), chunk_meta.data(), static_cast<size_t>(C) * sizeof(uint32_t));
    std::memcpy(chunk_scale_id_i32.data_ptr<int32_t>(), chunk_scale_id.data(), static_cast<size_t>(C) * sizeof(int32_t));
    std::memcpy(chunk_startbit_rel_u32.data_ptr<uint32_t>(), chunk_startbit_rel.data(), static_cast<size_t>(C) * sizeof(uint32_t));
    std::memcpy(chunk_out_base_u32.data_ptr<uint32_t>(), chunk_out_base.data(), static_cast<size_t>(C) * sizeof(uint32_t));

    // out_idx is dense identity mapping for now.
    torch::Tensor out_idx_i32 = torch::empty({n * m}, opts_i32);
    std::memcpy(out_idx_i32.data_ptr<int32_t>(), out_idx.data(), static_cast<size_t>(n * m) * sizeof(int32_t));

    // Reuse the stream count (S*16) already computed above.
    torch::Tensor stream_chunk_ofs_i64 = torch::zeros({num_streams + 1}, opts_i64);
    torch::Tensor stream_chunk_ids_i32 = torch::empty({C}, opts_i32);
    int64_t* ofs = stream_chunk_ofs_i64.data_ptr<int64_t>();
    int32_t* ids = stream_chunk_ids_i32.data_ptr<int32_t>();

    // Count chunks per stream.
    for (int64_t c = 0; c < C; ++c) {
        const int32_t sid = chunk_scale_id[c];
        if (sid < 0 || sid >= S) throw std::invalid_argument("chunk_scale_id out of range");
        const uint32_t qv = chunk_meta[c] & 0xFFu;
        if (qv >= 16u) throw std::invalid_argument("chunk_meta qv out of range");
        const int64_t stream = static_cast<int64_t>(sid) * 16 + static_cast<int64_t>(qv);
        ++ofs[stream + 1];
    }
    // Prefix sum.
    for (int64_t s = 0; s < num_streams; ++s) {
        ofs[s + 1] += ofs[s];
    }
    // Fill ids using cursors (stable-ish order).
    std::vector<int64_t> cursor(static_cast<size_t>(num_streams));
    for (int64_t s = 0; s < num_streams; ++s) cursor[static_cast<size_t>(s)] = ofs[s];
    for (int64_t c = 0; c < C; ++c) {
        const int32_t sid = chunk_scale_id[c];
        const uint32_t qv = chunk_meta[c] & 0xFFu;
        const int64_t stream = static_cast<int64_t>(sid) * 16 + static_cast<int64_t>(qv);
        const int64_t pos = cursor[static_cast<size_t>(stream)]++;
        ids[pos] = static_cast<int32_t>(c);
    }

    return std::make_tuple(
        decode_syms_i16,
        decode_nbits_u8,
        scale_values_u16,
        scale_bit_base_i64,
        chunk_startbit_rel_u32,
        chunk_out_base_u32,
        chunk_meta_u32,
        chunk_scale_id_i32,
        out_idx_i32,
        stream_chunk_ofs_i64,
        stream_chunk_ids_i32,
        bitstream_u64);
}

}  // namespace huffman_cpp
