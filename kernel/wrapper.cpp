#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>  // 添加这个头文件
#include <tuple>

namespace py = pybind11;  // 添加这个命名空间别名

// Huffman (implemented in build_huffman.cu)
namespace huffman_cpp {
    // AWQ int4 + scale -> lossless bf16 Huffman build (encode artifacts)
    // NOTE: this is a repo-local API (not upstream).
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    huffman_build_awq_decode_tables_entry(torch::Tensor bf16_params,
                                         torch::Tensor int4_values_u8,
                                         torch::Tensor scales_u16,
                                         int64_t max_len);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor>
    huffman_build_fp8_decode_tables_entry(torch::Tensor bf16_params,
                                          torch::Tensor fp8_params,
                                          torch::Tensor scales,
                                          int64_t max_len,
                                          int64_t threads_per_block);

    torch::Tensor huffman_decode_fp8_entry(torch::Tensor fp8_params_u8,
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
                                          int64_t max_len);

    // AWQ int4 + scale -> lossless bf16 Huffman build (bucket-based artifacts)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    huffman_build_awq_int4_bucket_decode_tables_entry(torch::Tensor bf16_params,
                                                      torch::Tensor int4_values_u8,
                                                      torch::Tensor scales_u16,
                                                      int64_t max_len,
                                                      int64_t threads_per_block);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor, torch::Tensor>
        huffman_build_awq_int4_bucket_decode_tables_by_scale_entry(torch::Tensor bf16_values,
                                                                   torch::Tensor int4_values,
                                                                   torch::Tensor scales_u16,
                                                                   int64_t max_len,
                                                                   int64_t threads,
                                                                   int64_t group_cap);

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
        int64_t group_size);

    // AWQ int4 bucket decode (CUDA)
    torch::Tensor huffman_decode_awq_int4_bucket_entry(torch::Tensor int4_values_u8,
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
                                                       int64_t max_len);
}

// Huffman AWQ shared-table decoder utilities (implemented in decode_huffman_encode_only.cu)
namespace huffman_decode_cpp {
    std::tuple<torch::Tensor, torch::Tensor>
    huffman_build_decode_tables_entry(torch::Tensor enc_ofs_i64,
                                     torch::Tensor enc_syms_i16,
                                     torch::Tensor enc_codes_u32,
                                     torch::Tensor enc_lens_u8,
                                     int64_t max_len);

    std::tuple<torch::Tensor, torch::Tensor>
    huffman_build_stream_chunk_index_entry(torch::Tensor chunk_meta_u32,
                                          torch::Tensor chunk_scale_id_i32,
                                          int64_t num_scales);

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
        int64_t max_len);
}

PYBIND11_MODULE(trie_cuda, m) {
    m.def("cuda_clock_rate_khz", []() {
        int dev = 0;
        cudaError_t e = cudaGetDevice(&dev);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaGetDevice failed: ") + cudaGetErrorString(e));
        cudaDeviceProp prop;
        e = cudaGetDeviceProperties(&prop, dev);
        if (e != cudaSuccess) throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(e));
        return static_cast<int64_t>(prop.clockRate);
    });
    m.def("init_cuda", &init_cuda);
    m.def("cleanup_cuda", &cleanup_cuda);
    m.def("build_trie_AWQ", &build_trie_AWQ_py);
    m.def("build_trie", &build_trie_py);
    m.def("query_trie", &query_trie_py);
    m.def("free_trie", [](uintptr_t ptr) {
        free_trie_wrapper(reinterpret_cast<void*>(ptr));
    });
    m.def("get_trie_count", [](uintptr_t ptr) {
        return get_trie_count(reinterpret_cast<void*>(ptr));
    });
    m.def("get_tot_path_length", &get_tot_path_length_py);
    m.def("save_trie_collection", [](uintptr_t ptr, const std::string& filename) {
        return save_trie_collection(reinterpret_cast<void*>(ptr), filename.c_str());
    });
    m.def("load_trie_collection", [](const std::string& filename) {
        auto ptr = load_trie_collection(filename.c_str());
        return reinterpret_cast<uintptr_t>(ptr);
    });
    m.def("save_trie_collection_awq", [](uintptr_t ptr, const std::string& filename) {
        return save_trie_collection_awq(reinterpret_cast<void*>(ptr), filename.c_str());
    });
    m.def("load_trie_collection_awq", [](const std::string& filename) {
        auto ptr = load_trie_collection_awq(filename.c_str());
        return reinterpret_cast<uintptr_t>(ptr);
    });
    m.def("load_bf16_from_int8", &load_bf16_from_int8_py);
    m.def("dequantize_awq", &dequantize_awq_py, "Dequantize AWQ int4 weights to bfloat16");

    // AWQ to FP8 bindings
    m.def("build_trie_AWQ_to_FP8", &build_trie_AWQ_to_FP8_py);
    m.def("save_trie_collection_awq_to_fp8", [](uintptr_t ptr, const std::string& filename) {
        return save_trie_collection_awq_to_fp8(reinterpret_cast<void*>(ptr), filename.c_str());
    });
    m.def("load_trie_collection_awq_to_fp8", [](const std::string& filename) {
        auto ptr = load_trie_collection_awq_to_fp8(filename.c_str());
        return reinterpret_cast<uintptr_t>(ptr);
    });

    // Huffman: build AWQ (int4 + scale) shared-table decode artifacts (CPU)
    m.def(
        "huffman_build_awq_decode_tables",
        [](torch::Tensor bf16_params,
           torch::Tensor int4_values_u8,
           torch::Tensor scales_u16,
           int64_t max_len) {
            auto tup = huffman_cpp::huffman_build_awq_decode_tables_entry(
                bf16_params, int4_values_u8, scales_u16, max_len);
            py::dict d;
            d["decode_syms_i16"] = std::get<0>(tup);
            d["decode_nbits_u8"] = std::get<1>(tup);
            d["scale_values_u16"] = std::get<2>(tup);
            d["scale_bit_base_i64"] = std::get<3>(tup);
            d["chunk_startbit_rel_u32"] = std::get<4>(tup);
            d["chunk_out_base_u32"] = std::get<5>(tup);
            d["chunk_meta_u32"] = std::get<6>(tup);
            d["chunk_scale_id_i32"] = std::get<7>(tup);
            d["out_idx_i32"] = std::get<8>(tup);
            d["stream_chunk_ofs_i64"] = std::get<9>(tup);
            d["stream_chunk_ids_i32"] = std::get<10>(tup);
            d["bitstream_u64"] = std::get<11>(tup);
            return d;
        },
        py::arg("bf16_params"),
        py::arg("int4_values_u8"),
        py::arg("scales_u16"),
        py::arg("max_len") = 11
    );

    // Huffman (AWQ): decode shared-table (CUDA)
    m.def(
        "huffman_decode_awq_shared_table",
        [](torch::Tensor bitstream_u64,
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
            return huffman_decode_cpp::huffman_decode_awq_shared_table_entry(
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
        },
        py::arg("bitstream_u64"),
        py::arg("decode_syms_i16"),
        py::arg("decode_nbits_u8"),
        py::arg("scale_bit_base_i64"),
        py::arg("chunk_startbit_rel_u32"),
        py::arg("chunk_out_base_u32"),
        py::arg("chunk_meta_u32"),
        py::arg("chunk_scale_id_i32"),
        py::arg("out_idx_i32"),
        py::arg("stream_chunk_ofs_i64"),
        py::arg("stream_chunk_ids_i32"),
        py::arg("out_len"),
        py::arg("max_len") = 11
    );

    // Huffman (AWQ int4): build bucket-based decode artifacts (CPU)
    m.def(
        "huffman_build_awq_int4_bucket_decode_tables",
        [](torch::Tensor bf16_params,
           torch::Tensor int4_values_u8,
           torch::Tensor scales_u16,
           int64_t max_len,
           int64_t threads_per_block) {
            auto tup = huffman_cpp::huffman_build_awq_int4_bucket_decode_tables_entry(
                bf16_params, int4_values_u8, scales_u16, max_len, threads_per_block);
            py::dict d;
            d["decode_syms_u16"] = std::get<0>(tup);
            d["decode_lens_u8"] = std::get<1>(tup);
            d["table_scale_u16"] = std::get<2>(tup);
            d["table_qv_u16"] = std::get<3>(tup);
            d["bucket_scale_u16"] = std::get<4>(tup);
            d["bucket_row_offsets_i32"] = std::get<5>(tup);
            d["row_indices_i32"] = std::get<6>(tup);
            d["bucket_table_i32"] = std::get<7>(tup);
            d["bucket_col_start_i32"] = std::get<8>(tup);
            d["bucket_col_len_i32"] = std::get<9>(tup);
            d["bucket_thread_startbit_u32"] = std::get<10>(tup);
            d["bucket_bit_base_i64"] = std::get<11>(tup);
            d["bitstream_u64"] = std::get<12>(tup);
            d["shape_nmk_i64"] = std::get<13>(tup);
            return d;
        },
        py::arg("bf16_params"),
        py::arg("int4_values_u8"),
        py::arg("scales_u16"),
        py::arg("max_len") = 11,
        py::arg("threads_per_block") = 256);

    m.def(
        "huffman_build_awq_int4_bucket_decode_tables_by_scale",
        [](torch::Tensor bf16_values,
           torch::Tensor int4_values,
           torch::Tensor scales_u16,
           int64_t max_len,
           int64_t threads,
           int64_t group_cap) {
            auto out = huffman_cpp::huffman_build_awq_int4_bucket_decode_tables_by_scale_entry(
                bf16_values, int4_values, scales_u16, max_len, threads, group_cap);
            py::dict d;
            d["decode_syms_u16"] = std::get<0>(out);
            d["decode_lens_u8"] = std::get<1>(out);
            d["table_scale_u16"] = std::get<2>(out);
            d["table_qv_u16"] = std::get<3>(out);
            d["bucket_scale_u16"] = std::get<4>(out);
            d["bucket_group_offsets_i32"] = std::get<5>(out);
            d["group_rows_i32"] = std::get<6>(out);
            d["group_col_start_i32"] = std::get<7>(out);
            d["bucket_table_i32"] = std::get<8>(out);
            d["bucket_thread_startbit_u32"] = std::get<9>(out);
            d["bucket_bit_base_i64"] = std::get<10>(out);
            d["bitstream_u64"] = std::get<11>(out);
            d["shape_nmkg_i64"] = std::get<12>(out);
            return d;
        },
        py::arg("bf16_values"),
        py::arg("int4_values"),
        py::arg("scales_u16"),
        py::arg("max_len") = 11,
        py::arg("threads") = 256,
        py::arg("group_cap") = 1024);

    // Huffman (AWQ int4): decode buckets -> bf16 weights (CUDA)
    m.def(
        "huffman_decode_awq_int4_bucket",
        [](torch::Tensor int4_values_u8,
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
            return huffman_cpp::huffman_decode_awq_int4_bucket_entry(
                int4_values_u8,
                bitstream_u64,
                decode_syms_u16,
                decode_lens_u8,
                bucket_table_i32,
                bucket_col_start_i32,
                bucket_col_len_i32,
                bucket_row_offsets_i32,
                row_indices_i32,
                bucket_thread_startbit_u32,
                bucket_bit_base_i64,
                n,
                m,
                max_len);
        },
        py::arg("int4_values_u8"),
        py::arg("bitstream_u64"),
        py::arg("decode_syms_u16"),
        py::arg("decode_lens_u8"),
        py::arg("bucket_table_i32"),
        py::arg("bucket_col_start_i32"),
        py::arg("bucket_col_len_i32"),
        py::arg("bucket_row_offsets_i32"),
        py::arg("row_indices_i32"),
        py::arg("bucket_thread_startbit_u32"),
        py::arg("bucket_bit_base_i64"),
        py::arg("n"),
        py::arg("m"),
        py::arg("max_len") = 11);

    m.def(
        "huffman_decode_awq_int4_bucket_by_scale",
        [](torch::Tensor int4_values_u8,
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
            return huffman_cpp::huffman_decode_awq_int4_bucket_by_scale_entry(
                int4_values_u8,
                bitstream_u64,
                decode_syms_u16,
                decode_lens_u8,
                bucket_table_i32,
                bucket_group_offsets_i32,
                group_rows_i32,
                group_col_start_i32,
                bucket_thread_startbit_u32,
                bucket_bit_base_i64,
                n,
                m,
                max_len,
                group_size);
        },
        py::arg("int4_values_u8"),
        py::arg("bitstream_u64"),
        py::arg("decode_syms_u16"),
        py::arg("decode_lens_u8"),
        py::arg("bucket_table_i32"),
        py::arg("bucket_group_offsets_i32"),
        py::arg("group_rows_i32"),
        py::arg("group_col_start_i32"),
        py::arg("bucket_thread_startbit_u32"),
        py::arg("bucket_bit_base_i64"),
        py::arg("n"),
        py::arg("m"),
        py::arg("max_len"),
        py::arg("group_size"));

    // Huffman: build decode tables per (FP8, Scale) group
    m.def(
        "huffman_build_fp8_decode_tables",
        [](torch::Tensor bf16_params,
           torch::Tensor fp8_params,
           torch::Tensor scales,
           int64_t max_len,
           int64_t threads_per_block) {
            auto tup = huffman_cpp::huffman_build_fp8_decode_tables_entry(
                bf16_params, fp8_params, scales, max_len, threads_per_block);
            py::dict d;
            d["decode_syms_u16"] = std::get<0>(tup);
            d["decode_lens_u8"] = std::get<1>(tup);
            d["table_scale_u16"] = std::get<2>(tup);
            d["table_fp8_u16"] = std::get<3>(tup);

            d["bucket_scale_u16"] = std::get<4>(tup);
            d["bucket_row_offsets_i32"] = std::get<5>(tup);
            d["row_indices_i32"] = std::get<6>(tup);
            d["bucket_table_i32"] = std::get<7>(tup);
            d["bucket_col_start_i32"] = std::get<8>(tup);
            d["bucket_col_len_i32"] = std::get<9>(tup);
            d["bucket_thread_startbit_u32"] = std::get<10>(tup);
            d["bucket_bit_base_i64"] = std::get<11>(tup);
            d["bitstream_u64"] = std::get<12>(tup);
            d["shape_nmk_i64"] = std::get<13>(tup);
            d["bucket_U_i32"] = std::get<14>(tup);
            d["bucket_unique_gid_i32"] = std::get<15>(tup);
            d["bucket_fp8_uidx_i16"] = std::get<16>(tup);
            return d;
        },
        py::arg("bf16_params"),
        py::arg("fp8_params"),
        py::arg("scales"),
        py::arg("max_len") = 12,
        py::arg("threads_per_block") = 256
    );

    // Huffman: decode FP8 buckets -> bf16 weights (CUDA)
    m.def(
        "huffman_decode_fp8",
        [](torch::Tensor fp8_params_u8,
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
            return huffman_cpp::huffman_decode_fp8_entry(
                fp8_params_u8,
                bitstream_u64,
                decode_syms_u16,
                decode_lens_u8,
                bucket_table_i32,
                bucket_col_start_i32,
                bucket_col_len_i32,
                bucket_row_offsets_i32,
                row_indices_i32,
                bucket_thread_startbit_u32,
                bucket_bit_base_i64,
                n,
                m,
                max_len);
        },
        py::arg("fp8_params_u8"),
        py::arg("bitstream_u64"),
        py::arg("decode_syms_u16"),
        py::arg("decode_lens_u8"),
        py::arg("bucket_table_i32"),
        py::arg("bucket_col_start_i32"),
        py::arg("bucket_col_len_i32"),
        py::arg("bucket_row_offsets_i32"),
        py::arg("row_indices_i32"),
        py::arg("bucket_thread_startbit_u32"),
        py::arg("bucket_bit_base_i64"),
        py::arg("n"),
        py::arg("m"),
        py::arg("max_len") = 12
    );
}
