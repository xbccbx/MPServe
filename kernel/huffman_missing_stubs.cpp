#include <torch/extension.h>

#include <stdexcept>
#include <tuple>

namespace huffman_cpp {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
huffman_encode_awq_entry(torch::Tensor /*values_u8*/,
                        torch::Tensor /*scales_u16*/,
                        int64_t /*max_len*/,
                        int64_t /*bucket_row_limit*/) {
    throw std::invalid_argument(
        "huffman_encode_awq_entry is not available in this build (stub)");
}

torch::Tensor huffman_decode_placeholder_entry(torch::Tensor /*bitstream_u64*/,
                                              torch::Tensor /*decode_syms_u16*/,
                                              torch::Tensor /*decode_lens_u8*/,
                                              torch::Tensor /*bucket_table_i32*/,
                                              torch::Tensor /*bucket_row_offsets_i32*/,
                                              torch::Tensor /*row_indices_i32*/,
                                              torch::Tensor /*bucket_thread_start_u32*/,
                                              torch::Tensor /*bucket_bit_offsets_i64*/,
                                              int64_t /*n*/,
                                              int64_t /*m*/,
                                              int64_t /*max_len*/) {
    throw std::invalid_argument(
        "huffman_decode_placeholder_entry is not available in this build (stub)");
}

}  // namespace huffman_cpp
