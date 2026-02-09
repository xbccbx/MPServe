#include <cstdint>
#include <cstdio>

extern "C" {

void* build_trie_wrapper_awq(uint32_t* /*a*/, uint16_t* /*b*/, uint16_t* /*scale*/, int /*n*/, int /*m*/) {
    std::fprintf(stderr, "build_trie_wrapper_awq: stub called (not implemented)\n");
    return nullptr;
}

bool save_trie_collection_awq(void* /*collection_ptr*/, const char* /*filename*/) {
    std::fprintf(stderr, "save_trie_collection_awq: stub called (not implemented)\n");
    return false;
}

void* load_trie_collection_awq(const char* /*filename*/) {
    std::fprintf(stderr, "load_trie_collection_awq: stub called (not implemented)\n");
    return nullptr;
}

void dequantize_awq_c(const int32_t* /*qweight*/, const int32_t* /*qzeros*/, const uint16_t* /*scales*/,
                     uint16_t* /*output*/, int /*n*/, int /*m*/, int /*group_size*/) {
    std::fprintf(stderr, "dequantize_awq_c: stub called (not implemented)\n");
}

void* build_trie_wrapper_awq_to_fp8(const int32_t* /*a*/, uint8_t* /*b*/, uint16_t* /*scale*/, int /*n*/, int /*m*/) {
    std::fprintf(stderr, "build_trie_wrapper_awq_to_fp8: stub called (not implemented)\n");
    return nullptr;
}

bool save_trie_collection_awq_to_fp8(void* /*collection_ptr*/, const char* /*filename*/) {
    std::fprintf(stderr, "save_trie_collection_awq_to_fp8: stub called (not implemented)\n");
    return false;
}

void* load_trie_collection_awq_to_fp8(const char* /*filename*/) {
    std::fprintf(stderr, "load_trie_collection_awq_to_fp8: stub called (not implemented)\n");
    return nullptr;
}

}  // extern "C"
