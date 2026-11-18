#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend.h"
#include "qairt-lowering.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace ggml::qairt;
static int failures = 0, checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); } } while (0)
static void upload(ggml_tensor * t, const std::vector<float> & values) {
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float)); }
    else {
        std::vector<ggml_fp16_t> half(values.size());
        for (size_t i = 0; i < half.size(); ++i) { half[i] = ggml_fp32_to_fp16(values[i]); }
        ggml_backend_tensor_set(t, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    }
}
static std::vector<float> download(ggml_tensor * t) {
    std::vector<float> values(size_t(ggml_nelements(t)));
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float)); }
    else { std::vector<ggml_fp16_t> half(values.size()); ggml_backend_tensor_get(t, half.data(), 0, half.size() * 2); for (size_t i = 0; i < values.size(); ++i) { values[i] = ggml_fp16_to_fp32(half[i]); } }
    return values;
}
static void attention_case(ggml_backend_t backend, int queries, int keys, int heads, int kv_heads, bool strided) {
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    const int width = 64;
    auto * q_base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, strided ? heads : queries, strided ? queries : heads);
    auto * q = strided ? ggml_permute(ctx, q_base, 0, 2, 1, 3) : q_base;
    auto * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, keys, kv_heads);
    auto * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, keys, kv_heads);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, queries);
    auto * y = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
    CHECK(primitive_supported(y));
    ggml_prec_set_acc(y, GGML_PREC_F32); CHECK(!primitive_supported(y));
    ggml_prec_set_acc(y, GGML_PREC_DEFAULT);
    ggml_flash_attn_ext_set_n_kv_max(y, 1); CHECK(!primitive_supported(y));
    ggml_flash_attn_ext_set_n_kv_max(y, 0); CHECK(primitive_supported(y));
    if (!backend) { ggml_free(ctx); return; }
    CHECK(ggml_backend_supports_op(backend, y));
    auto * cpu = ggml_backend_init_by_name("CPU", nullptr); CHECK(cpu);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); CHECK(buffer);
    if (!cpu || !buffer) { if (buffer) { ggml_backend_buffer_free(buffer); } if (cpu) { ggml_backend_free(cpu); } ggml_free(ctx); return; }
    std::vector<float> q_values(size_t(ggml_nelements(q_base))), k_values(size_t(ggml_nelements(k))), v_values(k_values.size());
    for (size_t i = 0; i < q_values.size(); ++i) { q_values[i] = 0.35f * std::sin(float(i) * 0.071f); }
    for (size_t i = 0; i < k_values.size(); ++i) { k_values[i] = 0.4f * std::cos(float(i) * 0.031f); v_values[i] = 0.5f * std::sin(float(i) * 0.019f) + 0.1f; }
    std::vector<float> mask_values(size_t(ggml_nelements(mask)), -std::numeric_limits<float>::infinity());
    for (int row = 0; row < queries; ++row) { for (int col = 0; col <= keys - queries + row; ++col) { mask_values[size_t(row * keys + col)] = 0.0f; } }
    upload(q_base, q_values); upload(k, k_values); upload(v, v_values); upload(mask, mask_values);
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, y);
    CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
    const auto expected = download(y);
    const auto status = ggml_backend_graph_compute(backend, graph); CHECK(status == GGML_STATUS_SUCCESS);
    if (status == GGML_STATUS_SUCCESS) {
        auto actual = download(y); double error = 0.0, energy = 0.0;
        for (size_t i = 0; i < actual.size(); ++i) { CHECK(std::isfinite(actual[i])); error += std::pow(double(actual[i]) - expected[i], 2); energy += double(expected[i]) * expected[i]; }
        std::printf("GQA q=%d k=%d heads=%d kv=%d strided=%d NMSE=%.9g\n", queries, keys, heads, kv_heads, int(strided), error / energy);
        CHECK(error <= 1e-5 * energy);
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        const auto saved = download(y);
        mask_values[0] = 1.0f; upload(mask, mask_values);
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
        CHECK(download(y) == saved);
    }
    ggml_backend_buffer_free(buffer); ggml_backend_free(cpu); ggml_free(ctx);
}
static void cache_case(ggml_backend_t backend, ggml_type index_type) {
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    auto * cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 64, 8);
    auto * updates = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 3);
    auto * indices = ggml_new_tensor_1d(ctx, index_type, 3);
    auto * y = ggml_set_rows(ctx, cache, updates, indices); CHECK(primitive_supported(y));
    auto * bad_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 1);
    auto * bad_norm = ggml_rms_norm(ctx, bad_input, 1e-5f);
    if (!backend) { ggml_free(ctx); return; }
    CHECK(ggml_backend_supports_op(backend, y));
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); auto * cpu = ggml_backend_init_by_name("CPU", nullptr); CHECK(buffer && cpu);
    if (!buffer || !cpu) { if (buffer) { ggml_backend_buffer_free(buffer); } if (cpu) { ggml_backend_free(cpu); } ggml_free(ctx); return; }
    auto set_indices = [&](std::vector<int64_t> values) {
        if (index_type == GGML_TYPE_I64) { ggml_backend_tensor_set(indices, values.data(), 0, values.size() * 8); }
        else { std::vector<int32_t> narrow; for (auto value : values) { narrow.push_back(int32_t(value)); } ggml_backend_tensor_set(indices, narrow.data(), 0, narrow.size() * 4); }
    };
    std::vector<float> initial(512), input(192);
    for (size_t i = 0; i < initial.size(); ++i) { initial[i] = 0.25f * std::sin(float(i) * 0.023f); }
    for (size_t i = 0; i < input.size(); ++i) { input[i] = 0.4f * std::cos(float(i) * 0.031f); }
    upload(cache, initial); upload(updates, input); set_indices({0, 3, 7});
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, y);
    CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS); const auto expected = download(cache);
    upload(cache, initial); CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS); CHECK(download(cache) == expected);
    set_indices({1, 4, 6}); CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS); const auto saved = download(cache);
    set_indices({1, 1, 6}); CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED); CHECK(download(cache) == saved);
    set_indices({1, 4, 8}); CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED); CHECK(download(cache) == saved);
    set_indices({1, 4, -1}); CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED); CHECK(download(cache) == saved);
    if (index_type == GGML_TYPE_I64) { set_indices({1, 4, INT64_MAX}); CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED); CHECK(download(cache) == saved); }
    set_indices({0, 2, 5}); upload(bad_input, std::vector<float>(64, std::numeric_limits<float>::quiet_NaN()));
    auto * bad_graph = ggml_new_graph(ctx); ggml_build_forward_expand(bad_graph, y); ggml_build_forward_expand(bad_graph, bad_norm);
    CHECK(ggml_backend_graph_compute(backend, bad_graph) == GGML_STATUS_FAILED); CHECK(download(cache) == saved);
    ggml_backend_buffer_free(buffer); ggml_backend_free(cpu); ggml_free(ctx);
}
static void connected_cache_attention(ggml_backend_t backend) {
    if (!backend) { return; }
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    auto * k_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, 8);
    auto * v_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 128, 8);
    auto * k_new = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 3);
    auto * v_new = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 3);
    auto * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 3);
    auto * k_write = ggml_set_rows(ctx, k_cache, k_new, indices);
    auto * v_write = ggml_set_rows(ctx, v_cache, v_new, indices);
    auto * k = ggml_permute(ctx, ggml_reshape_3d(ctx, k_write, 64, 2, 8), 0, 2, 1, 3);
    auto * v = ggml_permute(ctx, ggml_reshape_3d(ctx, v_write, 64, 2, 8), 0, 2, 1, 3);
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 3, 4);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 3);
    auto * y = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.125f, 0.0f, 0.0f);
    auto * cpu = ggml_backend_init_by_name("CPU", nullptr); auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); CHECK(cpu && buffer);
    if (!cpu || !buffer) { if (buffer) { ggml_backend_buffer_free(buffer); } if (cpu) { ggml_backend_free(cpu); } ggml_free(ctx); return; }
    std::vector<float> initial(1024), new_k(384), new_v(384), query(768), mask_values(24, -std::numeric_limits<float>::infinity());
    for (size_t i = 0; i < initial.size(); ++i) { initial[i] = 0.3f * std::sin(float(i) * 0.03f); }
    for (size_t i = 0; i < new_k.size(); ++i) { new_k[i] = 0.2f * std::cos(float(i) * 0.031f); new_v[i] = 0.4f * std::sin(float(i) * 0.019f); }
    for (size_t i = 0; i < query.size(); ++i) { query[i] = 0.3f * std::cos(float(i) * 0.021f); }
    for (size_t row = 0; row < 3; ++row) { for (size_t col = 0; col <= 5 + row; ++col) { mask_values[row * 8 + col] = 0.0f; } }
    upload(k_cache, initial); upload(v_cache, initial); upload(k_new, new_k); upload(v_new, new_v); upload(q, query); upload(mask, mask_values);
    int64_t positions[] = {5, 6, 7}; ggml_backend_tensor_set(indices, positions, 0, sizeof(positions));
    auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, y);
    CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS); const auto expected = download(y), expected_k = download(k_cache), expected_v = download(v_cache);
    upload(k_cache, initial); upload(v_cache, initial); CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const auto actual = download(y); double error = 0.0, energy = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) { error += std::pow(double(actual[i]) - expected[i], 2); energy += double(expected[i]) * expected[i]; }
    CHECK(error <= 1e-5 * energy); CHECK(download(k_cache) == expected_k); CHECK(download(v_cache) == expected_v);
    std::printf("GQA connected KV NMSE=%.9g\n", error / energy);
    const auto saved_k = download(k_cache), saved_v = download(v_cache), saved_y = download(y);
    upload(k_new, std::vector<float>(384, 0.75f)); mask_values[0] = 1.0f; upload(mask, mask_values);
    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
    CHECK(download(k_cache) == saved_k); CHECK(download(v_cache) == saved_v); CHECK(download(y) == saved_y);
    ggml_backend_buffer_free(buffer); ggml_backend_free(cpu); ggml_free(ctx);
}
int main(int argc, char ** argv) {
    const bool host = argc == 2 && std::strcmp(argv[1], "--host") == 0;
    ggml_backend_t backend = nullptr;
    if (!host) { ggml_backend_load_all(); backend = ggml_backend_init_by_name("QAIRT0", nullptr); CHECK(backend); if (!backend) { return 1; } }
    if (argc == 2 && std::strcmp(argv[1], "--single") == 0) { attention_case(backend, 4, 4, 4, 2, false); ggml_backend_free(backend); return failures ? 1 : 0; }
    for (int kv : {1, 2, 4}) { for (bool strided : {false, true}) { attention_case(backend, 4, 4, 4, kv, strided); attention_case(backend, 1, 8, 4, kv, strided); attention_case(backend, 3, 8, 4, kv, strided); } }
    cache_case(backend, GGML_TYPE_I32); cache_case(backend, GGML_TYPE_I64); connected_cache_attention(backend);
    if (backend) { ggml_backend_free(backend); }
    std::printf("QAIRT GQA: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
