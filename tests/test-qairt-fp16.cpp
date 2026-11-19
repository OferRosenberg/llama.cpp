#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-qairt.h"
#include "qairt-weights.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

using namespace ggml::qairt;
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
static int failures = 0, checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); } } while (0)
static context_ptr context() { return {ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free}; }
static void logger(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) { std::fputs(text, stderr); }
}
static void packing_tests() {
    std::vector<uint8_t> raw(18 * 2);
    const ggml_fp16_t scales[] = {ggml_fp32_to_fp16(0.25f), ggml_fp32_to_fp16(-0.5f)};
    for (size_t row = 0; row < 2; ++row) {
        std::memcpy(raw.data() + row * 18, &scales[row], 2);
        for (size_t i = 0; i < 16; ++i) { raw[row * 18 + 2 + i] = uint8_t(i | ((15 - i) << 4)); }
    }
    auto weights = prepare_q4_weights({raw.data(), raw.size()}, 32, 2);
    CHECK(weights.ok());
    if (weights.ok()) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t i = 0; i < 16; ++i) {
                CHECK(weights.value()[row * 32 + i].bits == ggml_fp32_to_fp16(float(int(i) - 8) * ggml_fp16_to_fp32(scales[row])));
                CHECK(weights.value()[row * 32 + i + 16].bits == ggml_fp32_to_fp16(float(7 - int(i)) * ggml_fp16_to_fp32(scales[row])));
            }
        }
    }
    CHECK(!prepare_q4_weights({raw.data(), raw.size() - 1}, 32, 2).ok());
    CHECK(!prepare_q4_weights({raw.data(), raw.size()}, 31, 2).ok());
    CHECK(!prepare_q4_weights({raw.data(), raw.size()}, SIZE_MAX, 2).ok());
}
static std::vector<float> cpu_product(ggml_backend_t cpu, ggml_tensor * weights, ggml_tensor * input) {
    auto c = context();
    auto * y = ggml_mul_mat(c.get(), weights, input);
    y->flags |= GGML_TENSOR_FLAG_COMPUTE;
    auto * graph = ggml_new_graph(c.get()); ggml_graph_add_node(graph, y);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(c.get(), cpu), ggml_backend_buffer_free);
    CHECK(buffer != nullptr);
    if (!buffer || ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) { CHECK(false); return {}; }
    std::vector<float> result(size_t(ggml_nelements(y)));
    ggml_backend_tensor_get(y, result.data(), 0, result.size() * sizeof(float));
    return result;
}
static void accuracy(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    CHECK(actual.size() == expected.size()); if (actual.size() != expected.size()) { return; }
    double error = 0, energy = 0, maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double delta = double(actual[i]) - expected[i];
        error += delta * delta; energy += double(expected[i]) * expected[i]; maximum = std::max(maximum, std::fabs(delta));
    }
    const double nmse = error / std::max(energy, 1e-12);
    std::printf("%s NMSE=%.9g max_abs=%.9g\n", label, nmse, maximum);
    CHECK(std::isfinite(nmse) && (energy ? nmse <= 1e-5 : maximum <= 0.004));
}

static void projection_test(ggml_backend_dev_t device, size_t width, size_t outputs, size_t rows, const char * label) {
    // Correctness cases release logger/backend/device/context resources together.
    backend_ptr backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), ggml_backend_free);
    CHECK(backend && cpu); if (!backend || !cpu) { return; }
    CHECK(ggml_backend_supports_buft(backend.get(), ggml_backend_dev_buffer_type(ggml_backend_get_device(cpu.get()))));
    auto wc = context(), c = context(), pc = context();
    auto * w = ggml_new_tensor_2d(wc.get(), GGML_TYPE_Q4_0, int64_t(width), int64_t(outputs));
    buffer_ptr wb(ggml_backend_alloc_ctx_tensors(wc.get(), backend.get()), ggml_backend_buffer_free);
    CHECK(wb != nullptr); if (!wb) { return; }
    ggml_backend_buffer_set_usage(wb.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> original_weights(width * outputs);
    for (size_t i = 0; i < original_weights.size(); ++i) { original_weights[i] = 0.03f * std::sin(float(i % 13007) * 0.137f); }
    std::vector<uint8_t> packed_weights(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q4_0, original_weights.data(), packed_weights.data(), 0, int64_t(outputs), int64_t(width), nullptr);
    original_weights.clear(); original_weights.shrink_to_fit();
    ggml_backend_tensor_set(w, packed_weights.data(), 0, packed_weights.size());
    auto * x = ggml_new_tensor_2d(c.get(), GGML_TYPE_F32, int64_t(width), int64_t(rows));
    auto * y = ggml_mul_mat(c.get(), w, x);
    auto * graph = ggml_new_graph(c.get()); ggml_graph_add_node(graph, y);
    if (width == 32) { ggml_graph_add_node(graph, ggml_reshape_3d(c.get(), y, 4, int64_t(outputs / 4), int64_t(rows))); }
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(c.get(), backend.get()), ggml_backend_buffer_free);
    CHECK(buffer && ggml_backend_supports_op(backend.get(), y)); if (!buffer) { return; }
    auto * fw = ggml_new_tensor_2d(pc.get(), GGML_TYPE_F16, int64_t(width), int64_t(outputs));
    auto * fx = ggml_new_tensor_2d(pc.get(), GGML_TYPE_F32, int64_t(width), int64_t(rows));
    buffer_ptr pb(ggml_backend_alloc_ctx_tensors(pc.get(), cpu.get()), ggml_backend_buffer_free);
    CHECK(pb != nullptr); if (!pb) { return; }
    auto prepared_weights = prepare_q4_weights({packed_weights.data(), packed_weights.size()}, width, outputs);
    CHECK(prepared_weights.ok()); if (!prepared_weights.ok()) { return; }
    ggml_backend_tensor_set(fw, prepared_weights.value().data(), 0, prepared_weights.value().size() * 2);
    auto reg = ggml_backend_dev_backend_reg(device);
    auto stats_fn = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
    ggml_backend_qairt_stats before {}, after {};
    CHECK(stats_fn && stats_fn(backend.get(), &before));
    std::vector<float> input(width * rows), last;
    for (int run = 0; run < 2; ++run) {
        for (size_t i = 0; i < input.size(); ++i) { input[i] = (run + 1) * std::sin(float(i % 4001) * 0.043f); }
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        const auto status = ggml_backend_graph_compute(backend.get(), graph);
        CHECK(status == GGML_STATUS_SUCCESS); if (status != GGML_STATUS_SUCCESS) { return; }
        last.resize(outputs * rows); ggml_backend_tensor_get(y, last.data(), 0, last.size() * sizeof(float));
        std::vector<float> prepared_input(input.size());
        for (size_t i = 0; i < input.size(); ++i) { prepared_input[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(input[i])); }
        ggml_backend_tensor_set(fx, prepared_input.data(), 0, prepared_input.size() * sizeof(float));
        std::printf("%s K=%zu N=%zu M=%zu run=%d ", label, width, outputs, rows, run);
        accuracy(last, cpu_product(cpu.get(), fw, fx), "prepared");
        accuracy(last, cpu_product(cpu.get(), w, x), "regular");
    }
    CHECK(stats_fn && stats_fn(backend.get(), &after));
    CHECK(after.preparations - before.preparations == 1 && after.cache_hits > before.cache_hits);
    std::vector<uint8_t> unchanged(packed_weights.size());
    ggml_backend_tensor_get(w, unchanged.data(), 0, unchanged.size()); CHECK(unchanged == packed_weights);
    if (width == 32) {
        const auto half = ggml_fp32_to_fp16(0.25f); std::memcpy(packed_weights.data(), &half, 2);
        ggml_backend_tensor_set(w, packed_weights.data(), 0, packed_weights.size());
        CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_qairt_stats mutated {}; CHECK(stats_fn && stats_fn(backend.get(), &mutated));
        CHECK(mutated.preparations == after.preparations + 1);
        std::vector<float> saved(outputs * rows); ggml_backend_tensor_get(y, saved.data(), 0, saved.size() * 4);
        input[0] = std::numeric_limits<float>::quiet_NaN();
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * 4);
        CHECK(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_FAILED);
        std::vector<float> actual(saved.size()); ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * 4);
        CHECK(saved == actual);
    }
}
static void fp16_activation_test(ggml_backend_dev_t device) {
    backend_ptr backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    CHECK(backend); if (!backend) { return; }
    auto c = context();
    auto * w = ggml_new_tensor_2d(c.get(), GGML_TYPE_Q4_0, 32, 1);
    auto * x = ggml_new_tensor_2d(c.get(), GGML_TYPE_F32, 32, 1);
    auto * y = ggml_mul_mat(c.get(), w, x);
    auto * graph = ggml_new_graph(c.get()); ggml_graph_add_node(graph, y);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(c.get(), backend.get()), ggml_backend_buffer_free);
    CHECK(buffer); if (!buffer) { return; }
    std::vector<float> weights(32, 0), input(32, 0);
    weights[1] = 1; input[0] = 127; input[1] = 0.3f;
    std::vector<uint8_t> packed(ggml_nbytes(w));
    ggml_quantize_chunk(GGML_TYPE_Q4_0, weights.data(), packed.data(), 0, 1, 32, nullptr);
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    const auto status = ggml_backend_graph_compute(backend.get(), graph);
    CHECK(status == GGML_STATUS_SUCCESS); if (status != GGML_STATUS_SUCCESS) { return; }
    float actual = 0; ggml_backend_tensor_get(y, &actual, 0, sizeof(actual));
    const float expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(input[1]));
    std::printf("direct FP16 activation actual=%.9g expected=%.9g\n", actual, expected);
    CHECK(std::fabs(actual - expected) <= 0.001f);
}
int main(int argc, char ** argv) {
    ggml_log_set(logger, nullptr);
    ggml_backend_load_all();
    packing_tests();
    if (argc == 2 && std::strcmp(argv[1], "--host") == 0) {
        std::printf("QAIRT FP16 host: %d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    }
    auto device = ggml_backend_dev_by_name("QAIRT0");
    CHECK(device != nullptr);
    if (device) {
        fp16_activation_test(device);
        projection_test(device, 32, 16, 13, "small");
        const bool shapes = argc == 2 && std::strcmp(argv[1], "--model-shapes") == 0;
        if (shapes) {
            struct shape { const char * name; size_t k, n; };
            for (const auto & s : {shape{"Q",2048,2048}, shape{"K",2048,1024}, shape{"V",2048,1024},
                                  shape{"O",2048,2048}, shape{"gate",2048,6144}, shape{"up",2048,6144}, shape{"down",6144,2048}}) {
                for (size_t rows : {size_t(1),size_t(13),size_t(128)}) { projection_test(device, s.k, s.n, rows, s.name); }
            }
        }
    }
    std::printf("QAIRT FP16: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
