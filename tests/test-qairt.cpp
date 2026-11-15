#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-qairt.h"
#include "qairt-lowering.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace ggml::qairt;
static int failures = 0;
static int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); ++failures; } } while (0)

static ggml_context * context() { return ggml_init({4 * 1024 * 1024, nullptr, true}); }
static void upload(ggml_tensor * t, const std::vector<float> & values) {
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_set(t, values.data(), 0, values.size() * 4); }
    else {
        std::vector<ggml_fp16_t> half(values.size());
        for (size_t i = 0; i < values.size(); ++i) { half[i] = ggml_fp32_to_fp16(values[i]); }
        ggml_backend_tensor_set(t, half.data(), 0, half.size() * 2);
    }
}
static std::vector<float> download(ggml_tensor * t) {
    const size_t n = size_t(ggml_nelements(t));
    std::vector<float> values(n);
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_get(t, values.data(), 0, n * 4); }
    else {
        std::vector<ggml_fp16_t> half(n);
        ggml_backend_tensor_get(t, half.data(), 0, n * 2);
        for (size_t i = 0; i < n; ++i) { values[i] = ggml_fp16_to_fp32(half[i]); }
    }
    return values;
}
static void expect(ggml_tensor * t, const std::vector<float> & reference, float tolerance = 0.004f) {
    auto actual = download(t);
    CHECK(actual.size() == reference.size());
    for (size_t i = 0; i < actual.size() && i < reference.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::fabs(actual[i] - reference[i]) <= tolerance * std::max(1.0f, std::fabs(reference[i])));
    }
}
static void host_tests() {
    auto * c = context();
    auto * w = ggml_new_tensor_2d(c, GGML_TYPE_F16, 4, 3);
    auto * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 2);
    auto * y = ggml_mul_mat(c, w, a);
    CHECK(primitive_supported(y));
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    CHECK(!primitive_supported(y));
    ggml_mul_mat_set_prec(y, GGML_PREC_DEFAULT);
    CHECK(!primitive_supported(ggml_add(c, a, ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 1))));
    auto * half = ggml_new_tensor_2d(c, GGML_TYPE_F16, 4, 2);
    CHECK(primitive_supported(ggml_add(c, half, ggml_new_tensor_2d(c, GGML_TYPE_F16, 4, 1))));
    CHECK(primitive_supported(ggml_cont(c, ggml_transpose(c, a))));
    CHECK(!primitive_supported(ggml_sqr(c, a)));
    auto saved = w->type;
    w->type = GGML_TYPE_Q4_0;
    CHECK(!primitive_supported(y));
    w->type = saved;
    auto stride = a->nb[1]; a->nb[1] += 4;
    CHECK(!primitive_supported(y)); a->nb[1] = stride;
    auto n = a->ne[2]; a->ne[2] = 2;
    CHECK(!primitive_supported(y)); a->ne[2] = n;
    n = a->ne[0]; a->ne[0] = INT64_MAX;
    CHECK(!float_layout(a)); a->ne[0] = n;
    CHECK(!primitive_supported(nullptr));
    ggml_free(c);
}
static void primitive_tests(ggml_backend_t backend, enum ggml_type type) {
    auto * c = context();
    auto * a = ggml_new_tensor_2d(c, type, 3, 2);
    auto * b = ggml_new_tensor_2d(c, type, 3, 1);
    auto * add = ggml_add(c, a, b);
    CHECK(!ggml_backend_supports_op(backend, ggml_sub(c, a, b)));
    auto * mul = ggml_mul(c, a, b);
    CHECK(!ggml_backend_supports_op(backend, ggml_div(c, a, b)));
    auto * transpose = ggml_cont(c, ggml_transpose(c, a));
    auto * g = ggml_new_graph(c);
    for (auto * t : {add, mul, transpose}) { CHECK(ggml_backend_supports_op(backend, t)); ggml_build_forward_expand(g, t); }
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer != nullptr);
    if (buffer) {
        upload(a, {1,2,3,4,5,6}); upload(b, {1,2,4});
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        expect(add, {2,4,7,5,7,10});
        expect(mul, {1,4,12,4,10,24});
        expect(transpose, {1,4,2,5,3,6}, 0);
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}
static void matmul_tests(ggml_backend_t backend, enum ggml_type weight_type, enum ggml_type activation_type = GGML_TYPE_F32) {
    auto * wc = context(); auto * c = context();
    auto * w = ggml_new_tensor_2d(wc, weight_type, 4, 3);
    auto weight_buffer = ggml_backend_alloc_ctx_tensors(wc, backend);
    CHECK(weight_buffer != nullptr);
    if (!weight_buffer) { ggml_free(wc); ggml_free(c); return; }
    ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * a = ggml_new_tensor_2d(c, activation_type, 4, 2);
    auto * y = ggml_mul_mat(c, w, a);
    auto * bias = ggml_new_tensor_2d(c, GGML_TYPE_F16, 3, 1);
    auto * cast = ggml_cast(c, y, GGML_TYPE_F16);
    auto * z = ggml_add(c, cast, bias);
    auto * g = ggml_new_graph(c); ggml_build_forward_expand(g, z);
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer != nullptr);
    if (buffer) {
        const std::vector<float> weights {1,0,0,1, 0,1,1,0, 1,-1,1,-1};
        upload(w, weights); upload(bias, {1,2,3});
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto stats_fn = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
        CHECK(stats_fn != nullptr);
        ggml_backend_qairt_stats before {}, after {};
        if (stats_fn) { CHECK(stats_fn(backend, &before)); }
        for (float scale : {1.0f, 2.0f, 1.0f}) {
            upload(a, {scale,2*scale,3*scale,4*scale,-scale,0,scale,2*scale});
            CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
            expect(y, {5*scale,5*scale,-2*scale,scale,scale,-2*scale});
            expect(z, {5*scale+1,5*scale+2,-2*scale+3,scale+1,scale+2,-2*scale+3});
        }
        if (stats_fn) {
            CHECK(stats_fn(backend, &after));
            CHECK(after.preparations - before.preparations <= 2);
            CHECK(after.cache_hits - before.cache_hits >= 4);
            CHECK(after.executions - before.executions == 6);
        }
        auto twice = weights; for (auto & value : twice) { value *= 2; }
        upload(w, twice);
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        expect(y, {10,10,-4,2,2,-4});
        if (stats_fn) { ggml_backend_qairt_stats now {}; CHECK(stats_fn(backend, &now)); CHECK(now.preparations == after.preparations + 1); }
        if (stats_fn) {
            ggml_backend_qairt_stats previous {}, next {};
            CHECK(stats_fn(backend, &previous));
            ggml_backend_buffer_reset(weight_buffer);
            CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
            CHECK(stats_fn(backend, &next));
            CHECK(next.preparations == previous.preparations + 1);
            expect(y, {10,10,-4,2,2,-4});
        }
        const auto old_y = download(y), old_z = download(z);
        upload(a, {2,4,6,8,-2,0,2,4});
        upload(bias, {std::numeric_limits<float>::quiet_NaN(),2,3});
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_FAILED);
        expect(y, old_y, 0); expect(z, old_z, 0);
        ggml_backend_buffer_free(buffer);
    }
    ggml_backend_buffer_free(weight_buffer); ggml_free(wc); ggml_free(c);
}
static void buffer_tests(ggml_backend_t backend) {
    auto * c = context();
    auto * a = ggml_new_tensor_1d(c, GGML_TYPE_F32, 4);
    auto * b = ggml_new_tensor_1d(c, GGML_TYPE_F32, 4);
    auto * quantized = ggml_new_tensor_1d(c, GGML_TYPE_Q4_0, 32);
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer != nullptr);
    if (buffer) {
        ggml_tensor * tensors[] = {a,b};
        const size_t offsets[] = {0,0}, sizes[] = {16,16}, positions[] = {0,16};
        const float source[] = {1,2,3,4,5,6,7,8};
        float result[8] {};
        ggml_backend_tensor_set_n(tensors, source, offsets, sizes, positions, 2);
        const ggml_tensor * read[] = {a,b};
        ggml_backend_tensor_get_n(read, result, offsets, sizes, positions, 2);
        CHECK(std::memcmp(source, result, sizeof(source)) == 0);
        const float patch = 9;
        ggml_backend_tensor_set(a, &patch, 4, 4);
        expect(a, {1,9,3,4}, 0);
        ggml_backend_tensor_copy(a, b); expect(b, {1,9,3,4}, 0);
        std::vector<uint8_t> bytes(ggml_nbytes(quantized), 0x5a), got(bytes.size());
        ggml_backend_tensor_set(quantized, bytes.data(), 0, bytes.size());
        ggml_backend_tensor_get(quantized, got.data(), 0, got.size());
        CHECK(got == bytes);
        ggml_backend_buffer_clear(buffer, 0);
        expect(a, {0,0,0,0}, 0);
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}

static void transposed_matmul_test(ggml_backend_t backend) {
    auto * c = context();
    auto * storage = ggml_new_tensor_2d(c, GGML_TYPE_F16, 3, 4);
    auto * weights = ggml_cont(c, ggml_transpose(c, storage));
    auto * input = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 2);
    auto * output = ggml_mul_mat(c, weights, input);
    CHECK(ggml_backend_supports_op(backend, output));
    auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer != nullptr);
    if (buffer) {
        upload(storage, {1,0,1,0,1,-1,0,1,1,1,0,-1});
        upload(input, {1,2,3,4,-1,0,1,2});
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        expect(output, {5,5,-2,1,1,-2});
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}
static void placement_test(ggml_backend_t qairt) {
    auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr); if (!cpu) { return; }
    auto * c = context(); auto * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 3, 2);
    auto * unsupported = ggml_sqr(c, a);
    CHECK(!ggml_backend_supports_op(qairt, unsupported));
    auto * g = ggml_new_graph(c); ggml_build_forward_expand(g, unsupported);
    ggml_backend_t backends[] = {qairt, cpu};
    auto scheduler = ggml_backend_sched_new(backends, nullptr, 2, 128, false, true);
    CHECK(scheduler != nullptr);
    if (scheduler) {
        CHECK(ggml_backend_sched_alloc_graph(scheduler, g));
        CHECK(ggml_backend_sched_get_tensor_backend(scheduler, unsupported) == cpu);
        upload(a, {1,2,3,4,5,6});
        CHECK(ggml_backend_sched_graph_compute(scheduler, g) == GGML_STATUS_SUCCESS);
        expect(unsupported, {1,4,9,16,25,36}, 0);
        ggml_backend_sched_free(scheduler);
    }
    ggml_backend_free(cpu);
    ggml_free(c);
}
int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--host") == 0) { host_tests(); }
    else {
        ggml_backend_load_all();
        auto dev = ggml_backend_dev_by_name("QAIRT0");
        if (argc == 2 && std::strcmp(argv[1], "--unavailable") == 0) { CHECK(dev == nullptr); }
        else {
            CHECK(dev != nullptr);
            auto backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
            CHECK(backend != nullptr);
            if (backend) {
                buffer_tests(backend); transposed_matmul_test(backend);
                primitive_tests(backend, GGML_TYPE_F16);
                matmul_tests(backend, GGML_TYPE_F16); matmul_tests(backend, GGML_TYPE_F32);
                matmul_tests(backend, GGML_TYPE_F16, GGML_TYPE_F16);
                placement_test(backend);
                ggml_backend_free(backend);
            }
        }
    }
    std::printf("QAIRT M1: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
