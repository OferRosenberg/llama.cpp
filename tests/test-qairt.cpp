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
    else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> values_bf16(values.size());
        for (size_t i = 0; i < values.size(); ++i) { values_bf16[i] = ggml_fp32_to_bf16(values[i]); }
        ggml_backend_tensor_set(t, values_bf16.data(), 0, values_bf16.size() * sizeof(ggml_bf16_t));
    }
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
    else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> values_bf16(n);
        ggml_backend_tensor_get(t, values_bf16.data(), 0, n * sizeof(ggml_bf16_t));
        for (size_t i = 0; i < n; ++i) { values[i] = ggml_bf16_to_fp32(values_bf16[i]); }
    }
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
    double squared_error = 0.0;
    double reference_energy = 0.0;
    for (size_t i = 0; i < actual.size() && i < reference.size(); ++i) {
        CHECK(std::isfinite(actual[i]) && std::fabs(actual[i] - reference[i]) <= tolerance * std::max(1.0f, std::fabs(reference[i])));
        const double difference = double(actual[i]) - double(reference[i]);
        squared_error += difference * difference;
        reference_energy += double(reference[i]) * double(reference[i]);
    }
    // Zero-reference tensors retain the absolute per-element check above.
    CHECK(reference_energy == 0.0 || squared_error <= 1e-5 * reference_energy);
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
    CHECK(primitive_supported(ggml_add(c, a, ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 1))));
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
    CHECK(!ggml_backend_supports_op(backend, add));
    for (auto * t : {mul, transpose}) { CHECK(ggml_backend_supports_op(backend, t)); ggml_build_forward_expand(g, t); }
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer != nullptr);
    if (buffer) {
        upload(a, {1,2,3,4,5,6}); upload(b, {1,2,4});
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
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
    auto * z = ggml_mul(c, cast, bias);
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
            expect(z, {5*scale,10*scale,-6*scale,scale,2*scale,-6*scale});
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
static void m2_cpu_weight_placement(ggml_backend_t qairt) {
    auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr); CHECK(cpu); if (!cpu) { return; }
    auto * wc = context(); auto * c = context();
    auto * w = ggml_new_tensor_2d(wc, GGML_TYPE_F32, 4, 3);
    auto weights = ggml_backend_alloc_ctx_tensors(wc, cpu); CHECK(weights);
    if (!weights) { ggml_free(wc); ggml_free(c); ggml_backend_free(cpu); return; }
    ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    upload(w, {1,0,0,1,0,1,1,0,1,-1,1,-1});
    auto * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 2); auto * y = ggml_mul_mat(c, w, a);
    auto * g = ggml_new_graph(c); ggml_build_forward_expand(g, y);
    ggml_backend_t backends[] = {qairt, cpu};
    auto scheduler = ggml_backend_sched_new(backends, nullptr, 2, 128, false, false); CHECK(scheduler);
    if (scheduler) {
        CHECK(ggml_backend_sched_alloc_graph(scheduler, g));
        CHECK(ggml_backend_sched_get_tensor_backend(scheduler, y) == cpu);
        upload(a, {1,2,3,4,-1,0,1,2});
        CHECK(ggml_backend_sched_graph_compute(scheduler, g) == GGML_STATUS_SUCCESS); expect(y, {5,5,-2,1,1,-2});
        ggml_backend_sched_free(scheduler);
    }
    ggml_backend_buffer_free(weights); ggml_free(wc); ggml_free(c); ggml_backend_free(cpu);
}
static void m2_connected_tests(ggml_backend_t backend) {
    auto * c = context(); auto * a = ggml_new_tensor_2d(c, GGML_TYPE_F16, 4, 2);
    auto * b = ggml_new_tensor_2d(c, GGML_TYPE_F16, 4, 1);
    auto * first = ggml_mul(c, a, b); auto * mul = ggml_mul(c, first, b);
    auto * g = ggml_new_graph(c); ggml_build_forward_expand(g, mul);
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
    if (buffer) {
        upload(a, {1,2,3,4,5,6,7,8}); upload(b, {1,2,3,4});
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
        expect(first, {1,4,9,16,5,12,21,32}); expect(mul, {1,8,27,64,5,24,63,128});
        const auto saved_first = download(first), saved_mul = download(mul);
        upload(a, {std::numeric_limits<float>::quiet_NaN(),2,3,4,5,6,7,8});
        CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_FAILED);
        expect(first, saved_first, 0); expect(mul, saved_mul, 0);
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}
static void m2_run_tests(ggml_backend_t backend) {
    auto * c = context();
    auto * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 2);
    auto * w = ggml_new_tensor_2d(c, GGML_TYPE_F16, 4, 3);
    auto * g = ggml_new_graph(c);
    std::vector<ggml_tensor *> outputs;
    for (int i = 0; i < 3; ++i) { outputs.push_back(ggml_mul_mat(c, w, a)); ggml_build_forward_expand(g, outputs.back()); }
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer); if (!buffer) { ggml_free(c); return; }
    upload(w, {1,0,0,1,0,1,1,0,1,-1,1,-1}); upload(a, {1,2,3,4,-1,0,1,2});
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
    ggml_backend_qairt_stats before {}, after {}; CHECK(stats && stats(backend, &before));
    CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
    for (auto * output : outputs) { expect(output, {5,5,-2,1,1,-2}); }
    CHECK(stats && stats(backend, &after));
    CHECK(after.preparations == before.preparations + 1); CHECK(after.executions == before.executions + 1);
    before = after; upload(a, {2,4,6,8,-2,0,2,4});
    CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
    for (auto * output : outputs) { expect(output, {10,10,-4,2,2,-4}); }
    CHECK(stats && stats(backend, &after)); CHECK(after.preparations == before.preparations);
    CHECK(after.cache_hits == before.cache_hits + 1);
    ggml_backend_buffer_free(buffer); ggml_free(c);
}
static void m2_metadata_tests() {
    auto * c = context(); auto * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 2);
    auto * view = ggml_reshape_3d(c, a, 2, 2, 2); CHECK(primitive_supported(view));
    CHECK(primitive_supported(ggml_permute(c, view, 0, 2, 1, 3)));
    const auto offset = view->view_offs; view->view_offs = 32; CHECK(!primitive_supported(view)); view->view_offs = offset;
    const auto dimension = view->ne[2]; view->ne[2] = INT64_MAX; CHECK(!primitive_supported(view)); view->ne[2] = dimension;
    ggml_free(c);
}
static void m2_bf16_tests(ggml_backend_t backend) {
    double maximum_nmse = 0.0, maximum_chain_nmse = 0.0;
    auto nmse = [](const std::vector<float> & actual, const std::vector<float> & reference) {
        double error = 0.0, energy = 0.0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double difference = double(actual[i]) - reference[i];
            error += difference * difference; energy += double(reference[i]) * reference[i];
        }
        return energy == 0.0 ? error : error / energy;
    };
    for (const bool transposed : {false, true}) {
        for (const bool constant : {false, true}) {
            for (const auto activation_type : {GGML_TYPE_F32, GGML_TYPE_BF16}) {
                auto * wc = context(); auto * c = context();
                auto * raw = ggml_new_tensor_2d(wc, GGML_TYPE_BF16, 64, 3);
                auto * w = raw;
                auto * activation_raw = ggml_new_tensor_2d(c, activation_type, transposed ? 2 : 64, transposed ? 64 : 2);
                auto * a = transposed ? ggml_transpose(c, activation_raw) : activation_raw;
                auto * identity = activation_type == GGML_TYPE_F32 ? ggml_new_tensor_2d(wc, GGML_TYPE_F16, 64, 64) : nullptr;
                auto * staged = identity ? ggml_mul_mat(c, identity, a) : nullptr;
                auto * input = staged ? staged : a;
                auto * y = ggml_mul_mat(c, w, input);
                CHECK(weight_layout(w)); CHECK(primitive_supported(raw)); CHECK(primitive_supported(w));
                CHECK(!float_layout(w)); CHECK(!row_layout(raw));
                CHECK(primitive_supported(y));
                CHECK(!primitive_supported(ggml_cont(c, w)));
                auto * half = ggml_new_tensor_2d(c, GGML_TYPE_F16, 64, 2);
                CHECK(!primitive_supported(ggml_mul_mat(c, w, half)));
                for (const auto precision : {GGML_PREC_BF16, GGML_PREC_F32, GGML_PREC_F16}) {
                    CHECK(ggml_prec_set_acc(y, precision)); CHECK(!primitive_supported(y));
                }
                CHECK(ggml_prec_set_acc(y, GGML_PREC_DEFAULT));
                for (const auto precision : {GGML_PREC_BF16, GGML_PREC_F32, GGML_PREC_F16}) {
                    CHECK(ggml_prec_set_src(y, precision, 1)); CHECK(!primitive_supported(y));
                }
                CHECK(ggml_prec_set_src(y, GGML_PREC_DEFAULT, 1));
                const size_t stride = w->nb[0]; w->nb[0] += 1; CHECK(!primitive_supported(y)); w->nb[0] = stride;
                auto * wp = activation_type == GGML_TYPE_F32 ? ggml_new_tensor_2d(wc, GGML_TYPE_F16, 64, 3) : nullptr;
                auto * yp = wp ? ggml_mul_mat(c, wp, input) : nullptr;
                auto * graph = ggml_new_graph(c);
                if (wp) { ggml_build_forward_expand(graph, wp); }
                ggml_build_forward_expand(graph, y);
                if (yp) { ggml_build_forward_expand(graph, yp); }
                if (!backend || !primitive_supported(y)) { ggml_free(wc); ggml_free(c); continue; }
                CHECK(ggml_backend_supports_op(backend, y));
                auto * rc = context();
                auto * rw = ggml_new_tensor_2d(rc, GGML_TYPE_BF16, 64, 3);
                auto * ra = ggml_new_tensor_2d(rc, activation_type, 64, 2);
                auto * reference_staged = identity ? ggml_new_tensor_2d(rc, GGML_TYPE_F32, 64, 2) : nullptr;
                auto * reference_input = reference_staged ? reference_staged : ra;
                auto * ry = ggml_mul_mat(rc, rw, reference_input);
                auto * rwp = wp ? ggml_new_tensor_2d(rc, GGML_TYPE_F16, 64, 3) : nullptr;
                auto * ryp = rwp ? ggml_mul_mat(rc, rwp, reference_input) : nullptr;
                auto * reference_graph = ggml_new_graph(rc); ggml_build_forward_expand(reference_graph, ry);
                if (ryp) { ggml_build_forward_expand(reference_graph, ryp); }
                auto cpu_device = ggml_backend_dev_by_name("CPU");
                auto cpu = cpu_device ? ggml_backend_dev_init(cpu_device, nullptr) : nullptr; CHECK(cpu);
                auto wb = ggml_backend_alloc_ctx_tensors(wc, backend);
                auto ab = ggml_backend_alloc_ctx_tensors(c, backend);
                auto rb = cpu ? ggml_backend_alloc_ctx_tensors(rc, cpu) : nullptr;
                CHECK(wb && ab && rb);
                if (wb && ab && rb) {
                    if (constant) { ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS); }
                    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                    auto stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
                    ggml_backend_qairt_stats before {}, after {};
                    std::vector<float> weights(192), activation(128, 1.00395f);
                    for (size_t i = 0; i < weights.size(); ++i) { weights[i] = 0.125f + 0.03f * std::sin(float(i) * 0.17f); }
                    auto set_weights = [&]() {
                        upload(raw, weights); upload(rw, weights);
                        if (wp) { upload(wp, weights); upload(rwp, weights); }
                    };
                    if (identity) {
                        std::vector<float> values(4096, 0.0f);
                        for (size_t i = 0; i < 64; ++i) { values[i * 64 + i] = 1.0f; }
                        upload(identity, values);
                    }
                    const uint64_t runs = identity ? 2 : 1;
                    auto set_activation = [&]() {
                        std::vector<float> storage(activation.size());
                        for (size_t row = 0; row < 2; ++row) {
                            for (size_t k = 0; k < 64; ++k) { storage[transposed ? k * 2 + row : row * 64 + k] = activation[row * 64 + k]; }
                        }
                        upload(activation_raw, storage);
                    };
                    set_weights(); set_activation(); upload(ra, activation);
                    if (reference_staged) { upload(reference_staged, activation); }
                    std::vector<uint8_t> original(ggml_nbytes(raw)), retained(original.size());
                    ggml_backend_tensor_get(raw, original.data(), 0, original.size());
                    CHECK(ggml_backend_supports_op(cpu, ry));
                    CHECK(ggml_backend_graph_compute(cpu, reference_graph) == GGML_STATUS_SUCCESS);
                    const auto original_reference = download(ry);
                    auto compare_result = [&]() {
                        if (staged) {
                            // Compare conversion with identical boundary inputs; qualify upstream FP16 error separately.
                            expect(staged, activation);
                            upload(reference_staged, download(staged));
                        }
                        CHECK(ggml_backend_graph_compute(cpu, reference_graph) == GGML_STATUS_SUCCESS);
                        const auto reference = download(ry), actual = download(y);
                        maximum_nmse = std::max(maximum_nmse, nmse(actual, reference));
                        expect(y, reference); if (yp) { expect(yp, download(ryp)); }
                    };
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        CHECK(stats && stats(backend, &before));
                        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                        if (repeat == 0) {
                            const auto actual = download(y);
                            const double chain_error = nmse(actual, original_reference);
                            maximum_chain_nmse = std::max(maximum_chain_nmse, chain_error);
                            std::printf("BF16 fixture: transpose=%d static=%d activation=%s original_cpu=%g qnn=%g chain_nmse=%.9g\n", transposed, constant, ggml_type_name(activation_type), original_reference[0], actual[0], chain_error);
                        }
                        compare_result();
                        CHECK(stats && stats(backend, &after));
                        CHECK(after.executions == before.executions + runs);
                        if (repeat > 0) { CHECK(after.preparations == before.preparations); CHECK(after.cache_hits == before.cache_hits + runs); }
                    }
                    ggml_backend_tensor_get(raw, retained.data(), 0, retained.size()); CHECK(retained == original);
                    before = after;
                    for (size_t i = 0; i < activation.size(); ++i) { activation[i] = 0.4f * std::cos(float(i) * 0.09f); }
                    set_activation(); upload(ra, activation);
                    CHECK(ggml_backend_graph_compute(cpu, reference_graph) == GGML_STATUS_SUCCESS);
                    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    compare_result();
                    CHECK(stats && stats(backend, &after)); CHECK(after.preparations == before.preparations);
                    before = after; weights[0] += 0.25f; set_weights();
                    CHECK(ggml_backend_graph_compute(cpu, reference_graph) == GGML_STATUS_SUCCESS);
                    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    compare_result();
                    CHECK(stats && stats(backend, &after));
                    CHECK(after.preparations == before.preparations + (constant ? runs : 0));
                    const auto saved_y = download(y), saved_plain = yp ? download(yp) : std::vector<float>{};
                    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), 1.0e10f}) {
                        activation[0] = invalid; set_activation();
                        if (staged) { upload(staged, std::vector<float>(128, 42.0f)); }
                        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                        CHECK(download(y) == saved_y); if (yp) { CHECK(download(yp) == saved_plain); }
                        if (staged) { CHECK(download(staged) == std::vector<float>(128, 42.0f)); }
                    }
                    activation[0] = 0.25f; set_activation();
                    const float saved_weight = weights[0];
                    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -1.0e10f}) {
                        weights[0] = invalid; set_weights();
                        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                        CHECK(download(y) == saved_y); if (yp) { CHECK(download(yp) == saved_plain); }
                        if (staged) { CHECK(download(staged) == std::vector<float>(128, 42.0f)); }
                    }
                    weights[0] = saved_weight; set_weights();
                    CHECK(ggml_prec_set_acc(y, GGML_PREC_BF16));
                    CHECK(!ggml_backend_supports_op(backend, y));
                    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                    CHECK(download(y) == saved_y);
                    CHECK(ggml_prec_set_acc(y, GGML_PREC_DEFAULT));
                    upload(ra, activation);
                    CHECK(ggml_backend_graph_compute(cpu, reference_graph) == GGML_STATUS_SUCCESS);
                    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    compare_result();
                }
                if (rb) { ggml_backend_buffer_free(rb); }
                if (ab) { ggml_backend_buffer_free(ab); }
                if (wb) { ggml_backend_buffer_free(wb); }
                if (cpu) { ggml_backend_free(cpu); }
                ggml_free(rc); ggml_free(wc); ggml_free(c);
            }
        }
    }
    if (backend) { std::printf("BF16 accuracy: matched_input_max_nmse=%.9g full_chain_max_nmse=%.9g full_chain_qualified=%d\n", maximum_nmse, maximum_chain_nmse, maximum_chain_nmse <= 1e-5); }
}
static void m2_quantized_tests(ggml_backend_t backend) {
    for (const auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        auto * wc = context(); auto * c = context();
        auto * w = ggml_new_tensor_2d(wc, type, 256, 3);
        CHECK(primitive_supported(w));
        auto * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, 2);
        auto * y = ggml_mul_mat(c, w, a);
        CHECK(primitive_supported(y));
        for (const auto precision : {GGML_PREC_F32, GGML_PREC_BF16, GGML_PREC_F16, GGML_PREC_Q8}) {
            CHECK(ggml_prec_set_src(y, precision, 1));
            CHECK(!primitive_supported(y));
        }
        CHECK(ggml_prec_set_src(y, GGML_PREC_DEFAULT, 1));
        auto * half_activation = ggml_new_tensor_2d(c, GGML_TYPE_F16, 256, 2);
        CHECK(!primitive_supported(ggml_mul_mat(c, w, half_activation)));
        const auto row_stride = w->nb[1]; w->nb[1] += 1; CHECK(!primitive_supported(y)); w->nb[1] = row_stride;
        const auto columns = w->ne[0]; w->ne[0] += 1; CHECK(!primitive_supported(y)); w->ne[0] = columns;
        if (!backend || !primitive_supported(y)) { ggml_free(wc); ggml_free(c); continue; }
        CHECK(ggml_backend_supports_op(backend, y));
        auto wb = ggml_backend_alloc_ctx_tensors(wc, backend);
        auto ab = ggml_backend_alloc_ctx_tensors(c, backend);
        CHECK(wb && ab);
        if (wb && ab) {
            ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::vector<float> source(768), activation(512), reconstructed(768), reference(6);
            for (size_t i = 0; i < source.size(); ++i) { source[i] = 0.125f * std::sin(float(i) * 0.17f); }
            for (size_t i = 0; i < activation.size(); ++i) { activation[i] = std::cos(float(i) * 0.09f); }
            std::vector<uint8_t> quantized(ggml_nbytes(w)), original(quantized.size());
            CHECK(ggml_quantize_chunk(type, source.data(), quantized.data(), 0, 3, 256, nullptr) == quantized.size());
            ggml_get_type_traits(type)->to_float(quantized.data(), reconstructed.data(), 768);
            for (size_t row = 0; row < 2; ++row) {
                for (size_t col = 0; col < 3; ++col) {
                    double sum = 0;
                    for (size_t k = 0; k < 256; ++k) { sum += double(reconstructed[col * 256 + k]) * activation[row * 256 + k]; }
                    reference[row * 3 + col] = float(sum);
                }
            }
            ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size()); upload(a, activation);
            auto * g = ggml_new_graph(c); ggml_build_forward_expand(g, y);
            for (int repeat = 0; repeat < 3; ++repeat) {
                CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
                expect(y, reference);
            }
            const auto saved_output = download(y);
            std::fill(activation.begin(), activation.end(), 1.0e-38f); upload(a, activation);
            CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_FAILED);
            CHECK(download(y) == saved_output);
            std::fill(activation.begin(), activation.end(), 0.0f); upload(a, activation);
            CHECK(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS);
            expect(y, std::vector<float>(6, 0.0f));
            ggml_backend_tensor_get(w, original.data(), 0, original.size()); CHECK(original == quantized);
        }
        if (wb) { ggml_backend_buffer_free(wb); }
        if (ab) { ggml_backend_buffer_free(ab); }
        ggml_free(wc); ggml_free(c);
    }
}
static void direct_fp16_tests(ggml_backend_t backend) {
    for (const auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (const bool constant : {true, false}) {
            auto * wc = context(); auto * c = context();
            auto * w = ggml_new_tensor_2d(wc, type, 256, 2);
            auto * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, 2);
            auto * y = ggml_mul_mat(c, w, a);
            auto wb = ggml_backend_alloc_ctx_tensors(wc, backend);
            auto ab = ggml_backend_alloc_ctx_tensors(c, backend);
            CHECK(wb && ab);
            if (wb && ab) {
                if (constant) { ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS); }
                std::vector<float> weights(512, 0.0f), activation(512, 0.0f), reconstructed(512), reference(4);
                weights[0] = 1.0f; weights[257] = 0.5f;
                activation[0] = 1.00395f; activation[1] = 0.23456f; activation[255] = 1.0f;
                activation[256] = -1.00395f; activation[257] = 0.45678f; activation[511] = 2.0f;
                std::vector<uint8_t> quantized(ggml_nbytes(w));
                if (ggml_is_quantized(type)) {
                    CHECK(ggml_quantize_chunk(type, weights.data(), quantized.data(), 0, 2, 256, nullptr) == quantized.size());
                    ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
                    ggml_get_type_traits(type)->to_float(quantized.data(), reconstructed.data(), 512);
                } else {
                    upload(w, weights); reconstructed = download(w);
                    ggml_backend_tensor_get(w, quantized.data(), 0, quantized.size());
                }
                for (size_t row = 0; row < 2; ++row) {
                    for (size_t col = 0; col < 2; ++col) {
                        double sum = 0.0;
                        for (size_t k = 0; k < 256; ++k) {
                            const float weight = ggml_fp16_to_fp32(ggml_fp32_to_fp16(reconstructed[col * 256 + k]));
                            const float input = ggml_fp16_to_fp32(ggml_fp32_to_fp16(activation[row * 256 + k]));
                            sum += double(weight) * input;
                        }
                        reference[row * 2 + col] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(sum)));
                    }
                }
                upload(a, activation);
                auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, y);
                for (int repeat = 0; repeat < 3; ++repeat) {
                    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    expect(y, reference);
                    const auto actual = download(y);
                    std::printf("direct FP16 fixture: weights=%s static=%d repeat=%d actual=%.9g expected=%.9g\n",
                                ggml_type_name(type), int(constant), repeat, actual[0], reference[0]);
                }
                std::fill(activation.begin(), activation.end(), 1.0e-38f); upload(a, activation);
                CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                expect(y, std::vector<float>(4, 0.0f));
                const auto saved = download(y);
                activation[0] = std::numeric_limits<float>::infinity(); upload(a, activation);
                CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                CHECK(download(y) == saved);
                std::vector<uint8_t> unchanged(quantized.size());
                ggml_backend_tensor_get(w, unchanged.data(), 0, unchanged.size());
                CHECK(unchanged == quantized);
            }
            if (wb) { ggml_backend_buffer_free(wb); }
            if (ab) { ggml_backend_buffer_free(ab); }
            ggml_free(wc); ggml_free(c);
        }
    }
}
static void m2_masked_softmax_tests(ggml_backend_t backend) {
    for (const auto mask_type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (const bool broadcast : {true, false}) {
            for (const bool inplace : {false, true}) {
                auto * c = context();
                auto * x = ggml_new_tensor_4d(c, GGML_TYPE_F32, 32, 3, 2, 2);
                auto * mask = ggml_new_tensor_4d(c, mask_type, 32, 4, broadcast ? 1 : 2, broadcast ? 1 : 2);
                auto * y = inplace ? ggml_soft_max_ext_inplace(c, x, mask, 0.5f, 0.0f) : ggml_soft_max_ext(c, x, mask, 0.5f, 0.0f);
                auto * staged = ggml_dup(c, ggml_reshape_2d(c, x, 32, 12));
                CHECK(primitive_supported(y));
                CHECK(!primitive_supported(ggml_soft_max_ext(c, x, mask, 0.5f, 2.0f)));
                auto * sinks = ggml_new_tensor_1d(c, GGML_TYPE_F32, 2);
                auto * with_sinks = ggml_soft_max_ext(c, x, mask, 0.5f, 0.0f);
                ggml_soft_max_add_sinks(with_sinks, sinks); CHECK(!primitive_supported(with_sinks));
                const auto stride = mask->nb[1]; mask->nb[1] += 4;
                CHECK(!primitive_supported(y)); mask->nb[1] = stride;
                auto * repeated = ggml_new_tensor_4d(c, GGML_TYPE_F32, 32, 3, 4, 2);
                auto * periodic_mask = ggml_new_tensor_4d(c, mask_type, 32, 3, 2, 1);
                CHECK(!primitive_supported(ggml_soft_max_ext(c, repeated, periodic_mask, 1.0f, 0.0f)));
                if (!backend) { ggml_free(c); continue; }
                const bool admitted = ggml_backend_supports_op(backend, y); CHECK(admitted);
                if (!admitted) { ggml_free(c); continue; }
                auto cpu = ggml_backend_init_by_name("CPU", nullptr); CHECK(cpu);
                auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
                if (cpu && buffer) {
                    auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, staged); ggml_build_forward_expand(graph, y);
                    std::vector<float> input(384), masks(size_t(ggml_nelements(mask)), std::numeric_limits<float>::quiet_NaN());
                    for (size_t batch = 0; batch < size_t(mask->ne[3]); ++batch) {
                        for (size_t head = 0; head < size_t(mask->ne[2]); ++head) {
                            for (size_t query = 0; query < 3; ++query) {
                                const size_t start = ((batch * size_t(mask->ne[2]) + head) * 4 + query) * 32;
                                for (size_t key = 0; key < 32; ++key) {
                                    masks[start + key] = key < 8 + query + head ? -0.25f * float(key % 3) : -std::numeric_limits<float>::infinity();
                                }
                            }
                        }
                    }
                    for (size_t row = 0; row < 12; ++row) {
                        const size_t query = row % 3, head = row / 3 % 2, batch = row / 6;
                        const size_t mask_start = ((batch % size_t(mask->ne[3]) * size_t(mask->ne[2]) + head % size_t(mask->ne[2])) * 4 + query) * 32;
                        for (size_t key = 0; key < 32; ++key) {
                            input[row * 32 + key] = std::isfinite(masks[mask_start + key]) ? 100000.0f + float(key) * 0.03125f : 1000000.0f;
                        }
                    }
                    upload(x, input); upload(mask, masks);
                    CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
                    auto reference = download(y); upload(x, input);
                    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS); expect(y, reference);
                    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                    auto stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
                    ggml_backend_qairt_stats before {}, after {}; CHECK(stats && stats(backend, &before));
                    masks[0] = -std::numeric_limits<float>::infinity(); upload(mask, masks); upload(x, input);
                    CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
                    reference = download(y); upload(x, input);
                    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS); expect(y, reference);
                    CHECK(stats && stats(backend, &after)); CHECK(after.preparations == before.preparations);
                    CHECK(after.cache_hits == before.cache_hits + 1); CHECK(after.executions == before.executions + 1);
                    for (const float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), 100000.0f}) {
                        masks[1] = bad; upload(mask, masks); upload(x, input); upload(staged, std::vector<float>(384, 42.0f));
                        const auto saved = download(y);
                        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                        CHECK(download(y) == saved); CHECK(download(staged) == std::vector<float>(384, 42.0f));
                    }
                    for (size_t key = 0; key < 32; ++key) { masks[key] = -std::numeric_limits<float>::infinity(); }
                    upload(mask, masks); upload(x, input); upload(staged, std::vector<float>(384, 42.0f));
                    const auto saved = download(y);
                    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                    CHECK(download(y) == saved); CHECK(download(staged) == std::vector<float>(384, 42.0f));
                }
                if (buffer) { ggml_backend_buffer_free(buffer); }
                if (cpu) { ggml_backend_free(cpu); }
                ggml_free(c);
            }
        }
    }
    // The same GGML tensor can provide logits and a finite additive mask.
    auto * c = context(); auto * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 2);
    auto * y = ggml_soft_max_ext(c, x, x, 0.5f, 0.0f); CHECK(primitive_supported(y));
    if (backend) {
        CHECK(ggml_backend_supports_op(backend, y));
        auto cpu = ggml_backend_init_by_name("CPU", nullptr); CHECK(cpu);
        auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
        if (cpu && buffer) {
            auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, y);
            std::vector<float> input(64);
            for (size_t i = 0; i < input.size(); ++i) { input[i] = float(i % 16) * 0.0625f; }
            upload(x, input); CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
            const auto reference = download(y);
            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS); expect(y, reference);
        }
        if (buffer) { ggml_backend_buffer_free(buffer); }
        if (cpu) { ggml_backend_free(cpu); }
    }
    ggml_free(c);
}
static void m2_row_tests(ggml_backend_t backend) {
    auto * c = context();
    auto * a = ggml_new_tensor_4d(c, GGML_TYPE_F32, 32, 3, 2, 2);
    auto * rms = ggml_rms_norm(c, a, 1.0e-5f);
    auto * norm = ggml_norm(c, a, 1.0e-5f);
    auto * softmax = ggml_soft_max_ext(c, a, nullptr, 0.5f, 0.0f);
    for (auto * node : {rms, norm, softmax}) { CHECK(primitive_supported(node)); }
    CHECK(!primitive_supported(ggml_rms_norm(c, a, 0.0f)));
    CHECK(!primitive_supported(ggml_norm(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 65537), 1.0e-5f)));
    CHECK(!primitive_supported(ggml_norm(c, a, std::numeric_limits<float>::quiet_NaN())));
    CHECK(!primitive_supported(ggml_soft_max_ext(c, a, nullptr, -1.0f, 0.0f)));
    auto * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 3);
    CHECK(!primitive_supported(ggml_soft_max_ext(c, a, mask, 1.0f, 2.0f)));
    CHECK(primitive_supported(ggml_soft_max_ext(c, a, mask, 1.0f, 0.0f)));
    CHECK(!primitive_supported(ggml_norm(c, ggml_transpose(c, ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 3)), 1.0e-5f)));
    const bool use_softmax = backend && ggml_backend_dev_supports_op(ggml_backend_get_device(backend), softmax);
    if (backend && primitive_supported(rms) && primitive_supported(norm)) {
        auto * graph = ggml_new_graph(c);
        for (auto * node : {rms, norm}) { ggml_build_forward_expand(graph, node); }
        if (use_softmax) { ggml_build_forward_expand(graph, softmax); }
        auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
        if (buffer) {
            std::vector<float> input(384), rms_reference(384), norm_reference(384), soft_reference(384);
            for (size_t row = 0; row < 12; ++row) {
                double sum = 0, square = 0, denominator = 0;
                for (size_t col = 0; col < 32; ++col) {
                    const float value = float(row) * 0.15f + std::sin(float(col) * 0.3f);
                    input[row * 32 + col] = value; sum += value; square += double(value) * value;
                    denominator += std::exp(double(value) * 0.5);
                }
                const double mean = sum / 32, variance = square / 32 - mean * mean;
                for (size_t col = 0; col < 32; ++col) {
                    const double value = input[row * 32 + col];
                    rms_reference[row * 32 + col] = float(value / std::sqrt(square / 32 + 1.0e-5));
                    norm_reference[row * 32 + col] = float((value - mean) / std::sqrt(variance + 1.0e-5));
                    soft_reference[row * 32 + col] = float(std::exp(value * 0.5) / denominator);
                }
            }
            upload(a, input);
            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            expect(rms, rms_reference); expect(norm, norm_reference);
            if (use_softmax) { expect(softmax, soft_reference); }
            auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            auto stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
            ggml_backend_qairt_stats before {}, after {}; CHECK(stats && stats(backend, &before));
            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            CHECK(stats && stats(backend, &after)); CHECK(after.preparations == before.preparations);
            CHECK(after.cache_hits == before.cache_hits + (use_softmax ? 3 : 2));
            const float epsilon = 0.25f; std::memcpy(rms->op_params, &epsilon, sizeof(epsilon));
            before = after; CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            CHECK(stats && stats(backend, &after)); CHECK(after.preparations == before.preparations + 1);
            if (use_softmax) {
                const float scale = 0.25f; std::memcpy(softmax->op_params, &scale, sizeof(scale));
                before = after;
                CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                CHECK(stats && stats(backend, &after)); CHECK(after.preparations == before.preparations + 1);
                // Large constant shifts must be removed before converting logits to FP16.
                auto * soft_graph = ggml_new_graph(c); ggml_build_forward_expand(soft_graph, softmax);
                for (auto & value : input) { value += 100000.0f; }
                for (size_t row = 0; row < 12; ++row) {
                    const double maximum = *std::max_element(input.begin() + row * 32, input.begin() + (row + 1) * 32);
                    double denominator = 0;
                    for (size_t col = 0; col < 32; ++col) { denominator += std::exp((input[row * 32 + col] - maximum) * scale); }
                    for (size_t col = 0; col < 32; ++col) { soft_reference[row * 32 + col] = float(std::exp((input[row * 32 + col] - maximum) * scale) / denominator); }
                }
                upload(a, input);
                CHECK(ggml_backend_graph_compute(backend, soft_graph) == GGML_STATUS_SUCCESS);
                expect(softmax, soft_reference);
            }
            if (!use_softmax) {
                // Scheduler admission and direct computation must both reject unqualified Softmax.
                auto * rejected_graph = ggml_new_graph(c); ggml_build_forward_expand(rejected_graph, softmax);
                upload(softmax, std::vector<float>(384, 42.0f));
                CHECK(ggml_backend_graph_compute(backend, rejected_graph) == GGML_STATUS_FAILED);
                CHECK(download(softmax) == std::vector<float>(384, 42.0f));
            }
            const auto saved_rms = download(rms), saved_norm = download(norm), saved_softmax = download(softmax);
            input[0] = std::numeric_limits<float>::quiet_NaN(); upload(a, input);
            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
            CHECK(download(rms) == saved_rms && download(norm) == saved_norm && download(softmax) == saved_softmax);
            ggml_backend_buffer_free(buffer);
        }
    }
    ggml_free(c);
}
static void m2_large_norm_test(ggml_backend_t backend) {
    auto * c = context();
    auto * a = ggml_new_tensor_4d(c, GGML_TYPE_F32, 4, 1, 65536, 1);
    auto * y = ggml_norm_inplace(c, a, 1.0e-6f);
    auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, y);
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
    if (buffer) {
        std::vector<float> input(size_t(ggml_nelements(a)));
        for (size_t row = 0; row < 65536; ++row) {
            for (size_t col = 0; col < 4; ++col) { input[row * 4 + col] = float(col) - 1.5f + float(row % 8) * 0.125f; }
        }
        upload(a, input);
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        const auto output = download(y);
        double maximum_error = 0;
        for (size_t i = 0; i < output.size(); ++i) {
            const double expected = (double(i % 4) - 1.5) / std::sqrt(1.25 + 1.0e-6);
            if (!std::isfinite(output[i])) { maximum_error = std::numeric_limits<double>::infinity(); break; }
            maximum_error = std::max(maximum_error, std::fabs(output[i] - expected));
        }
        CHECK(maximum_error < 0.004);
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}

static void m2_silu_tests(ggml_backend_t backend) {
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        auto * c = context();
        auto * x = ggml_new_tensor_4d(c, type, 32, 3, 2, 2);
        auto * y = ggml_silu(c, x);
        auto * inplace = ggml_silu_inplace(c, x);
        CHECK(primitive_supported(y)); CHECK(primitive_supported(inplace));
        CHECK(!primitive_supported(ggml_sigmoid(c, x)));
        const auto stride = x->nb[1]; x->nb[1] += ggml_type_size(type);
        CHECK(!primitive_supported(y)); x->nb[1] = stride;
        y->src[1] = x; CHECK(!primitive_supported(y)); y->src[1] = nullptr;
        if (backend && primitive_supported(y)) {
            CHECK(ggml_backend_dev_supports_op(ggml_backend_get_device(backend), y));
            auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, y);
            auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
            if (buffer) {
                std::vector<float> input(384), reference(384);
                auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                auto stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
                ggml_backend_qairt_stats before {}, after {};
                for (int pass = 0; pass < 2; ++pass) {
                    for (size_t i = 0; i < input.size(); ++i) {
                        float value = float(int(i % 65) - 32) * (pass ? 0.125f : 0.5f);
                        if (type == GGML_TYPE_F16) { value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)); }
                        input[i] = value; reference[i] = float(double(value) / (1.0 + std::exp(-double(value))));
                    }
                    upload(x, input);
                    CHECK(stats && stats(backend, &before));
                    CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                    expect(y, reference);
                    CHECK(stats && stats(backend, &after));
                    CHECK(after.executions == before.executions + 1);
                    CHECK(after.preparations == before.preparations + (pass == 0 ? 1 : 0));
                    if (pass) { CHECK(after.cache_hits == before.cache_hits + 1); }
                }
                auto * inplace_graph = ggml_new_graph(c); ggml_build_forward_expand(inplace_graph, inplace);
                CHECK(ggml_backend_graph_compute(backend, inplace_graph) == GGML_STATUS_SUCCESS);
                expect(inplace, reference);
                const auto saved = download(y);
                input[0] = std::numeric_limits<float>::quiet_NaN(); upload(x, input);
                CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                CHECK(download(y) == saved);
                ggml_backend_buffer_free(buffer);
            }
        }
        ggml_free(c);
    }
}
static void m2_residual_tests(ggml_backend_t backend) {
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        auto * c = context();
        auto * branch = ggml_new_tensor_4d(c, type, 32, 3, 2, 2);
        auto * skip = ggml_new_tensor_4d(c, type, 32, 1, 1, 2);
        auto * y = ggml_add(c, branch, skip);
        auto * main = ggml_silu(c, branch);
        auto * joined = ggml_add(c, main, skip);
        CHECK(primitive_supported(y));
        const auto stride = skip->nb[3]; skip->nb[3] += ggml_type_size(type);
        CHECK(!primitive_supported(y)); skip->nb[3] = stride;
        y->src[2] = branch; CHECK(!primitive_supported(y)); y->src[2] = nullptr;
        if (backend && primitive_supported(y) && ggml_backend_dev_supports_op(ggml_backend_get_device(backend), y)) {
            auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, y);
            auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
            if (buffer) {
                std::vector<float> a(384), b(64), reference(384);
                for (size_t i = 0; i < a.size(); ++i) { a[i] = float(int(i % 33) - 16) * 0.125f; }
                for (size_t i = 0; i < b.size(); ++i) { b[i] = float(int(i % 17) - 8) * 0.25f; }
                for (size_t i = 0; i < a.size(); ++i) { reference[i] = a[i] + b[(i / 192) * 32 + i % 32]; }
                upload(branch, a); upload(skip, b);
                CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                expect(y, reference, 0.0f);
                auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                auto stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
                ggml_backend_qairt_stats before {}, after {}; CHECK(stats && stats(backend, &before));
                for (size_t i = 0; i < b.size(); ++i) { b[i] += 1.0f; }
                for (auto & value : reference) { value += 1.0f; }
                upload(skip, b);
                CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                expect(y, reference, 0.0f);
                CHECK(stats && stats(backend, &after));
                CHECK(after.preparations == before.preparations && after.cache_hits == before.cache_hits + 1);
                auto * joined_graph = ggml_new_graph(c); ggml_build_forward_expand(joined_graph, joined);
                CHECK(ggml_backend_graph_compute(backend, joined_graph) == GGML_STATUS_SUCCESS);
                std::vector<float> joined_reference(a.size());
                for (size_t i = 0; i < a.size(); ++i) {
                    joined_reference[i] = float(double(a[i]) / (1.0 + std::exp(-double(a[i])))) + b[(i / 192) * 32 + i % 32];
                }
                expect(joined, joined_reference);
                const auto saved_main = download(main), saved_joined = download(joined);
                b[0] = std::numeric_limits<float>::quiet_NaN(); upload(skip, b);
                CHECK(ggml_backend_graph_compute(backend, joined_graph) == GGML_STATUS_FAILED);
                CHECK(download(main) == saved_main && download(joined) == saved_joined);
                b[0] = 0.0f; upload(skip, b);
                const auto saved = download(y);
                a[0] = std::numeric_limits<float>::quiet_NaN(); upload(branch, a);
                CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                CHECK(download(y) == saved);
                ggml_backend_buffer_free(buffer);
            }
        }
        if (backend && !ggml_backend_dev_supports_op(ggml_backend_get_device(backend), y)) {
            auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, y);
            auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
            if (buffer) {
                upload(y, std::vector<float>(384, 42.0f));
                CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                CHECK(download(y) == std::vector<float>(384, 42.0f));
                ggml_backend_buffer_free(buffer);
            }
        }
        ggml_free(c);
    }
}
static void m2_rope_tests(ggml_backend_t backend) {
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int mode : {GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX}) {
            for (int dimensions : {32, 64}) {
                auto * c = context();
                auto * x = ggml_new_tensor_4d(c, type, 64, 3, 4, 2);
                auto * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4);
                auto * factors = ggml_new_tensor_1d(c, GGML_TYPE_F32, dimensions / 2);
                auto * y = ggml_rope_ext(c, x, ids, factors, dimensions, mode, 128,
                                        10000.0f, 0.5f, 1.0f, 1.1f, 32.0f, 1.0f);
                ggml_rope_set_truncate(y, false);
                auto * inplace = ggml_rope_ext_inplace(c, x, ids, factors, dimensions, mode, 128,
                                                     10000.0f, 0.5f, 1.0f, 1.1f, 32.0f, 1.0f);
                ggml_rope_set_truncate(inplace, false);
                CHECK(primitive_supported(y)); CHECK(primitive_supported(inplace));
                const auto stride = ids->nb[0]; ids->nb[0] = 8;
                CHECK(!primitive_supported(y)); ids->nb[0] = stride;
                const auto saved_type = ids->type; ids->type = GGML_TYPE_F32;
                CHECK(!primitive_supported(y)); ids->type = saved_type;
                ggml_rope_set_offset(y, 2); CHECK(!primitive_supported(y)); ggml_rope_set_offset(y, 0);
                int bad_mode = GGML_ROPE_TYPE_MROPE;
                std::memcpy(reinterpret_cast<uint8_t *>(y->op_params) + 8, &bad_mode, 4);
                CHECK(!primitive_supported(y)); std::memcpy(reinterpret_cast<uint8_t *>(y->op_params) + 8, &mode, 4);
                auto * activated = ggml_silu(c, x);
                auto * chained = ggml_rope_ext(c, activated, ids, nullptr, dimensions, mode, 0,
                                              10000.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
                if (backend && primitive_supported(y) && ggml_backend_dev_supports_op(ggml_backend_get_device(backend), y)) {
                    CHECK(ggml_backend_dev_supports_op(ggml_backend_get_device(backend), y));
                    auto * cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                    auto * cpu = cpu_device ? ggml_backend_dev_init(cpu_device, nullptr) : nullptr; CHECK(cpu);
                    auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, y);
                    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
                    if (buffer && cpu) {
                        std::vector<float> input(1536), ff(dimensions / 2, 1.0f);
                        for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 71) - 35) * 0.0625f + 0.001f; }
                        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                        auto stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
                        for (int pass = 0; pass < 4; ++pass) {
                            std::vector<int32_t> positions = pass == 0 ? std::vector<int32_t>{7, 0, 7, 2} :
                                                            pass == 1 ? std::vector<int32_t>{19, 3, 11, 19} :
                                                                        std::vector<int32_t>{2049, 127, 1024, 0};
                            if (pass == 3) { ff[0] = 1.5f; }
                            upload(x, input); upload(factors, ff);
                            ggml_backend_tensor_set(ids, positions.data(), 0, positions.size() * sizeof(int32_t));
                            CHECK(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
                            const auto reference = download(y);
                            upload(y, std::vector<float>(input.size(), 42.0f));
                            ggml_backend_qairt_stats before {}, after {};
                            CHECK(stats && stats(backend, &before));
                            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
                            expect(y, reference);
                            if (type == GGML_TYPE_F32 && dimensions < 64) {
                                const auto actual = download(y);
                                for (size_t i = 0; i < actual.size(); ++i) {
                                    if (i % 64 >= size_t(dimensions)) { CHECK(actual[i] == input[i]); }
                                }
                            }
                            CHECK(stats && stats(backend, &after));
                            CHECK(after.executions == before.executions + 1);
                            CHECK(after.preparations == before.preparations + ((pass == 0 || pass == 3) ? 1 : 0));
                            if (pass == 1 || pass == 2) { CHECK(after.cache_hits == before.cache_hits + 1); }
                        }
                        auto * inplace_graph = ggml_new_graph(c); ggml_build_forward_expand(inplace_graph, inplace);
                        upload(x, input); CHECK(ggml_backend_graph_compute(cpu, inplace_graph) == GGML_STATUS_SUCCESS);
                        const auto inplace_reference = download(inplace);
                        upload(x, input); CHECK(ggml_backend_graph_compute(backend, inplace_graph) == GGML_STATUS_SUCCESS);
                        expect(inplace, inplace_reference);
                        auto * chain_graph = ggml_new_graph(c); ggml_build_forward_expand(chain_graph, chained);
                        upload(x, input); CHECK(ggml_backend_graph_compute(cpu, chain_graph) == GGML_STATUS_SUCCESS);
                        const auto chain_reference = download(chained);
                        upload(x, input); CHECK(ggml_backend_graph_compute(backend, chain_graph) == GGML_STATUS_SUCCESS);
                        expect(chained, chain_reference);
                        const auto saved_activated = download(activated), saved_chained = download(chained);
                        std::vector<int32_t> bad_positions {-1, 0, 1, 2};
                        ggml_backend_tensor_set(ids, bad_positions.data(), 0, bad_positions.size() * sizeof(int32_t));
                        CHECK(ggml_backend_graph_compute(backend, chain_graph) == GGML_STATUS_FAILED);
                        CHECK(download(activated) == saved_activated && download(chained) == saved_chained);
                        const auto saved = download(y);
                        for (int32_t invalid : {-1, INT32_MAX}) {
                            std::vector<int32_t> positions {invalid, 0, 1, 2};
                            ggml_backend_tensor_set(ids, positions.data(), 0, positions.size() * sizeof(int32_t));
                            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                            CHECK(download(y) == saved);
                        }
                        std::vector<int32_t> positions {0, 1, 2, 3};
                        ggml_backend_tensor_set(ids, positions.data(), 0, positions.size() * sizeof(int32_t));
                        ff[0] = 0.0f; upload(factors, ff);
                        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                        CHECK(download(y) == saved);
                        ff[0] = 1.0f; upload(factors, ff);
                        input[0] = std::numeric_limits<float>::quiet_NaN(); upload(x, input);
                        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                        CHECK(download(y) == saved);
                    }
                    if (buffer) { ggml_backend_buffer_free(buffer); }
                    if (cpu) { ggml_backend_free(cpu); }
                }
                if (backend && !ggml_backend_dev_supports_op(ggml_backend_get_device(backend), y)) {
                    auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, y);
                    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
                    if (buffer) {
                        upload(y, std::vector<float>(1536, 42.0f));
                        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_FAILED);
                        CHECK(download(y) == std::vector<float>(1536, 42.0f));
                        ggml_backend_buffer_free(buffer);
                    }
                }
                ggml_free(c);
            }
        }
    }
}
int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--bf16-host") == 0) { m2_bf16_tests(nullptr); }
    else if (argc == 2 && std::strcmp(argv[1], "--bf16") == 0) {
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("QAIRT0");
        auto backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr; CHECK(backend);
        if (backend) { m2_bf16_tests(backend); ggml_backend_free(backend); }
    }
    else if (argc == 2 && (std::strcmp(argv[1], "--direct-fp16") == 0 || std::strcmp(argv[1], "--default-fp16") == 0)) {
        CHECK(_putenv_s("GGML_QAIRT_DIRECT_FP16", std::strcmp(argv[1], "--default-fp16") == 0 ? "" : "1") == 0);
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("QAIRT0");
        auto backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr; CHECK(backend);
        if (backend) { direct_fp16_tests(backend); ggml_backend_free(backend); }
    }
    else if (argc == 2 && std::strcmp(argv[1], "--masked-softmax") == 0) {
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("QAIRT0");
        auto backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr; CHECK(backend);
        if (backend) { m2_masked_softmax_tests(backend); ggml_backend_free(backend); }
    }
    else if (argc == 2 && std::strcmp(argv[1], "--rope") == 0) {
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("QAIRT0");
        auto backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr; CHECK(backend);
        if (backend) { m2_rope_tests(backend); ggml_backend_free(backend); }
    }
    else if (argc == 2 && std::strcmp(argv[1], "--residual") == 0) {
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("QAIRT0");
        auto backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr; CHECK(backend);
        if (backend) { m2_residual_tests(backend); ggml_backend_free(backend); }
    }
    else if (argc == 2 && std::strcmp(argv[1], "--silu") == 0) {
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("QAIRT0");
        auto backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr; CHECK(backend);
        if (backend) { m2_silu_tests(backend); ggml_backend_free(backend); }
    }
    else if (argc == 2 && std::strcmp(argv[1], "--rows") == 0) {
        ggml_backend_load_all(); auto * dev = ggml_backend_dev_by_name("QAIRT0");
        auto backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr; CHECK(backend);
        if (backend) { m2_row_tests(backend); m2_large_norm_test(backend); ggml_backend_free(backend); }
    }
    else if (argc == 2 && std::strcmp(argv[1], "--m2-host") == 0) { m2_bf16_tests(nullptr); m2_quantized_tests(nullptr); m2_metadata_tests(); m2_row_tests(nullptr); m2_masked_softmax_tests(nullptr); m2_silu_tests(nullptr); m2_residual_tests(nullptr); m2_rope_tests(nullptr); }
    else if (argc == 2 && std::strcmp(argv[1], "--host") == 0) { host_tests(); }
    else {
        ggml_backend_load_all();
        auto dev = ggml_backend_dev_by_name("QAIRT0");
        if (argc == 2 && std::strcmp(argv[1], "--unavailable") == 0) { CHECK(dev == nullptr); }
        else {
            CHECK(dev != nullptr);
            auto backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
            CHECK(backend != nullptr);
            if (backend) {
                m2_bf16_tests(backend); m2_quantized_tests(backend); m2_metadata_tests(); m2_row_tests(backend); m2_silu_tests(backend); m2_residual_tests(backend); m2_rope_tests(backend); m2_run_tests(backend); m2_connected_tests(backend); m2_cpu_weight_placement(backend);
                buffer_tests(backend); transposed_matmul_test(backend);
                primitive_tests(backend, GGML_TYPE_F16);
                matmul_tests(backend, GGML_TYPE_F16); matmul_tests(backend, GGML_TYPE_F32);
                matmul_tests(backend, GGML_TYPE_F16, GGML_TYPE_F16);
                placement_test(backend);
                ggml_backend_free(backend);
            }
        }
    }
    std::printf("QAIRT M1/M2: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
