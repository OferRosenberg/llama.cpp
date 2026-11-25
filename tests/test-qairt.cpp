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

static void m2_host_tests() {
    ggml_backend_load_all();
    auto * c = context();
    auto * weights = ggml_new_tensor_2d(c, GGML_TYPE_Q4_0, 2048, 1024);
    auto * input = ggml_new_tensor_2d(c, GGML_TYPE_F32, 2048, 13);
    auto * output = ggml_mul_mat(c, weights, input);
    CHECK(primitive_supported(output));
    CHECK(ggml_prec_set_acc(output, GGML_PREC_F32));
    CHECK(!primitive_supported(output));
    CHECK(ggml_prec_set_acc(output, GGML_PREC_DEFAULT));
    CHECK(ggml_prec_set_src(output, GGML_PREC_F32, 1));
    CHECK(!primitive_supported(output));
    ggml_free(c);
}

static void host_tests() {
    auto * c = context();
    auto * w = ggml_new_tensor_2d(c, GGML_TYPE_F16, 4, 3);
    auto * a = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 2);
    auto * y = ggml_mul_mat(c, w, a);
    CHECK(primitive_supported(y));
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    CHECK(primitive_supported(y));
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
    ggml_set_output(y);
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
            CHECK(after.preparations - before.preparations == 1);
            CHECK(after.cache_hits - before.cache_hits == 2);
            CHECK(after.executions - before.executions == 3);
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
static void partition_dispatch_test(ggml_backend_t backend) {
    auto get_stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_qairt_get_stats"));
    CHECK(get_stats != nullptr);
    if (!get_stats) { return; }
    for (bool inactive_branch : {false, true}) {
        auto * c = context();
        auto * input = ggml_new_tensor_1d(c, GGML_TYPE_F16, 32);
        auto * increment = ggml_new_tensor_1d(c, GGML_TYPE_F16, 32);
        auto * poison = ggml_new_tensor_1d(c, GGML_TYPE_F16, 32);
        auto * output = ggml_add(c, input, increment);
        auto * graph = ggml_new_graph(c);
        if (inactive_branch) {
            output = ggml_add(c, output, increment);
            ggml_tensor * branches[] = {output, ggml_dup(c, poison)};
            CHECK(ggml_build_forward_select(graph, branches, 2, 0) == output);
        } else {
            ggml_build_forward_expand(graph, output);
        }
        auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
        CHECK(buffer != nullptr);
        if (buffer) {
            upload(input, std::vector<float>(32, 0.25f));
            upload(increment, std::vector<float>(32, 0.125f));
            upload(poison, std::vector<float>(32, std::numeric_limits<float>::quiet_NaN()));
            ggml_backend_qairt_stats before {}, after {};
            CHECK(get_stats(backend, &before));
            const auto status = ggml_backend_graph_compute(backend, graph);
            CHECK(status == GGML_STATUS_SUCCESS);
            CHECK(get_stats(backend, &after));
            const uint64_t qnn_runs = inactive_branch ? 1 : 0;
            CHECK(after.preparations == before.preparations + qnn_runs);
            CHECK(after.executions == before.executions + qnn_runs);
            if (status == GGML_STATUS_SUCCESS) {
                expect(output, std::vector<float>(32, inactive_branch ? 0.5f : 0.375f), 0);
            }
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(c);
    }
}

static void runtime_path_test(ggml_backend_t backend) {
    CHECK(!ggml_backend_supports_buft(backend, ggml_backend_cpu_buffer_type()));
    auto get_stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_qairt_get_stats"));
    CHECK(get_stats != nullptr);
    if (!get_stats) { return; }
    for (int length : {129, 160}) {
        auto * c = context();
        auto * input = ggml_new_tensor_1d(c, GGML_TYPE_F16, 32);
        auto * increment = ggml_new_tensor_1d(c, GGML_TYPE_F16, 32);
        auto * output = input;
        for (int i = 0; i < length; ++i) { output = ggml_add(c, output, increment); }
        auto * graph = ggml_new_graph_custom(c, 1024, false);
        ggml_build_forward_expand(graph, output);
        auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
        CHECK(buffer != nullptr);
        if (buffer) {
            upload(input, std::vector<float>(32, 0.25f));
            upload(increment, std::vector<float>(32, 0.125f));
            ggml_backend_qairt_stats before {}, after {};
            CHECK(get_stats(backend, &before));
            const auto status = ggml_backend_graph_compute(backend, graph);
            CHECK(status == GGML_STATUS_SUCCESS);
            CHECK(get_stats(backend, &after));
            // The second chain reuses the first 128-Add graph.
            CHECK(after.preparations == before.preparations + 1);
            CHECK(after.executions == before.executions + (length == 129 ? 1 : 2));
            CHECK(after.exported_tensors - before.exported_tensors == (length == 129 ? 1 : 2));
            CHECK(after.materialized_outputs - before.materialized_outputs == 1);
            CHECK(after.fp16_handoffs - before.fp16_handoffs == (length == 129 ? 0 : 1));
            if (status == GGML_STATUS_SUCCESS) { expect(output, std::vector<float>(32, 0.25f + length * 0.125f), 0); }
            ggml_backend_buffer_free(buffer);
        }
        ggml_free(c);
    }
}

static void boundary_alias_test(ggml_backend_t backend) {
    auto get_stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_qairt_get_stats"));
    CHECK(get_stats != nullptr); if (!get_stats) { return; }
    auto * c = context();
    auto * input = ggml_new_tensor_2d(c, GGML_TYPE_F32, 4, 8);
    auto * increment = ggml_new_tensor_1d(c, GGML_TYPE_F32, 1);
    auto * output = input;
    ggml_tensor * observed = nullptr;
    for (int i = 0; i < 128; ++i) {
        output = ggml_add(c, output, increment);
        if (i == 63) { observed = output; ggml_set_output(observed); }
    }
    output = ggml_cont(c, ggml_transpose(c, output));
    for (int i = 0; i < 32; ++i) { output = ggml_add(c, output, increment); }
    auto * graph = ggml_new_graph_custom(c, 1024, false);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer != nullptr);
    if (buffer) {
        upload(increment, {0.125f});
        for (float start : {0.25f, 1.25f}) {
            std::vector<float> values(32), expected(32);
            for (size_t i = 0; i < values.size(); ++i) { values[i] = start + i / 16.0f; }
            upload(input, values);
            ggml_backend_qairt_stats before {}, after {};
            CHECK(get_stats(backend, &before));
            CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            CHECK(get_stats(backend, &after));
            CHECK(after.exported_tensors - before.exported_tensors == 3);
            CHECK(after.materialized_outputs - before.materialized_outputs == 2);
            for (size_t i = 0; i < values.size(); ++i) { expected[i] = values[i] + 8.0f; }
            expect(observed, expected, 0);
            for (size_t row = 0; row < 4; ++row) {
                for (size_t col = 0; col < 8; ++col) { expected[row * 8 + col] = values[col * 4 + row] + 20.0f; }
            }
            expect(output, expected, 0);
        }
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}
static void boundary_observation_test(ggml_backend_t backend) {
    auto get_stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_qairt_get_stats"));
    CHECK(get_stats != nullptr); if (!get_stats) { return; }
    auto * c = context();
    auto * input = ggml_new_tensor_1d(c, GGML_TYPE_F32, 32);
    auto * increment = ggml_new_tensor_1d(c, GGML_TYPE_F32, 32);
    auto * observed = ggml_add(c, input, increment);
    auto * output = ggml_add(c, ggml_add(c, observed, increment), increment);
    auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer != nullptr);
    if (buffer) {
        upload(input, std::vector<float>(32, 0.25f)); upload(increment, std::vector<float>(32, 0.125f));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        expect(output, std::vector<float>(32, 0.625f), 0);
        ggml_backend_qairt_stats before {}, after {};
        CHECK(get_stats(backend, &before));
        ggml_set_output(observed);
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        CHECK(get_stats(backend, &after));
        CHECK(after.preparations == before.preparations + 1);
        CHECK(after.exported_tensors == before.exported_tensors + 2);
        expect(observed, std::vector<float>(32, 0.375f), 0);
        expect(output, std::vector<float>(32, 0.625f), 0);
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}
static void boundary_reuse_test(ggml_backend_t backend) {
    auto * c = context();
    auto * input = ggml_new_tensor_1d(c, GGML_TYPE_F16, 32);
    auto * increment = ggml_new_tensor_1d(c, GGML_TYPE_F16, 32);
    auto * output = input;
    ggml_tensor * skip = nullptr;
    for (int i = 0; i < 544; ++i) {
        output = ggml_add(c, output, increment);
        if (i == 127) { skip = output; }
    }
    output = ggml_add(c, output, skip);
    auto * graph = ggml_new_graph_custom(c, 1024, false); ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer != nullptr);
    if (buffer) {
        upload(input, std::vector<float>(32, 0.25f));
        upload(increment, std::vector<float>(32, 0.125f));
        CHECK(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
        expect(output, std::vector<float>(32, 84.5f), 0);
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}
static void boundary_scheduler_test(ggml_backend_t backend) {
    auto cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(cpu != nullptr); if (!cpu) { return; }
    auto * c = context();
    auto * input = ggml_new_tensor_1d(c, GGML_TYPE_F32, 32);
    auto * increment = ggml_new_tensor_1d(c, GGML_TYPE_F32, 32);
    auto * hybrid = ggml_add(c, ggml_add(c, input, increment), increment);
    auto * output = ggml_sqr(c, hybrid);
    auto * graph = ggml_new_graph(c); ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = {backend, cpu};
    auto scheduler = ggml_backend_sched_new(backends, nullptr, 2, 1024, false, true);
    CHECK(scheduler != nullptr);
    if (scheduler) {
        ggml_backend_sched_set_tensor_backend(scheduler, hybrid, backend);
        CHECK(ggml_backend_sched_alloc_graph(scheduler, graph));
        upload(input, std::vector<float>(32, 0.25f));
        upload(increment, std::vector<float>(32, 0.125f));
        CHECK(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
        expect(output, std::vector<float>(32, 0.25f), 0);
        ggml_backend_sched_free(scheduler);
    }
    ggml_backend_free(cpu); ggml_free(c);
}

static void grouped_runtime_test(ggml_backend_t backend) {
    auto * c = context();
    auto * weights = ggml_new_tensor_3d(c, GGML_TYPE_F16, 2, 2, 8);
    auto * input = ggml_new_tensor_3d(c, GGML_TYPE_F32, 2, 2, 16);
    auto * output = input;
    for (int i = 0; i < 32; ++i) { output = ggml_mul_mat(c, weights, output); }
    auto * graph = ggml_new_graph_custom(c, 1024, false);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(c, backend);
    CHECK(buffer != nullptr);
    if (buffer) {
        std::vector<float> identity(32, 0.0f);
        for (size_t i = 0; i < identity.size(); i += 4) { identity[i] = identity[i + 3] = 1.0f; }
        upload(weights, identity);
        upload(input, std::vector<float>(64, 0.5f));
        const auto status = ggml_backend_graph_compute(backend, graph);
        CHECK(status == GGML_STATUS_SUCCESS);
        if (status == GGML_STATUS_SUCCESS) { expect(output, std::vector<float>(64, 0.5f), 0); }
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}
int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--m2-host") == 0) { m2_host_tests(); }
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
                if (argc == 2 && std::strcmp(argv[1], "--runtime-path") == 0) {
                    partition_dispatch_test(backend); runtime_path_test(backend); boundary_alias_test(backend); boundary_observation_test(backend); boundary_reuse_test(backend); boundary_scheduler_test(backend); grouped_runtime_test(backend);
                    ggml_backend_free(backend);
                    std::printf("QAIRT native runtime: %d checks, %d failures\n", checks, failures);
                    return failures ? 1 : 0;
                }
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
