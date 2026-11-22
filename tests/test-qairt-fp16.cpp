#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-qairt.h"
#include "qairt-weights.h"
#include "qairt-target.h"
#include <cstdlib>

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
    if (level == GGML_LOG_LEVEL_ERROR || std::strstr(text, "qairt: inventory")) { std::fputs(text, stderr); }
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
static void w4_packing_tests() {
    std::vector<uint8_t> raw(36);
    for (size_t row = 0; row < 2; ++row) {
        const auto d = ggml_fp32_to_fp16(row ? -0.5f : 0.25f);
        std::memcpy(raw.data()+row*18,&d,2);
        for (size_t j = 0; j < 16; ++j) { raw[row*18+2+j] = uint8_t(j | ((15-j)<<4)); }
    }
    const auto original = raw;
    auto packed = prepare_w4_weights({raw.data(),raw.size()},32,2);
    auto reference = prepare_q4_weights({raw.data(),raw.size()},32,2);
    CHECK(packed.ok() && reference.ok());
    if (packed.ok() && reference.ok()) {
        CHECK(packed.value().codes.size() == 64 && packed.value().bytes() == 80);
        CHECK(packed.value().quantization->scales[0].scale == 0.25f);
        CHECK(packed.value().quantization->scales[1].scale == 0.5f);
        CHECK(packed.value().quantization->scales[1].offset == 1.0f);
        for (size_t i = 0; i < 64; ++i) {
            const auto scale = packed.value().quantization->scales[i/32];
            CHECK(ggml_fp16_to_fp32(ggml_fp32_to_fp16((packed.value().codes[i]+scale.offset)*scale.scale)) == ggml_fp16_to_fp32(reference.value()[i].bits));
        }
        tensor_spec spec {"w",QNN_TENSOR_TYPE_STATIC,QNN_DATATYPE_SFIXED_POINT_8,{1,1,2,32},packed.value().quantization};
        CHECK(tensor_record::checked_bytes(spec).ok() && tensor_record::checked_bytes(spec).value() == 64);
        spec.role = QNN_TENSOR_TYPE_APP_WRITE; CHECK(!tensor_record::checked_bytes(spec).ok());
        spec.role = QNN_TENSOR_TYPE_STATIC; spec.dimensions[2] = 3; CHECK(!tensor_record::checked_bytes(spec).ok());
    }
    CHECK(!prepare_w4_weights({raw.data(),raw.size()-1},32,2).ok());
    CHECK(!prepare_w4_weights({raw.data(),raw.size()},SIZE_MAX,2).ok());
    w4_weight_upload upload;
    CHECK(upload.update({raw.data(),raw.size()},32,2,18,18).ok());
    CHECK(!upload.get(32,2) && upload.preparations() == 0);
    CHECK(upload.update({raw.data(),raw.size()},32,2,0,18).ok());
    auto cached = upload.get(32,2);
    CHECK(cached && upload.preparations() == 1 && upload.get(32,2) == cached);
    const auto d = ggml_fp32_to_fp16(0.5f); std::memcpy(raw.data(),&d,2);
    CHECK(upload.update({raw.data(),raw.size()},32,2,0,2).ok());
    CHECK(upload.get(32,2) != cached && upload.preparations() == 2);
    CHECK(cached && cached->quantization->scales[0].scale == 0.25f);
    const auto nan = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN()); std::memcpy(raw.data(),&nan,2);
    CHECK(!upload.update({raw.data(),raw.size()},32,2,0,2).ok() && !upload.get(32,2));
    CHECK(original[0] != raw[0] || original[1] != raw[1]);
}

static void q41_packing_tests() {
    std::vector<uint8_t> raw(20 * 4);
    for (size_t b = 0; b < 4; ++b) {
        const auto d = ggml_fp32_to_fp16(b == 2 ? 0.0f : b == 3 ? -0.25f : 0.25f);
        const auto m = ggml_fp32_to_fp16(b == 2 ? 1.125f : -1.9375f);
        std::memcpy(raw.data() + 20 * b, &d, 2); std::memcpy(raw.data() + 20 * b + 2, &m, 2);
        for (size_t j = 0; j < 16; ++j) { raw[20 * b + 4 + j] = uint8_t(j | ((15 - j) << 4)); }
    }
    const auto original = raw;
    auto packed = prepare_w4_weights({raw.data(),raw.size()},32,4,GGML_TYPE_Q4_1);
    auto ref = prepare_q4_weights({raw.data(),raw.size()},32,4,GGML_TYPE_Q4_1);
    CHECK(packed.ok() && ref.ok());
    if (packed.ok() && ref.ok()) {
        CHECK(packed.value().quantization->scales[0].offset == 0.25f);
        for (size_t i = 0; i < 128; ++i) {
            const auto q = packed.value().quantization->scales[i / 32];
            CHECK(ggml_fp32_to_fp16((packed.value().codes[i] + q.offset) * q.scale) == ref.value()[i].bits);
        }
    }
    w4_weight_upload upload;
    CHECK(upload.update({raw.data(),raw.size()},32,4,40,40,GGML_TYPE_Q4_1).ok());
    CHECK(!upload.get(32,4,GGML_TYPE_Q4_1));
    CHECK(upload.update({raw.data(),raw.size()},32,4,0,40,GGML_TYPE_Q4_1).ok());
    CHECK(upload.get(32,4,GGML_TYPE_Q4_1) && !upload.get(32,4,GGML_TYPE_Q4_0));
    CHECK(raw == original);
    CHECK(!prepare_w4_weights({raw.data(),raw.size()-1},32,4,GGML_TYPE_Q4_1).ok());
    CHECK(!prepare_w4_weights({raw.data(),raw.size()},32,4,GGML_TYPE_Q8_0).ok());
    const auto nan = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
    std::memcpy(raw.data()+2,&nan,2);
    CHECK(!prepare_w4_weights({raw.data(),raw.size()},32,4,GGML_TYPE_Q4_1).ok());
    const auto huge = ggml_fp32_to_fp16(65504.0f); std::memcpy(raw.data()+2,&huge,2);
    CHECK(!prepare_w4_weights({raw.data(),raw.size()},32,4,GGML_TYPE_Q4_1).ok());
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
static void w4_backend_test(ggml_backend_dev_t device, ggml_type type = GGML_TYPE_Q4_0) {
    CHECK(std::strstr(ggml_backend_dev_description(device),"W4FP16") != nullptr);
    backend_ptr backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    CHECK(backend); if (!backend) { return; }
    auto wc = context(), c = context();
    auto * w = ggml_new_tensor_2d(wc.get(),type,32,16);
    buffer_ptr wb(ggml_backend_alloc_ctx_tensors(wc.get(),backend.get()),ggml_backend_buffer_free);
    CHECK(wb); if (!wb) { return; }
    // Host buffers bypass tensor upload callbacks in the non-mmap model loader.
    CHECK(!ggml_backend_buffer_is_host(wb.get()));
    CHECK(ggml_backend_supports_buft(backend.get(),ggml_backend_dev_buffer_type(device)));
    ggml_backend_buffer_set_usage(wb.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> raw(ggml_nbytes(w));
    const size_t block_bytes = ggml_type_size(type);
    for (size_t row = 0; row < 16; ++row) {
        const auto d = ggml_fp32_to_fp16(row%2 ? -0.25f : 0.25f); std::memcpy(raw.data()+row*block_bytes,&d,2);
        if (type == GGML_TYPE_Q4_1) {
            const auto m = ggml_fp32_to_fp16(row == 2 ? 1.125f : -1.9375f);
            std::memcpy(raw.data()+row*block_bytes+2,&m,2);
            if (row == 2) { const auto zero = ggml_fp32_to_fp16(0); std::memcpy(raw.data()+row*block_bytes,&zero,2); }
        }
        for (size_t j = 0; j < 16; ++j) { raw[row*block_bytes+(type == GGML_TYPE_Q4_1 ? 4 : 2)+j] = uint8_t(j | ((15-j)<<4)); }
    }
    ggml_backend_tensor_set(w,raw.data()+raw.size()/2,raw.size()/2,raw.size()/2);
    ggml_backend_tensor_set(w,raw.data(),0,raw.size()/2);
    auto * x = ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,32,32);
    auto * y = ggml_mul_mat(c.get(),w,x);
    auto * graph = ggml_new_graph(c.get()); ggml_graph_add_node(graph,y);
    auto * x2 = ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,32,1);
    auto * y2 = ggml_mul_mat(c.get(),w,x2);
    auto * graph2 = ggml_new_graph(c.get()); ggml_graph_add_node(graph2,y2);
    buffer_ptr b(ggml_backend_alloc_ctx_tensors(c.get(),backend.get()),ggml_backend_buffer_free);
    CHECK(b); if (!b) { return; }
    std::vector<float> identity(32*32);
    for (size_t i = 0; i < 32; ++i) { identity[i*32+i] = 1.0f; }
    ggml_backend_tensor_set(x,identity.data(),0,identity.size()*4);
    const auto fn = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device),"ggml_backend_qairt_get_stats"));
    CHECK(fn); if (!fn) { return; }
    for (int run = 0; run < 2; ++run) {
        CHECK(ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS);
        std::vector<float> actual(16*32), expected(actual.size()); ggml_backend_tensor_get(y,actual.data(),0,actual.size()*4);
        auto reference = prepare_q4_weights({raw.data(),raw.size()},32,16,type); CHECK(reference.ok());
        if (reference.ok()) { for (size_t row = 0; row < 16; ++row) { for (size_t j = 0; j < 32; ++j) { expected[j*16+row] = ggml_fp16_to_fp32(reference.value()[row*32+j].bits); } } accuracy(actual,expected,"W4FP16 identity"); }
    }
    ggml_backend_qairt_stats stats {}; CHECK(fn(backend.get(),&stats));
    CHECK(stats.w4_preparations == 1 && stats.w4_executions == 2 && stats.cache_hits == 1);
    std::vector<float> selected(32); selected[0] = 1;
    ggml_backend_tensor_set(x2,selected.data(),0,selected.size()*4);
    CHECK(ggml_backend_graph_compute(backend.get(),graph2) == GGML_STATUS_SUCCESS);
    std::vector<float> second(16); ggml_backend_tensor_get(y2,second.data(),0,second.size()*4);
    auto reference2 = prepare_q4_weights({raw.data(),raw.size()},32,16,type); CHECK(reference2.ok());
    std::vector<float> expected2(16);
    if (reference2.ok()) { for (size_t row = 0; row < 16; ++row) { expected2[row] = ggml_fp16_to_fp32(reference2.value()[row*32].bits); } accuracy(second,expected2,"W4FP16 second profile"); }
    CHECK(fn(backend.get(),&stats) && stats.w4_preparations == 2 && stats.w4_executions == 3);
    const auto d = ggml_fp32_to_fp16(0.5f); std::memcpy(raw.data(),&d,2);
    ggml_backend_tensor_set(w,raw.data(),0,2);
    CHECK(ggml_backend_graph_compute(backend.get(),graph) == GGML_STATUS_SUCCESS);
    CHECK(fn(backend.get(),&stats) && stats.w4_preparations == 3 && stats.w4_executions == 4);
    std::vector<float> updated(16*32), expected_updated(updated.size()); ggml_backend_tensor_get(y,updated.data(),0,updated.size()*4);
    auto reference_updated = prepare_q4_weights({raw.data(),raw.size()},32,16,type); CHECK(reference_updated.ok());
    if (reference_updated.ok()) { for (size_t row = 0; row < 16; ++row) { for (size_t j = 0; j < 32; ++j) { expected_updated[j*16+row] = ggml_fp16_to_fp32(reference_updated.value()[row*32+j].bits); } } accuracy(updated,expected_updated,"W4FP16 mutated weight"); }
    std::vector<uint8_t> unchanged(raw.size()); ggml_backend_tensor_get(w,unchanged.data(),0,unchanged.size()); CHECK(raw == unchanged);
    std::printf("W4FP16 native preparations=%llu executions=%llu\n",(unsigned long long)stats.w4_preparations,(unsigned long long)stats.w4_executions);
}

static void weight_reuse_test(ggml_backend_dev_t device) {
    backend_ptr backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    CHECK(backend); if (!backend) { return; }
    const auto config = config_parser::from_environment(); CHECK(config.ok());
    if (!config.ok()) { return; }
    CHECK(config.value().app_write_weights && !config.value().enable_w4fp16);
    auto wc_a = context(), wc_b = context(), c = context();
    auto * a = ggml_new_tensor_2d(wc_a.get(),GGML_TYPE_Q4_0,32,16);
    auto * b = ggml_new_tensor_2d(wc_b.get(),GGML_TYPE_Q4_0,32,16);
    buffer_ptr weights_a(ggml_backend_alloc_ctx_tensors(wc_a.get(),backend.get()),ggml_backend_buffer_free);
    buffer_ptr weights_b(ggml_backend_alloc_ctx_tensors(wc_b.get(),backend.get()),ggml_backend_buffer_free);
    CHECK(weights_a && weights_b); if (!weights_a || !weights_b) { return; }
    ggml_backend_buffer_set_usage(weights_a.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_set_usage(weights_b.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<uint8_t> raw_a(ggml_nbytes(a)), raw_b(ggml_nbytes(b));
    for (size_t row=0;row<16;++row) {
        const auto scale_a=ggml_fp32_to_fp16(0.25f),scale_b=ggml_fp32_to_fp16(-0.5f);
        std::memcpy(raw_a.data()+18*row,&scale_a,2);std::memcpy(raw_b.data()+18*row,&scale_b,2);
        for(size_t j=0;j<16;++j){raw_a[18*row+2+j]=uint8_t(j|((15-j)<<4));raw_b[18*row+2+j]=uint8_t((15-j)|(j<<4));}
    }
    ggml_backend_tensor_set(a,raw_a.data(),0,raw_a.size());ggml_backend_tensor_set(b,raw_b.data(),0,raw_b.size());
    auto * x_a=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,32,2);
    auto * x_b=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,32,2);
    auto * y_a=ggml_mul_mat(c.get(),a,x_a);auto * y_b=ggml_mul_mat(c.get(),b,x_b);
    auto * g_a=ggml_new_graph(c.get());ggml_graph_add_node(g_a,y_a);
    auto * g_b=ggml_new_graph(c.get());ggml_graph_add_node(g_b,y_b);
    buffer_ptr io(ggml_backend_alloc_ctx_tensors(c.get(),backend.get()),ggml_backend_buffer_free);
    CHECK(io); if (!io) { return; }
    std::vector<float> input(64);input[0]=1;input[33]=1;
    ggml_backend_tensor_set(x_a,input.data(),0,input.size()*4);ggml_backend_tensor_set(x_b,input.data(),0,input.size()*4);
    for(int run=0;run<4;++run) {
        const bool use_b=run==1;
        if(run==3){const auto scale=ggml_fp32_to_fp16(1.0f);std::memcpy(raw_a.data(),&scale,2);ggml_backend_tensor_set(a,raw_a.data(),0,2);}
        CHECK(ggml_backend_graph_compute(backend.get(),use_b?g_b:g_a)==GGML_STATUS_SUCCESS);
        const auto & raw=use_b?raw_b:raw_a;auto decoded=prepare_q4_weights({raw.data(),raw.size()},32,16);CHECK(decoded.ok());
        if(!decoded.ok()){return;}
        std::vector<float> actual(32),expected(32);ggml_backend_tensor_get(use_b?y_b:y_a,actual.data(),0,actual.size()*4);
        for(size_t token=0;token<2;++token){for(size_t row=0;row<16;++row){expected[token*16+row]=ggml_fp16_to_fp32(decoded.value()[row*32+token].bits);}}
        accuracy(actual,expected,use_b?"dynamic weight B":"dynamic weight A");
    }
    const auto fn=reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device),"ggml_backend_qairt_get_stats"));
    CHECK(fn);if(!fn){return;}
    ggml_backend_qairt_stats stats {};CHECK(fn(backend.get(),&stats));
    CHECK(stats.preparations==(config.value().share_weight_graphs?1:3));
    CHECK(stats.cache_hits==(config.value().share_weight_graphs?3:1) && stats.executions==4);
    std::printf("APP_WRITE A/B/A/mutation: preparations=%llu hits=%llu executions=%llu sharing=%d\n",(unsigned long long)stats.preparations,(unsigned long long)stats.cache_hits,(unsigned long long)stats.executions,config.value().share_weight_graphs);
}

int main(int argc, char ** argv) {
    ggml_log_set(logger, nullptr);
    const char * app_write = std::getenv("GGML_QAIRT_APP_WRITE_WEIGHTS");
    const std::string saved_app_write = app_write ? app_write : "";
    const char * sharing = std::getenv("GGML_QAIRT_SHARE_WEIGHT_GRAPHS");
    const std::string saved_sharing = sharing ? sharing : "";
    _putenv_s("GGML_QAIRT_APP_WRITE_WEIGHTS", "0");
    _putenv_s("GGML_QAIRT_SHARE_WEIGHT_GRAPHS", "0");
    const char * old_target = std::getenv("GGML_QAIRT_DEVICE");
    const std::string saved_target = old_target ? old_target : "";
    _putenv_s("GGML_QAIRT_DEVICE", "HTP");
    const char * old_flag = std::getenv("GGML_QAIRT_W4FP16");
    const std::string saved_flag = old_flag ? old_flag : "";
    const wchar_t * old_directory = _wgetenv(L"GGML_QAIRT_LIBRARY_DIR");
    const std::wstring saved_directory = old_directory ? old_directory : L"";
    _wputenv_s(L"GGML_QAIRT_LIBRARY_DIR", L"C:/QNN-unused");
    _putenv_s("GGML_QAIRT_W4FP16", "invalid");
    CHECK(!config_parser::from_environment().ok());
    _putenv_s("GGML_QAIRT_W4FP16", "1");
    const auto enabled = config_parser::from_environment(); CHECK(enabled.ok() && enabled.value().enable_w4fp16);
    _putenv_s("GGML_QAIRT_W4FP16", "0");
    const auto disabled = config_parser::from_environment(); CHECK(disabled.ok() && !disabled.value().enable_w4fp16);
    _putenv_s("GGML_QAIRT_APP_WRITE_WEIGHTS", "1");
    _putenv_s("GGML_QAIRT_W4FP16", "1"); CHECK(!config_parser::from_environment().ok());
    _putenv_s("GGML_QAIRT_W4FP16", "0");
    _putenv_s("GGML_QAIRT_SHARE_WEIGHT_GRAPHS", "1"); CHECK(config_parser::from_environment().ok());
    _putenv_s("GGML_QAIRT_APP_WRITE_WEIGHTS", "0"); CHECK(!config_parser::from_environment().ok());
    _putenv_s("GGML_QAIRT_APP_WRITE_WEIGHTS", saved_app_write.c_str());
    _putenv_s("GGML_QAIRT_SHARE_WEIGHT_GRAPHS", saved_sharing.c_str());
    _putenv_s("GGML_QAIRT_W4FP16", saved_flag.c_str());
    _wputenv_s(L"GGML_QAIRT_LIBRARY_DIR", saved_directory.c_str());
    _putenv_s("GGML_QAIRT_DEVICE", saved_target.c_str());
    ggml_backend_load_all();
    auto q41_context = context();
    CHECK(q4_weight_layout(ggml_new_tensor_2d(q41_context.get(), GGML_TYPE_Q4_1, 32, 2)));
    packing_tests();
    w4_packing_tests();
    q41_packing_tests();
    if (argc == 2 && std::strcmp(argv[1], "--host") == 0) {
        std::printf("QAIRT FP16 host: %d checks, %d failures\n", checks, failures);
        return failures ? 1 : 0;
    }
    auto device = ggml_backend_dev_by_name("QAIRT0");
    CHECK(device != nullptr);
    if (device && argc == 2 && std::strcmp(argv[1],"--weight-reuse") == 0) {
        weight_reuse_test(device);
        std::printf("QAIRT APP_WRITE: %d checks, %d failures\n",checks,failures);
        return failures ? 1 : 0;
    }
    if (device && argc == 2 && (std::strcmp(argv[1],"--w4fp16") == 0 || std::strcmp(argv[1],"--q41") == 0)) {
        w4_backend_test(device, std::strcmp(argv[1],"--q41") == 0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q4_0);
        std::printf("QAIRT W4FP16: %d checks, %d failures\n",checks,failures);
        return failures ? 1 : 0;
    }
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
