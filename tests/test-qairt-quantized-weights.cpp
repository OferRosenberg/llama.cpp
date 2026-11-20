#include "ggml.h"
#include "ggml-backend.h"
#include "llama-model.h"
#include "qairt-graph.h"
#include "qairt-weights.h"
#include "QnnOpDef.h"
#include "QnnProperty.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ggml::qairt;
using timer = std::chrono::steady_clock;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { ++failures; if (failures <= 16) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); } } } while (0)

struct options {
    const char * op = "matmul";
    const char * encoding = "bw4";
    const char * scales = "mixed";
    const char * model = nullptr;
    const char * weight_name = "blk.27.attn_q.weight";
    uint32_t k = 128, n = 32, m = 13, runs = 7;
    bool sign_extend = false;
    bool transpose = true, one_hot = false, float_offset = false, signed_scales = false, trace_values = false, unsigned_carrier = false, dynamic_weights = false, tensor_v1 = false;
};
struct quantized_weights {
    std::vector<uint8_t> data;
    std::vector<Qnn_FloatScaleOffset_t> scales;
};
static double elapsed(timer::time_point start) {
    return std::chrono::duration<double, std::milli>(timer::now() - start).count();
}
static bool is_packed(const options & o) { return std::strcmp(o.encoding, "bw4") != 0 && std::strcmp(o.encoding, "bw8") != 0; }
static bool is_transposed(const options & o) { return std::strcmp(o.op, "fc") == 0 || (std::strcmp(o.op, "matmul") == 0 && o.transpose); }
static std::vector<uint8_t> source_weights(const options & o) {
    std::vector<uint8_t> raw(size_t(o.k) * o.n / 32 * 18);
    for (size_t b = 0; b < raw.size() / 18; ++b) {
        float scale = float(3 + b % 13) / 1024.0f;
        if (std::strcmp(o.scales, "negative") == 0 || (std::strcmp(o.scales, "mixed") == 0 && b % 2)) { scale = -scale; }
        if (std::strcmp(o.scales, "zero") == 0 || (std::strcmp(o.scales, "mixed") == 0 && b % 17 == 0)) { scale = 0; }
        const auto half = ggml_fp32_to_fp16(scale);
        std::memcpy(raw.data() + b * 18, &half, 2);
        for (size_t j = 0; j < 16; ++j) { raw[b * 18 + 2 + j] = uint8_t(((j + b * 3) % 16) | (((15 - j + b * 5) % 16) << 4)); }
    }
    return raw;
}
static quantized_weights repack(const std::vector<uint8_t> & raw, const options & o) {
    quantized_weights output;
    const size_t count = size_t(o.k) * o.n;
    const bool packed = is_packed(o), transposed = is_transposed(o), bw8 = std::strcmp(o.encoding, "bw8") == 0;
    output.data.resize(packed ? count / 2 : count);
    output.scales.resize(count / 32);
    for (size_t b = 0; b < count / 32; ++b) {
        ggml_fp16_t half;
        std::memcpy(&half, raw.data() + b * 18, 2);
        const float d = ggml_fp16_to_fp32(half);
        // Q4_0 deltas can be negative. Remap integer codes to keep the public scale positive.
        const float scale = d == 0 ? 1.0f : o.signed_scales ? d : std::fabs(d);
        output.scales[b] = {scale, o.unsigned_carrier ? (d == 0 ? 0.0f : d > 0 ? -8.0f : -7.0f) : d < 0 && !bw8 && !o.signed_scales ? (o.float_offset ? scale : 1.0f) : 0.0f};
        for (size_t j = 0; j < 32; ++j) {
            const uint8_t byte = raw[b * 18 + 2 + j % 16];
            const int u = j < 16 ? byte & 15 : byte >> 4;
            const int q = d == 0 ? 0 : o.unsigned_carrier ? (d > 0 ? u : 15 - u) : d > 0 || o.signed_scales ? u - 8 : bw8 ? 8 - u : 7 - u;
            const size_t source = b * 32 + j;
            const size_t index = transposed ? source : (source % o.k) * o.n + source / o.k;
            if (packed) { output.data[index / 2] |= uint8_t((q & 15) << (4 * (index % 2))); }
            else { output.data[index] = uint8_t(bw8 || o.sign_extend ? q : q & 15); }
        }
    }
    return output;
}
static void packing_tests() {
    for (const char * scales : {"positive", "negative", "mixed", "zero"}) {
        for (bool unsigned_carrier : {false, true}) {
            for (bool signed_scales : {false, true}) {
                if (unsigned_carrier && signed_scales) { continue; }
                for (const char * encoding : {"bw4", "bw8", "packed4", "packed-float4"}) {
                    for (bool transpose : {true, false}) {
                        for (bool sign_extend : {false, true}) {
                            options o; o.encoding = encoding; o.transpose = transpose; o.scales = scales;
                            o.unsigned_carrier = unsigned_carrier; o.signed_scales = signed_scales; o.sign_extend = sign_extend;
                            if (sign_extend && (unsigned_carrier || is_packed(o))) { continue; }
                            const auto raw = source_weights(o);
                            const auto original = raw;
                            const auto packed = repack(raw, o);
                            auto reference = prepare_q4_weights({raw.data(), raw.size()}, o.k, o.n);
                            CHECK(reference.ok()); if (!reference.ok()) { return; }
                            for (size_t i = 0; i < size_t(o.k) * o.n; ++i) {
                                const size_t index = transpose ? i : (i % o.k) * o.n + i / o.k;
                                const unsigned bits = is_packed(o) ? (packed.data[index / 2] >> (4 * (index % 2))) & 15 : packed.data[index];
                                const int q = unsigned_carrier ? int(bits) : std::strcmp(encoding, "bw8") == 0 || sign_extend ? (bits >= 128 ? int(bits) - 256 : int(bits)) : (bits >= 8 ? int(bits) - 16 : int(bits));
                                const auto & scale = packed.scales[i / 32];
                                const auto expected = ggml_fp32_to_fp16((q + scale.offset) * scale.scale);
                                CHECK(ggml_fp16_to_fp32(expected) == ggml_fp16_to_fp32(reference.value()[i].bits));
                            }
                            CHECK(raw == original);
                        }
                    }
                }
            }
        }
    }
    std::printf("Q4_0 integer repacking: %d failures\n", failures);
}
static void bind(Qnn_Tensor_t & t, void * data, size_t bytes) {
    const Qnn_ClientBuffer_t buffer = {data, uint32_t(bytes)};
    if (t.version == QNN_TENSOR_VERSION_1) { t.v1.clientBuf = buffer; }
    else { t.v2.clientBuf = buffer; }
}
static Qnn_Tensor_t tensor(const char * name, Qnn_TensorType_t role, Qnn_DataType_t type, std::vector<uint32_t> & dims, void * data, size_t bytes) {
    Qnn_Tensor_t t = QNN_TENSOR_INIT;
    t.version = QNN_TENSOR_VERSION_2;
    t.v2.name = name; t.v2.type = role; t.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t.v2.dataType = type; t.v2.rank = uint32_t(dims.size()); t.v2.dimensions = dims.data();
    t.v2.memType = QNN_TENSORMEMTYPE_RAW; t.v2.clientBuf = {data, uint32_t(bytes)};
    return t;
}
struct run_result {
    const char * stage = "none";
    Qnn_ErrorHandle_t code = QNN_SUCCESS;
    std::vector<uint16_t> output;
    double prepare_ms = 0, execute_ms = 0;
};
static void print_error(const qairt_error & e) {
    std::printf("runtime_failure operation=%s code=%llu message=%s\n", e.operation.c_str(), (unsigned long long)e.qnn_error.value_or(0), e.message.c_str());
}
static run_result execute(const options & o, const std::vector<uint16_t> & input, const std::vector<uint8_t> & raw, bool quantized) {
    run_result r;
    auto config = config_parser::from_environment();
    if (!config.ok()) { print_error(config.error()); r.stage = "configuration"; r.code = QNN_COMMON_ERROR_GENERAL; return r; }
    auto runtime_result = runtime_factory::create(std::make_shared<const runtime_config>(config.value()));
    if (!runtime_result.ok()) { print_error(runtime_result.error()); r.stage = "runtime"; r.code = QNN_COMMON_ERROR_GENERAL; return r; }
    auto runtime = std::move(runtime_result.value());
    auto context_result = context_owner::create(runtime);
    if (!context_result.ok()) { print_error(context_result.error()); r.stage = "context"; r.code = QNN_COMMON_ERROR_GENERAL; return r; }
    auto context = std::move(context_result.value());
    auto domain = runtime->domain();
    auto cleanup = [&]() {
        auto c = context->close(); CHECK(c.ok()); context.reset();
        auto s = runtime->close(); CHECK(s.ok()); runtime.reset(); domain.reset();
    };
    auto call = [&](const char * stage, auto && fn) {
        r.stage = stage; r.code = domain->call(fn);
        return r.code == QNN_SUCCESS;
    };
    const auto capability = domain->call([](const auto & api) { return api.propertyHasCapability(QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK); });
    std::printf("BW_FLOAT_BLOCK capability=%llu", (unsigned long long)capability);
    if (runtime->platform().htp_architecture) { std::printf(" arch=%u", *runtime->platform().htp_architecture); }
    if (runtime->platform().soc_model) { std::printf(" soc=%u", *runtime->platform().soc_model); }
    std::printf("\n");
    Qnn_GraphHandle_t graph = nullptr;
    if (!call("graphCreate", [&](const auto & api) { return api.graphCreate(context->handle(), "quantized_weights", runtime->graph_configs(), &graph); })) { cleanup(); return r; }
    const bool fc = std::strcmp(o.op, "fc") == 0;
    const bool conv = std::strcmp(o.op, "conv") == 0;
    std::vector<uint32_t> xd = fc ? std::vector<uint32_t>{o.m, o.k} : std::vector<uint32_t>{1, 1, o.m, o.k};
    std::vector<uint32_t> wd = fc ? std::vector<uint32_t>{o.n, o.k} : conv ? std::vector<uint32_t>{1, 1, o.k, o.n} : is_transposed(o) ? std::vector<uint32_t>{1, 1, o.n, o.k} : std::vector<uint32_t>{1, 1, o.k, o.n};
    std::vector<uint32_t> yd = fc ? std::vector<uint32_t>{o.m, o.n} : std::vector<uint32_t>{1, 1, o.m, o.n};
    std::vector<uint32_t> block = fc ? std::vector<uint32_t>{1, 32} : is_transposed(o) ? std::vector<uint32_t>{1, 1, 1, 32} : std::vector<uint32_t>{1, 1, 32, 1};
    auto x = tensor("x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, xd, nullptr, 0);
    auto y = tensor("y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, yd, nullptr, 0);
    std::vector<uint16_t> fp16;
    quantized_weights packed;
    if (quantized) { packed = repack(raw, o); }
    else {
        auto reference = prepare_q4_weights({raw.data(), raw.size()}, o.k, o.n);
        CHECK(reference.ok()); if (!reference.ok()) { cleanup(); r.stage = "reference"; r.code = QNN_COMMON_ERROR_GENERAL; return r; }
        fp16.resize(size_t(o.k) * o.n);
        for (size_t i = 0; i < fp16.size(); ++i) { fp16[is_transposed(o) ? i : (i % o.k) * o.n + i / o.k] = reference.value()[i].bits; }
    }
    auto w = tensor("w", quantized && o.dynamic_weights ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_STATIC, quantized ? (o.unsigned_carrier ? (is_packed(o) ? QNN_DATATYPE_UFIXED_POINT_4 : QNN_DATATYPE_UFIXED_POINT_8) : (is_packed(o) ? QNN_DATATYPE_SFIXED_POINT_4 : QNN_DATATYPE_SFIXED_POINT_8)) : QNN_DATATYPE_FLOAT_16,
                    wd, quantized ? (void *)packed.data.data() : (void *)fp16.data(), quantized ? packed.data.size() : fp16.size() * 2);
    if (quantized) {
        w.v2.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
        if (std::strcmp(o.encoding, "packed-float4") == 0) {
            w.v2.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_FLOAT_BLOCK;
            w.v2.quantizeParams.floatBlockEncoding = {block.data(), packed.scales.data()};
        } else {
            w.v2.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK;
            w.v2.quantizeParams.bwFloatBlockEncoding = {std::strcmp(o.encoding, "bw8") == 0 ? 8u : 4u, block.data(), packed.scales.data()};
        }
    }
    if (quantized && o.dynamic_weights) { w.v2.clientBuf = {nullptr, 0}; }
    const auto original_scales = packed.scales;
    for (auto * t : {&x, &w, &y}) {
        if (o.tensor_v1) { const auto v2 = t->v2; t->version = QNN_TENSOR_VERSION_1; t->v1.id = v2.id; t->v1.name = v2.name; t->v1.type = v2.type; t->v1.dataFormat = v2.dataFormat; t->v1.dataType = v2.dataType; t->v1.quantizeParams = v2.quantizeParams; t->v1.rank = v2.rank; t->v1.dimensions = v2.dimensions; t->v1.memType = v2.memType; t->v1.clientBuf = v2.clientBuf; }
        if (!call(t == &w ? "tensorCreateGraphTensor(weights)" : "tensorCreateGraphTensor", [&](const auto & api) { return api.tensorCreateGraphTensor(graph, t); })) { cleanup(); return r; }
    }
    if (quantized) {
        size_t changed = 0;
        for (size_t i = 0; i < packed.scales.size(); ++i) {
            changed += packed.scales[i].scale != original_scales[i].scale || packed.scales[i].offset != original_scales[i].offset;
        }
        std::printf("METADATA changed=%zu/%zu\n", changed, packed.scales.size());
    }
    std::vector<Qnn_Param_t> params;
    if (!fc && !conv) {
        params.resize(2);
        for (auto & p : params) { p = QNN_PARAM_INIT; p.paramType = QNN_PARAMTYPE_SCALAR; p.scalarParam.dataType = QNN_DATATYPE_BOOL_8; }
        params[0].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0; params[0].scalarParam.bool8Value = 0;
        params[1].name = QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1; params[1].scalarParam.bool8Value = o.transpose;
    }
    std::vector<std::vector<uint32_t>> param_dims = {{2}, {2, 2}, {2}};
    std::vector<std::vector<uint32_t>> param_values = {{1, 1}, {0, 0, 0, 0}, {1, 1}};
    if (conv) {
        params.resize(4);
        const char * names[] = {QNN_OP_CONV_2D_PARAM_STRIDE, QNN_OP_CONV_2D_PARAM_PAD_AMOUNT, QNN_OP_CONV_2D_PARAM_DILATION};
        for (size_t i = 0; i < 3; ++i) {
            params[i] = QNN_PARAM_INIT; params[i].name = names[i]; params[i].paramType = QNN_PARAMTYPE_TENSOR;
            params[i].tensorParam = tensor(names[i], QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_UINT_32, param_dims[i], param_values[i].data(), param_values[i].size() * 4);
            if (!call("tensorCreateGraphTensor(parameter)", [&](const auto & api) { return api.tensorCreateGraphTensor(graph, &params[i].tensorParam); })) { cleanup(); return r; }
        }
        params[3] = QNN_PARAM_INIT; params[3].name = QNN_OP_CONV_2D_PARAM_GROUP; params[3].paramType = QNN_PARAMTYPE_SCALAR;
        params[3].scalarParam.dataType = QNN_DATATYPE_UINT_32; params[3].scalarParam.uint32Value = 1;
    }
    Qnn_Tensor_t inputs[] = {x, w};
    Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
    op.v1.name = "projection"; op.v1.packageName = "qti.aisw"; op.v1.typeName = fc ? QNN_OP_FULLY_CONNECTED : conv ? QNN_OP_CONV_2D : QNN_OP_MAT_MUL;
    op.v1.numOfInputs = 2; op.v1.inputTensors = inputs; op.v1.numOfOutputs = 1; op.v1.outputTensors = &y;
    op.v1.numOfParams = uint32_t(params.size()); op.v1.params = params.data();
    if (!call("backendValidateOpConfig", [&](const auto & api) { return api.backendValidateOpConfig(runtime->backend(), op); }) ||
        !call("graphAddNode", [&](const auto & api) { return api.graphAddNode(graph, op); })) { cleanup(); return r; }
    auto start = timer::now();
    if (!call("graphFinalize", [&](const auto & api) { return api.graphFinalize(graph, nullptr, nullptr); })) { cleanup(); return r; }
    r.prepare_ms = elapsed(start);
    r.output.resize(size_t(o.m) * o.n);
    bind(x, const_cast<uint16_t *>(input.data()), input.size() * 2);
    bind(y, r.output.data(), r.output.size() * 2);
    std::vector<Qnn_Tensor_t> execution_inputs = {x};
    if (quantized && o.dynamic_weights) { bind(w, packed.data.data(), packed.data.size()); execution_inputs.push_back(w); }
    std::vector<double> timings;
    for (uint32_t i = 0; i <= o.runs; ++i) {
        start = timer::now();
        if (!call("graphExecute", [&](const auto & api) { return api.graphExecute(graph, execution_inputs.data(), uint32_t(execution_inputs.size()), &y, 1, nullptr, nullptr); })) { cleanup(); return r; }
        if (i) { timings.push_back(elapsed(start)); }
    }
    std::sort(timings.begin(), timings.end()); r.execute_ms = timings[timings.size() / 2];
    cleanup();
    return r;
}
int main(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--host") == 0) { packing_tests(); return failures ? 1 : 0; }
        if (std::strcmp(argv[i], "--no-transpose") == 0) { o.transpose = false; continue; }
        if (std::strcmp(argv[i], "--one-hot") == 0) { o.one_hot = true; continue; }
        if (std::strcmp(argv[i], "--sign-extend") == 0) { o.sign_extend = true; continue; }
        if (std::strcmp(argv[i], "--unsigned") == 0) { o.unsigned_carrier = true; continue; }
        if (std::strcmp(argv[i], "--dynamic-weights") == 0) { o.dynamic_weights = true; continue; }
        if (std::strcmp(argv[i], "--tensor-v1") == 0) { o.tensor_v1 = true; continue; }
        if (std::strcmp(argv[i], "--signed-scales") == 0) { o.signed_scales = true; continue; }
        if (std::strcmp(argv[i], "--trace-values") == 0) { o.trace_values = true; continue; }
        if (std::strcmp(argv[i], "--float-offset") == 0) { o.float_offset = true; continue; }
        if (i + 1 == argc) { return 2; }
        const char * name = argv[i]; const char * value = argv[++i];
        if (std::strcmp(name, "--op") == 0) { o.op = value; }
        else if (std::strcmp(name, "--model") == 0) { o.model = value; }
        else if (std::strcmp(name, "--tensor") == 0) { o.weight_name = value; }
        else if (std::strcmp(name, "--encoding") == 0) { o.encoding = value; }
        else if (std::strcmp(name, "--scales") == 0) { o.scales = value; }
        else if (std::strcmp(name, "--k") == 0) { o.k = uint32_t(std::strtoul(value, nullptr, 10)); }
        else if (std::strcmp(name, "--n") == 0) { o.n = uint32_t(std::strtoul(value, nullptr, 10)); }
        else if (std::strcmp(name, "--m") == 0) { o.m = uint32_t(std::strtoul(value, nullptr, 10)); }
        else if (std::strcmp(name, "--runs") == 0) { o.runs = uint32_t(std::strtoul(value, nullptr, 10)); }
        else { std::fprintf(stderr, "Unknown option %s\n", name); return 2; }
    }
    if ((o.sign_extend && (o.unsigned_carrier || is_packed(o))) || (o.unsigned_carrier && (o.signed_scales || o.float_offset)) || (o.signed_scales && o.float_offset)) {
        std::fprintf(stderr, "Conflicting scale/offset diagnostic options\n"); return 2;
    }
    std::vector<uint8_t> raw;
    if (o.model) {
        llama_backend_init();
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {nullptr}; mp.devices = devices; mp.n_gpu_layers = 0; mp.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(o.model, mp), llama_model_free);
        if (!model) { return 2; }
        ggml_tensor * weight = nullptr;
        for (const auto & entry : model->tensors_by_name) { if (entry.first == o.weight_name) { weight = entry.second; break; } }
        if (!weight || weight->type != GGML_TYPE_Q4_0 || weight->ne[2] != 1 || weight->ne[3] != 1 || !ggml_is_contiguous(weight) || weight->ne[0] <= 0 || weight->ne[0] > 8192 || weight->ne[1] <= 0 || weight->ne[1] > 8192 || weight->ne[0] * weight->ne[1] > 32 * 1024 * 1024) {
            std::fprintf(stderr, "Model tensor must be a contiguous Q4_0 matrix: %s\n", o.weight_name); return 2;
        }
        o.k = uint32_t(weight->ne[0]); o.n = uint32_t(weight->ne[1]);
        raw.resize(ggml_nbytes(weight)); ggml_backend_tensor_get(weight, raw.data(), 0, raw.size());
        std::printf("MODEL tensor=%s K=%u N=%u type=Q4_0\n", o.weight_name, o.k, o.n);
    }
    if ((std::strcmp(o.op, "matmul") && std::strcmp(o.op, "fc") && std::strcmp(o.op, "conv")) ||
        (std::strcmp(o.encoding, "bw4") && std::strcmp(o.encoding, "bw8") && std::strcmp(o.encoding, "packed4") && std::strcmp(o.encoding, "packed-float4")) ||
        (std::strcmp(o.scales, "positive") && std::strcmp(o.scales, "negative") && std::strcmp(o.scales, "mixed") && std::strcmp(o.scales, "zero")) ||
        !o.k || o.k % 32 || !o.n || !o.m || !o.runs || o.runs > 100 || o.k > 8192 || o.n > 8192 || o.m > 8192 || size_t(o.k) * o.n > 32 * 1024 * 1024) { return 2; }
    if (o.one_hot) { o.m = o.k; }
    std::printf("CASE op=%s encoding=%s scales=%s offset=%s K=%u N=%u M=%u transpose=%d\n", o.op, o.encoding, o.scales, o.float_offset ? "float" : "integer", o.k, o.n, o.m, is_transposed(o));
    auto start = timer::now(); if (raw.empty()) { raw = source_weights(o); } const auto original = raw;
    size_t positive = 0, negative = 0, zero = 0;
    for (size_t b = 0; b < raw.size() / 18; ++b) { ggml_fp16_t half; std::memcpy(&half, raw.data() + b * 18, 2); const float d = ggml_fp16_to_fp32(half); positive += d > 0; negative += d < 0; zero += d == 0; }
    std::printf("SCALES positive=%zu negative=%zu zero=%zu signed_scales=%d unsigned=%d dynamic=%d tensor_version=%u sign_extend=%d\n", positive, negative, zero, o.signed_scales, o.unsigned_carrier, o.dynamic_weights, o.tensor_v1 ? 1 : 2, o.sign_extend);
    start = timer::now(); const auto packed = repack(raw, o); const double repack_ms = elapsed(start);
    start = timer::now(); auto weights = prepare_q4_weights({raw.data(), raw.size()}, o.k, o.n); const double decode_ms = elapsed(start);
    CHECK(weights.ok()); if (!weights.ok()) { return 1; }
    std::printf("PAYLOAD source_bytes=%zu qnn_weight_bytes=%zu scale_offset_bytes=%zu fp16_bytes=%zu repack_ms=%.6f decode_ms=%.6f\n", raw.size(), packed.data.size(), packed.scales.size() * sizeof(Qnn_FloatScaleOffset_t), weights.value().size() * 2, repack_ms, decode_ms);
    std::vector<uint16_t> input(size_t(o.k) * o.m);
    for (size_t i = 0; i < input.size(); ++i) { input[i] = ggml_fp32_to_fp16(o.one_hot ? (i / o.k == i % o.k ? 1.0f : 0.0f) : 0.25f * std::sin(float(i % 4001) * 0.043f)); }
    const auto baseline = execute(o, input, raw, false);
    std::printf("BASELINE stage=%s code=%llu prepare_ms=%.6f execute_ms=%.6f\n", baseline.stage, (unsigned long long)baseline.code, baseline.prepare_ms, baseline.execute_ms);
    if (baseline.code != QNN_SUCCESS) { return 2; }
    const auto candidate = execute(o, input, raw, true);
    std::printf("QUANTIZED stage=%s code=%llu prepare_ms=%.6f execute_ms=%.6f\n", candidate.stage, (unsigned long long)candidate.code, candidate.prepare_ms, candidate.execute_ms);
    CHECK(raw == original);
    if (candidate.code != QNN_SUCCESS) { return 2; }
    double error = 0, energy = 0, maximum = 0;
    size_t unequal = 0;
    for (size_t i = 0; i < baseline.output.size(); ++i) {
        const double ref = ggml_fp16_to_fp32(baseline.output[i]), actual = ggml_fp16_to_fp32(candidate.output[i]);
        const double delta = actual - ref; error += delta * delta; energy += ref * ref; maximum = std::max(maximum, std::fabs(delta));
        if (o.trace_values && baseline.output[i] != candidate.output[i] && unequal < 16) { std::printf("DIFF index=%zu reference=%.9g actual=%.9g reference_bits=%04x actual_bits=%04x\n", i, ref, actual, baseline.output[i], candidate.output[i]); }
        unequal += baseline.output[i] != candidate.output[i];
        CHECK(std::isfinite(actual) && std::isfinite(ref));
    }
    const double nmse = error / std::max(energy, 1e-12);
    CHECK(nmse <= 1e-5);
    std::printf("RESULT NMSE=%.12g max_abs=%.12g unequal=%zu/%zu failures=%d\n", nmse, maximum, unequal, baseline.output.size(), failures);
    return failures ? 1 : 0;
}
