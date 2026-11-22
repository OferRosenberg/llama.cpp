#include "qairt-graph.h"
#include "ggml-impl.h"

#include <limits>
#include <cmath>
#include <QnnOpDef.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <System/QnnSystemInterface.h>
#include <HTP/QnnHtpSystemContext.h>

namespace ggml::qairt {

void context_owner::dispose(context_owner * owner) noexcept {
    if (owner->close_raw() == QNN_SUCCESS) { delete owner; }
    else { std::fprintf(stderr, "qairt: retaining failed context and runtime ancestry until process exit\n"); }
}

Qnn_ErrorHandle_t context_owner::close_raw() noexcept {
    if (active_graphs_) { return QNN_COMMON_ERROR_RESOURCE_UNAVAILABLE; }
    auto code = context_.close_raw();
    if (code != QNN_SUCCESS) { return code; }
    if (admitted_) { --runtime_->active_contexts_; admitted_ = false; }
    return QNN_SUCCESS;
}

status context_owner::close() {
    if (active_graphs_) { return status::failure(host_error(error_stage::cleanup, "context.close", "Active graph owners remain")); }
    return check_qnn(close_raw(), error_stage::cleanup, "contextFree");
}

result<std::shared_ptr<context_owner>> context_owner::create(std::shared_ptr<runtime_owner> runtime) {
    using context_result = result<std::shared_ptr<context_owner>>;
    if (!runtime || runtime->closed()) { return context_result::failure(host_error(error_stage::context_create, "contextCreate", "No live runtime")); }
    auto owner = std::shared_ptr<context_owner>(new context_owner(), context_owner::dispose);
    owner->runtime_ = runtime;
    auto domain = runtime->domain();
    Qnn_ContextHandle_t context = nullptr;
    auto code = domain->call([&](const auto & api) { return api.contextCreate(runtime->backend(), runtime->device(), runtime->context_configs(), &context); });
    if (code != QNN_SUCCESS) { return context_result::failure(sdk_error(error_stage::context_create, "contextCreate", code)); }
    owner->context_.adopt_into_empty(context, domain, { runtime->profile() });
    ++runtime->active_contexts_;
    owner->admitted_ = true;
    return context_result::success(std::move(owner));
}

result<std::shared_ptr<graph_owner>> graph_owner::create(std::shared_ptr<context_owner> context, std::string name) {
    using graph_result = result<std::shared_ptr<graph_owner>>;
    if (!context || !context->handle() || name.empty() || name.size() > 128 || name.find('\0') != std::string::npos) {
        return graph_result::failure(host_error(error_stage::graph_create, "graphCreate", "Invalid context or graph name"));
    }
    auto graph = std::shared_ptr<graph_owner>(new graph_owner());
    graph->name_ = std::move(name);
    auto runtime = context->runtime();
    auto code = runtime->domain()->call([&](const auto & api) {
        return api.graphCreate(context->handle(), graph->name_.c_str(), runtime->graph_configs(), &graph->graph_);
    });
    if (code != QNN_SUCCESS) { graph->graph_ = nullptr; return graph_result::failure(sdk_error(error_stage::graph_create, "graphCreate", code)); }
    graph->context_ = std::move(context);
    static std::atomic<uint64_t> sequence {0};
    graph->diagnostic_id_ = ++sequence;
    if (diagnostic_phase()) {
        GGML_LOG_INFO("qairt: inventory graph_create phase=%s graph=%llu context=%p handle=%p\n", diagnostic_phase(),
            (unsigned long long) graph->diagnostic_id_, graph->context_->handle(), graph->graph_);
    }
    ++graph->context_->active_graphs_;
    return graph_result::success(std::move(graph));
}

graph_owner::~graph_owner() noexcept {
    GGML_ASSERT(!active_bindings_);
    if (context_) { --context_->active_graphs_; }
}

status graph_owner::close() {
    if (active_bindings_) { return status::failure(host_error(error_stage::cleanup, "graph.close", "Active bindings remain")); }
    if (context_) { --context_->active_graphs_; context_.reset(); }
    graph_ = nullptr;
    state_ = graph_state::closed;
    tensors_.clear(); input_ids_.clear(); output_ids_.clear();
    return status::success();
}

const tensor_record * graph_owner::find_tensor(tensor_id id) const noexcept {
    if (state_ == graph_state::closed) { return nullptr; }
    for (const auto & tensor : tensors_) { if (tensor->id().value == id.value) { return tensor.get(); } }
    return nullptr;
}

result<tensor_id> graph_owner::register_tensor(tensor_spec spec) { return register_tensor(std::move(spec), {}); }

result<tensor_id> graph_owner::register_tensor(tensor_spec spec, buffer_view_const payload) {
    if (state_ != graph_state::building) {
        return result<tensor_id>::failure(host_error(error_stage::tensor_registration, "register_tensor", "Graph is not accepting tensors"));
    }
    if (tensors_.size() >= 1024) { return result<tensor_id>::failure(host_error(error_stage::tensor_registration, "register_tensor", "Graph tensor limit")); }
    auto bytes = tensor_record::checked_bytes(spec);
    if (!bytes.ok()) { return result<tensor_id>::failure(bytes.error()); }
    for (const auto & tensor : tensors_) {
        if (tensor->spec().name == spec.name) { return result<tensor_id>::failure(host_error(error_stage::tensor_registration, "tensor.name", "Duplicate tensor name")); }
    }
    auto record = std::unique_ptr<tensor_record>(new tensor_record(std::move(spec)));
    if (record->spec().role == QNN_TENSOR_TYPE_STATIC) {
        auto checked = validate_buffer(*record, payload);
        if (!checked.ok()) { return result<tensor_id>::failure(checked.error()); }
    } else if (payload.data || payload.bytes) {
        return result<tensor_id>::failure(host_error(error_stage::tensor_registration, "tensor.payload", "Only STATIC tensors accept registration data"));
    }
    // No allocation can fail between successful SDK registration and publishing its record.
    tensors_.reserve(tensors_.size() + 1);
    input_ids_.reserve(input_ids_.size() + 1);
    output_ids_.reserve(output_ids_.size() + 1);
    auto descriptor = record->descriptor(payload);
    auto code = context_->runtime()->domain()->call([&](const auto & api) { return api.tensorCreateGraphTensor(graph_, &descriptor); });
    if (code != QNN_SUCCESS) { state_ = graph_state::failed; return result<tensor_id>::failure(sdk_error(error_stage::tensor_registration, "tensorCreateGraphTensor", code)); }
    record->id_ = { descriptor.v1.id };
    const auto id = record->id();
    const bool duplicate = find_tensor(id) != nullptr;
    const bool metadata_changed = !record->dimensions_unchanged();
    if (record->spec().role == QNN_TENSOR_TYPE_APP_WRITE) { input_ids_.push_back(id); }
    if (record->spec().role == QNN_TENSOR_TYPE_APP_READ) { output_ids_.push_back(id); }
    tensors_.push_back(std::move(record));
    if (id.value == null_tensor_id.value || duplicate || metadata_changed || descriptor.version != QNN_TENSOR_VERSION_1) {
        state_ = graph_state::failed;
        return result<tensor_id>::failure(host_error(error_stage::tensor_registration, "tensorCreateGraphTensor", "Unexpected ID or fixed tensor metadata"));
    }
    return result<tensor_id>::success(id);
}

op_spec op_spec::matmul(tensor_id a, tensor_id w, tensor_id y) {
    Qnn_Scalar_t no = QNN_SCALAR_INIT;
    no.dataType = QNN_DATATYPE_BOOL_8; no.bool8Value = 0;
    Qnn_Scalar_t yes = QNN_SCALAR_INIT;
    yes.dataType = QNN_DATATYPE_BOOL_8; yes.bool8Value = 1;
    return { "matmul", "qti.aisw", "MatMul", {a, w}, {y}, {{"transpose_in0", no}, {"transpose_in1", yes}} };
}

op_spec op_spec::sigmoid(tensor_id x, tensor_id y) {
    return {"sigmoid", "qti.aisw", QNN_OP_SIGMOID, {x}, {y}, {}, {}};
}
op_spec op_spec::normalization(tensor_id a, tensor_id gamma, tensor_id beta, tensor_id y, bool rms, float epsilon, uint32_t axis) {
    op_spec spec;
    spec.name = "normalize"; spec.package_name = "qti.aisw";
    spec.type_name = rms ? QNN_OP_RMS_NORM : QNN_OP_LAYER_NORM;
    spec.inputs = {a, gamma, beta}; spec.outputs = {y};
    Qnn_Scalar_t value = QNN_SCALAR_INIT;
    value.dataType = QNN_DATATYPE_FLOAT_32; value.floatValue = epsilon;
    spec.parameters.push_back({"epsilon", value});
    spec.tensor_parameters.push_back({"axes", {1}, {axis}});
    return spec;
}

op_spec op_spec::softmax(tensor_id a, tensor_id y, float beta, uint32_t axis_value) {
    op_spec spec;
    spec.name = "softmax"; spec.package_name = "qti.aisw"; spec.type_name = QNN_OP_SOFTMAX;
    spec.inputs = {a}; spec.outputs = {y};
    Qnn_Scalar_t axis = QNN_SCALAR_INIT, scale = QNN_SCALAR_INIT;
    axis.dataType = QNN_DATATYPE_UINT_32; axis.uint32Value = axis_value;
    scale.dataType = QNN_DATATYPE_FLOAT_32; scale.floatValue = beta;
    spec.parameters = {{"axis", axis}, {"beta", scale}};
    return spec;
}

op_spec op_spec::rotary_embedding(tensor_id x, tensor_id cosine, tensor_id sine, tensor_id y, uint32_t dimensions, bool interleaved) {
    op_spec spec;
    spec.name = "rope"; spec.package_name = "qti.aisw"; spec.type_name = QNN_OP_ROTARY_EMBEDDING;
    spec.inputs = {x, cosine, sine}; spec.outputs = {y};
    Qnn_Scalar_t dim = QNN_SCALAR_INIT, pairs = QNN_SCALAR_INIT;
    dim.dataType = QNN_DATATYPE_UINT_32; dim.uint32Value = dimensions;
    pairs.dataType = QNN_DATATYPE_BOOL_8; pairs.bool8Value = interleaved ? 1 : 0;
    // The SDK documents interleaved but does not define a parameter-name macro.
    spec.parameters = {{QNN_OP_ROTARY_EMBEDDING_PARAM_ROTARY_EMBEDDING_DIM, dim}, {"interleaved", pairs}};
    return spec;
}

op_spec op_spec::group_query_attention(std::vector<tensor_id> inputs, std::vector<tensor_id> outputs, uint32_t heads, uint32_t kv_heads, float scale) {
    op_spec spec {"gqa", "qti.aisw", QNN_OP_GROUP_QUERY_ATTENTION, std::move(inputs), std::move(outputs), {}, {}};
    Qnn_Scalar_t h = QNN_SCALAR_INIT, kv = QNN_SCALAR_INIT, rotary = QNN_SCALAR_INIT, factor = QNN_SCALAR_INIT;
    h.dataType = kv.dataType = rotary.dataType = QNN_DATATYPE_UINT_32;
    h.uint32Value = heads; kv.uint32Value = kv_heads; rotary.uint32Value = 0;
    factor.dataType = QNN_DATATYPE_FLOAT_32; factor.floatValue = scale;
    spec.parameters = {{QNN_OP_GROUP_QUERY_ATTENTION_PARAM_NUM_HEADS, h}, {QNN_OP_GROUP_QUERY_ATTENTION_PARAM_KV_NUM_HEADS, kv},
                       {QNN_OP_GROUP_QUERY_ATTENTION_PARAM_DO_ROTARY, rotary}, {QNN_OP_GROUP_QUERY_ATTENTION_PARAM_SCALE, factor}};
    return spec;
}

result<std::unique_ptr<op_packet>> op_packet::create_attention(const graph_owner & graph, const op_spec & spec) {
    using packet_result = result<std::unique_ptr<op_packet>>;
    auto fail = []() { return packet_result::failure(host_error(error_stage::op_validation, "GQA packet", "Invalid GQA shapes, roles or parameters")); };
    if (spec.name.empty() || spec.name.size() > 128 || spec.name.find('\0') != std::string::npos || spec.package_name != "qti.aisw" ||
        spec.inputs.size() != 10 || spec.outputs.size() != 3 || spec.parameters.size() != 4 || !spec.tensor_parameters.empty()) { return fail(); }
    const char * names[] = {QNN_OP_GROUP_QUERY_ATTENTION_PARAM_NUM_HEADS, QNN_OP_GROUP_QUERY_ATTENTION_PARAM_KV_NUM_HEADS,
                           QNN_OP_GROUP_QUERY_ATTENTION_PARAM_DO_ROTARY, QNN_OP_GROUP_QUERY_ATTENTION_PARAM_SCALE};
    for (size_t i = 0; i < 4; ++i) {
        if (spec.parameters[i].name != names[i] || spec.parameters[i].value.dataType != (i == 3 ? QNN_DATATYPE_FLOAT_32 : QNN_DATATYPE_UINT_32)) { return fail(); }
    }
    const auto heads = spec.parameters[0].value.uint32Value, kv_heads = spec.parameters[1].value.uint32Value;
    if (!heads || !kv_heads || heads % kv_heads || spec.parameters[2].value.uint32Value != 0 ||
        !std::isfinite(spec.parameters[3].value.floatValue) || spec.parameters[3].value.floatValue <= 0.0f) { return fail(); }
    const auto * q = graph.find_tensor(spec.inputs[0]);
    if (!q || q->spec().dtype != QNN_DATATYPE_FLOAT_16 || q->spec().dimensions.size() != 3) { return fail(); }
    const auto & qd = q->spec().dimensions;
    if (qd[2] % heads || qd[2] / heads == 0) { return fail(); }
    const uint32_t width = qd[2] / heads;
    uint32_t past = 0;
    if ((spec.inputs[5].value == null_tensor_id.value) != (spec.inputs[6].value == null_tensor_id.value)) { return fail(); }
    if (spec.inputs[5].value != null_tensor_id.value) {
        const auto * k = graph.find_tensor(spec.inputs[5]);
        if (!k || k->spec().dimensions.size() != 4) { return fail(); }
        past = k->spec().dimensions[2];
    }
    for (size_t i = 0; i < spec.inputs.size(); ++i) {
        if (spec.inputs[i].value == null_tensor_id.value) { if (i < 5) { return fail(); } continue; }
        const auto * input = graph.find_tensor(spec.inputs[i]);
        if (!input || (input->spec().role != QNN_TENSOR_TYPE_APP_WRITE && input->spec().role != QNN_TENSOR_TYPE_STATIC)) { return fail(); }
        const Qnn_DataType_t dtype = i == 1 || i == 2 ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_FLOAT_16;
        const std::vector<uint32_t> dims = i == 0 ? qd : i == 1 ? std::vector<uint32_t>{qd[0]} : i == 2 ? std::vector<uint32_t>{} :
            i < 5 ? std::vector<uint32_t>{qd[0], qd[1], kv_heads * width} : std::vector<uint32_t>{qd[0], kv_heads, past, width};
        if (i >= 7 || input->spec().dtype != dtype || input->spec().dimensions != dims) { return fail(); }
    }
    if (past > UINT32_MAX - qd[1]) { return fail(); }
    for (size_t i = 0; i < 3; ++i) {
        const auto * output = graph.find_tensor(spec.outputs[i]);
        if (!output || output->spec().dtype != QNN_DATATYPE_FLOAT_16 || output->spec().role != QNN_TENSOR_TYPE_APP_READ ||
            output->spec().dimensions != (i == 0 ? qd : std::vector<uint32_t>{qd[0], kv_heads, past + qd[1], width})) { return fail(); }
    }
    auto packet = std::unique_ptr<op_packet>(new op_packet()); packet->spec_ = spec;
    for (const auto & parameter : packet->spec_.parameters) {
        Qnn_Param_t p = QNN_PARAM_INIT; p.paramType = QNN_PARAMTYPE_SCALAR; p.name = parameter.name.c_str(); p.scalarParam = parameter.value;
        packet->parameters_.push_back(p);
    }
    for (auto id : spec.inputs) {
        Qnn_Tensor_t input = QNN_TENSOR_INIT;
        if (id.value == null_tensor_id.value) { input.v1.type = QNN_TENSOR_TYPE_NULL; input.v1.memType = QNN_TENSORMEMTYPE_RAW; input.v1.dataType = QNN_DATATYPE_FLOAT_16; }
        else { input = graph.find_tensor(id)->descriptor(); }
        packet->inputs_.push_back(input);
    }
    for (auto id : spec.outputs) { packet->outputs_.push_back(graph.find_tensor(id)->descriptor()); }
    return packet_result::success(std::move(packet));
}

op_spec op_spec::scatter_rows(tensor_id cache, tensor_id indices, tensor_id updates, tensor_id output) {
    Qnn_Scalar_t reduction = QNN_SCALAR_INIT; reduction.dataType = QNN_DATATYPE_UINT_32; reduction.uint32Value = QNN_OP_SCATTER_ND_REDUCTION_NONE;
    return {"scatter_rows", "qti.aisw", QNN_OP_SCATTER_ND, {cache, indices, updates}, {output}, {{QNN_OP_SCATTER_ND_PARAM_REDUCTION, reduction}}, {}};
}
result<std::unique_ptr<op_packet>> op_packet::create_scatter(const graph_owner & graph, const op_spec & spec) {
    using packet_result = result<std::unique_ptr<op_packet>>;
    auto fail = []() { return packet_result::failure(host_error(error_stage::op_validation, "ScatterNd packet", "Invalid cache scatter recipe")); };
    if (spec.name.empty() || spec.name.size() > 128 || spec.name.find('\0') != std::string::npos || spec.package_name != "qti.aisw" ||
        spec.inputs.size() != 3 || spec.outputs.size() != 1 || spec.parameters.size() != 1 || !spec.tensor_parameters.empty()) { return fail(); }
    const auto * data = graph.find_tensor(spec.inputs[0]); const auto * indices = graph.find_tensor(spec.inputs[1]);
    const auto * updates = graph.find_tensor(spec.inputs[2]); const auto * output = graph.find_tensor(spec.outputs[0]);
    for (const auto * input : {data, indices, updates}) { if (!input || (input->spec().role != QNN_TENSOR_TYPE_APP_WRITE && input->spec().role != QNN_TENSOR_TYPE_APP_READ && input->spec().role != QNN_TENSOR_TYPE_NATIVE && input->spec().role != QNN_TENSOR_TYPE_STATIC)) { return fail(); } }
    if (!output || data->spec().dtype != QNN_DATATYPE_FLOAT_16 || updates->spec().dtype != QNN_DATATYPE_FLOAT_16 || indices->spec().dtype != QNN_DATATYPE_INT_32 ||
        (output->spec().role != QNN_TENSOR_TYPE_APP_READ && output->spec().role != QNN_TENSOR_TYPE_NATIVE) || output->spec().dtype != QNN_DATATYPE_FLOAT_16 || data->spec().dimensions.size() != 2 ||
        indices->spec().dimensions.size() != 2 || indices->spec().dimensions[1] != 1 ||
        updates->spec().dimensions != std::vector<uint32_t>{indices->spec().dimensions[0], data->spec().dimensions[1]} || output->spec().dimensions != data->spec().dimensions ||
        spec.parameters[0].name != QNN_OP_SCATTER_ND_PARAM_REDUCTION || spec.parameters[0].value.dataType != QNN_DATATYPE_UINT_32 || spec.parameters[0].value.uint32Value != 0) { return fail(); }
    auto packet = std::unique_ptr<op_packet>(new op_packet()); packet->spec_ = spec;
    Qnn_Param_t parameter = QNN_PARAM_INIT; parameter.paramType = QNN_PARAMTYPE_SCALAR; parameter.name = packet->spec_.parameters[0].name.c_str(); parameter.scalarParam = packet->spec_.parameters[0].value;
    packet->parameters_.push_back(parameter);
    for (auto id : spec.inputs) { packet->inputs_.push_back(graph.find_tensor(id)->descriptor()); }
    packet->outputs_.push_back(output->descriptor()); return packet_result::success(std::move(packet));
}

op_spec op_spec::transpose(tensor_id input, tensor_id output, std::vector<uint32_t> permutation) {
    const uint32_t rank = uint32_t(permutation.size());
    return {"transpose", "qti.aisw", QNN_OP_TRANSPOSE, {input}, {output}, {}, {{QNN_OP_TRANSPOSE_PARAM_PERM, {rank}, std::move(permutation)}}};
}
op_spec op_spec::gather_rows(tensor_id data, tensor_id indices, tensor_id output) {
    Qnn_Scalar_t axis = QNN_SCALAR_INIT; axis.dataType = QNN_DATATYPE_INT_32; axis.int32Value = 0;
    return {"gather_rows", "qti.aisw", QNN_OP_GATHER, {data, indices}, {output}, {{QNN_OP_GATHER_PARAM_AXIS, axis}}, {}};
}
result<std::unique_ptr<op_packet>> op_packet::create_gather(const graph_owner & graph, const op_spec & spec) {
    using packet_result = result<std::unique_ptr<op_packet>>;
    auto fail = []() { return packet_result::failure(host_error(error_stage::op_validation,"Gather packet","Invalid row Gather recipe")); };
    if (spec.name.empty() || spec.name.size() > 128 || spec.name.find('\0') != std::string::npos || spec.package_name != "qti.aisw" ||
        spec.inputs.size()!=2 || spec.outputs.size()!=1 || spec.parameters.size()!=1 || !spec.tensor_parameters.empty() ||
        spec.parameters[0].name!=QNN_OP_GATHER_PARAM_AXIS || spec.parameters[0].value.dataType!=QNN_DATATYPE_INT_32 || spec.parameters[0].value.int32Value!=0) { return fail(); }
    const auto * data=graph.find_tensor(spec.inputs[0]);const auto * indices=graph.find_tensor(spec.inputs[1]);const auto * output=graph.find_tensor(spec.outputs[0]);
    if (!data || !indices || !output || data->spec().dtype!=QNN_DATATYPE_FLOAT_16 || output->spec().dtype!=QNN_DATATYPE_FLOAT_16 ||
        indices->spec().dtype!=QNN_DATATYPE_INT_32 || data->spec().dimensions.size()!=2 || indices->spec().dimensions.size()!=1 ||
        output->spec().dimensions!=std::vector<uint32_t>{indices->spec().dimensions[0],data->spec().dimensions[1]} ||
        (data->spec().role!=QNN_TENSOR_TYPE_APP_WRITE && data->spec().role!=QNN_TENSOR_TYPE_NATIVE && data->spec().role!=QNN_TENSOR_TYPE_APP_READ && data->spec().role!=QNN_TENSOR_TYPE_STATIC) || indices->spec().role!=QNN_TENSOR_TYPE_APP_WRITE || (output->spec().role!=QNN_TENSOR_TYPE_APP_READ && output->spec().role!=QNN_TENSOR_TYPE_NATIVE)) { return fail(); }
    auto packet=std::unique_ptr<op_packet>(new op_packet());packet->spec_=spec;
    Qnn_Param_t axis=QNN_PARAM_INIT;axis.paramType=QNN_PARAMTYPE_SCALAR;axis.name=packet->spec_.parameters[0].name.c_str();axis.scalarParam=packet->spec_.parameters[0].value;
    packet->parameters_.push_back(axis);packet->inputs_={data->descriptor(),indices->descriptor()};packet->outputs_={output->descriptor()};
    return packet_result::success(std::move(packet));
}

result<std::unique_ptr<op_packet>> op_packet::create(const graph_owner & graph, const op_spec & spec) {
    if (spec.type_name == QNN_OP_GATHER) { return create_gather(graph, spec); }
    if (spec.type_name == QNN_OP_SCATTER_ND) { return create_scatter(graph, spec); }
    if (spec.type_name == QNN_OP_GROUP_QUERY_ATTENTION) { return create_attention(graph, spec); }
    using packet_result = result<std::unique_ptr<op_packet>>;
    const bool reshape = spec.type_name == QNN_OP_RESHAPE;
    const bool slice = spec.type_name == QNN_OP_STRIDED_SLICE;
    const bool concat = spec.type_name == QNN_OP_CONCAT;
    if (reshape || slice || concat) {
        auto fail = []() { return packet_result::failure(host_error(error_stage::op_validation, "shape packet", "Invalid shape transformation")); };
        if (spec.name.empty() || spec.name.size() > 128 || spec.name.find('\0') != std::string::npos || spec.package_name != "qti.aisw" ||
            spec.inputs.empty() || spec.outputs.size() != 1 || (!concat && spec.inputs.size() != 1) ||
            spec.parameters.size() != (concat ? 1 : 0) || spec.tensor_parameters.size() != (slice ? 1 : 0)) { return fail(); }
        const auto * output = graph.find_tensor(spec.outputs[0]);
        if (!output || output->spec().dtype != QNN_DATATYPE_FLOAT_16 ||
            (output->spec().role != QNN_TENSOR_TYPE_NATIVE && output->spec().role != QNN_TENSOR_TYPE_APP_READ)) { return fail(); }
        const auto & yd = output->spec().dimensions;
        size_t total = 0;
        uint32_t axis = 0;
        if (concat) {
            const auto & value = spec.parameters[0];
            if (value.name != QNN_OP_CONCAT_PARAM_AXIS || value.value.dataType != QNN_DATATYPE_UINT_32 || value.value.uint32Value >= yd.size()) { return fail(); }
            axis = value.value.uint32Value;
        }
        for (auto id : spec.inputs) {
            const auto * input = graph.find_tensor(id);
            if (!input || input->spec().dtype != output->spec().dtype) { return fail(); }
            const auto & d = input->spec().dimensions;
            if (reshape) {
                auto in_bytes = input->required_bytes(), out_bytes = output->required_bytes();
                if (!in_bytes.ok() || !out_bytes.ok() || in_bytes.value() != out_bytes.value()) { return fail(); }
            } else if (slice) {
                const auto & ranges = spec.tensor_parameters[0];
                if (d.size() != yd.size() || ranges.dtype != QNN_DATATYPE_INT_32 || ranges.name != QNN_OP_STRIDED_SLICE_PARAM_RANGES ||
                    ranges.dimensions != std::vector<uint32_t>{uint32_t(d.size()), 3} || ranges.values.size() != d.size()*3) { return fail(); }
                for (size_t i = 0; i < d.size(); ++i) {
                    const auto begin = ranges.values[i*3], end = ranges.values[i*3+1], step = ranges.values[i*3+2];
                    if (!step || end <= begin || end > d[i] || yd[i] != (end-begin+step-1)/step) { return fail(); }
                }
            } else {
                if (d.size() != yd.size()) { return fail(); }
                for (size_t i = 0; i < d.size(); ++i) { if (i != axis && d[i] != yd[i]) { return fail(); } }
                total += d[axis];
            }
        }
        if (concat && total != yd[axis]) { return fail(); }
        auto packet = std::unique_ptr<op_packet>(new op_packet()); packet->spec_ = spec;
        for (auto id : spec.inputs) { packet->inputs_.push_back(graph.find_tensor(id)->descriptor()); }
        packet->outputs_.push_back(output->descriptor());
        for (size_t i = 0; i < spec.parameters.size(); ++i) {
            Qnn_Param_t value = QNN_PARAM_INIT; value.paramType = QNN_PARAMTYPE_SCALAR;
            value.name = packet->spec_.parameters[i].name.c_str(); value.scalarParam = packet->spec_.parameters[i].value;
            packet->parameters_.push_back(value);
        }
        for (auto & storage : packet->spec_.tensor_parameters) {
            Qnn_Param_t value = QNN_PARAM_INIT; value.paramType = QNN_PARAMTYPE_TENSOR; value.name = storage.name.c_str();
            value.tensorParam = QNN_TENSOR_INIT;
            packet->parameters_.push_back(value);
        }
        return packet_result::success(std::move(packet));
    }
    const bool matmul = spec.type_name == QNN_OP_MAT_MUL;
    const bool elementwise = spec.type_name == "ElementWiseAdd" || spec.type_name == "ElementWiseSubtract" ||
                             spec.type_name == "ElementWiseMultiply" || spec.type_name == "ElementWiseDivide";
    const bool norm = spec.type_name == QNN_OP_RMS_NORM || spec.type_name == QNN_OP_LAYER_NORM;
    const bool softmax = spec.type_name == QNN_OP_SOFTMAX;
    const bool sigmoid = spec.type_name == QNN_OP_SIGMOID;
    const bool rope = spec.type_name == QNN_OP_ROTARY_EMBEDDING;
    const bool transpose = spec.type_name == QNN_OP_TRANSPOSE;
    const bool unary = norm || softmax || sigmoid || rope || transpose;
    if (spec.name.empty() || spec.name.size() > 128 || spec.name.find('\0') != std::string::npos ||
        spec.package_name != "qti.aisw" || (!matmul && !elementwise && !unary) ||
        spec.inputs.size() != (norm || rope ? 3 : unary ? 1 : 2) || spec.outputs.size() != 1 ||
        spec.parameters.size() != (norm ? 1 : matmul || softmax || rope ? 2 : 0) ||
        spec.tensor_parameters.size() != (norm || transpose ? 1 : 0)) {
        return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Unsupported primitive recipe"));
    }
    const auto * a = graph.find_tensor(spec.inputs[0]);
    const auto * w = unary ? nullptr : graph.find_tensor(spec.inputs[1]);
    const auto * y = graph.find_tensor(spec.outputs[0]);
    auto input_role = [](const tensor_record * tensor) {
        return tensor && (tensor->spec().role == QNN_TENSOR_TYPE_APP_WRITE || tensor->spec().role == QNN_TENSOR_TYPE_STATIC ||
                          tensor->spec().role == QNN_TENSOR_TYPE_NATIVE || tensor->spec().role == QNN_TENSOR_TYPE_APP_READ);
    };
    const bool block_weight = matmul && w && w->spec().dtype == QNN_DATATYPE_SFIXED_POINT_8 &&
        w->spec().role == QNN_TENSOR_TYPE_STATIC && w->spec().quantization;
    if (!input_role(a) || (!unary && !input_role(w)) || !y ||
        (y->spec().role != QNN_TENSOR_TYPE_APP_READ && y->spec().role != QNN_TENSOR_TYPE_NATIVE) ||
        a->spec().dtype != y->spec().dtype || (!unary && (a->spec().dtype != QNN_DATATYPE_FLOAT_16 || (w->spec().dtype != a->spec().dtype && !block_weight)))) {
        return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Invalid primitive IDs, types or roles"));
    }
    const auto & ad = a->spec().dimensions;
    const auto & yd = y->spec().dimensions;
    if (ad.empty() || ad.size() > 4 || yd.size() != ad.size() ||
        (matmul && (ad.size() != (block_weight ? 4 : 2) || w->spec().dimensions.size() != ad.size())) ||
        (elementwise && w->spec().dimensions.size() != ad.size())) {
        return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Unsupported numerical rank"));
    }
    if (matmul) {
        const auto & wd = w->spec().dimensions;
        const size_t base = block_weight ? 2 : 0;
        if ((block_weight && (ad[0] != 1 || ad[1] != 1 || wd[0] != 1 || wd[1] != 1 || yd[0] != 1 || yd[1] != 1)) ||
            ad[base+1] != wd[base+1] || yd[base] != ad[base] || yd[base+1] != wd[base] ||
            spec.parameters[0].name != "transpose_in0" || spec.parameters[1].name != "transpose_in1" ||
            spec.parameters[0].value.dataType != QNN_DATATYPE_BOOL_8 || spec.parameters[1].value.dataType != QNN_DATATYPE_BOOL_8 ||
            spec.parameters[0].value.bool8Value != 0 || spec.parameters[1].value.bool8Value != 1) {
            return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Invalid MatMul shapes or parameters"));
        }
        matmul_spec shape;
        shape.m = ad[base]; shape.k = ad[base+1]; shape.n = wd[base];
        auto admitted = graph.context()->runtime()->policy().admit_matmul(shape);
        if (!admitted.ok()) { return packet_result::failure(admitted.error()); }
    } else if (unary) {
        if (transpose) {
            const auto & permutation = spec.tensor_parameters[0];
            if (permutation.name != QNN_OP_TRANSPOSE_PARAM_PERM || permutation.dimensions != std::vector<uint32_t>{uint32_t(ad.size())} || permutation.values.size() != ad.size()) {
                return packet_result::failure(host_error(error_stage::op_validation,"op_packet","Invalid Transpose permutation"));
            }
            std::vector<bool> used(ad.size(),false);
            for (size_t i=0;i<ad.size();++i) {
                const uint32_t axis=permutation.values[i];
                if (axis>=ad.size() || used[axis] || yd[i]!=ad[axis]) { return packet_result::failure(host_error(error_stage::op_validation,"op_packet","Invalid Transpose dimensions")); }
                used[axis]=true;
            }
        } else if (ad != yd) { return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Rowwise shapes differ")); }
        if (rope) {
            const auto & dim = spec.parameters[0]; const auto & pairs = spec.parameters[1];
            if (ad.size() != 4 || dim.name != QNN_OP_ROTARY_EMBEDDING_PARAM_ROTARY_EMBEDDING_DIM ||
                dim.value.dataType != QNN_DATATYPE_UINT_32 || dim.value.uint32Value < 2 ||
                dim.value.uint32Value > ad[3] || dim.value.uint32Value % 2 ||
                pairs.name != "interleaved" || pairs.value.dataType != QNN_DATATYPE_BOOL_8 || pairs.value.bool8Value > 1) {
                return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Invalid RoPE shape or parameters"));
            }
            for (size_t i = 1; i < 3; ++i) {
                const auto * table = graph.find_tensor(spec.inputs[i]);
                if (!input_role(table) || table->spec().dtype != a->spec().dtype ||
                    table->spec().dimensions != std::vector<uint32_t>{ad[0], ad[2], dim.value.uint32Value / 2}) {
                    return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Invalid selected RoPE table"));
                }
            }
        } else if (norm) {
            for (size_t i = 1; i < 3; ++i) {
                const auto * affine = graph.find_tensor(spec.inputs[i]);
                if (!input_role(affine) || affine->spec().dtype != a->spec().dtype || affine->spec().dimensions != std::vector<uint32_t>{ad.back()}) {
                    return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Invalid normalization affine input"));
                }
            }
            const auto & epsilon = spec.parameters[0]; const auto & axes = spec.tensor_parameters[0];
            if (epsilon.name != "epsilon" || epsilon.value.dataType != QNN_DATATYPE_FLOAT_32 ||
                !std::isfinite(epsilon.value.floatValue) || epsilon.value.floatValue <= 0.0f ||
                axes.name != "axes" || axes.dimensions != std::vector<uint32_t>{1} || axes.values != std::vector<uint32_t>{uint32_t(ad.size() - 1)}) {
                return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Invalid normalization parameters"));
            }
        } else if (softmax) {
            const auto & axis = spec.parameters[0]; const auto & beta = spec.parameters[1];
            if (axis.name != "axis" || axis.value.dataType != QNN_DATATYPE_UINT_32 || axis.value.uint32Value != ad.size() - 1 ||
                beta.name != "beta" || beta.value.dataType != QNN_DATATYPE_FLOAT_32 ||
                !std::isfinite(beta.value.floatValue) || beta.value.floatValue <= 0.0f) {
                return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Invalid Softmax parameters"));
            }
        }
    } else {
        const auto & wd = w->spec().dimensions;
        for (size_t i = 0; i < ad.size(); ++i) {
            if (yd[i] != ad[i] || (wd[i] != ad[i] && wd[i] != 1)) {
                return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Invalid elementwise broadcast"));
            }
        }
    }
    auto packet = std::unique_ptr<op_packet>(new op_packet());
    packet->spec_ = spec;
    packet->parameters_.resize(spec.parameters.size() + spec.tensor_parameters.size());
    for (size_t i = 0; i < spec.parameters.size(); ++i) {
        Qnn_Param_t parameter = QNN_PARAM_INIT;
        parameter.paramType = QNN_PARAMTYPE_SCALAR;
        parameter.name = packet->spec_.parameters[i].name.c_str();
        parameter.scalarParam = packet->spec_.parameters[i].value;
        packet->parameters_[i] = parameter;
    }
    for (size_t i = 0; i < spec.tensor_parameters.size(); ++i) {
        auto & storage = packet->spec_.tensor_parameters[i];
        Qnn_Param_t parameter = QNN_PARAM_INIT;
        parameter.paramType = QNN_PARAMTYPE_TENSOR; parameter.name = storage.name.c_str();
        parameter.tensorParam = QNN_TENSOR_INIT;
        parameter.tensorParam.v1.name = storage.name.c_str(); parameter.tensorParam.v1.type = QNN_TENSOR_TYPE_STATIC;
        parameter.tensorParam.v1.dataType = storage.dtype;
        parameter.tensorParam.v1.rank = uint32_t(storage.dimensions.size()); parameter.tensorParam.v1.dimensions = storage.dimensions.data();
        parameter.tensorParam.v1.memType = QNN_TENSORMEMTYPE_RAW;
        parameter.tensorParam.v1.clientBuf = {storage.values.data(), uint32_t(storage.values.size() * sizeof(uint32_t))};
        packet->parameters_[spec.parameters.size() + i] = parameter;
    }
    packet->inputs_.push_back(a->descriptor());
    for (size_t i = 1; i < spec.inputs.size(); ++i) { packet->inputs_.push_back(graph.find_tensor(spec.inputs[i])->descriptor()); }
    packet->outputs_ = {y->descriptor()};
    return packet_result::success(std::move(packet));
}
status op_packet::register_parameters(graph_owner & graph) {
    for (size_t i = 0; i < spec_.tensor_parameters.size(); ++i) {
        auto & storage = spec_.tensor_parameters[i];
        const buffer_view_const payload {storage.values.data(), storage.values.size() * sizeof(uint32_t)};
        auto id = graph.register_tensor({spec_.name + "_" + storage.name, QNN_TENSOR_TYPE_STATIC,
                                         storage.dtype, storage.dimensions}, payload);
        if (!id.ok()) { return status::failure(id.error()); }
        parameters_[spec_.parameters.size() + i].tensorParam = graph.find_tensor(id.value())->descriptor(payload);
    }
    return status::success();
}
Qnn_OpConfig_t op_packet::descriptor() noexcept {
    Qnn_OpConfig_t config = QNN_OPCONFIG_INIT;
    config.v1.name = spec_.name.c_str(); config.v1.packageName = spec_.package_name.c_str(); config.v1.typeName = spec_.type_name.c_str();
    config.v1.numOfParams = static_cast<uint32_t>(parameters_.size()); config.v1.params = parameters_.data();
    config.v1.numOfInputs = static_cast<uint32_t>(inputs_.size()); config.v1.inputTensors = inputs_.data();
    config.v1.numOfOutputs = static_cast<uint32_t>(outputs_.size()); config.v1.outputTensors = outputs_.data();
    return config;
}

status graph_owner::add_node(const op_spec & spec) {
    if (state_ != graph_state::building) { return status::failure(host_error(error_stage::node_addition, "graphAddNode", "Graph is not accepting a node")); }
    if (produced_ids_.size() >= 1024) { return status::failure(host_error(error_stage::node_addition, "graphAddNode", "Graph node limit")); }
    for (const auto id : spec.inputs) {
        if (spec.type_name == QNN_OP_GROUP_QUERY_ATTENTION && id.value == null_tensor_id.value) { continue; }
        const auto * tensor = find_tensor(id);
        if (!tensor) { return status::failure(host_error(error_stage::node_addition, "graphAddNode", "Unknown input")); }
        if (tensor->spec().role == QNN_TENSOR_TYPE_NATIVE || tensor->spec().role == QNN_TENSOR_TYPE_APP_READ) {
            bool produced = false;
            for (const auto previous : produced_ids_) { produced |= previous.value == id.value; }
            if (!produced) { return status::failure(host_error(error_stage::node_addition, "graphAddNode", "Input precedes its producer")); }
        }
    }
    for (const auto id : spec.outputs) {
        for (const auto previous : produced_ids_) {
            if (previous.value == id.value) { return status::failure(host_error(error_stage::node_addition, "graphAddNode", "Duplicate output producer")); }
        }
    }
    produced_ids_.reserve(produced_ids_.size() + spec.outputs.size());
    auto packet = op_packet::create(*this, spec);
    if (!packet.ok()) { return status::failure(packet.error()); }
    auto registered = packet.value()->register_parameters(*this);
    if (!registered.ok()) { return registered; }
    auto config = packet.value()->descriptor();
    auto runtime = context_->runtime();
    auto code = runtime->domain()->call([&](const auto & api) { return api.backendValidateOpConfig(runtime->backend(), config); });
    if (code != QNN_SUCCESS) { state_ = graph_state::failed; return status::failure(sdk_error(error_stage::op_validation, "backendValidateOpConfig", code)); }
    code = runtime->domain()->call([&](const auto & api) { return api.graphAddNode(graph_, config); });
    if (code != QNN_SUCCESS) { state_ = graph_state::failed; return status::failure(sdk_error(error_stage::node_addition, "graphAddNode", code)); }
    for (const auto id : spec.outputs) { produced_ids_.push_back(id); }
    node_added_ = true;
    return status::success();
}

static void measure_compiled_memory(const std::shared_ptr<context_owner> & context, uint64_t graph_id, const std::string & graph_name) {
    const char * enabled = std::getenv("GGML_QAIRT_MEMORY_PROBE");
    if (!enabled || std::strcmp(enabled, "1") != 0) { return; }
    try {
        const auto runtime = context->runtime();
        auto loaded = dll_module::load(runtime->config().library_directory / "QnnSystem.dll");
        if (!loaded.ok()) { throw std::runtime_error("QnnSystem.dll load failed"); }
        const auto module = loaded.value();
        auto resolved = module->resolve<decltype(&QnnSystemInterface_getProviders)>("QnnSystemInterface_getProviders");
        if (!resolved.ok()) { throw std::runtime_error("System provider resolution failed"); }
        const QnnSystemInterface_t ** providers = nullptr;
        uint32_t count = 0;
        if (resolved.value()(&providers, &count) != QNN_SUCCESS || !count) { throw std::runtime_error("No System provider"); }
        if (!providers || !providers[0]) { throw std::runtime_error("Invalid System provider"); }
        const auto & system = providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
        if (!system.systemContextCreate || !system.systemContextGetBinaryInfo || !system.systemContextFree) { throw std::runtime_error("Missing System metadata functions"); }
        Qnn_ContextBinarySize_t size = 0, written = 0;
        auto code = runtime->domain()->call([&](const auto & api) { return api.contextGetBinarySize ? api.contextGetBinarySize(context->handle(), &size) : QNN_COMMON_ERROR_NOT_SUPPORTED; });
        if (code != QNN_SUCCESS || !size) { throw std::runtime_error("contextGetBinarySize failed"); }
        std::vector<uint8_t> binary(size);
        code = runtime->domain()->call([&](const auto & api) { return api.contextGetBinary ? api.contextGetBinary(context->handle(), binary.data(), size, &written) : QNN_COMMON_ERROR_NOT_SUPPORTED; });
        if (code != QNN_SUCCESS || !written || written > size) { throw std::runtime_error("contextGetBinary failed"); }
        QnnSystemContext_Handle_t handle = nullptr;
        if (system.systemContextCreate(&handle) != QNN_SUCCESS) { throw std::runtime_error("systemContextCreate failed"); }
        const QnnSystemContext_BinaryInfo_t * info = nullptr;
        Qnn_ContextBinarySize_t info_size = 0;
        code = system.systemContextGetBinaryInfo(handle, binary.data(), written, &info, &info_size);
        if (code == QNN_SUCCESS && info) {
            GGML_LOG_INFO("qairt: inventory memory_probe graph=%llu serialized_bytes=%llu binary_version=%u\n",
                (unsigned long long) graph_id, (unsigned long long) written, (unsigned) info->version);
            if (info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
                const auto & details = info->contextBinaryInfoV3;
                for (uint32_t i = 0; i < details.numGraphs; ++i) {
                    const auto & graph = details.graphs[i];
                    if (graph.version != QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) { continue; }
                    const auto & graph_info = graph.graphInfoV3;
                    if (!graph_info.graphName || graph_name != graph_info.graphName) { continue; }
                    if (graph_info.graphBlobInfoSize < sizeof(QnnHtpSystemContext_GraphBlobInfo_t) + sizeof(QnnHtpSystemContext_GraphBlobInfoV2_t) || !graph_info.graphBlobInfo) { continue; }
                    QnnHtpSystemContext_GraphBlobInfo_t settings {};
                    QnnHtpSystemContext_GraphBlobInfoV2_t memory {};
                    std::memcpy(&settings, graph_info.graphBlobInfo, sizeof(settings));
                    // SDK stores V2 metadata after the V1 graph settings.
                    std::memcpy(&memory, static_cast<const uint8_t *>(graph_info.graphBlobInfo) + sizeof(settings), sizeof(memory));
                    GGML_LOG_INFO("qairt: inventory memory_probe graph=%llu spill_fill_bytes=%llu\n", (unsigned long long) graph_id, (unsigned long long) settings.contextBinaryGraphBlobInfoV1.spillFillBufferSize);
                    if (memory.version != QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V2) { continue; }
                    GGML_LOG_INFO("qairt: inventory memory_probe graph=%llu metadata_index=%u graph_name=%s opdata_bytes=%llu const_bytes=%llu ddr_tensor_bytes=%llu shared_weights_bytes=%llu io_bytes=%llu\n",
                        (unsigned long long) graph_id, i, graph_info.graphName ? graph_info.graphName : "unknown", (unsigned long long) memory.opDataSize, (unsigned long long) memory.constSize,
                        (unsigned long long) memory.ddrTensorSize, (unsigned long long) memory.sharedWeightsSize, (unsigned long long) memory.ioTensorSize);
                }
            }
        } else { GGML_LOG_WARN("qairt: inventory memory_probe metadata_error=%llu\n", (unsigned long long) code); }
        system.systemContextFree(handle);
    } catch (const std::exception & error) { GGML_LOG_WARN("qairt: inventory memory_probe unavailable: %s\n", error.what()); }
}

status graph_owner::finalize() {
    if (state_ != graph_state::building || !node_added_) { return status::failure(host_error(error_stage::graph_finalize, "graphFinalize", "Graph needs its primitive node")); }
    auto runtime = context_->runtime();
    if (diagnostic_phase()) {
        size_t counts[4] {}, sizes[4] {};
        for (const auto & tensor : tensors_) {
            const auto role = tensor->spec().role;
            const size_t slot = role == QNN_TENSOR_TYPE_STATIC ? 0 : role == QNN_TENSOR_TYPE_APP_WRITE ? 1 : role == QNN_TENSOR_TYPE_APP_READ ? 2 : 3;
            ++counts[slot]; sizes[slot] += tensor->required_bytes().value();
        }
        GGML_LOG_INFO("qairt: inventory finalize_begin phase=%s graph=%llu static_count=%zu static_bytes=%zu input_count=%zu input_bytes=%zu output_count=%zu output_bytes=%zu native_count=%zu native_logical_bytes=%zu\n",
            diagnostic_phase(), (unsigned long long) diagnostic_id_, counts[0], sizes[0], counts[1], sizes[1], counts[2], sizes[2], counts[3], sizes[3]);
    }
    auto code = runtime->domain()->call([&](const auto & api) { return api.graphFinalize(graph_, runtime->profile(), nullptr); });
    if (diagnostic_phase()) {
        GGML_LOG_INFO("qairt: inventory finalize_end phase=%s graph=%llu code=%llu\n", diagnostic_phase(),
            (unsigned long long) diagnostic_id_, (unsigned long long) code);
    }
    if (code == QNN_SUCCESS) { measure_compiled_memory(context_, diagnostic_id_, name_); }
    state_ = code == QNN_SUCCESS ? graph_state::finalized : graph_state::failed;
    return check_qnn(code, error_stage::graph_finalize, "graphFinalize");
}

result<std::unique_ptr<execution_binding>> execution_binding::create(std::shared_ptr<const graph_owner> graph) {
    using binding_result = result<std::unique_ptr<execution_binding>>;
    if (!graph || graph->state_ != graph_state::finalized || graph->input_ids_.empty() || graph->input_ids_.size() > 256 || graph->output_ids_.empty() || graph->output_ids_.size() > 128) {
        return binding_result::failure(host_error(error_stage::io_binding, "binding.create", "Graph is not finalized with supported I/O"));
    }
    auto binding = std::unique_ptr<execution_binding>(new execution_binding());
    binding->graph_ = std::move(graph);
    for (auto id : binding->graph_->input_ids_) { binding->inputs_.push_back(binding->graph_->find_tensor(id)->descriptor()); }
    for (auto id : binding->graph_->output_ids_) { binding->outputs_.push_back(binding->graph_->find_tensor(id)->descriptor()); }
    binding->input_buffers_.resize(binding->inputs_.size());
    binding->output_buffers_.resize(binding->outputs_.size());
    ++binding->graph_->active_bindings_;
    binding->admitted_ = true;
    return binding_result::success(std::move(binding));
}

execution_binding::~execution_binding() noexcept { if (admitted_) { --graph_->active_bindings_; } }

status execution_binding::bind_input(tensor_id id, buffer_view_const buffer) { return bind(id, buffer, false); }
status execution_binding::bind_output(tensor_id id, buffer_view buffer) { return bind(id, {buffer.data, buffer.bytes}, true); }

status execution_binding::bind_typed(tensor_id id, const void * data, size_t count, size_t element_size, Qnn_DataType_t dtype, bool output) {
    const auto * tensor = graph_->find_tensor(id);
    if (!tensor || tensor->spec().dtype != dtype || count > std::numeric_limits<size_t>::max() / element_size) {
        return status::failure(host_error(error_stage::io_binding, "bind_typed", "Invalid tensor type or element count"));
    }
    return bind(id, { data, count * element_size }, output);
}

status execution_binding::bind(tensor_id id, buffer_view_const buffer, bool output) {
    const auto * tensor = graph_->find_tensor(id);
    if (!tensor || tensor->spec().role != (output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_APP_WRITE)) {
        return status::failure(host_error(error_stage::io_binding, "bind", "Unknown ID or incorrect I/O role"));
    }
    auto checked = validate_buffer(*tensor, buffer);
    if (!checked.ok()) { return checked; }
    auto & descriptors = output ? outputs_ : inputs_;
    for (size_t i = 0; i < descriptors.size(); ++i) {
        if (descriptors[i].v1.id != id.value) { continue; }
        if ((output ? output_buffers_[i].data : input_buffers_[i].data)) {
            return status::failure(host_error(error_stage::io_binding, "bind", "Duplicate binding"));
        }
        descriptors[i] = tensor->descriptor(buffer);
        if (output) { output_buffers_[i] = { const_cast<void *>(buffer.data), buffer.bytes }; }
        else { input_buffers_[i] = buffer; }
        return status::success();
    }
    return status::failure(host_error(error_stage::io_binding, "bind", "Tensor is absent from graph I/O"));
}

static bool overlaps(buffer_view_const a, buffer_view_const b) {
    const auto pa = reinterpret_cast<uintptr_t>(a.data);
    const auto pb = reinterpret_cast<uintptr_t>(b.data);
    return pa < pb + b.bytes && pb < pa + a.bytes;
}

status execution_binding::validate_complete() const {
    for (size_t i = 0; i < inputs_.size(); ++i) {
        const auto * tensor = graph_->find_tensor({ inputs_[i].v1.id });
        if (!tensor) { return status::failure(host_error(error_stage::io_binding, "validate", "Unknown input ID")); }
        auto checked = validate_buffer(*tensor, input_buffers_[i]);
        if (!checked.ok()) { return checked; }
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
        const auto * tensor = graph_->find_tensor({ outputs_[i].v1.id });
        if (!tensor) { return status::failure(host_error(error_stage::io_binding, "validate", "Unknown output ID")); }
        buffer_view_const output {output_buffers_[i].data, output_buffers_[i].bytes};
        auto checked = validate_buffer(*tensor, output);
        if (!checked.ok()) { return checked; }
        for (auto input : input_buffers_) {
            if (overlaps(input, output)) { return status::failure(host_error(error_stage::io_binding, "validate", "Input/output buffers overlap")); }
        }
        for (size_t j = 0; j < i; ++j) {
            if (overlaps(output, {output_buffers_[j].data, output_buffers_[j].bytes})) {
                return status::failure(host_error(error_stage::io_binding, "validate", "Output buffers overlap"));
            }
        }
    }
    return status::success();
}

status graph_owner::execute(execution_binding & binding) const {
    if (state_ != graph_state::finalized || binding.graph_.get() != this) {
        return status::failure(host_error(error_stage::graph_execute, "graphExecute", "Binding belongs to another or unfinalized graph"));
    }
    auto checked = binding.validate_complete();
    if (!checked.ok()) { return checked; }
    auto runtime = context_->runtime();
    auto code = runtime->domain()->call([&](const auto & api) {
        return api.graphExecute(graph_, binding.inputs_.data(), static_cast<uint32_t>(binding.inputs_.size()),
            binding.outputs_.data(), static_cast<uint32_t>(binding.outputs_.size()), runtime->profile(), nullptr);
    });
    if (code != QNN_SUCCESS) { return status::failure(sdk_error(error_stage::graph_execute, "graphExecute", code)); }
    for (const auto & tensor : tensors_) {
        if (!tensor->dimensions_unchanged()) { return status::failure(host_error(error_stage::graph_execute, "graphExecute", "SDK modified fixed dimensions")); }
    }
    for (size_t i = 0; i < binding.outputs_.size(); ++i) {
        const auto & tensor = binding.outputs_[i];
        const auto * record = find_tensor(output_ids_[i]);
        const auto original = record->descriptor();
        if (tensor.version != QNN_TENSOR_VERSION_1 || tensor.v1.id != output_ids_[i].value ||
            tensor.v1.rank != original.v1.rank || tensor.v1.dimensions != original.v1.dimensions ||
            tensor.v1.dataType != original.v1.dataType || tensor.v1.type != original.v1.type ||
            tensor.v1.clientBuf.data != binding.output_buffers_[i].data || tensor.v1.clientBuf.dataSize != binding.output_buffers_[i].bytes) {
            return status::failure(host_error(error_stage::graph_execute, "graphExecute", "SDK modified fixed output metadata"));
        }
    }
    return status::success();
}

}
