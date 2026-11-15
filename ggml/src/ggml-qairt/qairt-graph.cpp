#include "qairt-graph.h"

#include <limits>
#include <cstdio>

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
    if (state_ != graph_state::building || node_added_) {
        return result<tensor_id>::failure(host_error(error_stage::tensor_registration, "register_tensor", "Graph is not accepting tensors"));
    }
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
    if (duplicate || metadata_changed || descriptor.version != QNN_TENSOR_VERSION_1) {
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

result<std::unique_ptr<op_packet>> op_packet::create(const graph_owner & graph, const op_spec & spec) {
    using packet_result = result<std::unique_ptr<op_packet>>;
    if (spec.name.empty() || spec.name.size() > 128 || spec.name.find('\0') != std::string::npos ||
        spec.package_name != "qti.aisw" || spec.type_name != "MatMul" ||
        spec.inputs.size() != 2 || spec.outputs.size() != 1 || spec.parameters.size() != 2) {
        return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Only the M0 MatMul recipe is supported"));
    }
    const auto * a = graph.find_tensor(spec.inputs[0]);
    const auto * w = graph.find_tensor(spec.inputs[1]);
    const auto * y = graph.find_tensor(spec.outputs[0]);
    if (!a || !w || !y || a->spec().role != QNN_TENSOR_TYPE_APP_WRITE || w->spec().role != QNN_TENSOR_TYPE_STATIC ||
        y->spec().role != QNN_TENSOR_TYPE_APP_READ || a->spec().dimensions != std::vector<uint32_t>({2,4}) ||
        w->spec().dimensions != std::vector<uint32_t>({3,4}) || y->spec().dimensions != std::vector<uint32_t>({2,3}) ||
        spec.parameters[0].name != "transpose_in0" || spec.parameters[1].name != "transpose_in1" ||
        spec.parameters[0].value.dataType != QNN_DATATYPE_BOOL_8 || spec.parameters[1].value.dataType != QNN_DATATYPE_BOOL_8 ||
        spec.parameters[0].value.bool8Value != 0 || spec.parameters[1].value.bool8Value != 1) {
        return packet_result::failure(host_error(error_stage::op_validation, "op_packet", "Invalid MatMul IDs, shapes, roles or parameters"));
    }
    auto packet = std::unique_ptr<op_packet>(new op_packet());
    packet->spec_ = spec;
    packet->parameters_.resize(spec.parameters.size());
    for (size_t i = 0; i < spec.parameters.size(); ++i) {
        Qnn_Param_t parameter = QNN_PARAM_INIT;
        parameter.paramType = QNN_PARAMTYPE_SCALAR;
        parameter.name = packet->spec_.parameters[i].name.c_str();
        parameter.scalarParam = packet->spec_.parameters[i].value;
        packet->parameters_[i] = parameter;
    }
    packet->inputs_ = { a->descriptor(), w->descriptor() };
    packet->outputs_ = { y->descriptor() };
    return packet_result::success(std::move(packet));
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
    if (state_ != graph_state::building || node_added_) { return status::failure(host_error(error_stage::node_addition, "graphAddNode", "Graph is not accepting a node")); }
    auto admitted = context_->runtime()->policy().admit_matmul(matmul_spec{});
    if (!admitted.ok()) { return admitted; }
    auto packet = op_packet::create(*this, spec);
    if (!packet.ok()) { return status::failure(packet.error()); }
    auto config = packet.value()->descriptor();
    auto runtime = context_->runtime();
    auto code = runtime->domain()->call([&](const auto & api) { return api.backendValidateOpConfig(runtime->backend(), config); });
    if (code != QNN_SUCCESS) { state_ = graph_state::failed; return status::failure(sdk_error(error_stage::op_validation, "backendValidateOpConfig", code)); }
    code = runtime->domain()->call([&](const auto & api) { return api.graphAddNode(graph_, config); });
    if (code != QNN_SUCCESS) { state_ = graph_state::failed; return status::failure(sdk_error(error_stage::node_addition, "graphAddNode", code)); }
    node_added_ = true;
    return status::success();
}

status graph_owner::finalize() {
    if (state_ != graph_state::building || !node_added_) { return status::failure(host_error(error_stage::graph_finalize, "graphFinalize", "Graph needs its MatMul node")); }
    auto runtime = context_->runtime();
    auto code = runtime->domain()->call([&](const auto & api) { return api.graphFinalize(graph_, runtime->profile(), nullptr); });
    state_ = code == QNN_SUCCESS ? graph_state::finalized : graph_state::failed;
    return check_qnn(code, error_stage::graph_finalize, "graphFinalize");
}

result<std::unique_ptr<execution_binding>> execution_binding::create(std::shared_ptr<const graph_owner> graph) {
    using binding_result = result<std::unique_ptr<execution_binding>>;
    if (!graph || graph->state_ != graph_state::finalized || graph->input_ids_.size() != 1 || graph->output_ids_.size() != 1) {
        return binding_result::failure(host_error(error_stage::io_binding, "binding.create", "Graph is not finalized with the M0 I/O"));
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
