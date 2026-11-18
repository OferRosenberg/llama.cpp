#include "qairt-graph.h"
#include "ggml.h"
#include <cstdio>
#include <vector>

using namespace ggml::qairt;

int main() {
    auto config = process_config::capture_once();
    if (!config.ok()) { return 1; }
    auto runtime = runtime_factory::create(config.value());
    if (!runtime.ok()) { std::fprintf(stderr, "%s\n", runtime.error().message.c_str()); return 1; }
    auto context = context_owner::create(runtime.value());
    if (!context.ok()) { return 1; }
    auto graph = graph_owner::create(context.value(), "m2_multiple_outputs");
    if (!graph.ok()) { return 1; }
    std::vector<tensor_id> inputs, outputs;
    for (int i = 0; i < 4; ++i) {
        auto id = graph.value()->register_tensor({"input" + std::to_string(i), QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, {2, 2}});
        if (!id.ok()) { return 1; }
        inputs.push_back(id.value());
    }
    for (int i = 0; i < 2; ++i) {
        auto id = graph.value()->register_tensor({"output" + std::to_string(i), QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, {2, 2}});
        if (!id.ok()) { return 1; }
        outputs.push_back(id.value());
    }
    for (int i = 0; i < 2; ++i) {
        auto op = op_spec::matmul(inputs[size_t(i) * 2], inputs[size_t(i) * 2 + 1], outputs[size_t(i)]);
        op.name = "matmul" + std::to_string(i);
        auto added = graph.value()->add_node(op);
        if (!added.ok()) { std::fprintf(stderr, "M2 graph: %s\n", added.error().message.c_str()); return 1; }
    }
    if (!graph.value()->finalize().ok()) { return 1; }
    auto binding = execution_binding::create(graph.value());
    if (!binding.ok()) { std::fprintf(stderr, "M2 binding: %s\n", binding.error().message.c_str()); return 1; }
    std::vector<std::vector<fp16_bits>> data(4, std::vector<fp16_bits>(4));
    std::vector<std::vector<fp16_bits>> result(2, std::vector<fp16_bits>(4));
    for (auto & input : data) { for (auto & value : input) { value.bits = ggml_fp32_to_fp16(1.0f); } }
    for (size_t i = 0; i < data.size(); ++i) {
        if (!binding.value()->bind_input(inputs[i], array_view<const fp16_bits>(data[i].data(), data[i].size())).ok()) { return 1; }
    }
    for (size_t i = 0; i < result.size(); ++i) {
        if (!binding.value()->bind_output(outputs[i], array_view<fp16_bits>(result[i].data(), result[i].size())).ok()) { return 1; }
    }
    if (!graph.value()->execute(*binding.value()).ok()) { return 1; }
    for (const auto & output : result) { for (const auto value : output) { if (ggml_fp16_to_fp32(value.bits) != 2.0f) { return 1; } } }
    std::printf("QAIRT M2 graph: two nodes, four inputs, two outputs passed\n");
    return 0;
}
