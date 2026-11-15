#pragma once

#include "qairt-graph.h"
#include "qairt-lowering.h"
#include "ggml-qairt.h"
#include <memory>
#include <vector>

namespace ggml::qairt {

struct buffer_context {
    void * storage = nullptr;
    size_t size = 0;
    uint64_t identity = 0;
    uint64_t generation = 0;
    void touch() noexcept;
};

bool qairt_buffer(ggml_backend_buffer_t buffer) noexcept;
bool readable_tensor(const ggml_tensor * tensor) noexcept;

struct pending_output {
    ggml_tensor * tensor;
    std::vector<uint8_t> bytes;
};

struct prepared_primitive {
    std::vector<uint64_t> key;
    std::shared_ptr<graph_owner> graph;
    std::vector<tensor_id> input_ids;
    tensor_id output_id;
};

class backend_context final {
public:
    explicit backend_context(std::shared_ptr<runtime_owner> runtime) : runtime_(std::move(runtime)) {}
    backend_context(const backend_context &) = delete;
    backend_context & operator=(const backend_context &) = delete;
    enum ggml_status compute(ggml_cgraph * graph);
    void invalidate() noexcept { cache_.clear(); }
    ggml_backend_qairt_stats stats() const noexcept { return stats_; }
private:
    result<std::vector<fp16_bits>> pack(const ggml_tensor * tensor, const std::vector<pending_output> & pending) const;
    result<prepared_primitive *> prepare(const ggml_tensor * tensor, const std::vector<pending_output> & pending);
    std::shared_ptr<runtime_owner> runtime_;
    std::vector<std::unique_ptr<prepared_primitive>> cache_;
    ggml_backend_qairt_stats stats_ {};
};

}
