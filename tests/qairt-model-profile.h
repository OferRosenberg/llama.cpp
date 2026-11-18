#pragma once

#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>

enum class qairt_block_half { outside, attention, ffn };
enum class qairt_block_profile { cpu, attention, ffn, block, full, front, back, norm, qkv, rope, state, tail, qk, q, k, v, kv, softmax, output, residual };
enum class qairt_stage : uint32_t { none = 0, norm = 1, q = 2, k = 4, v = 8, rope = 16, kv = 32, softmax = 64, copy = 128, output = 256, residual = 512 };

inline bool qairt_profile_name(const char * name, const char * marker) {
    const size_t size = std::strlen(marker);
    return std::strncmp(name, marker, size) == 0 && (name[size] == '\0' || name[size] == '#');
}

inline const char * qairt_half_name(qairt_block_half half) {
    switch (half) { case qairt_block_half::attention: return "attention"; case qairt_block_half::ffn: return "ffn"; default: return "outside"; }
}

inline bool qairt_parse_profile(const char * setting, qairt_block_profile & profile) {
    if (!setting) { profile = qairt_block_profile::full; return true; }
    const std::pair<const char *, qairt_block_profile> choices[] = {
        {"cpu", qairt_block_profile::cpu}, {"attention", qairt_block_profile::attention}, {"ffn", qairt_block_profile::ffn},
        {"block", qairt_block_profile::block}, {"full", qairt_block_profile::full},
        {"attention-front", qairt_block_profile::front}, {"attention-back", qairt_block_profile::back},
        {"norm", qairt_block_profile::norm}, {"qkv", qairt_block_profile::qkv}, {"rope", qairt_block_profile::rope},
        {"state", qairt_block_profile::state}, {"output-tail", qairt_block_profile::tail},
        {"qk", qairt_block_profile::qk}, {"q", qairt_block_profile::q}, {"k", qairt_block_profile::k}, {"v", qairt_block_profile::v},
        {"kv", qairt_block_profile::kv}, {"softmax", qairt_block_profile::softmax}, {"output", qairt_block_profile::output}, {"residual", qairt_block_profile::residual},
    };
    for (const auto & choice : choices) { if (std::strcmp(setting, choice.first) == 0) { profile = choice.second; return true; } }
    return false;
}

inline int qairt_named_layer(const char * name) {
    const char * number = nullptr;
    if (std::strncmp(name, "blk.", 4) == 0) { number = name + 4; }
    else if (const auto * suffix = std::strrchr(name, '-')) { number = suffix + 1; }
    else if (const auto * suffix = std::strstr(name, "_l")) { number = suffix + 2; }
    if (!number || *number < '0' || *number > '9') { return -1; }
    char * end = nullptr; const long layer = std::strtol(number, &end, 10);
    return layer >= 0 && layer <= 15 && (*end == '\0' || *end == '#' || *end == '.' || *end == ' ') ? int(layer) : -1;
}

inline int qairt_tensor_layer(const ggml_tensor * tensor) {
    std::vector<const ggml_tensor *> pending {tensor};
    std::unordered_set<const ggml_tensor *> visited;
    for (size_t i = 0; i < pending.size() && visited.size() <= 16384; ++i) {
        const auto * node = pending[i]; if (!node || !visited.insert(node).second) { continue; }
        const int layer = qairt_named_layer(ggml_get_name(node)); if (layer >= 0) { return layer; }
        if (node->op != GGML_OP_NONE) { for (const auto * src : node->src) { if (src) { pending.push_back(src); } } }
    }
    return -1;
}

inline qairt_block_half qairt_block_half_of(const ggml_tensor * tensor, int first_layer = 15) {
    if (!tensor) { return qairt_block_half::outside; }
    const char * root = ggml_get_name(tensor);
    if (qairt_profile_name(root, "norm") || std::strncmp(root, "result_", 7) == 0) { return qairt_block_half::outside; }
    const int layer = qairt_tensor_layer(tensor); if (layer < first_layer || layer > 15) { return qairt_block_half::outside; }
    char boundary[32], input[32];
    std::snprintf(boundary, sizeof(boundary), "ffn_inp-%d", layer);
    std::snprintf(input, sizeof(input), "l_out-%d", layer - 1);
    if (qairt_profile_name(root, boundary)) { return qairt_block_half::attention; }
    std::vector<const ggml_tensor *> pending {tensor};
    std::unordered_set<const ggml_tensor *> visited;
    while (!pending.empty()) {
        const auto * node = pending.back(); pending.pop_back();
        if (!node || !visited.insert(node).second) { continue; }
        if (visited.size() > 16384) { return qairt_block_half::outside; }
        const char * name = ggml_get_name(node);
        if (qairt_profile_name(name, boundary)) { return qairt_block_half::ffn; }
        if (qairt_profile_name(name, input) || node->op == GGML_OP_NONE) { continue; }
        for (const auto * src : node->src) { if (src) { pending.push_back(src); } }
    }
    return qairt_block_half::attention;
}

inline qairt_stage qairt_stage_of(const ggml_tensor * tensor, int first_layer = 15) {
    if (!tensor || qairt_block_half_of(tensor, first_layer) != qairt_block_half::attention) { return qairt_stage::none; }
    switch (tensor->op) {
        case GGML_OP_RMS_NORM: case GGML_OP_NORM: return qairt_stage::norm;
        case GGML_OP_ROPE: return qairt_stage::rope;
        case GGML_OP_SET_ROWS: return qairt_stage::kv;
        case GGML_OP_SOFT_MAX: return qairt_stage::softmax;
        case GGML_OP_CONT: case GGML_OP_DUP: case GGML_OP_CPY: return qairt_stage::copy;
        case GGML_OP_ADD: return qairt_stage::residual;
        case GGML_OP_MUL_MAT: {
            const char * name = ggml_get_name(tensor);
            if (std::strncmp(name, "Qcur-", 5) == 0) { return qairt_stage::q; }
            if (std::strncmp(name, "Kcur-", 5) == 0) { return qairt_stage::k; }
            if (std::strncmp(name, "Vcur-", 5) == 0) { return qairt_stage::v; }
            if (std::strncmp(name, "attn_out-", 9) == 0) { return qairt_stage::output; }
            return qairt_stage::none;
        }
        default: return qairt_stage::none;
    }
}

inline uint32_t qairt_profile_mask(qairt_block_profile mode) {
    switch (mode) {
        case qairt_block_profile::front: return 1 | 2 | 4 | 8 | 16;
        case qairt_block_profile::back: return 32 | 64 | 128 | 256 | 512;
        case qairt_block_profile::norm: return 1;
        case qairt_block_profile::qkv: return 2 | 4 | 8;
        case qairt_block_profile::rope: return 16;
        case qairt_block_profile::state: return 32 | 64;
        case qairt_block_profile::tail: return 128 | 256 | 512;
        case qairt_block_profile::qk: return 2 | 4;
        case qairt_block_profile::q: return 2; case qairt_block_profile::k: return 4; case qairt_block_profile::v: return 8;
        case qairt_block_profile::kv: return 32; case qairt_block_profile::softmax: return 64;
        case qairt_block_profile::output: return 256; case qairt_block_profile::residual: return 512;
        default: return 1023;
    }
}

inline bool qairt_profile_metadata(ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// Test-only admission and execution scope. Restore callbacks after all test contexts are freed.
class qairt_profile_filter {
    using supports_fn = bool (*)(ggml_backend_dev_t, const ggml_tensor *);
    using buft_fn = bool (*)(ggml_backend_dev_t, ggml_backend_buffer_type_t);
    using init_fn = ggml_backend_t (*)(ggml_backend_dev_t, const char *);
    using compute_fn = ggml_status (*)(ggml_backend_t, ggml_cgraph *);
    using observe_fn = void (*)(ggml_tensor *, void *);
    struct admission { qairt_block_half half; qairt_stage stage; };

    inline static qairt_profile_filter * active = nullptr;
    ggml_backend_dev_t device;
    qairt_block_profile mode;
    int first_layer;
    bool force_selected_ops = false;
    supports_fn original_supports;
    ggml_backend_dev_t cpu_device = nullptr;
    supports_fn original_cpu_supports = nullptr;
    buft_fn original_buft;
    init_fn original_init;
    compute_fn original_compute = nullptr;
    observe_fn before = nullptr, after = nullptr;
    void * observer = nullptr;
    std::map<std::string, size_t> executed;
    std::unordered_map<const ggml_tensor *, admission> admitted;
    size_t attention_nodes = 0, ffn_nodes = 0, outside_nodes = 0, forbidden_nodes = 0, replay_groups = 0;
    uint32_t observed_stages = 0;
    size_t stage_nodes[10] {};

    bool allows(admission info) const {
        return allows(info.half) && (mode <= qairt_block_profile::full || (qairt_profile_mask(mode) & uint32_t(info.stage)));
    }
    static bool supports(ggml_backend_dev_t dev, const ggml_tensor * tensor) {
        try {
            if (!active->original_supports(dev, tensor)) { return false; }
            const admission info {qairt_block_half_of(tensor, active->first_layer), qairt_stage_of(tensor, active->first_layer)};
            if (qairt_profile_metadata(tensor->op) ? !active->allows(info.half) : !active->allows(info)) { return false; }
            active->admitted[tensor] = info; return true;
        } catch (...) { return false; }
    }
    // Fine-grained experiments must not silently inherit their CPU producer's placement.
    // Reject only the selected operations that the real QAIRT backend supports.
    static bool cpu_supports(ggml_backend_dev_t dev, const ggml_tensor * tensor) {
        if (active->force_selected_ops && !qairt_profile_metadata(tensor->op) && supports(active->device, tensor)) { return false; }
        return active->original_cpu_supports(dev, tensor);
    }
    static bool supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
        return (active->original_buft && active->original_buft(dev, buft)) ||
            (active->mode > qairt_block_profile::full && ggml_backend_buft_is_host(buft));
    }
    static ggml_backend_t init(ggml_backend_dev_t dev, const char * params) {
        auto * backend = active->original_init(dev, params);
        if (backend) { active->original_compute = backend->iface.graph_compute; backend->iface.graph_compute = compute; }
        return backend;
    }
    static ggml_status compute(ggml_backend_t backend, ggml_cgraph * graph) {
        ggml_tensor * replay = nullptr; size_t count = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * tensor = ggml_graph_node(graph, i); if (!qairt_profile_metadata(tensor->op)) { replay = tensor; ++count; }
        }
        if (active->before && count == 1) { active->before(replay, active->observer); }
        if (active->before && count > 1) { ++active->replay_groups; std::fprintf(stderr, "local replay: unexpected group size=%zu\n", count); }
        const auto status = active->original_compute(backend, graph); if (status != GGML_STATUS_SUCCESS) { return status; }
        if (active->after && count == 1) { active->after(replay, active->observer); }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const auto * tensor = ggml_graph_node(graph, i); if (qairt_profile_metadata(tensor->op)) { continue; }
            const auto found = active->admitted.find(tensor);
            const admission info = found == active->admitted.end() ? admission {qairt_block_half::outside, qairt_stage::none} : found->second;
            if (info.half == qairt_block_half::attention) { ++active->attention_nodes; }
            else if (info.half == qairt_block_half::ffn) { ++active->ffn_nodes; } else { ++active->outside_nodes; }
            if (!active->allows(info)) { ++active->forbidden_nodes; }
            active->observed_stages |= uint32_t(info.stage);
            for (size_t j = 0; j < 10; ++j) { if (uint32_t(info.stage) == (1u << j)) { ++active->stage_nodes[j]; } }
            const std::string key = std::string(qairt_half_name(info.half)) + " " + ggml_op_desc(tensor) + " " + ggml_get_name(tensor);
            if (++active->executed[key] == 1) { std::printf("HTP profile operation: %s\n", key.c_str()); }
        }
        return status;
    }

public:
    qairt_profile_filter(ggml_backend_dev_t dev, qairt_block_profile profile, int first = 15) :
        device(dev), mode(profile), first_layer(first), original_supports(dev->iface.supports_op),
        original_buft(dev->iface.supports_buft), original_init(dev->iface.init_backend) {
        GGML_ASSERT(!active); active = this;
        if (mode > qairt_block_profile::full) {
            cpu_device = ggml_backend_dev_by_name("CPU");
            if (cpu_device) {
                original_cpu_supports = cpu_device->iface.supports_op;
                cpu_device->iface.supports_op = cpu_supports;
            }
        }
        device->iface.supports_op = supports; device->iface.supports_buft = supports_buft; device->iface.init_backend = init;
    }
    qairt_profile_filter(const qairt_profile_filter &) = delete;
    qairt_profile_filter & operator=(const qairt_profile_filter &) = delete;
    ~qairt_profile_filter() {
        if (cpu_device) { cpu_device->iface.supports_op = original_cpu_supports; }
        device->iface.supports_op = original_supports; device->iface.supports_buft = original_buft; device->iface.init_backend = original_init; active = nullptr;
    }
    bool allows(qairt_block_half half) const {
        return mode == qairt_block_profile::full || (mode == qairt_block_profile::block && half != qairt_block_half::outside) ||
            ((mode == qairt_block_profile::attention || mode > qairt_block_profile::full) && half == qairt_block_half::attention) ||
            (mode == qairt_block_profile::ffn && half == qairt_block_half::ffn);
    }
    bool allows_stage(qairt_stage stage) const { return allows(qairt_block_half::attention) && (qairt_profile_mask(mode) & uint32_t(stage)); }
    bool is_cpu() const { return mode == qairt_block_profile::cpu; }
    static void force_selected(bool enabled) { if (active) { active->force_selected_ops = enabled; } }
    static bool diagnostic_candidate(const ggml_tensor * tensor) {
        return active && !qairt_profile_metadata(tensor->op) && active->admitted.count(tensor);
    }
    void set_observer(observe_fn start, observe_fn finish, void * user) { before = start; after = finish; observer = user; }
    bool verified() const {
        if (forbidden_nodes || replay_groups) { return false; }
        if (mode == qairt_block_profile::cpu) { return attention_nodes + ffn_nodes + outside_nodes == 0; }
        if (mode == qairt_block_profile::attention) { return attention_nodes > 0 && ffn_nodes + outside_nodes == 0; }
        if (mode == qairt_block_profile::ffn) { return ffn_nodes > 0 && attention_nodes + outside_nodes == 0; }
        if (mode > qairt_block_profile::full) { return attention_nodes > 0 && ffn_nodes + outside_nodes == 0 && (observed_stages & qairt_profile_mask(mode)) == qairt_profile_mask(mode); }
        return attention_nodes > 0 && ffn_nodes > 0 && (mode == qairt_block_profile::full || outside_nodes == 0);
    }
    void report(const char * setting) const {
        std::printf("HTP profile totals: profile=%s attention=%zu ffn=%zu outside=%zu forbidden=%zu\n", setting, attention_nodes, ffn_nodes, outside_nodes, forbidden_nodes);
        std::printf("HTP stage totals: norm=%zu q=%zu k=%zu v=%zu rope=%zu kv=%zu softmax=%zu copy=%zu output=%zu residual=%zu replay_groups=%zu\n",
            stage_nodes[0], stage_nodes[1], stage_nodes[2], stage_nodes[3], stage_nodes[4], stage_nodes[5], stage_nodes[6], stage_nodes[7], stage_nodes[8], stage_nodes[9], replay_groups);
    }
};
