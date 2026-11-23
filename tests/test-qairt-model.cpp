#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include "qairt-weights.h"
#include "ggml-qairt.h"
#include "qairt-fp16-reference.h"
#include "llama-context.h"
#include <filesystem>
#include <fstream>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using timing_clock = std::chrono::steady_clock;
static double elapsed_ms(timing_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(timing_clock::now() - start).count();
}
static int failures = 0;
#define CHECK(x) do { if (!(x)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); } } while (0)
static std::atomic<bool> qnn_log_phase {false};
static bool set_probe_phase(const char * phase) {
#ifdef _WIN32
    return _putenv_s("GGML_QAIRT_DIAGNOSTIC_PHASE", phase ? phase : "") == 0;
#else
    return phase ? setenv("GGML_QAIRT_DIAGNOSTIC_PHASE", phase, 1) == 0 : unsetenv("GGML_QAIRT_DIAGNOSTIC_PHASE") == 0;
#endif
}
static void logger(ggml_log_level level, const char * text, void *) {
    if ((qnn_log_phase.load() && std::strstr(text, "qairt:")) || std::strstr(text, "qairt: inventory") || std::strstr(text, "qairt: archive ") || level == GGML_LOG_LEVEL_ERROR || std::strstr(text, "rejected ") || std::strstr(text, "qairt: executing") || std::strstr(text, "qairt: connected graph") || std::strstr(text, "qairt: partition span") || std::strstr(text, "qairt: W4FP16 initialized")) { std::fputs(text, stderr); }
}
struct observation;
static bool observe(ggml_tensor *, bool, void *);
static bool observe_backend(ggml_tensor *, void *);
class execution_observer final {
public:
    explicit execution_observer(ggml_backend_dev_t device, int layer, bool all, int layers = 1) : device_(device), init_(device->iface.init_backend), support_(device->iface.supports_op), layer_(layer), first_(layer - layers + 1), all_(all) {
        active_ = this; device_->iface.init_backend = initialize; device_->iface.supports_op = supports;
    }
    ~execution_observer() {
        device_->iface.init_backend = init_; device_->iface.supports_op = support_;
        for (auto & backend : original_) { backend.first->iface = backend.second; }
        active_ = nullptr;
    }
    void clear() { projections.clear(); operations.clear(); others = 0; graph_counts = {}; }
    bool archive(int action, ggml_cgraph * graph=nullptr, const char * path=nullptr) {
        if(original_.size()!=1){std::fprintf(stderr,"Archive experiment requires one QAIRT scheduler backend\n");return false;}
        auto * backend=original_.begin()->first;
        const auto fn=reinterpret_cast<ggml_backend_qairt_archive_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device_),"ggml_backend_qairt_archive"));
        if(!fn){std::fprintf(stderr,"Archive procedure missing action=%d\n",action);return false;}
        backend->iface.graph_compute=original_.at(backend).graph_compute;
        const bool result=fn(backend,action,graph,path);
        backend->iface.graph_compute=compute;
        if (!result) { std::fprintf(stderr, "Archive action=%d failed nodes=%d\n", action, graph ? ggml_graph_n_nodes(graph) : 0); }
        return result;
    }
    bool snapshot_stats(ggml_backend_qairt_stats & stats) {
        if (original_.size() != 1) { return false; }
        auto * backend = original_.begin()->first;
        const auto get_stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device_), "ggml_backend_qairt_get_stats"));
        const auto previous = backend->iface.graph_compute;
        backend->iface.graph_compute = original_.at(backend).graph_compute;
        const bool result = get_stats && get_stats(backend, &stats);
        backend->iface.graph_compute = previous;
        return result;
    }
    bool prepare_archives(llama_context * context, uint32_t tokens, const std::filesystem::path & directory) {
        if (!tokens || tokens > 128 || llama_n_ctx(context) != 256) { return false; }
        const char * diagnostic_setting = std::getenv("GGML_QAIRT_DIAGNOSTICS");
        const bool diagnostics = diagnostic_setting && std::strcmp(diagnostic_setting, "1") == 0;
        struct diagnostic_scope {
            std::string phase;
            bool logging;
            bool & capture;
            explicit diagnostic_scope(bool & captured) : logging(qnn_log_phase.load()), capture(captured) {
                const char * previous = std::getenv("GGML_QAIRT_DIAGNOSTIC_PHASE");
                phase = previous ? previous : "";
            }
            ~diagnostic_scope() {
                capture = false;
                qnn_log_phase.store(logging);
                CHECK(set_probe_phase(phase.empty() ? nullptr : phase.c_str()));
            }
        } diagnostic_guard(capture_);
        std::filesystem::create_directories(directory);
        std::ofstream manifest(directory / "manifest.txt", std::ios::trunc);
        manifest << "QAIRT phase archives v1\nblocks=" << (layer_ + 1) << "\nprefill_tokens=" << tokens << "\nkv_extent=256\n";
        for (int phase = 0; phase < 2; ++phase) {
            const char * name = phase ? "decode" : "prefill";
            if (diagnostics && !set_probe_phase(phase ? "compile_decode" : "compile_prefill")) { return false; }
            const auto context_start = timing_clock::now();
            if (!archive(phase ? GGML_QAIRT_ARCHIVE_BEGIN_DECODE : GGML_QAIRT_ARCHIVE_BEGIN_PREFILL)) { return false; }
            const double context_ms = elapsed_ms(context_start);
            const auto build_start = timing_clock::now();
            const auto memory = llama_get_memory(context)->init_full();
            captured_.clear(); capture_ = true;
            auto * graph = context->graph_reserve(phase ? 1 : tokens, 1, 1, memory.get());
            captured_.clear();
            const bool allocated = graph && ggml_backend_sched_alloc_graph(context->get_sched(), graph);
            capture_ = false;
            const double build_ms = elapsed_ms(build_start);
            if (diagnostics) {
                std::fprintf(stderr, "QNN archive capture phase=%s graph=%d allocated=%d splits=%d captured=%zu backends=%zu\n",
                    name, graph != nullptr, allocated, ggml_backend_sched_get_n_splits(context->get_sched()), captured_.size(), original_.size());
            }
            if (!allocated || captured_.empty()) { return false; }
            ggml_backend_qairt_stats before {}, after {};
            if (!snapshot_stats(before)) { return false; }
            const auto compile_start = timing_clock::now();
            size_t nodes = 0, partitions = 0;
            for (auto & split : captured_) {
                nodes += size_t(ggml_graph_n_nodes(&split));
                // Match scheduler evaluation boundaries without evaluating tensor data.
                for (int begin = 0; begin < split.n_nodes;) {
                    int end = begin;
                    while (end + 1 < split.n_nodes && !observe(split.nodes[end], true, fused_observation)) { ++end; }
                    auto view = split;
                    view.nodes = split.nodes + begin;
                    view.n_nodes = end + 1 - begin;
                    if (!archive(GGML_QAIRT_ARCHIVE_PREPARE, &view)) { return false; }
                    ++partitions;
                    begin = end + 1;
                }
            }
            const double compile_ms = elapsed_ms(compile_start);
            if (!snapshot_stats(after)) { return false; }
            CHECK(after.executions == before.executions);
            std::printf("QNN timing phase=compile_%s wall_ms=%.3f context_ms=%.3f graph_build_ms=%.3f backend_prepare_ms=%.3f finalize_ms=%.3f graphs=%llu executions=%llu\n",
                name, compile_ms, context_ms, build_ms,
                double(after.preparation_time_us - before.preparation_time_us) / 1000.0,
                double(after.finalization_time_us - before.finalization_time_us) / 1000.0,
                (unsigned long long) (after.preparations - before.preparations),
                (unsigned long long) (after.executions - before.executions));
            std::fflush(stdout);
            const auto path = (directory / (phase ? "decode.bin" : "prefill.bin")).u8string();
            captured_.clear();
            const auto save_start = timing_clock::now();
            if (!archive(GGML_QAIRT_ARCHIVE_SAVE, nullptr, path.c_str())) { return false; }
            const double save_ms = elapsed_ms(save_start);
            manifest << name << "_graphs=" << partitions << "\n" << name << "_compile_ms=" << compile_ms
                     << "\n" << name << "_archive_ms=" << save_ms << "\n";
            std::printf("QNN timing phase=archive_%s wall_ms=%.3f\n", name, save_ms);
            std::printf("QNN archive phase=%s graphs=%zu ggml_nodes=%zu bytes=%llu executions=0\n",
                name, partitions, nodes, (unsigned long long) std::filesystem::file_size(std::filesystem::u8path(path)));
            std::fflush(stdout);
        }
        ggml_backend_sched_reset(context->get_sched());
        llama_get_memory(context)->clear(true);
        if (diagnostics && !set_probe_phase("load_archives")) { return false; }
        if (diagnostics) { qnn_log_phase.store(true); }
        const auto load_start = timing_clock::now();
        const bool loaded = archive(GGML_QAIRT_ARCHIVE_LOAD);
        const double load_ms = elapsed_ms(load_start);
        qnn_log_phase.store(false);
        // diagnostic_guard restores the caller state on success and failure.
        manifest << "load_both_ms=" << load_ms << "\n";
        std::printf("QNN timing phase=load_both wall_ms=%.3f status=%s\n", load_ms, loaded ? "ok" : "failed");
        std::fflush(stdout);
        return loaded;
    }
    bool collect_observations = true;
    void * fused_observation = nullptr;
    ggml_backend_qairt_stats graph_counts {};
    std::map<std::string,int> operations;
    std::map<std::string, int> projections;
    int others = 0;
private:
    static bool selected(const ggml_tensor * t, int depth = 0) {
        if (!t || depth > 6) { return false; }
        const std::string name = ggml_get_name(t);
        for (int layer = active_->first_; layer <= active_->layer_; ++layer) {
            const std::string weight = "blk." + std::to_string(layer) + ".";
            if (name.find(weight) == 0) { return true; }
        }
        if (t->op==GGML_OP_GET_ROWS && t->src[0] && std::string(ggml_get_name(t->src[0]))=="l_out-"+std::to_string(active_->layer_-1)) { return true; }
        const auto dash = name.rfind('-');
        if (dash != std::string::npos && dash + 1 < name.size() && name[dash+1] >= '0' && name[dash+1] <= '9') {
            char * end = nullptr; const long layer = std::strtol(name.c_str()+dash+1,&end,10);
            if (end && (*end == '\0' || *end == ' ')) { return layer >= active_->first_ && layer <= active_->layer_; }
        }
        const auto cache = name.find("_l");
        if (cache != std::string::npos) {
            char * end = nullptr; const long layer = std::strtol(name.c_str()+cache+2,&end,10);
            if (end != name.c_str()+cache+2) { return layer >= active_->first_ && layer <= active_->layer_; }
        }
        for (const auto * source : t->src) { if (selected(source,depth+1)) { return true; } }
        return false;
    }
    static bool supports(ggml_backend_dev_t device, const ggml_tensor * t) {
        const bool admitted = active_->support_(device,t);
        if (!admitted) { return false; }
        if (t->op == GGML_OP_NONE) { return true; }
        if (!active_->all_) {
            return t->op==GGML_OP_RESHAPE || t->op==GGML_OP_VIEW || t->op==GGML_OP_TRANSPOSE || t->op==GGML_OP_PERMUTE ||
                   t->op==GGML_OP_CONT || t->op==GGML_OP_DUP || t->op==GGML_OP_CPY || (t->op==GGML_OP_MUL_MAT && (t->src[0]->type==GGML_TYPE_Q4_0 || t->src[0]->type==GGML_TYPE_Q4_1));
        }
        return selected(t);
    }
    static ggml_backend_t initialize(ggml_backend_dev_t device, const char * params) {
        auto backend = active_->init_(device, params);
        if (backend) {
            active_->original_[backend] = backend->iface;
            backend->iface.graph_compute = compute; backend->iface.free = release; backend->iface.graph_optimize = optimize;
        }
        return backend;
    }
    static void optimize(ggml_backend_t backend, ggml_cgraph * graph, ggml_backend_graph_optimize_params * params) {
        const auto original=active_->original_.at(backend).graph_optimize;
        if(original){original(backend,graph,params);}
        if(active_->capture_){active_->captured_.push_back(*graph);}
    }
    static ggml_status compute(ggml_backend_t backend, ggml_cgraph * graph) {
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        const auto get_stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
        ggml_backend_qairt_stats before {}, after {};
        backend->iface.graph_compute = active_->original_.at(backend).graph_compute;
        CHECK(get_stats && get_stats(backend, &before));
        const auto status = active_->original_.at(backend).graph_compute(backend, graph);
        CHECK(get_stats && get_stats(backend, &after));
        backend->iface.graph_compute = compute;
        active_->graph_counts.preparations += after.preparations - before.preparations;
        active_->graph_counts.cache_hits += after.cache_hits - before.cache_hits;
        active_->graph_counts.executions += after.executions - before.executions;
        active_->graph_counts.w4_preparations += after.w4_preparations - before.w4_preparations;
        active_->graph_counts.w4_executions += after.w4_executions - before.w4_executions;
        active_->graph_counts.preparation_time_us += after.preparation_time_us - before.preparation_time_us;
        active_->graph_counts.finalization_time_us += after.finalization_time_us - before.finalization_time_us;
        active_->graph_counts.finalization_failures += after.finalization_failures - before.finalization_failures;
        if (status == GGML_STATUS_SUCCESS && active_->collect_observations) {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * t = ggml_graph_node(graph, i);
                if (active_->fused_observation) { observe_backend(t, active_->fused_observation); }
                if (t->op == GGML_OP_MUL_MAT && (t->src[0]->type == GGML_TYPE_Q4_0 || t->src[0]->type == GGML_TYPE_Q4_1)) { ++active_->projections[ggml_get_name(t->src[0])]; }
                else if (t->op != GGML_OP_NONE && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_VIEW && t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_PERMUTE && t->op != GGML_OP_CONT && t->op != GGML_OP_DUP && t->op != GGML_OP_CPY) { ++active_->others; }
                if (t->op != GGML_OP_NONE && t->op != GGML_OP_RESHAPE && t->op != GGML_OP_VIEW && t->op != GGML_OP_TRANSPOSE && t->op != GGML_OP_PERMUTE) {
                    ++active_->operations[std::string(ggml_op_name(t->op)) + " " + ggml_get_name(t)];
                }
            }
        }
        return status;
    }
    static void release(ggml_backend_t backend) {
        const auto original = active_->original_.at(backend);
        active_->original_.erase(backend); original.free(backend);
    }
    inline static execution_observer * active_ = nullptr;
    ggml_backend_dev_t device_;
    decltype(ggml_backend_device_i::init_backend) init_;
    decltype(ggml_backend_device_i::supports_op) support_;
    int layer_;
    int first_;
    bool all_;
    std::map<ggml_backend_t, ggml_backend_i> original_;
    bool capture_ = false;
    std::vector<ggml_cgraph> captured_;
};


struct metrics { double nmse = 0, relative = 0, maximum = 0, kl = 0; };
static metrics measure(const std::vector<float> & reference, const std::vector<float> & actual, bool logits) {
    metrics result;
    if (reference.empty() || reference.size() != actual.size()) {
        result.nmse = result.relative = result.maximum = result.kl = HUGE_VAL; return result;
    }
    double error = 0, energy = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) {
            result.nmse = result.relative = result.maximum = result.kl = HUGE_VAL; return result;
        }
        const double delta = double(reference[i]) - actual[i];
        error += delta * delta; energy += double(reference[i]) * reference[i];
        result.maximum = std::max(result.maximum, std::fabs(delta));
    }
    result.nmse = error / std::max(energy, 1e-12); result.relative = std::sqrt(result.nmse);
    if (logits) {
        const double rmax = *std::max_element(reference.begin(), reference.end());
        const double amax = *std::max_element(actual.begin(), actual.end());
        double rsum = 0, asum = 0;
        for (size_t i = 0; i < actual.size(); ++i) { rsum += std::exp(reference[i] - rmax); asum += std::exp(actual[i] - amax); }
        const double rz = rmax + std::log(rsum), az = amax + std::log(asum);
        for (size_t i = 0; i < actual.size(); ++i) { result.kl += std::exp(reference[i] - rz) * (reference[i] - rz - actual[i] + az); }
    }
    return result;
}
static std::vector<float> download(ggml_tensor * tensor) {
    std::vector<float> result(size_t(ggml_nelements(tensor)));
    if (tensor->type == GGML_TYPE_F32) { ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * 4); }
    else {
        std::vector<ggml_fp16_t> half(result.size()); ggml_backend_tensor_get(tensor, half.data(), 0, half.size() * 2);
        for (size_t i = 0; i < half.size(); ++i) { result[i] = ggml_fp16_to_fp32(half[i]); }
    }
    return result;
}
struct observation {
    int layer = 0, first = 0;
    bool fused = false, in_backend = false, timing_only = false;
    fp16_reference * fp16 = nullptr;
    backend_ptr cpu {nullptr, ggml_backend_free};
    std::map<std::string, std::vector<float>> tensors;
    std::map<std::string, int> occurrences;
    std::map<std::string, std::vector<float>> projection_inputs;
    std::map<std::string, ggml_type> projection_types;
    int htp_projections = 0, other_htp = 0, local_count = 0;
    double local_worst = 0;
    void clear() { tensors.clear(); occurrences.clear(); projection_inputs.clear(); projection_types.clear(); htp_projections = other_htp = local_count = 0; local_worst = 0; }
};
static void local_projection(ggml_tensor * tensor, observation & seen) {
    if (seen.fp16) {
        const auto m = measure(seen.fp16->compute(tensor), download(tensor), false);
        ++seen.local_count; seen.local_worst = std::max(seen.local_worst, m.nmse);
        std::printf("local %s NMSE=%.9g max_abs=%.9g reference=FP16-emulated\n", ggml_get_name(tensor), m.nmse, m.maximum);
        CHECK(m.nmse <= 1e-5); return;
    }
    auto * c = ggml_init({1024 * 1024, nullptr, true});
    CHECK(c); if (!c) { return; }
    auto * output = ggml_dup_tensor(c, tensor);
    output->op = tensor->op; output->flags = GGML_TENSOR_FLAG_COMPUTE;
    std::memcpy(output->op_params, tensor->op_params, sizeof(output->op_params));
    std::copy(std::begin(tensor->src), std::end(tensor->src), std::begin(output->src));
    auto buffer = ggml_backend_alloc_ctx_tensors(c, seen.cpu.get());
    CHECK(buffer);
    if (buffer) {
        auto * graph = ggml_new_graph(c); ggml_graph_add_node(graph, output);
        CHECK(ggml_backend_graph_compute(seen.cpu.get(), graph) == GGML_STATUS_SUCCESS);
        const auto m = measure(download(output), download(tensor), false);
        ++seen.local_count; seen.local_worst = std::max(seen.local_worst, m.nmse);
        std::printf("local %s NMSE=%.9g max_abs=%.9g\n", ggml_get_name(tensor), m.nmse, m.maximum);
        CHECK(m.nmse <= 1e-5);
        ggml_backend_buffer_free(buffer);
    }
    ggml_free(c);
}
static bool observe(ggml_tensor * tensor, bool ask, void * user) {
    auto & seen = *static_cast<observation *>(user);
    if (!ask && seen.timing_only) { return true; }
    const std::string name = ggml_get_name(tensor), suffix = "-" + std::to_string(seen.layer);
    bool selected = false;
    for (int layer = seen.first; layer <= seen.layer; ++layer) {
        const auto ending = "-" + std::to_string(layer);
        selected |= name.size() >= ending.size() && name.compare(name.size() - ending.size(), ending.size(), ending) == 0;
    }
    const bool projection = tensor->op == GGML_OP_MUL_MAT && (tensor->src[0]->type == GGML_TYPE_Q4_0 || tensor->src[0]->type == GGML_TYPE_Q4_1);
    const std::string weight_prefix = "blk." + std::to_string(seen.layer) + ".";
    bool selected_projection = false;
    if (projection) { for (int layer = seen.first; layer <= seen.layer; ++layer) {
        selected_projection |= std::string(ggml_get_name(tensor->src[0])).find("blk." + std::to_string(layer) + ".") == 0;
    } }
    if (tensor->op == GGML_OP_MUL_MAT && tensor->src[0]) {
        const std::string weight_name = ggml_get_name(tensor->src[0]);
        for (int layer = seen.first; layer <= seen.layer; ++layer) {
            if (weight_name.find("blk." + std::to_string(layer) + ".") == 0) { seen.projection_types[weight_name] = tensor->src[0]->type; }
        }
    }
    const bool result = name == "result_norm" || name == "result_output";
    bool cache = false;
    if (tensor->op == GGML_OP_SET_ROWS && tensor->src[0]) { for (int layer = seen.first; layer <= seen.layer; ++layer) {
        const std::string ending = "-" + std::to_string(layer);
        cache |= std::strstr(ggml_get_name(tensor->src[0]), ending.c_str()) != nullptr;
    } }
    if ((!selected && !selected_projection && !result && !cache) || !ggml_is_contiguous(tensor) ||
        (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16) || ggml_nelements(tensor) > 4 * 1024 * 1024) { return false; }
    if (ask) {
        // Full-model validation uses bounded connected graphs at block boundaries.
        const bool boundary = seen.first != seen.layer && selected && name.find("l_out-") == 0 && name != "l_out-" + std::to_string(seen.layer);
        return !seen.fused || name == "result_output" || boundary;
    }
    if (seen.fused && !seen.in_backend && tensor->buffer && std::strcmp(ggml_backend_buffer_name(tensor->buffer), "QAIRT") == 0) { return true; }
    if (selected_projection && ggml_is_contiguous(tensor->src[1])) { seen.projection_inputs[ggml_get_name(tensor->src[0])] = download(tensor->src[1]); }
    const bool htp = tensor->buffer && std::strcmp(ggml_backend_buffer_name(tensor->buffer), "QAIRT") == 0;
    if (htp && projection) { ++seen.htp_projections; if (seen.cpu) { local_projection(tensor, seen); } }
    else if (htp && tensor->op != GGML_OP_NONE && tensor->op != GGML_OP_RESHAPE && tensor->op != GGML_OP_VIEW &&
             tensor->op != GGML_OP_TRANSPOSE && tensor->op != GGML_OP_PERMUTE && tensor->op != GGML_OP_CONT && tensor->op != GGML_OP_DUP && tensor->op != GGML_OP_CPY) { ++seen.other_htp; }
    const std::string identity = selected_projection ? std::string("projection:") + ggml_get_name(tensor->src[0]) : (cache ? "cache:" : "") + name;
    const std::string key = identity + "#" + std::to_string(seen.occurrences[identity]++);
    seen.tensors[key] = download(tensor);
    return true;
}

static bool observe_backend(ggml_tensor * tensor, void * user) {
    auto & seen = *static_cast<observation *>(user);
    seen.in_backend = true;
    const bool result = observe(tensor, false, user);
    seen.in_backend = false; return result;
}
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab, text.c_str(), int32_t(text.size()), nullptr, 0, true, true);
    if (n >= 0 || n == INT32_MIN) { return {}; }
    std::vector<llama_token> tokens(size_t(-n));
    n = llama_tokenize(vocab, text.c_str(), int32_t(text.size()), tokens.data(), int32_t(tokens.size()), true, true);
    if (n < 0) { return {}; }
    tokens.resize(size_t(n)); return tokens;
}
static int decode(llama_context * ctx, const std::vector<llama_token> & tokens, int position) {
    auto batch = llama_batch_init(int32_t(tokens.size()), 0, 1);
    batch.n_tokens = int32_t(tokens.size());
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[size_t(i)]; batch.pos[i] = position + i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = i == batch.n_tokens - 1;
    }
    const int status = llama_decode(ctx, batch); llama_batch_free(batch); return status;
}
static bool time_archive_execution(llama_context * context, const std::vector<llama_token> & prefix,
                                   const std::vector<llama_token> & decode_tokens, execution_observer & executed,
                                   observation & seen, int graphs_per_frame) {
    struct timing_scope {
        execution_observer & executed;
        observation & seen;
        bool collect, timing_only;
        timing_scope(execution_observer & e, observation & s) : executed(e), seen(s),
            collect(e.collect_observations), timing_only(s.timing_only) {
            executed.collect_observations = false;
            seen.timing_only = true;
        }
        ~timing_scope() { executed.collect_observations = collect; seen.timing_only = timing_only; }
    } scope(executed, seen);
    seen.clear(); executed.clear();
    llama_memory_clear(llama_get_memory(context), true);
    if (!executed.archive(GGML_QAIRT_ARCHIVE_SELECT_PREFILL)) { return false; }
    const auto prefill_start = timing_clock::now();
    const int prefill_status = decode(context, prefix, 0);
    const double prefill_ms = elapsed_ms(prefill_start);
    if (prefill_status != 0) { std::fprintf(stderr, "Timed prefill failed status=%d\n", prefill_status); return false; }
    std::printf("QNN throughput phase=prefill tokens=%zu elapsed_ms=%.3f tps=%.6f observations=off cache=loaded\n",
        prefix.size(), prefill_ms, prefill_ms > 0 ? double(prefix.size()) * 1000.0 / prefill_ms : 0);
    std::fflush(stdout);
    if (!executed.archive(GGML_QAIRT_ARCHIVE_SELECT_DECODE)) { return false; }
    const auto decode_start = timing_clock::now();
    int position = int(prefix.size());
    for (const auto token : decode_tokens) {
        const int status = decode(context, {token}, position++);
        if (status != 0) { std::fprintf(stderr, "Timed decode failed position=%d status=%d\n", position - 1, status); return false; }
    }
    const double decode_ms = elapsed_ms(decode_start);
    std::printf("QNN throughput phase=decode tokens=%zu elapsed_ms=%.3f tps=%.6f observations=off cache=loaded teacher_forced=1\n",
        decode_tokens.size(), decode_ms, decode_ms > 0 ? double(decode_tokens.size()) * 1000.0 / decode_ms : 0);
    const auto & stats = executed.graph_counts;
    std::printf("QNN throughput reuse graphs_created=%llu cache_hits=%llu executions=%llu finalization_failures=%llu\n",
        (unsigned long long) stats.preparations, (unsigned long long) stats.cache_hits,
        (unsigned long long) stats.executions, (unsigned long long) stats.finalization_failures);
    const uint64_t expected = uint64_t(graphs_per_frame) * (decode_tokens.size() + 1);
    const bool reused = stats.preparations == 0 && stats.finalization_failures == 0 &&
        stats.executions == expected && stats.cache_hits == expected && seen.tensors.empty() && seen.local_count == 0;
    CHECK(reused);
    std::fflush(stdout);
    return reused;
}
static context_ptr make_context(llama_model * model, observation & seen, int capacity, bool all = false) {
    auto p = llama_context_default_params();
    p.n_ctx = uint32_t(capacity); p.n_batch = p.n_ubatch = 128;
    p.n_seq_max = 1; p.n_threads = p.n_threads_batch = 8;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    p.type_k = p.type_v = GGML_TYPE_F16; p.offload_kqv = all; p.op_offload = all;
    p.cb_eval = observe; p.cb_eval_user_data = &seen;
    return {llama_init_from_model(model, p), llama_free};
}
static void compare_observations(const observation & cpu, const observation & actual, bool control, const char * label) {
    CHECK(cpu.tensors.size() == actual.tensors.size());
    double worst_relative = 0, worst_absolute = 0;
    std::string worst_name;
    int compared = 0, caches = 0;
    for (const auto & entry : cpu.tensors) {
        const auto found = actual.tensors.find(entry.first);
        CHECK(found != actual.tensors.end()); if (found == actual.tensors.end()) { continue; }
        caches += entry.first.find("cache:") == 0;
        const bool logits = entry.first.find("result_output") != std::string::npos;
        const auto m = measure(entry.second, found->second, logits);
        if (m.relative > worst_relative) { worst_relative = m.relative; worst_name = entry.first; }
        worst_absolute = std::max(worst_absolute, m.maximum);
        if (control) { CHECK(entry.second == found->second); }
        else {
            const bool passed = m.maximum <= 0.75 && (logits ? m.kl <= 0.001 : m.relative <= 0.02);
            if (!passed) {
                double half_floor = 0, peak = 0;
                for (float value : entry.second) {
                    peak = std::max(peak, std::fabs(double(value)));
                    half_floor = std::max(half_floor, std::fabs(double(value) - ggml_fp16_to_fp32(ggml_fp32_to_fp16(value))));
                }
                std::printf("checkpoint failure %s %s max_abs=%.9g relative=%.9g NMSE=%.9g KL=%.9g CPU_peak=%.9g FP16_round_floor=%.9g\n", label, entry.first.c_str(), m.maximum, m.relative, m.nmse, m.kl, peak, half_floor);
            }
            CHECK(passed);
        }
        ++compared;
    }
    CHECK(compared >= 7 && caches >= 2);
    std::printf("%s checkpoints=%d caches=%d max_abs=%.9g worst_relative=%.9g tensor=%s\n",
                label, compared, caches, worst_absolute, worst_relative, worst_name.c_str());
}
static void compare_activation_edges(const observation & cpu, const observation & actual, const char * label) {
    for (const auto & item : cpu.projection_inputs) {
        const auto found = actual.projection_inputs.find(item.first);
        CHECK(found != actual.projection_inputs.end()); if (found == actual.projection_inputs.end()) { continue; }
        CHECK(item.second.size() == found->second.size()); if (item.second.size() != found->second.size()) { continue; }
        std::vector<float> a(item.second.size()), b(found->second.size());
        size_t changed = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const auto ah = ggml_fp32_to_fp16(item.second[i]), bh = ggml_fp32_to_fp16(found->second[i]);
            changed += ah != bh; a[i] = ggml_fp16_to_fp32(ah); b[i] = ggml_fp16_to_fp32(bh);
        }
        const auto input = measure(item.second, found->second, false), packed = measure(a, b, false);
        std::printf("FP16 activation edge %s %s input_NMSE=%.9g prepared_NMSE=%.9g changed_values=%zu\n",
                    label, item.first.c_str(), input.nmse, packed.nmse, changed);
    }
}
struct options { bool control = false, all = true, fused = false, fp16 = false, prepare_probe = false; int layers = 1, prefill = 0, tail = 0, steps = 8, capacity = 1024; std::string prompt, context_binaries; };
static bool parse(int argc, char ** argv, options & options) {
    try {
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--fp16-reference") { options.fp16 = true; continue; }
            if (arg == "--qnn-prepare-probe") { options.prepare_probe = true; continue; }
            if (arg == "--fused") { options.fused = true; continue; }
            if (arg == "--cpu-control") { options.control = true; continue; }
            if (i + 1 >= argc) { return false; }
            const std::string value = argv[++i];
            if (arg == "--ops") { if (value != "projections" && value != "all") { return false; } options.all = value == "all"; continue; }
            if (arg == "--context-binaries") { options.context_binaries = value; continue; }
            if (arg == "--prompt") { options.prompt = value; continue; }
            if (arg == "--offload-last" && value == "all") { options.layers = -1; continue; }
            size_t used = 0; const int n = std::stoi(value, &used); if (used != value.size()) { return false; }
            if (arg == "--offload-last") { options.layers = n; }
            else if (arg == "--prefill") { options.prefill = n; }
            else if (arg == "--tail") { options.tail = n; }
            else if (arg == "--decode-steps") { options.steps = n; }
            else if (arg == "--ctx") { options.capacity = n; }
            else { return false; }
        }
    } catch (...) { return false; }
    return (!options.fp16 || options.all) && (!options.fused || options.all) && (options.layers == -1 || (options.layers >= 1 && options.layers <= 256)) && options.steps >= 1 && options.steps <= 128 && (options.capacity == 1024 || (!options.context_binaries.empty() && options.capacity == 256)) &&
           (options.context_binaries.empty() || (!options.control && options.fused && options.all && options.layers == -1 && !options.prefill && !options.tail && !options.prepare_probe)) &&
           (options.prefill == 0 || options.prefill == 128) && (options.tail == 0 || options.tail == 13);
}
static int fp16_reference_self_test() {
    ggml_backend_load_all();
    fp16_reference ref;
    auto * c = ggml_init({1024 * 1024, nullptr, true}); CHECK(c); if (!c) { return 1; }
    auto * w = ggml_new_tensor_2d(c, GGML_TYPE_Q4_0, 32, 2);
    auto * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 2);
    auto * product = ggml_mul_mat(c, w, x);
    auto * norm = ggml_rms_norm(c, x, 1e-6f);
    auto * gate = ggml_swiglu_split(c, x, x);
    auto * transposed = ggml_transpose(c, x);
    auto * contiguous = ggml_cont(c, transposed);
    auto * positions = ggml_new_tensor_1d(c, GGML_TYPE_I64, 2);
    auto * cache = ggml_new_tensor_2d(c, GGML_TYPE_F16, 32, 4);
    auto * scatter = ggml_set_rows(c, cache, x, positions);
    auto * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, 32, 2);
    auto * softmax = ggml_soft_max_ext(c, x, mask, 0.17677669f, 0.0f);
    auto * backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    auto * buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer);
    if (!buffer) { ggml_backend_free(backend); ggml_free(c); return 1; }
    std::vector<float> weights(64), input(64), masking(64, 0.0f);
    for (size_t i = 0; i < input.size(); ++i) { weights[i] = float(int(i % 13) - 6) * 0.137f; input[i] = float(int(i % 19) - 9) * 0.1731f; }
    std::vector<uint8_t> quantized(ggml_nbytes(w));
    ggml_get_type_traits(GGML_TYPE_Q4_0)->from_float_ref(weights.data(), quantized.data(), 64);
    ggml_backend_tensor_set(w, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * 4);
    const int64_t indices[] = {1, 3}; ggml_backend_tensor_set(positions, indices, 0, sizeof(indices));
    std::vector<ggml_fp16_t> state(128, ggml_fp32_to_fp16(0.625f)); ggml_backend_tensor_set(cache, state.data(), 0, state.size() * 2);
    masking[31] = masking[63] = -HUGE_VALF; ggml_backend_tensor_set(mask, masking.data(), 0, masking.size() * 4);
    const auto actual = ref.compute(product), half_weights = fp16_reference::read(w), half_inputs = fp16_reference::read(x);
    std::vector<float> expected(4);
    for (size_t row = 0; row < 2; ++row) { for (size_t column = 0; column < 2; ++column) {
        float sum = 0; for (size_t k = 0; k < 32; ++k) { sum += half_weights[row * 32 + k] * half_inputs[column * 32 + k]; }
        expected[column * 2 + row] = fp16_reference::round(sum);
    } }
    CHECK(actual == expected);
    const auto normalized = ref.compute(norm), activated = ref.compute(gate), permuted = ref.compute(contiguous), updated = ref.compute(scatter), probabilities = ref.compute(softmax);
    for (size_t i = 0; i < input.size(); ++i) {
        CHECK(normalized[i] == fp16_reference::round(normalized[i]));
        const float sigmoid = fp16_reference::round(1.0f / (1.0f + std::exp(-half_inputs[i])));
        CHECK(activated[i] == fp16_reference::round(fp16_reference::round(half_inputs[i] * sigmoid) * half_inputs[i]));
        CHECK(permuted[(i % 32) * 2 + i / 32] == half_inputs[i]);
        CHECK(updated[32 + i % 32 + (i / 32) * 64] == half_inputs[i]);
    }
    for (size_t i = 0; i < 32; ++i) { CHECK(updated[i] == 0.625f && updated[64 + i] == 0.625f); }
    CHECK(probabilities[31] == 0.0f && probabilities[63] == 0.0f);
    for (size_t row = 0; row < 2; ++row) { float sum = 0; for (size_t i = 0; i < 32; ++i) { sum += probabilities[row * 32 + i]; } CHECK(std::fabs(sum - 1.0f) < 0.001f); }
    ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(c);
    std::printf("FP16 reference self-test: failures=%d\n", failures); return failures ? 1 : 0;
}
int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--fp16-reference-self-test") == 0) { return fp16_reference_self_test(); }
    options opt;
    if (argc < 2 || !parse(argc, argv, opt)) {
        std::fprintf(stderr, "Usage: test-qairt-model MODEL [--cpu-control] [--fused] [--fp16-reference] [--qnn-prepare-probe] [--offload-last N|all] [--ops all|projections] [--prefill 128] [--tail 13] [--decode-steps N] [--prompt TEXT] [--ctx 1024] [--context-binaries DIR --ctx 256]\n");
        return 2;
    }
    if (opt.prepare_probe) { CHECK(set_probe_phase("init")); }
    llama_log_set(logger, nullptr); ggml_log_set(logger, nullptr); ggml_backend_load_all(); llama_backend_init();
    ggml_backend_dev_t cpu_devices[] = {nullptr};
    auto mp = llama_model_default_params(); mp.devices = cpu_devices; mp.n_gpu_layers = 0;
    model_ptr reference(llama_model_load_from_file(argv[1], mp), llama_model_free);
    CHECK(reference); if (!reference) { return 1; }
    const int last = llama_model_n_layer(reference.get()) - 1;
    CHECK(last >= 0);
    if (opt.layers == -1) { opt.layers = last + 1; }
    CHECK(opt.layers <= last + 1); if (opt.layers > last + 1) { return 2; }
    const int first = last - opt.layers + 1;
    std::printf("Model validation: blocks=%d selected=%d-%d hybrid_blocks=%d output_tokens=%d reference=%s\n", last + 1, first, last, opt.layers, opt.steps, opt.fp16 ? "FP16-emulated" : "ordinary-CPU");
    const auto * vocab = llama_model_get_vocab(reference.get());
    const int vocabulary = llama_vocab_n_tokens(vocab);
    ggml_backend_dev_t htp = opt.control ? nullptr : ggml_backend_dev_by_name("QAIRT0");
    CHECK(opt.control || htp); if (!opt.control && !htp) { return 1; }
    std::unique_ptr<execution_observer> executed;
    if (htp) { executed = std::make_unique<execution_observer>(htp,last,opt.all,opt.layers); }
    std::vector<std::string> patterns;
    for (int layer = first; layer <= last; ++layer) {
        if (opt.all) { patterns.push_back("^blk\\." + std::to_string(layer) + "\\..*"); }
        else { for (const char * name : {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"}) {
            patterns.push_back("^blk\\." + std::to_string(layer) + "\\." + name + "\\.weight$");
        } }
    }
    std::vector<llama_model_tensor_buft_override> overrides;
    if (htp) {
        for (const auto & pattern : patterns) { overrides.push_back({pattern.c_str(), ggml_backend_dev_buffer_type(htp)}); }
    }
    overrides.push_back({nullptr, nullptr});
    ggml_backend_dev_t devices[] = {htp, nullptr};
    mp.n_gpu_layers = !opt.control && opt.all ? opt.layers + 1 : 0;
    mp.devices = opt.control ? cpu_devices : devices; mp.tensor_buft_overrides = opt.control ? nullptr : overrides.data();
    model_ptr actual(llama_model_load_from_file(argv[1], mp), llama_model_free);
    CHECK(actual); if (!actual) { return 1; }
    observation cpu_seen, actual_seen; cpu_seen.layer = actual_seen.layer = last; cpu_seen.first = actual_seen.first = first;
    if (!opt.control) { actual_seen.cpu.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr)); CHECK(actual_seen.cpu); }
    std::unique_ptr<fp16_reference_execution> fp16;
    if (opt.fp16) { fp16 = std::make_unique<fp16_reference_execution>(last, opt.layers); actual_seen.fp16 = &fp16->reference(); }
    auto cpu_ctx = make_context(reference.get(), cpu_seen, opt.capacity);
    if (fp16 && !opt.control) { fp16->stop_capture(); }
    auto actual_ctx = make_context(actual.get(), actual_seen, opt.capacity, !opt.control && opt.all);
    if (fp16) { fp16->stop_capture(); }
    CHECK(cpu_ctx && actual_ctx); if (!cpu_ctx || !actual_ctx) { return 1; }
    if (opt.fused && !opt.control) { actual_seen.fused = true; executed->fused_observation = &actual_seen; }
    const std::string question = opt.prompt.empty() ? "What is 17 + 25? Answer with one short sentence." : opt.prompt;
    const std::string prompt = "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    std::printf("User prompt: %s\n", question.c_str());
    auto tokens = tokenize(vocab, prompt);
    CHECK(!tokens.empty()); if (tokens.empty()) { return 1; }
    if (opt.prefill) {
        const auto filler = tokenize(vocab, "The sky is blue. Water is wet. Arithmetic adds numbers. ");
        CHECK(!filler.empty()); if (filler.empty()) { return 1; }
        while (tokens.size() < size_t(opt.prefill + opt.tail)) { tokens.insert(tokens.end(), filler.begin(), filler.end()); }
        tokens.resize(size_t(opt.prefill + opt.tail));
    }
    const size_t prefix_size = opt.prefill ? size_t(opt.prefill) : tokens.size();
    std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + prefix_size);
    if(!opt.context_binaries.empty()) {
        if(prefix.size()+size_t(opt.steps)>256 || prefix.size()>128){std::fprintf(stderr,"Archive profile exceeds fixed 256-entry KV range\n");return 2;}
        if(!executed->prepare_archives(actual_ctx.get(),uint32_t(prefix.size()),std::filesystem::u8path(opt.context_binaries))) {
            std::fprintf(stderr,"QNN phase archive preparation/load failed before query execution\n");return 1;
        }
        std::printf("QNN archives loaded before first query execution\n");std::fflush(stdout);
    }
    std::vector<float> initial_cpu, initial_actual;
    int position = 0, disagreements = 0; bool decode_failed = false;
    llama_token actual_next = 0;
    auto frame = [&](const std::vector<llama_token> & input, const char * label, bool decode_step) {
        cpu_seen.clear(); actual_seen.clear();
        if (executed) { executed->clear(); }
        const size_t reference_before = fp16 ? fp16->nodes : 0;
        const int cpu_status = decode(cpu_ctx.get(), input, position);
        int q41_projections = 0;
        for (const auto & projection : cpu_seen.projection_types) {
            CHECK(projection.second == GGML_TYPE_Q4_0 || projection.second == GGML_TYPE_Q4_1);
            q41_projections += projection.second == GGML_TYPE_Q4_1;
        }
        CHECK(cpu_seen.projection_types.size() == size_t(7 * opt.layers));
        const int htp_projections = 7 * opt.layers;
        if (std::strcmp(label, "prefill") == 0) {
            std::printf("Projection inventory: Q4_0=%d Q4_1=%d all_HTP=%d\n", htp_projections-q41_projections, q41_projections, htp_projections);
            for (const auto & projection : cpu_seen.projection_types) {
                if (projection.second == GGML_TYPE_Q4_1) { std::printf("HTP projection: %s type=Q4_1\n", projection.first.c_str()); }
            }
        }
        if (executed) { CHECK(executed->projections.empty() && executed->others == 0); }
        if (opt.prepare_probe) {
            CHECK(set_probe_phase(label));
            qnn_log_phase.store(std::strcmp(label, "decode0") == 0);
            std::fprintf(stderr, "qairt: inventory phase_begin phase=%s\n", label);
        }
        if(!opt.context_binaries.empty() && !executed->archive(decode_step?GGML_QAIRT_ARCHIVE_SELECT_DECODE:GGML_QAIRT_ARCHIVE_SELECT_PREFILL)) { decode_failed=true; return llama_token(0); }
        const int actual_status = decode(actual_ctx.get(), input, position);
        if (opt.prepare_probe) {
            std::fprintf(stderr, "qairt: inventory phase_end phase=%s status=%d\n", label, actual_status);
            qnn_log_phase.store(false);
            CHECK(set_probe_phase("idle"));
        }
        if (executed) {
            actual_seen.htp_projections = 0; actual_seen.other_htp = executed->others;
            for (const auto & projection : executed->projections) {
                bool selected_weight = false;
                for (int layer = first; layer <= last; ++layer) { selected_weight |= projection.first.find("blk." + std::to_string(layer) + ".") == 0; }
                CHECK(selected_weight && projection.second == 1);
                actual_seen.htp_projections += projection.second;
            }
        }
        if (fp16) {
            const size_t nodes = fp16->nodes - reference_before;
            std::printf("%s FP16 reference emulated_nodes=%zu cached_weights=%zu scope=blocks%d-%d+final_norm\n", label, nodes, fp16->reference().cached_weights(), first, last);
            CHECK(nodes == size_t((26 * opt.layers + 4) * (opt.control ? 2 : 1)));
        }
        CHECK(cpu_status == 0 && actual_status == 0);
        if (cpu_status || actual_status) {
            if (executed) {
                std::printf("%s QNN graphs created=%llu cache_hits=%llu executions=%llu finalization_failures=%llu\n", label,
                            (unsigned long long) executed->graph_counts.preparations,
                            (unsigned long long) executed->graph_counts.cache_hits,
                            (unsigned long long) executed->graph_counts.executions,
                            (unsigned long long) executed->graph_counts.finalization_failures);
                std::printf("%s QNN compilation preparation_us=%llu finalization_us=%llu\n", label,
                            (unsigned long long) executed->graph_counts.preparation_time_us,
                            (unsigned long long) executed->graph_counts.finalization_time_us);
                std::printf("%s W4FP16 preparations=%llu executions=%llu\n", label,
                            (unsigned long long) executed->graph_counts.w4_preparations,
                            (unsigned long long) executed->graph_counts.w4_executions);
            }
            std::printf("Model failure: cpu_status=%d actual_status=%d failures=%d\n", cpu_status, actual_status, failures);
            std::fflush(stdout);
            std::fprintf(stderr, "Model decode failed; remaining comparisons invalid.\n");
            decode_failed = true; return llama_token(0);
        }
        position += int(input.size());
        const float * cpu = llama_get_logits_ith(cpu_ctx.get(), -1), * hybrid = llama_get_logits_ith(actual_ctx.get(), -1);
        CHECK(cpu && hybrid); if (!cpu || !hybrid) { return llama_token(0); }
        std::vector<float> a(cpu, cpu + vocabulary), b(hybrid, hybrid + vocabulary);
        const auto m = measure(a, b, true);
        const llama_token choice = llama_token(std::max_element(a.begin(), a.end()) - a.begin());
        const llama_token other = llama_token(std::max_element(b.begin(), b.end()) - b.begin());
        actual_next = other;
        if (decode_step) { disagreements += choice != other; }
        std::printf("%s logits max_abs=%.9g KL=%.9g relative=%.9g greedy=%d/%d HTP=%d other_HTP=%d local_NMSE=%.9g\n",
                    label, m.maximum, m.kl, m.relative, choice, other, actual_seen.htp_projections, actual_seen.other_htp, actual_seen.local_worst);
        if (opt.control) { CHECK(a == b); CHECK(actual_seen.htp_projections == 0); }
        else { CHECK(m.maximum <= 0.75 && m.kl <= 0.001); CHECK(actual_seen.htp_projections == htp_projections && actual_seen.local_count == htp_projections); }
        if (opt.control || !opt.all) { CHECK(actual_seen.other_htp == 0); }
        else {
            // Require every numerical operation in the selected dense block and final norm.
            CHECK(actual_seen.other_htp == 18 * opt.layers + 4);
            std::map<std::string, int> by_operation;
            for (const auto & op : executed->operations) {
                by_operation[op.first.substr(0, op.first.find(' '))] += op.second;
            }
            CHECK(by_operation["MUL_MAT"] == 9 * opt.layers);
            CHECK(by_operation["RMS_NORM"] == 4 * opt.layers + 1 && by_operation["MUL"] == 4 * opt.layers + 1);
            CHECK(by_operation["ROPE"] == 2 * opt.layers && by_operation["SOFT_MAX"] == opt.layers);
            CHECK(by_operation["GLU"] == opt.layers && by_operation["ADD"] == 2 * opt.layers);
            CHECK(by_operation["SET_ROWS"] == 2 * opt.layers && by_operation["GET_ROWS"] == 2);
        }
        if (executed) {
            if (opt.fused && opt.all) { CHECK(executed->graph_counts.executions == uint64_t(opt.layers)); }
            std::printf("%s QNN graphs created=%llu cache_hits=%llu executions=%llu\n", label,
                        (unsigned long long) executed->graph_counts.preparations,
                        (unsigned long long) executed->graph_counts.cache_hits,
                        (unsigned long long) executed->graph_counts.executions);
            std::printf("%s QNN compilation preparation_us=%llu finalization_us=%llu\n", label,
                        (unsigned long long) executed->graph_counts.preparation_time_us,
                        (unsigned long long) executed->graph_counts.finalization_time_us);
            if (std::strstr(ggml_backend_dev_description(htp),"W4FP16")) {
                CHECK(executed->graph_counts.w4_executions == uint64_t(htp_projections));
            }
            std::printf("%s W4FP16 preparations=%llu executions=%llu\n", label,
                        (unsigned long long) executed->graph_counts.w4_preparations,
                        (unsigned long long) executed->graph_counts.w4_executions);
        }
        if (executed && opt.all) {
            for (const auto & op : executed->operations) {
                const auto name = op.first.substr(0, op.first.find(' '));
                const bool host = name == "CONT" || name == "DUP" || name == "CPY";
                std::printf("%s %s count=%d\n", host ? "QAIRT host helper" : "HTP placement",
                            op.first.c_str(), op.second);
            }
        }
        compare_observations(cpu_seen, actual_seen, opt.control, label);
        if (!opt.control) { compare_activation_edges(cpu_seen, actual_seen, label); }
        if (initial_cpu.empty()) { initial_cpu = a; initial_actual = b; }
        std::fflush(stdout); return choice;
    };
    auto next = frame(prefix, "prefill", false);
    if (decode_failed) { return 1; }
    if (opt.tail) { next = frame({tokens.begin() + prefix_size, tokens.end()}, "tail13", false); }
    if (decode_failed) { return 1; }
    std::string response, actual_response;
    int output_disagreements = 0;
    std::vector<llama_token> performance_tokens;
    for (int step = 0; step < opt.steps; ++step) {
        output_disagreements += next != actual_next;
        char actual_piece[256]; const int actual_n = llama_token_to_piece(vocab, actual_next, actual_piece, sizeof(actual_piece), 0, false);
        if (actual_n > 0 && actual_n <= int(sizeof(actual_piece))) { actual_response.append(actual_piece, size_t(actual_n)); }
        char piece[256]; const int n = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, false);
        if (n > 0 && n <= int(sizeof(piece))) { response.append(piece, size_t(n)); }
        const std::string label = "decode" + std::to_string(step);
        performance_tokens.push_back(next);
        next = frame({next}, label.c_str(), true);
        if (decode_failed) { return 1; }
        if (opt.prepare_probe) {
            std::printf("QNN preparation probe: prefill_status=0 decode_status=0 decode_steps=1 failures=%d\n", failures);
            std::fflush(stdout);
            return failures ? 1 : 0;
        }
    }
    CHECK(disagreements <= 1);
    std::printf("teacher-forced response: %s\n", response.c_str());
    std::printf("hybrid teacher-forced response: %s\n", actual_response.c_str());
    std::printf("generated token comparison: tokens=%d disagreements=%d identical=%d\n", opt.steps, output_disagreements, int(response == actual_response));
    llama_memory_clear(llama_get_memory(cpu_ctx.get()), true); llama_memory_clear(llama_get_memory(actual_ctx.get()), true);
    position = 0; frame(prefix, "reset", false);
    if (decode_failed) { return 1; }
    const float * reset_cpu = llama_get_logits_ith(cpu_ctx.get(), -1), * reset_actual = llama_get_logits_ith(actual_ctx.get(), -1);
    CHECK(reset_cpu && reset_actual);
    if (reset_cpu && reset_actual) {
        CHECK(std::equal(initial_cpu.begin(), initial_cpu.end(), reset_cpu));
        CHECK(std::equal(initial_actual.begin(), initial_actual.end(), reset_actual));
    }
    if (!opt.context_binaries.empty()) {
        std::printf("QNN throughput replay begin: CPU comparisons and tensor observations disabled\n");
        std::fflush(stdout);
        if (!time_archive_execution(actual_ctx.get(), prefix, performance_tokens, *executed, actual_seen, opt.layers)) { return 1; }
    }
    std::printf("Qwen3 M2a1: CPU_control=%d layers=%d selected=%d prefill=%zu tail=%d decode=%d disagreements=%d failures=%d\n",
                int(opt.control), last + 1, last, prefix_size, opt.tail, opt.steps, disagreements, failures);
    return failures ? 1 : 0;
}
