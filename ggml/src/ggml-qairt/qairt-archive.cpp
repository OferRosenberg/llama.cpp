#include "qairt-backend.h"
#include "ggml-impl.h"
#include <chrono>
#include <cstdlib>

namespace ggml::qairt {
void backend_context::invalidate() noexcept {
    cache_.clear();shared_context_.reset();
    for(auto & phase:archives_){phase.entries.clear();phase.context.reset();phase.saved=false;}
    archive_phase_=-1;preparing_=false;archives_loaded_=false;
}

status backend_context::archive(int action,ggml_cgraph * graph,const char * path) {
    const auto invalid=[](const char * message){return status::failure(host_error(error_stage::binary_restore,"phase archive",message));};
    if(action==GGML_QAIRT_ARCHIVE_BEGIN_PREFILL || action==GGML_QAIRT_ARCHIVE_BEGIN_DECODE) {
        const int phase=action==GGML_QAIRT_ARCHIVE_BEGIN_PREFILL?0:1;
        if(archive_phase_!=-1 || archives_loaded_ || !cache_.empty() || shared_context_ || archives_[phase].saved ||
           (phase==0 && archive_mode_) || (phase==1 && !archives_[0].saved)) { return invalid("Invalid phase construction order"); }
        const char * probe=std::getenv("GGML_QAIRT_MEMORY_PROBE");
        if(probe && std::string(probe)=="1"){return invalid("Disable per-graph memory serialization during phase compilation");}
        auto created=context_owner::create(runtime_);if(!created.ok()){return status::failure(created.error());}
        shared_context_=std::move(created).value();archive_mode_=true;preparing_=true;archive_phase_=phase;
        GGML_LOG_INFO("qairt: archive begin phase=%s context=%p\n",phase?"decode":"prefill",shared_context_->handle());
        return status::success();
    }
    if(action==GGML_QAIRT_ARCHIVE_PREPARE) {
        if(!preparing_ || archive_phase_<0 || !graph){return invalid("No construction phase for preparation");}
        if(compute(graph)!=GGML_STATUS_SUCCESS){return invalid("Phase graph preparation failed");}
        return status::success();
    }
    if(action==GGML_QAIRT_ARCHIVE_SAVE) {
        if(!preparing_ || archive_phase_<0 || !shared_context_ || cache_.empty() || !path || !*path){return invalid("No prepared phase to save");}
        auto & phase=archives_[archive_phase_];const auto start=std::chrono::steady_clock::now();
        auto saved=shared_context_->save_binary(std::filesystem::u8path(path));if(!saved.ok()){return status::failure(saved.error());}
        const auto binary_end = std::chrono::steady_clock::now();
        for(const auto & entry:cache_){const auto released=entry->graph->suspend();if(!released.ok()){invalidate();return released;}}
        const auto handle=shared_context_->handle();const auto freed=shared_context_->close();
        if(!freed.ok()){invalidate();return freed;}
        shared_context_.reset();phase.path=std::filesystem::u8path(path);phase.digest=std::move(saved).value();phase.entries=std::move(cache_);phase.saved=true;
        const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count();
        GGML_LOG_INFO("qairt: archive released phase=%s context=%p graphs=%zu save_release_ms=%lld\n",archive_phase_?"decode":"prefill",handle,phase.entries.size(),(long long)elapsed);
        GGML_LOG_INFO("qairt: archive timing phase=%s_binary binary_ms=%.3f release_ms=%.3f\n", archive_phase_ ? "decode" : "prefill",
            std::chrono::duration<double, std::milli>(binary_end - start).count(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - binary_end).count());
        archive_phase_=-1;preparing_=false;
        return status::success();
    }
    if(action==GGML_QAIRT_ARCHIVE_LOAD) {
        if(!archive_mode_ || archive_phase_!=-1 || preparing_ || archives_loaded_ || !archives_[0].saved || !archives_[1].saved) { return invalid("Both construction contexts must be saved and freed before loading"); }
        size_t phase_index = 0;
        for(auto & phase:archives_) {
            const auto start=std::chrono::steady_clock::now();auto loaded=context_owner::from_binary(runtime_,phase.path,phase.digest);
            if(!loaded.ok()){const auto error=loaded.error();invalidate();return status::failure(error);}
            phase.context=std::move(loaded).value();
            if(phase.context->binary_graphs().size()!=phase.entries.size()){invalidate();return invalid("Archive graph count mismatch");}
            for(const auto & entry:phase.entries) {
                auto restored=entry->graph->restore(phase.context);
                if(!restored.ok()){const auto error=restored.error();invalidate();return status::failure(error);}
                const auto remap=[&](tensor_id old){for(const auto & pair:restored.value()){if(pair.first.value==old.value){return pair.second;}}throw std::runtime_error("Missing loaded tensor binding");};
                for(auto & id:entry->outputs){id=remap(id);}
                for(auto & input:entry->partition->inputs){input.id=remap(input.id);}
                for(auto & rope:entry->partition->ropes){rope.table.cosine_input=remap(rope.table.cosine_input);rope.table.sine_input=remap(rope.table.sine_input);}
            }
            const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count();
            GGML_LOG_INFO("qairt: archive ready graphs=%zu load_retrieve_ms=%lld\n",phase.entries.size(),(long long)elapsed);
            GGML_LOG_INFO("qairt: archive timing phase=load_%s wall_ms=%.3f\n", phase_index++ ? "decode" : "prefill",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        archives_loaded_=true;return status::success();
    }
    if(action==GGML_QAIRT_ARCHIVE_SELECT_PREFILL || action==GGML_QAIRT_ARCHIVE_SELECT_DECODE) {
        if(!archives_loaded_ || preparing_){return invalid("Both archives must be loaded before execution");}
        const int phase=action==GGML_QAIRT_ARCHIVE_SELECT_PREFILL?0:1;
        if(archive_phase_==phase){return status::success();}
        if(archive_phase_>=0){archives_[archive_phase_].entries=std::move(cache_);}
        cache_=std::move(archives_[phase].entries);archive_phase_=phase;
        GGML_LOG_INFO("qairt: archive select phase=%s graphs=%zu\n",phase?"decode":"prefill",cache_.size());
        return status::success();
    }
    return invalid("Unknown archive action");
}
}