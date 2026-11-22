#include "llama-model.h"
#include "ggml-qairt.h"
#include "qairt-weights.h"
#include "qairt-fp16-reference.h"

#include <array>
#include <cstdlib>

using namespace ggml::qairt;
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
static int failures = 0;
#define CHECK(x) do { if (!(x)) { ++failures; std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); } } while (0)
static context_ptr context() { return {ggml_init({2 * 1024 * 1024,nullptr,true}),ggml_free}; }
static void logger(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) { std::fputs(text,stderr); }
}
struct weight {
    std::string name;
    ggml_type type;
    int64_t k, n;
    std::vector<uint8_t> data;
};
static std::array<weight,4> synthetic_weights(bool stress) {
    std::array<weight,4> result {{{"ffn_gate.weight",GGML_TYPE_Q4_0,128,256,{}},
        {"ffn_up.weight",GGML_TYPE_Q4_0,128,256,{}}, {"ffn_down.weight",GGML_TYPE_Q4_1,256,128,{}},
        {"ffn_norm.weight",GGML_TYPE_F32,128,1,{}}}};
    for (size_t j=0;j<result.size();++j) {
        auto & w=result[j]; std::vector<float> values(size_t(w.k*w.n));
        for(size_t i=0;i<values.size();++i) {
            values[i]=j==3 ? 1.0f+0.03f*std::sin(float(i)) :
                0.025f*std::sin(float((i*17+j*53)%4093)*0.113f)+0.006f*std::cos(float(i%31)*0.17f);
        }
        w.data.resize(ggml_row_size(w.type,w.k)*size_t(w.n));
        if(w.type==GGML_TYPE_F32) { std::memcpy(w.data.data(),values.data(),w.data.size()); }
        else if (stress) { ggml_quantize_chunk(w.type,values.data(),w.data.data(),0,w.n,w.k,nullptr); }
        else {
            // Exact FP16 block values isolate conversion correctness from cancellation sensitivity.
            const size_t bytes=ggml_type_size(w.type); uint32_t state=uint32_t(12345+j*7919);
            for(size_t b=0;b<w.data.size()/bytes;++b) {
                const float d=float(1+b%4)/512.0f;
                const auto delta=ggml_fp32_to_fp16(w.type==GGML_TYPE_Q4_0 && b%2 ? -d : d);
                std::memcpy(w.data.data()+b*bytes,&delta,2);
                if(w.type==GGML_TYPE_Q4_1) {
                    const auto minimum=ggml_fp32_to_fp16(-7.5f*d);
                    std::memcpy(w.data.data()+b*bytes+2,&minimum,2);
                }
                for(size_t k=0;k<16;++k) {
                    state=state*1664525u+1013904223u;
                    w.data[b*bytes+(w.type==GGML_TYPE_Q4_1?4:2)+k]=uint8_t(state>>24);
                }
            }
        }
    }
    return result;
}
static std::array<weight,4> model_weights(const char * path) {
    auto params=llama_model_default_params(); ggml_backend_dev_t devices[]={nullptr};
    params.devices=devices; params.n_gpu_layers=0; params.use_extra_bufts=false;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(path,params),llama_model_free);
    if(!model) { throw std::runtime_error("Cannot load model"); }
    std::array<weight,4> result;
    const char * names[]={"blk.2.ffn_gate.weight","blk.2.ffn_up.weight","blk.2.ffn_down.weight","blk.2.ffn_norm.weight"};
    for(size_t i=0;i<result.size();++i) {
        const ggml_tensor * t=nullptr;
        for (const auto & entry:model->tensors_by_name) { if (entry.first==names[i]) { t=entry.second; break; } }
        if(!t || !ggml_is_contiguous(t) || t->ne[2]!=1 || t->ne[3]!=1 ||
            t->ne[0]<=0 || t->ne[0]>8192 || t->ne[1]<=0 || t->ne[1]>8192) {
            throw std::runtime_error("Invalid model FFN tensor");
        }
        auto & w=result[i]; w.name=names[i]; w.type=t->type; w.k=t->ne[0]; w.n=t->ne[1];
        w.data.resize(ggml_nbytes(t)); ggml_backend_tensor_get(t,w.data.data(),0,w.data.size());
        std::printf("WEIGHT %s type=%s K=%lld N=%lld source_bytes=%zu\n",names[i],ggml_type_name(w.type),(long long)w.k,(long long)w.n,w.data.size());
    }
    if(result[0].type!=GGML_TYPE_Q4_0 || result[1].type!=GGML_TYPE_Q4_0 || result[2].type!=GGML_TYPE_Q4_1) {
        throw std::runtime_error("Expected Q4_0 gate/up and Q4_1 down");
    }
    return result;
}
struct output {
    std::array<std::vector<float>,7> values;
    ggml_backend_qairt_stats stats {};
};
static output execute(const std::array<weight,4> & source, int64_t tokens, bool cpu, bool fp16_weights) {
    backend_ptr backend(cpu ? ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU,nullptr) :
        ggml_backend_init_by_name("QAIRT0",nullptr),ggml_backend_free);
    if(!backend) { throw std::runtime_error("Cannot initialize backend"); }
    auto wc=context(), c=context(); std::array<ggml_tensor *,4> weights;
    for(size_t i=0;i<weights.size();++i) {
        const auto & w=source[i];
        weights[i]=ggml_new_tensor_2d(wc.get(),fp16_weights && ggml_is_quantized(w.type) ? GGML_TYPE_F16 : w.type,w.k,w.n);
        ggml_set_name(weights[i],w.name.c_str());
    }
    buffer_ptr wb(ggml_backend_alloc_ctx_tensors(wc.get(),backend.get()),ggml_backend_buffer_free);
    if(!wb) { throw std::runtime_error("Cannot allocate weights"); }
    ggml_backend_buffer_set_usage(wb.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for(size_t i=0;i<weights.size();++i) {
        const auto & w=source[i];
        if(fp16_weights && ggml_is_quantized(w.type)) {
            auto prepared=prepare_q4_weights({w.data.data(),w.data.size()},size_t(w.k),size_t(w.n),w.type);
            if(!prepared.ok()) { throw std::runtime_error(prepared.error().message); }
            ggml_backend_tensor_set(weights[i],prepared.value().data(),0,prepared.value().size()*2);
        } else { ggml_backend_tensor_set(weights[i],w.data.data(),0,w.data.size()); }
    }
    auto * x=ggml_new_tensor_2d(c.get(),GGML_TYPE_F32,source[0].k,tokens);
    ggml_set_name(x,"ffn_input"); ggml_set_input(x);
    std::array<ggml_tensor *,7> nodes;
    nodes[0]=ggml_rms_norm(c.get(),x,1e-6f);
    nodes[1]=ggml_mul(c.get(),nodes[0],weights[3]);
    nodes[2]=ggml_mul_mat(c.get(),weights[0],nodes[1]);
    nodes[3]=ggml_mul_mat(c.get(),weights[1],nodes[1]);
    nodes[4]=ggml_swiglu_split(c.get(),nodes[2],nodes[3]);
    nodes[5]=ggml_mul_mat(c.get(),weights[2],nodes[4]);
    nodes[6]=ggml_add(c.get(),nodes[5],x);
    const char * names[]={"rmsnorm","norm_scale","gate","up","swiglu","down","residual"};
    for(size_t i=0;i<nodes.size();++i) { ggml_set_name(nodes[i],names[i]); ggml_set_output(nodes[i]); }
    auto * graph=ggml_new_graph(c.get()); ggml_build_forward_expand(graph,nodes.back());
    buffer_ptr io(ggml_backend_alloc_ctx_tensors(c.get(),backend.get()),ggml_backend_buffer_free);
    if(!io) { throw std::runtime_error("Cannot allocate FFN tensors"); }
    std::vector<float> input(size_t(ggml_nelements(x)));
    for(size_t i=0;i<input.size();++i) { input[i]=fp16_reference::round(0.25f*std::sin(float(i%4001)*0.043f)); }
    ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
    output result;
    if(cpu) {
        fp16_reference reference;
        for(auto * node:nodes) { fp16_reference::write(node,reference.compute(node)); }
    } else {
        for(auto * node:nodes) { CHECK(ggml_backend_supports_op(backend.get(),node)); }
        for(int run=0;run<2;++run) {
            if(ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS) { throw std::runtime_error("FFN QNN execution failed"); }
        }
        const auto fn=reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get())),"ggml_backend_qairt_get_stats"));
        if(!fn || !fn(backend.get(),&result.stats)) { throw std::runtime_error("Cannot read QAIRT stats"); }
        CHECK(result.stats.preparations==1 && result.stats.cache_hits==1 && result.stats.executions==2);
        CHECK(result.stats.w4_executions==(fp16_weights?0:6));
        CHECK(result.stats.finalization_failures==0);
        std::printf("GRAPH fp16_weights=%d nodes=%d preparations=%llu cache_hits=%llu w4_executions=%llu finalize_ms=%.3f\n",
            fp16_weights,ggml_graph_n_nodes(graph),(unsigned long long)result.stats.preparations,
            (unsigned long long)result.stats.cache_hits,(unsigned long long)result.stats.w4_executions,
            result.stats.finalization_time_us/1000.0);
    }
    for(size_t i=0;i<nodes.size();++i) { result.values[i]=fp16_reference::read(nodes[i],false); }
    return result;
}
static void compare(const output & actual, const output & reference, const char * label) {
    const char * names[]={"rmsnorm","norm_scale","gate","up","swiglu","down","residual"};
    for(size_t j=0;j<actual.values.size();++j) {
        CHECK(actual.values[j].size()==reference.values[j].size());
        if(actual.values[j].size()!=reference.values[j].size()) { continue; }
        double error=0,energy=0,maximum=0;
        for(size_t i=0;i<actual.values[j].size();++i) {
            const double a=actual.values[j][i], r=reference.values[j][i],delta=a-r;
            CHECK(std::isfinite(a) && std::isfinite(r));
            error+=delta*delta; energy+=r*r; maximum=std::max(maximum,std::fabs(delta));
        }
        const double nmse=error/std::max(energy,1e-12);
        std::printf("ACCURACY %s checkpoint=%s NMSE=%.12g max_abs=%.12g\n",label,names[j],nmse,maximum);
        CHECK(std::isfinite(nmse) && nmse<=1e-5);
    }
}
int main(int argc,char ** argv) {
    try {
        const char * model=nullptr; int64_t tokens=1; bool stress=false;
        for(int i=1;i<argc;++i) {
            if(std::strcmp(argv[i],"--stress-sinus")==0) { stress=true; continue; }
            if(i+1==argc) { return 2; }
            if(std::strcmp(argv[i],"--model")==0) { model=argv[++i]; }
            else if(std::strcmp(argv[i],"--m")==0) { tokens=std::strtoll(argv[++i],nullptr,10); }
            else { return 2; }
        }
        if(tokens<1 || tokens>128) { return 2; }
        ggml_log_set(logger,nullptr); llama_log_set(logger,nullptr); llama_backend_init();
        const auto weights=model ? model_weights(model) : synthetic_weights(stress);
        std::printf("FFN model=%s tokens=%lld\n",model?model:stress?"synthetic cancellation stress":"synthetic exact FP16",(long long)tokens);
        const auto cpu=execute(weights,tokens,true,false);
        const auto baseline=execute(weights,tokens,false,true);
        const auto candidate=execute(weights,tokens,false,false);
        compare(baseline,cpu,"FP16_QNN_vs_CPU");
        compare(candidate,cpu,"W4_QNN_vs_CPU");
        compare(candidate,baseline,"W4_QNN_vs_FP16_QNN");
        std::printf("RESULT failures=%d\n",failures);
        return failures?1:0;
    } catch(const std::exception & error) { std::fprintf(stderr,"ERROR %s\n",error.what()); return 2; }
}