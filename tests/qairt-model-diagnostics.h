#pragma once
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Materialize logical elements before execution; scheduler copies and in-place ops
// may otherwise overwrite operands before the CPU reference can consume them.
inline std::vector<unsigned char> qairt_diagnostic_dense(const ggml_tensor * tensor) {
    const size_t bytes = ggml_nbytes(tensor);
    if (ggml_is_contiguous(tensor)) {
        std::vector<unsigned char> result(bytes);
        ggml_backend_tensor_get(tensor, result.data(), 0, bytes);
        return result;
    }
    const size_t element = ggml_type_size(tensor->type);
    GGML_ASSERT(!ggml_is_quantized(tensor->type));
    std::vector<unsigned char> result(size_t(ggml_nelements(tensor)) * element);
    size_t cursor = 0;
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3)
    for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2)
    for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
        const size_t offset = size_t(i3)*tensor->nb[3] + size_t(i2)*tensor->nb[2] + size_t(i1)*tensor->nb[1];
        if (tensor->nb[0] == element) {
            const size_t row = size_t(tensor->ne[0])*element;
            ggml_backend_tensor_get(tensor, result.data()+cursor, offset, row); cursor += row;
        } else {
            for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                ggml_backend_tensor_get(tensor, result.data()+cursor, offset+size_t(i0)*tensor->nb[0], element); cursor += element;
            }
        }
    }
    return result;
}

inline std::vector<float> qairt_diagnostic_values(const ggml_tensor * tensor) {
    auto bytes = qairt_diagnostic_dense(tensor);
    std::vector<float> result(size_t(ggml_nelements(tensor)));
    if (tensor->type == GGML_TYPE_F32) { std::memcpy(result.data(), bytes.data(), bytes.size()); }
    else {
        GGML_ASSERT(tensor->type == GGML_TYPE_F16);
        for (size_t i = 0; i < result.size(); ++i) {
            ggml_fp16_t value; std::memcpy(&value, bytes.data()+2*i, 2); result[i] = ggml_fp16_to_fp32(value);
        }
    }
    return result;
}

inline double qairt_diagnostic_nmse(const std::vector<float> & expected, const std::vector<float> & actual) {
    GGML_ASSERT(expected.size() == actual.size());
    double squared = 0, energy = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i])) { return INFINITY; }
        const double difference = double(actual[i])-expected[i];
        squared += difference*difference; energy += double(expected[i])*expected[i];
    }
    return squared/std::max(energy, 1e-12);
}

class qairt_local_diagnostics {
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    ggml_backend_t cpu;
    context_ptr context {nullptr, ggml_free};
    buffer_ptr buffer {nullptr, ggml_backend_buffer_free};
    ggml_tensor * reference = nullptr;
    ggml_cgraph * graph = nullptr;
    struct metric { size_t count=0; double maximum=0; };
    std::map<std::string, metric> metrics;
    size_t errors=0, checks=0;
    void record(const std::string & label, double error, bool gate) {
        auto & m=metrics[label]; ++m.count; m.maximum=std::max(m.maximum,error);
        if (gate) { ++checks; if (!(error <= 1e-5)) { ++errors; } }
        std::printf("local diagnostic %s: NMSE=%.9g gate=%d\n",label.c_str(),error,int(gate));
    }
    void conversions(ggml_tensor * tensor) {
        if (tensor->op != GGML_OP_MUL_MAT || !ggml_is_quantized(tensor->src[0]->type) ||
            tensor->src[1]->type != GGML_TYPE_F32) { return; }
        const auto * input=tensor->src[1];
        if (input->ne[0] % ggml_blck_size(GGML_TYPE_Q8_K)) { return; }
        const auto raw=qairt_diagnostic_values(input);
        auto quantized_values=raw, half_values=raw, cpu_values=raw;
        const size_t width=size_t(input->ne[0]);
        std::vector<block_q8_K> packed(width/size_t(ggml_blck_size(GGML_TYPE_Q8_K)));

        const auto * cpu_traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        for(size_t offset=0;offset<raw.size();offset+=width) {
            quantize_row_q8_K_ref(raw.data()+offset,packed.data(),input->ne[0]);
            dequantize_row_q8_K(packed.data(),quantized_values.data()+offset,input->ne[0]);
            if (!cpu_traits->from_float) { ++errors; return; }
            cpu_traits->from_float(raw.data()+offset,packed.data(),input->ne[0]);
            dequantize_row_q8_K(packed.data(),cpu_values.data()+offset,input->ne[0]);
        }
        for(size_t i=0;i<raw.size();++i) {half_values[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(quantized_values[i]));}
        const std::string name=ggml_get_name(tensor);
        record(name+" input_to_Q8_K",qairt_diagnostic_nmse(raw,quantized_values),false);
        record(name+" Q8_K_to_FP16",qairt_diagnostic_nmse(quantized_values,half_values),false);
        record(name+" CPU_vs_ref_Q8_K",qairt_diagnostic_nmse(cpu_values,quantized_values),false);
        const auto * weight=tensor->src[0];
        const size_t rows=std::min<size_t>(size_t(weight->ne[1]),16);
        const size_t row_bytes=ggml_row_size(weight->type,weight->ne[0]);
        std::vector<unsigned char> bytes(row_bytes);
        std::vector<float> floats(size_t(weight->ne[0])*rows), halves(floats.size());
        for(size_t row=0;row<rows;++row) {
            ggml_backend_tensor_get(weight,bytes.data(),row*weight->nb[1],row_bytes);
            ggml_get_type_traits(weight->type)->to_float(bytes.data(),floats.data()+row*size_t(weight->ne[0]),weight->ne[0]);
        }
        for(size_t i=0;i<floats.size();++i) {halves[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(floats[i]));}
        record(name+" sampled_weight_to_FP16",qairt_diagnostic_nmse(floats,halves),false);
    }
public:
    explicit qairt_local_diagnostics(ggml_backend_t backend):cpu(backend){}
    static void before(ggml_tensor * tensor, void * user) {static_cast<qairt_local_diagnostics *>(user)->prepare(tensor);}
    static void after(ggml_tensor * tensor, void * user) {static_cast<qairt_local_diagnostics *>(user)->finish(tensor);}
    void prepare(ggml_tensor * tensor) {
        buffer.reset(); context.reset(); reference=nullptr; graph=nullptr;
        context.reset(ggml_init({16*ggml_tensor_overhead()+ggml_graph_overhead_custom(1,false)+4096,nullptr,true}));
        if(!context) {++errors;return;}
        reference=ggml_dup_tensor(context.get(),tensor);
        reference->op=tensor->op; reference->flags=GGML_TENSOR_FLAG_COMPUTE;
        std::memcpy(reference->op_params,tensor->op_params,sizeof(reference->op_params));
        std::vector<std::pair<ggml_tensor *,std::vector<unsigned char>>> copies;
        // Preserve untouched rows for SET_ROWS; this is an independent cache snapshot.
        if(tensor->op==GGML_OP_SET_ROWS) {copies.emplace_back(reference,qairt_diagnostic_dense(tensor));}
        for(size_t i=0;i<GGML_MAX_SRC;++i) {
            auto * src=tensor->src[i]; if(!src) {continue;}
            if(ggml_is_quantized(src->type)) {reference->src[i]=src;continue;}
            reference->src[i]=ggml_dup_tensor(context.get(),src);
            copies.emplace_back(reference->src[i],qairt_diagnostic_dense(src));
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(),cpu));
        if(!buffer) {++errors;reference=nullptr;return;}
        for(const auto & copy:copies) {ggml_backend_tensor_set(copy.first,copy.second.data(),0,copy.second.size());}
        graph=ggml_new_graph_custom(context.get(),1,false); ggml_graph_add_node(graph,reference);
        conversions(tensor);
    }
    void finish(ggml_tensor * tensor) {
        if(!reference) {return;}
        if(ggml_backend_graph_compute(cpu,graph)!=GGML_STATUS_SUCCESS) {++errors;return;}
        const auto expected=qairt_diagnostic_values(reference),actual=qairt_diagnostic_values(tensor);
        const std::string name=std::string(ggml_op_desc(tensor))+" "+ggml_get_name(tensor);
        record(name,qairt_diagnostic_nmse(expected,actual),true);
        if(tensor->op==GGML_OP_RMS_NORM) {
            auto * input=reference->src[0];
            auto values=qairt_diagnostic_values(input);
            for(auto & value:values) {value=ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));}
            ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));
            if(ggml_backend_graph_compute(cpu,graph)!=GGML_STATUS_SUCCESS) {++errors;return;}
            auto rounded=qairt_diagnostic_values(reference);
            for(auto & value:rounded) {value=ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));}
            record(name+" simulated_FP16_boundary",qairt_diagnostic_nmse(expected,rounded),false);
            record(name+" HTP_vs_simulated_FP16",qairt_diagnostic_nmse(rounded,actual),false);
        }
    }
    void simulate_fp16_norm(ggml_tensor * tensor) {
        const auto expected = qairt_diagnostic_values(tensor);
        prepare(tensor); if (!reference) { return; }
        auto * input = reference->src[0];
        auto values = qairt_diagnostic_values(input);
        for (auto & value : values) { value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)); }
        ggml_backend_tensor_set(input, values.data(), 0, values.size()*sizeof(float));
        if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) { ++errors; return; }
        auto rounded = qairt_diagnostic_values(reference);
        for (auto & value : rounded) { value = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)); }
        record("CPU simulated norm " + std::string(ggml_get_name(tensor)), qairt_diagnostic_nmse(expected, rounded), true);
        ggml_backend_tensor_set(tensor, rounded.data(), 0, rounded.size()*sizeof(float));
    }
    bool report() const {
        for(const auto & item:metrics) {std::printf("local diagnostic maximum: %s count=%zu NMSE=%.9g\n",item.first.c_str(),item.second.count,item.second.maximum);}
        std::printf("local diagnostic totals: checks=%zu failures=%zu\n",checks,errors);
        return checks>0 && errors==0;
    }
};


struct qairt_quantization_snapshot {
    std::vector<float> raw, reconstructed;
    std::vector<block_q8_K> blocks;
};
inline qairt_quantization_snapshot qairt_snapshot_quantization(const ggml_tensor * input) {
    qairt_quantization_snapshot result;
    if (input->type != GGML_TYPE_F32 || input->ne[0] % ggml_blck_size(GGML_TYPE_Q8_K)) {return result;}
    result.raw=qairt_diagnostic_values(input); result.reconstructed=result.raw;
    const size_t width=size_t(input->ne[0]), block_size=size_t(ggml_blck_size(GGML_TYPE_Q8_K));
    result.blocks.resize(result.raw.size()/block_size);
    for(size_t offset=0;offset<result.raw.size();offset+=width) {
        quantize_row_q8_K_ref(result.raw.data()+offset,result.blocks.data()+offset/block_size,input->ne[0]);
        dequantize_row_q8_K(result.blocks.data()+offset/block_size,result.reconstructed.data()+offset,input->ne[0]);
    }
    return result;
}
inline void qairt_compare_quantization(const char * name, const qairt_quantization_snapshot & cpu,
                                      const qairt_quantization_snapshot & hybrid) {
    if(cpu.raw.empty() || cpu.raw.size()!=hybrid.raw.size()) {return;}
    size_t changed=0,scales=0;
    for(size_t b=0;b<cpu.blocks.size();++b) {
        scales+=cpu.blocks[b].d!=hybrid.blocks[b].d;
        for(size_t j=0;j<size_t(ggml_blck_size(GGML_TYPE_Q8_K));++j) {changed+=cpu.blocks[b].qs[j]!=hybrid.blocks[b].qs[j];}
    }
    std::printf("quantization propagation %s: raw_NMSE=%.9g Q8_K_NMSE=%.9g changed_codes=%zu/%zu changed_scales=%zu/%zu\n",
        name,qairt_diagnostic_nmse(cpu.raw,hybrid.raw),qairt_diagnostic_nmse(cpu.reconstructed,hybrid.reconstructed),
        changed,cpu.raw.size(),scales,cpu.blocks.size());
}

