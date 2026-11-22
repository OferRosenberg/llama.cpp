#include "qairt-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <QnnOpDef.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstring>
#include <functional>
#include <limits>
#include <unordered_map>

namespace ggml::qairt {
namespace {
qairt_error partition_error(const char * message) {
    return host_error(error_stage::numerical_check, "connected partition", message);
}
template<class T> T checked(result<T> value) {
    if (!value.ok()) { throw value.error(); }
    return std::move(value).value();
}
void checked(status value) { if (!value.ok()) { throw value.error(); } }
std::vector<uint32_t> logical_shape(const ggml_tensor * t) {
    return {uint32_t(t->ne[3]), uint32_t(t->ne[2]), uint32_t(t->ne[1]), uint32_t(t->ne[0])};
}
std::vector<uint32_t> matrix_shape(const ggml_tensor * t) {
    return {uint32_t(ggml_nelements(t) / t->ne[0]), uint32_t(t->ne[0])};
}
bool constant_tensor(const ggml_tensor * t, const std::vector<pending_output> & pending) {
    const auto buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (!qairt_buffer(buffer) || ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) { return false; }
    const auto begin = reinterpret_cast<uintptr_t>(t->data);
    for (const auto & output : pending) {
        const auto other = reinterpret_cast<uintptr_t>(output.tensor->data);
        if ((other >= begin && other - begin < ggml_nbytes(t)) || (begin > other && begin - other < output.bytes.size())) { return false; }
    }
    return true;
}
struct tensor_inventory {
    std::vector<const ggml_tensor *> tensors;
    std::unordered_map<const ggml_tensor *, size_t> slots;
    size_t add(const ggml_tensor * t) {
        if (!t) { return SIZE_MAX; }
        const auto found = slots.find(t);
        if (found != slots.end()) { return found->second; }
        if (tensors.size() >= 1024) { throw partition_error("Tensor inventory limit"); }
        const size_t slot = tensors.size(); slots[t] = slot; tensors.push_back(t); return slot;
    }
    explicit tensor_inventory(const std::vector<ggml_tensor *> & nodes) {
        for (const auto * t : nodes) { add(t); }
        for (size_t i = 0; i < nodes.size(); ++i) {
            for (const auto * source : nodes[i]->src) {
                const auto * t = source;
                for (size_t depth = 0; t && depth < 32; ++depth) {
                    add(t);
                    if (!metadata_op(t->op) || !t->view_src) { break; }
                    t = t->view_src;
                }
            }
        }
    }
};

class partition_builder final {
public:
    partition_builder(prepared_run & entry, const std::vector<ggml_tensor *> & nodes, const tensor_inventory & inventory,
                      std::function<std::vector<fp16_bits>(const ggml_tensor *)> pack,
                      const std::vector<pending_output> & pending, bool w4fp16, bool app_write_weights) : w4fp16_(w4fp16), app_write_weights_(app_write_weights), entry_(entry), nodes_(nodes), inventory_(inventory), pack_(std::move(pack)), pending_(pending) {
        for (const auto * t : nodes) { members_[t] = true; }
    }
    void lower() {
        for (size_t i = 0; i < nodes_.size(); ++i) {
            auto * t = nodes_[i];
            const size_t first_op=entry_.partition->qnn_nodes;
            if (metadata_op(t->op)) { continue; }
            const auto y = tensor(logical_shape(t), QNN_TENSOR_TYPE_APP_READ);
            entry_.outputs.push_back(y); entry_.partition->output_nodes.push_back(i);
            const auto packed = t->op == GGML_OP_MUL_MAT ? native_weights(t->src[0]) : nullptr;
            auto a = packed ? null_tensor_id : input(t->src[0]);
            if (packed) { native_matmul(t,*packed,y); }
            else if (t->op == GGML_OP_FLASH_ATTN_EXT) { flash(i,t,a,y); }
            else if (t->op == GGML_OP_MUL_MAT) { matmul(t, a, input(t->src[1]), y); }
            else if (t->op == GGML_OP_ROPE) { rope(i, t, a, y); }
            else if (t->op == GGML_OP_SET_ROWS) {
                const auto cache = reshape(input(t->src[2]),matrix_shape(t->src[2]));
                const auto indices = external(t->src[1],partition_input_kind::scatter_indices,i,{uint32_t(t->src[1]->ne[0]),1});
                const auto scattered = tensor(matrix_shape(t));
                add(op_spec::scatter_rows(cache,indices,reshape(a,matrix_shape(t->src[0])),scattered));
                reshape(scattered,logical_shape(t),y);
                state_.push_back({t,y});
            }
            else if (t->op == GGML_OP_GET_ROWS) {
                const auto indices = external(t->src[1], partition_input_kind::indices, i, {uint32_t(t->src[1]->ne[0])});
                const auto rows = tensor(matrix_shape(t));
                add(op_spec::gather_rows(reshape(a, matrix_shape(t->src[0])), indices, rows));
                reshape(rows, logical_shape(t), y);
            } else if (t->op == GGML_OP_CONT || t->op == GGML_OP_DUP || t->op == GGML_OP_CPY) {
                reshape(a, logical_shape(t), y);
            } else if (t->op == GGML_OP_RMS_NORM || t->op == GGML_OP_NORM) {
                const uint32_t channels = uint32_t(t->ne[0]), rows = uint32_t(ggml_nelements(t)/channels), height = row_height(rows);
                const std::vector<uint32_t> shape {1, height, rows/height, channels};
                std::vector<fp16_bits> gamma(channels, {ggml_fp32_to_fp16(1.0f)}), beta(channels);
                const auto g = constant({channels}, gamma), b = constant({channels}, beta), normalized = tensor(shape);
                float epsilon; std::memcpy(&epsilon, t->op_params, sizeof(epsilon));
                add(op_spec::normalization(reshape(a, shape), g, b, normalized, t->op == GGML_OP_RMS_NORM, epsilon, 3));
                reshape(normalized, logical_shape(t), y);
            } else if (t->op == GGML_OP_SOFT_MAX) {
                float scale; std::memcpy(&scale, t->op_params, sizeof(scale));
                auto logits = a;
                if (scale != 1.0f) {
                    auto scaled = tensor(logical_shape(t));
                    add({"scale", "qti.aisw", "ElementWiseMultiply", {logits, constant({1,1,1,1}, {{ggml_fp32_to_fp16(scale)}})}, {scaled}, {}, {}});
                    logits = scaled;
                }
                if (t->src[1]) {
                    const auto * mask = t->src[1];
                    const std::vector<uint32_t> mask_shape {uint32_t(mask->ne[3]),uint32_t(mask->ne[2]),uint32_t(t->ne[1]),uint32_t(mask->ne[0])};
                    tensor_id id;
                    if (internal(mask)) {
                        id=input(mask);
                        if(logical_shape(mask)!=mask_shape){id=slice(id,{0,0,0,0},mask_shape);}
                    }else{id=external(mask,partition_input_kind::mask,i,mask_shape);}
                    const auto masked = tensor(logical_shape(t));
                    add({"mask", "qti.aisw", "ElementWiseAdd", {logits,id}, {masked}, {}, {}}); logits = masked;
                }
                add(op_spec::softmax(logits, y, 1.0f, 3));
            } else if (silu_op(t) || swiglu_op(t)) {
                const auto sigmoid = tensor(logical_shape(t)); add(op_spec::sigmoid(a,sigmoid));
                if (swiglu_op(t)) {
                    const auto gate = tensor(logical_shape(t));
                    add({"gate", "qti.aisw", "ElementWiseMultiply", {a,sigmoid}, {gate}, {}, {}});
                    add({"up", "qti.aisw", "ElementWiseMultiply", {gate,input(t->src[1])}, {y}, {}, {}});
                } else { add({"silu", "qti.aisw", "ElementWiseMultiply", {a,sigmoid}, {y}, {}, {}}); }
            } else if (elementwise_type(t->op)) {
                add({"elementwise", "qti.aisw", elementwise_type(t->op), {a,input(t->src[1])}, {y}, {}, {}});
            } else { throw partition_error("Operation requires a separate recipe"); }
            values_[t] = y;
            entry_.partition->spans.push_back({i,first_op,entry_.partition->qnn_nodes-first_op});
        }
    }
private:
    tensor_id tensor(std::vector<uint32_t> shape, Qnn_TensorType_t role = QNN_TENSOR_TYPE_NATIVE) {
        return checked(entry_.graph->register_tensor({"tensor"+std::to_string(next_tensor_++),role,QNN_DATATYPE_FLOAT_16,std::move(shape)}));
    }
    tensor_id constant(std::vector<uint32_t> shape, const std::vector<fp16_bits> & payload) {
        entry_.static_bytes += payload.size()*2;
        return checked(entry_.graph->register_tensor({"constant"+std::to_string(next_tensor_++),QNN_TENSOR_TYPE_STATIC,QNN_DATATYPE_FLOAT_16,std::move(shape)},
                                                     {payload.data(),payload.size()*2}));
    }
    void add(op_spec spec) {
        spec.name = "op"+std::to_string(entry_.partition->qnn_nodes++);
        checked(entry_.graph->add_node(spec));
    }
    tensor_id reshape(tensor_id x, const std::vector<uint32_t> & shape, tensor_id y = null_tensor_id) {
        const auto & current = entry_.graph->find_tensor(x)->spec().dimensions;
        if (y.value == null_tensor_id.value && shape == current) { return x; }
        if (y.value == null_tensor_id.value) { y = tensor(shape); }
        add({"reshape","qti.aisw",QNN_OP_RESHAPE,{x},{y},{},{}}); return y;
    }
    tensor_id transpose(tensor_id x, const std::vector<uint32_t> & permutation) {
        const auto current = entry_.graph->find_tensor(x)->spec().dimensions;
        std::vector<uint32_t> shape; for (auto axis : permutation) { shape.push_back(current[axis]); }
        const auto y = tensor(shape); add(op_spec::transpose(x,y,permutation)); return y;
    }
    tensor_id slice(tensor_id x, const std::vector<uint32_t> & begin, const std::vector<uint32_t> & extent) {
        std::vector<uint32_t> ranges; for (size_t i=0;i<begin.size();++i) { ranges.insert(ranges.end(),{begin[i],begin[i]+extent[i],1}); }
        const auto y = tensor(extent);
        add({"slice","qti.aisw",QNN_OP_STRIDED_SLICE,{x},{y},{},{{QNN_OP_STRIDED_SLICE_PARAM_RANGES,{uint32_t(begin.size()),3},std::move(ranges),QNN_DATATYPE_INT_32}}}); return y;
    }
    tensor_id external(const ggml_tensor * t, partition_input_kind kind, size_t node, std::vector<uint32_t> shape) {
        const auto dtype = (kind == partition_input_kind::indices || kind == partition_input_kind::scatter_indices) ? QNN_DATATYPE_INT_32 : QNN_DATATYPE_FLOAT_16;
        const bool weight = !app_write_weights_ && kind == partition_input_kind::numerical && constant_tensor(t,pending_);
        tensor_spec spec {"input"+std::to_string(next_tensor_++),weight ? QNN_TENSOR_TYPE_STATIC : QNN_TENSOR_TYPE_APP_WRITE,dtype,std::move(shape)};
        tensor_id id;
        if (weight) {
            auto payload = pack_(t); entry_.static_bytes += payload.size()*2;
            id = checked(entry_.graph->register_tensor(std::move(spec),{payload.data(),payload.size()*2}));
        } else {
            id = checked(entry_.graph->register_tensor(std::move(spec)));
            entry_.partition->inputs.push_back({inventory_.slots.at(t),id,kind,node});
        }
        return id;
    }
    static const ggml_tensor * root(const ggml_tensor * t) {
        for (size_t depth=0; t && t->view_src && depth<32; ++depth) { t=t->view_src; }
        return t;
    }
    bool internal(const ggml_tensor * t) const {
        for(size_t depth=0;t && depth<32;++depth){
            if(values_.count(t) || members_.count(t)){return true;}
            t=t->view_src;
        }
        return false;
    }
    tensor_id input(const ggml_tensor * t) {
        if (!t) { throw partition_error("Missing source"); }
        for (auto it=state_.rbegin();it!=state_.rend();++it) {
            if (root(t)!=root(it->first)) { continue; }
            const auto a=reinterpret_cast<uintptr_t>(t->data),b=reinterpret_cast<uintptr_t>(it->first->data);
            if (a>=b && a-b<=ggml_nbytes(it->first) && ggml_nbytes(t)<=ggml_nbytes(it->first)-(a-b)) {
                if (ggml_are_same_shape(t,it->first) && ggml_are_same_stride(t,it->first) && a==b) { return it->second; }
                return view(t,it->first,it->second);
            }
        }
        const auto found = values_.find(t); if (found != values_.end()) { return found->second; }
        if (!t) { throw partition_error("Missing source"); }
        if (metadata_op(t->op) && t->view_src) {
            const auto * parent = t->view_src;
            if (members_.count(parent) || values_.count(parent)) {
                const auto value = view(t,parent,input(parent)); values_[t]=value; return value;
            }
            for (size_t depth=0;parent && depth<32;++depth) {
                if (values_.count(parent)) { const auto value=view(t,parent,values_.at(parent));values_[t]=value;return value; }
                parent=parent->view_src;
            }
        }
        if (members_.count(t) && !metadata_op(t->op)) { throw partition_error("Source precedes producer"); }
        const auto value = external(t,partition_input_kind::numerical,0,logical_shape(t)); values_[t]=value; return value;
    }
    tensor_id view(const ggml_tensor * t, const ggml_tensor * parent, tensor_id value) {
        if (t->op == GGML_OP_RESHAPE && ggml_nelements(t) == ggml_nelements(parent)) { return reshape(value,logical_shape(t)); }
        // Recover a physical contiguous storage shape, then slice and transpose into logical order.
        if (!ggml_is_contiguous(parent)) {
            std::vector<uint32_t> permutation(4), used(4,0);
            for (size_t i=0;i<4;++i) {
                bool matched=false;
                for (size_t j=0;j<4;++j) {
                    if (!used[j] && t->nb[3-i]==parent->nb[3-j] && t->ne[3-i]==parent->ne[3-j]) { permutation[i]=uint32_t(j);used[j]=1;matched=true;break; }
                }
                if (!matched) { throw partition_error("Unrepresentable strided view"); }
            }
            return transpose(value,permutation);
        }
        const auto element = ggml_type_size(parent->type);
        std::array<size_t,4> order {0,1,2,3};
        std::stable_sort(order.begin(),order.end(),[t](size_t a,size_t b){return t->nb[a]!=t->nb[b] ? t->nb[a]<t->nb[b] : t->ne[a]<t->ne[b];});
        if (t->nb[order[0]]!=element) { throw partition_error("Strided view has no dense feature axis"); }
        std::vector<uint32_t> full(4),begin(4),extent(4),permutation(4);
        size_t offset=t->view_offs;
        if (t->data && parent->data) {
            const auto a=reinterpret_cast<uintptr_t>(t->data),b=reinterpret_cast<uintptr_t>(parent->data);
            if (a<b) { throw partition_error("View precedes source storage"); } offset=a-b;
        }
        for (size_t sorted=0;sorted<4;++sorted) {
            const size_t axis=order[sorted],qaxis=3-sorted,stride=t->nb[axis];
            const size_t upper=sorted==3 ? size_t(ggml_nelements(parent))*element : t->nb[order[sorted+1]];
            if (!stride || upper%stride || upper/stride==0) { throw partition_error("Unsupported physical view strides"); }
            full[qaxis]=uint32_t(upper/stride); extent[qaxis]=uint32_t(t->ne[axis]); begin[qaxis]=uint32_t(offset/stride%full[qaxis]);
            if (extent[qaxis]>full[qaxis]-begin[qaxis]) {
                GGML_LOG_ERROR("qairt: view %s of %s axis=%zu extent=%u full=%u begin=%u offset=%zu parent=[%lld,%lld,%lld,%lld] target=[%lld,%lld,%lld,%lld] strides=[%zu,%zu,%zu,%zu]\n",
                    ggml_get_name(t),ggml_get_name(parent),axis,extent[qaxis],full[qaxis],begin[qaxis],offset,
                    (long long)parent->ne[0],(long long)parent->ne[1],(long long)parent->ne[2],(long long)parent->ne[3],
                    (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],t->nb[0],t->nb[1],t->nb[2],t->nb[3]);
                throw partition_error("View exceeds physical source");
            }
            permutation[3-axis]=uint32_t(qaxis);
        }
        auto reshaped=reshape(value,full);
        if (full!=extent || offset) { reshaped=slice(reshaped,begin,extent); }
        return transpose(reshaped,permutation);
    }
    std::shared_ptr<const w4_weights> native_weights(const ggml_tensor * t) const {
        if (!w4fp16_ || !q4_weight_layout(t) || !constant_tensor(t,pending_)) { return nullptr; }
        const auto b = t->view_src ? t->view_src->buffer : t->buffer;
        const auto * c = static_cast<const buffer_context *>(b->context);
        const size_t offset = reinterpret_cast<uintptr_t>(t->data) - reinterpret_cast<uintptr_t>(c->storage);
        std::lock_guard<std::mutex> lock(c->weights_mutex);
        const auto found = c->weights.find(offset);
        return found == c->weights.end() ? nullptr : found->second.upload.get(size_t(t->ne[0]),size_t(t->ne[1]),t->type);
    }
    void native_matmul(const ggml_tensor * t, const w4_weights & weights, tensor_id y) {
        const uint32_t k = uint32_t(weights.width), n = uint32_t(weights.rows), m = uint32_t(ggml_nelements(t->src[1]) / k);
        const auto activation = reshape(input(t->src[1]),{1,1,m,k});
        tensor_spec spec {"w4weight"+std::to_string(next_tensor_++),QNN_TENSOR_TYPE_STATIC,QNN_DATATYPE_SFIXED_POINT_8,{1,1,n,k},weights.quantization};
        const auto w = checked(entry_.graph->register_tensor(std::move(spec),{weights.codes.data(),weights.codes.size()}));
        entry_.static_bytes += weights.bytes();
        const auto output = tensor({1,1,m,n});
        add(op_spec::matmul(activation,w,output));
        reshape(output,logical_shape(t),y); ++entry_.partition->w4_matmuls;
    }
    void matmul(const ggml_tensor * t,tensor_id a,tensor_id b,tensor_id y) {
        const auto * left=t->src[0];const auto * right=t->src[1];
        const uint32_t k=uint32_t(left->ne[0]),n=uint32_t(left->ne[1]),m=uint32_t(right->ne[1]);
        a=reshape(a,matrix_shape(left));b=reshape(b,matrix_shape(right));
        const size_t groups=size_t(right->ne[2])*size_t(right->ne[3]);
        std::vector<tensor_id> results;
        for (size_t batch=0;batch<size_t(right->ne[3]);++batch) {
            for (size_t head=0;head<size_t(right->ne[2]);++head) {
                const size_t ah=head/(size_t(right->ne[2])/size_t(left->ne[2]));
                const size_t ab=batch/(size_t(right->ne[3])/size_t(left->ne[3]));
                const auto aa=groups==1 ? a : slice(a,{uint32_t((ab*size_t(left->ne[2])+ah)*n),0},{n,k});
                const auto bb=groups==1 ? b : slice(b,{uint32_t((batch*size_t(right->ne[2])+head)*m),0},{m,k});
                const auto output=tensor({m,n});add(op_spec::matmul(bb,aa,output));results.push_back(output);
            }
        }
        auto joined=results[0];
        if (results.size()>1) {
            joined=tensor(matrix_shape(t));Qnn_Scalar_t axis=QNN_SCALAR_INIT;axis.dataType=QNN_DATATYPE_UINT_32;axis.uint32Value=0;
            add({"concat","qti.aisw",QNN_OP_CONCAT,std::move(results),{joined},{{QNN_OP_CONCAT_PARAM_AXIS,axis}},{}});
        }
        reshape(joined,logical_shape(t),y);
    }
    void flash(size_t node,const ggml_tensor * t,tensor_id q,tensor_id y) {
        const auto * query=t->src[0];const auto * key=t->src[1];const auto * value=t->src[2];
        const uint32_t width=uint32_t(query->ne[0]),queries=uint32_t(query->ne[1]),heads=uint32_t(query->ne[2]),keys=uint32_t(key->ne[1]);
        const std::vector<uint32_t> score_shape {1,heads,queries,keys};
        ggml_tensor score=*t;score.ne[0]=keys;score.ne[1]=queries;score.ne[2]=heads;score.ne[3]=1;score.src[0]=const_cast<ggml_tensor *>(key);score.src[1]=const_cast<ggml_tensor *>(query);
        const auto logits=tensor(score_shape);matmul(&score,input(key),q,logits);
        float scale;std::memcpy(&scale,t->op_params,sizeof(scale));
        const auto scaled=tensor(score_shape);
        add({"scale","qti.aisw","ElementWiseMultiply",{logits,constant({1,1,1,1},{{ggml_fp32_to_fp16(scale)}})},{scaled},{},{}});
        const auto mask=external(t->src[3],partition_input_kind::mask,node,{1,1,queries,keys});
        const auto masked=tensor(score_shape),probabilities=tensor(score_shape);
        add({"mask","qti.aisw","ElementWiseAdd",{scaled,mask},{masked},{},{}});
        add(op_spec::softmax(masked,probabilities,1.0f,3));
        ggml_tensor transposed=*value;transposed.ne[0]=keys;transposed.ne[1]=width;
        ggml_tensor output=*query;output.src[0]=&transposed;output.src[1]=&score;
        const auto attended=tensor({1,heads,queries,width});
        matmul(&output,transpose(input(value),{0,1,3,2}),probabilities,attended);
        reshape(transpose(attended,{0,2,1,3}),logical_shape(t),y);
    }
    void rope(size_t node,const ggml_tensor * t,tensor_id x,tensor_id y) {
        const auto p=rope_params(t);const auto shape=logical_shape(t);
        const std::vector<uint32_t> selected {shape[0],shape[1],uint32_t(p.dimensions/2)};
        const auto cosine=external(t,partition_input_kind::cosine,node,selected);
        const auto sine=external(t,partition_input_kind::sine,node,selected);
        partition_rope tables;tables.node=node;tables.table.cosine_input=cosine;tables.table.sine_input=sine;
        entry_.partition->ropes.push_back(std::move(tables));
        const auto transposed=transpose(x,{0,2,1,3});
        const auto rotated=tensor({shape[0],shape[2],shape[1],shape[3]});
        add(op_spec::rotary_embedding(transposed,cosine,sine,rotated,uint32_t(p.dimensions),p.mode==GGML_ROPE_TYPE_NORMAL));
        const auto restored=transpose(rotated,{0,2,1,3});reshape(restored,shape,y);
    }
    bool w4fp16_ = false;
    bool app_write_weights_ = false;
    prepared_run & entry_;
    const std::vector<ggml_tensor *> & nodes_;
    const tensor_inventory & inventory_;
    std::function<std::vector<fp16_bits>(const ggml_tensor *)> pack_;
    const std::vector<pending_output> & pending_;
    std::unordered_map<const ggml_tensor *,bool> members_;
    std::unordered_map<const ggml_tensor *,tensor_id> values_;
    std::vector<std::pair<const ggml_tensor *,tensor_id>> state_;
    size_t next_tensor_=0;
};
}

status backend_context::run_partition(const std::vector<ggml_tensor *> & nodes,std::vector<pending_output> & pending) {
    try {
        tensor_inventory inventory(nodes);
        std::vector<uint64_t> key {0x504152544954494fULL,3,uint64_t(runtime_->config().enable_w4fp16),uint64_t(runtime_->config().app_write_weights),uint64_t(runtime_->config().share_weight_graphs),nodes.size(),inventory.tensors.size()};
        for (const auto * t:inventory.tensors) {
            key.push_back(t->op);key.push_back(t->type);key.push_back(t->view_offs);
            for(size_t i=0;i<4;++i){key.push_back(t->ne[i]);key.push_back(t->nb[i]);}
            for(size_t i=0;i<sizeof(t->op_params)/4;++i){uint32_t word;std::memcpy(&word,reinterpret_cast<const uint8_t *>(t->op_params)+i*4,4);key.push_back(word);}
            for(const auto * source:t->src){const auto found=inventory.slots.find(source);key.push_back(found==inventory.slots.end()?SIZE_MAX:found->second);}
            const auto parent=inventory.slots.find(t->view_src);key.push_back(parent==inventory.slots.end()?SIZE_MAX:parent->second);
            if(constant_tensor(t,pending) && !runtime_->config().share_weight_graphs) {
                const auto buffer=t->view_src?t->view_src->buffer:t->buffer;const auto * allocation=static_cast<const buffer_context *>(buffer->context);
                key.push_back(1);key.push_back(allocation->identity);key.push_back(allocation->generation);
                key.push_back(reinterpret_cast<uintptr_t>(t->data)-reinterpret_cast<uintptr_t>(allocation->storage));
            }else{key.push_back(0);}
        }
        prepared_run * entry=nullptr;
        for(size_t i=0;i<cache_.size();++i){if(cache_[i]->key==key){auto hit=std::move(cache_[i]);cache_.erase(cache_.begin()+i);cache_.push_back(std::move(hit));entry=cache_.back().get();++stats_.cache_hits;break;}}
        if(!entry) {
            if(!reserve_cache(0)){throw partition_error("Graph cache limit");}
            const auto preparation_start = std::chrono::steady_clock::now();
            auto fresh=std::make_unique<prepared_run>();fresh->key=std::move(key);fresh->partition=std::make_unique<partition_data>();
            std::shared_ptr<context_owner> graph_context;
            if (runtime_->config().shared_context) {
                if (!shared_context_) { shared_context_ = checked(context_owner::create(runtime_)); }
                graph_context = shared_context_;
            } else { graph_context = checked(context_owner::create(runtime_)); }
            const auto graph_name = runtime_->config().shared_context ? "connected_partition_" + std::to_string(++next_graph_name_) : "connected_partition";
            fresh->graph=checked(graph_owner::create(std::move(graph_context),graph_name));
            partition_builder builder(*fresh,nodes,inventory,[&](const ggml_tensor * t){return checked(pack(t,pending));},pending,runtime_->config().enable_w4fp16,runtime_->config().app_write_weights);
            builder.lower();
            if(!reserve_cache(fresh->static_bytes)){throw partition_error("Static partition payload exceeds cache limit");}
            if (diagnostic_phase()) {
                uint64_t retained_bytes = 0;
                for (const auto & cached : cache_) { retained_bytes += cached->static_bytes; }
                GGML_LOG_INFO("qairt: inventory partition phase=%s graph=%llu cache_entries=%zu cache_static_bytes=%llu candidate_static_bytes=%zu first=%s last=%s ggml=%zu qnn=%zu w4=%zu\n",
                    diagnostic_phase(), (unsigned long long) fresh->graph->diagnostic_id(), cache_.size(), (unsigned long long) retained_bytes,
                    fresh->static_bytes, ggml_get_name(nodes.front()), ggml_get_name(nodes.back()), nodes.size(), fresh->partition->qnn_nodes, fresh->partition->w4_matmuls);
            }
            const auto finalization_start = std::chrono::steady_clock::now();
            const auto finalized = fresh->graph->finalize();
            const auto preparation_end = std::chrono::steady_clock::now();
            stats_.preparation_time_us += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(preparation_end - preparation_start).count());
            stats_.finalization_time_us += uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(preparation_end - finalization_start).count());
            stats_.finalization_failures += !finalized.ok();
            checked(finalized);
            if(runtime_->config().verbose_diagnostics) {
                GGML_LOG_INFO("qairt: connected graph GGML=%zu QNN=%zu outputs=%zu W4=%zu first=%s last=%s\n",nodes.size(),fresh->partition->qnn_nodes,
                    fresh->outputs.size(),fresh->partition->w4_matmuls,ggml_get_name(nodes.front()),ggml_get_name(nodes.back()));
                for(const auto & span:fresh->partition->spans) {
                    GGML_LOG_INFO("qairt: partition span node=%zu name=%s ggml=%s first=%zu count=%zu\n",span.node,
                        ggml_get_name(nodes[span.node]),ggml_op_name(nodes[span.node]->op),span.first_op,span.count);
                }
            }
            stats_.w4_preparations += fresh->partition->w4_matmuls;
            cache_.push_back(std::move(fresh));entry=cache_.back().get();++stats_.preparations;
        }
        // Refresh selected position tables; cache owns values, never GGML tensor pointers.
        std::unordered_map<uint32_t,std::vector<fp16_bits>> selected_tables;
        for(auto & recipe:entry->partition->ropes) {
            const auto * t=nodes[recipe.node];const auto p=rope_params(t);const size_t pairs=size_t(p.dimensions/2),tokens=size_t(t->ne[2]);
            std::vector<int32_t> positions(tokens);ggml_backend_tensor_get(t->src[1],positions.data(),0,tokens*4);
            size_t rows=1024;for(int32_t position:positions){if(position<0 || position>=(1<<20)){throw partition_error("RoPE position outside supported range");}while(rows<=size_t(position)){rows*=2;}}
            std::vector<float> factors;
            if(t->src[2]) {factors=checked(float_parameters(t->src[2],pending));factors.resize(pairs);
                for(float factor:factors){if(!std::isfinite(factor) || factor<=0.0f){throw partition_error("Invalid RoPE frequency factor");}}
            }
            auto & table=recipe.table;
            if(factors!=recipe.factors){table.rows=0;recipe.factors=factors;}
            if(rows>table.rows) {
                const size_t bytes=rows*pairs*4,old=table.cosine.size()*4;
                if(bytes>64*1024*1024 || (bytes>old && !reserve_cache(bytes-old,entry))){throw partition_error("RoPE table cache limit");}
                if(bytes>old){entry->static_bytes+=bytes-old;}else{entry->static_bytes-=old-bytes;}
                table.cosine.resize(rows*pairs);table.sine.resize(rows*pairs);
                float correction[2] {};if(p.extension!=0.0f){ggml_rope_yarn_corr_dims(p.dimensions,p.context,p.base,p.beta_fast,p.beta_slow,p.truncate,correction);}
                const float step=std::pow(p.base,-2.0f/p.dimensions);
                for(size_t position=table.rows;position<rows;++position){float theta=float(position);
                    for(size_t pair=0;pair<pairs;++pair){const float extrapolated=theta/(factors.empty()?1.0f:factors[pair]);
                        const float interpolated=p.scale*extrapolated;float angle=interpolated,magnitude=p.attention;
                        if(p.extension!=0.0f){const float ramp=1.0f-std::min(1.0f,std::max(0.0f,(float(pair)-correction[0])/std::max(0.001f,correction[1]-correction[0])));
                            const float mix=ramp*p.extension;angle=interpolated*(1.0f-mix)+extrapolated*mix;magnitude*=1.0f+0.1f*std::log(1.0f/p.scale);}
                        const float cosine=std::cos(angle)*magnitude,sine=std::sin(angle)*magnitude;
                        if(!std::isfinite(cosine) || !std::isfinite(sine) || std::fabs(cosine)>65504.0f || std::fabs(sine)>65504.0f){throw partition_error("Invalid FP16 RoPE table value");}
                        table.cosine[position*pairs+pair].bits=ggml_fp32_to_fp16(cosine);
                        table.sine[position*pairs+pair].bits=ggml_fp32_to_fp16(sine);theta*=step;
                    }}table.rows=rows;
            }
            auto & cosine=selected_tables[table.cosine_input.value];auto & sine=selected_tables[table.sine_input.value];
            cosine.resize(size_t(t->ne[3])*tokens*pairs);sine.resize(cosine.size());
            for(size_t batch=0;batch<size_t(t->ne[3]);++batch){for(size_t token=0;token<tokens;++token){
                std::copy_n(table.cosine.data()+size_t(positions[token])*pairs,pairs,cosine.data()+(batch*tokens+token)*pairs);
                std::copy_n(table.sine.data()+size_t(positions[token])*pairs,pairs,sine.data()+(batch*tokens+token)*pairs);
            }}
        }
        auto binding=checked(execution_binding::create(entry->graph));
        std::vector<std::vector<uint8_t>> inputs;inputs.reserve(entry->partition->inputs.size());
        for(const auto & input:entry->partition->inputs) {
            const auto * t=inventory.tensors[input.slot];std::vector<uint8_t> bytes;
            if(input.kind==partition_input_kind::cosine || input.kind==partition_input_kind::sine) {
                const auto & payload=selected_tables.at(input.id.value);bytes.resize(payload.size()*2);std::memcpy(bytes.data(),payload.data(),bytes.size());
            }else if(input.kind==partition_input_kind::indices || input.kind==partition_input_kind::scatter_indices) {
                const auto * node=nodes[input.node];const size_t count=size_t(ggml_nelements(t));std::vector<int32_t> indices(count);
                if (input.kind==partition_input_kind::indices) {
                    ggml_backend_tensor_get(t,indices.data(),0,count*4);
                    for(auto value:indices){if(value<0 || value>=node->src[0]->ne[1]){throw partition_error("Gather index outside source rows");}}
                } else {
                    const size_t rows=size_t(node->src[2]->ne[1]),element=ggml_type_size(t->type);
                    std::vector<uint8_t> raw(count*element);ggml_backend_tensor_get(t,raw.data(),0,raw.size());std::vector<bool> seen(rows,false);
                    for(size_t i=0;i<count;++i){int64_t value;
                        if(t->type==GGML_TYPE_I64){std::memcpy(&value,raw.data()+i*8,8);}else{int32_t v;std::memcpy(&v,raw.data()+i*4,4);value=v;}
                        if(value<0 || uint64_t(value)>=rows || seen[size_t(value)]){throw partition_error("KV scatter index out of range or duplicate");}
                        seen[size_t(value)]=true;indices[i]=int32_t(value);
                    }
                }
                bytes.resize(count*4);std::memcpy(bytes.data(),indices.data(),bytes.size());
            }else if(input.kind==partition_input_kind::mask) {
                const size_t columns=size_t(t->ne[0]),queries=entry->graph->find_tensor(input.id)->spec().dimensions[2];
                const size_t element=ggml_type_size(t->type);auto raw=checked(mask_bytes(t,pending));
                std::vector<fp16_bits> payload(columns*queries*size_t(t->ne[2])*size_t(t->ne[3]));
                for(size_t row=0;row<payload.size()/columns;++row){bool finite=false;
                    for(size_t col=0;col<columns;++col){const size_t source=(row/queries*size_t(t->ne[1])+row%queries)*columns+col;float value;
                        if(element==4){std::memcpy(&value,raw.data()+source*4,4);}else{ggml_fp16_t half;std::memcpy(&half,raw.data()+source*2,2);value=ggml_fp16_to_fp32(half);}
                        if(value==-std::numeric_limits<float>::infinity()){value=-65504.0f;}else{if(!std::isfinite(value) || std::fabs(value)>65504.0f){throw partition_error("Invalid FP16 mask");}finite=true;}
                        payload[row*columns+col].bits=ggml_fp32_to_fp16(value);
                    }if(!finite){throw partition_error("Softmax row has no unmasked value");}
                }bytes.resize(payload.size()*2);std::memcpy(bytes.data(),payload.data(),bytes.size());
            }else{auto payload=checked(pack(t,pending));bytes.resize(payload.size()*2);std::memcpy(bytes.data(),payload.data(),bytes.size());}
            inputs.push_back(std::move(bytes));checked(binding->bind_input(input.id,{inputs.back().data(),inputs.back().size()}));
        }
        std::vector<std::vector<fp16_bits>> outputs;outputs.reserve(entry->outputs.size());
        for(size_t i=0;i<entry->outputs.size();++i){outputs.emplace_back(size_t(ggml_nelements(nodes[entry->partition->output_nodes[i]])));
            checked(binding->bind_output(entry->outputs[i],{outputs.back().data(),outputs.back().size()*2}));}
        checked(entry->graph->execute(*binding));++stats_.executions;stats_.w4_executions+=entry->partition->w4_matmuls;
        for(size_t i=0;i<outputs.size();++i){auto * t=nodes[entry->partition->output_nodes[i]];std::vector<uint8_t> bytes(dense_bytes(t));
            for(size_t j=0;j<outputs[i].size();++j){const float value=ggml_fp16_to_fp32(outputs[i][j].bits);
                if(!std::isfinite(value)){throw partition_error("Nonfinite partition output");}
                if(t->type==GGML_TYPE_F32){std::memcpy(bytes.data()+j*4,&value,4);}else{std::memcpy(bytes.data()+j*2,&outputs[i][j].bits,2);}}
            if(t->op==GGML_OP_SET_ROWS) {
                const auto * cache=t->src[2];const auto * indices=t->src[1];const size_t rows=size_t(cache->ne[1]),width=size_t(cache->ne[0]);
                auto original=checked(pack(cache,pending));std::vector<bool> updated(rows,false);
                std::vector<uint8_t> raw(size_t(indices->ne[0])*ggml_type_size(indices->type));ggml_backend_tensor_get(indices,raw.data(),0,raw.size());
                for(size_t j=0;j<size_t(indices->ne[0]);++j){int64_t value;
                    if(indices->type==GGML_TYPE_I64){std::memcpy(&value,raw.data()+j*8,8);}else{int32_t v;std::memcpy(&v,raw.data()+j*4,4);value=v;}updated[size_t(value)]=true;}
                for(size_t row=0;row<rows;++row){if(!updated[row]){for(size_t col=0;col<width;++col){
                    if(original[row*width+col].bits!=outputs[i][row*width+col].bits){throw partition_error("KV scatter modified untouched row");}
                }}}
            }
            pending.push_back({t,std::move(bytes)});
        }
        return status::success();
    }catch(const qairt_error & error){return status::failure(error);}
}
}
