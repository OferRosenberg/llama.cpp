#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-qairt.h"
#include "qairt-lowering.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace ggml::qairt;
static int failures = 0, cases = 0;
static void logger(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR) { std::fputs(text, stderr); }
}
#define CHECK(x) do { if (!(x)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); } } while (0)
static float half(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }
static std::vector<float> values(size_t n, float scale = 1.0f) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) { x[i] = half(scale * (std::sin(float(i) * 0.17f) + 0.2f * std::cos(float(i) * 0.31f))); }
    return x;
}
static void upload(ggml_tensor * t, const std::vector<float> & x) {
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_set(t, x.data(), 0, x.size() * 4); }
    else {
        std::vector<ggml_fp16_t> h(x.size());
        for (size_t i = 0; i < x.size(); ++i) { h[i] = ggml_fp32_to_fp16(x[i]); }
        ggml_backend_tensor_set(t, h.data(), 0, h.size() * 2);
    }
}
static std::vector<float> download(ggml_tensor * t) {
    std::vector<float> x(size_t(ggml_nelements(t)));
    if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_get(t, x.data(), 0, x.size() * 4); }
    else {
        std::vector<ggml_fp16_t> h(x.size()); ggml_backend_tensor_get(t, h.data(), 0, h.size() * 2);
        for (size_t i = 0; i < x.size(); ++i) { x[i] = ggml_fp16_to_fp32(h[i]); }
    }
    return x;
}
static void compare(const char * name, const std::vector<float> & actual, const std::vector<float> & expected) {
    double error = 0, energy = 0, maximum = 0;
    CHECK(actual.size() == expected.size());
    for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        CHECK(std::isfinite(actual[i]));
        const double d = double(actual[i]) - expected[i]; error += d*d; energy += double(expected[i])*expected[i]; maximum = std::max(maximum, std::fabs(d));
    }
    const double nmse = error / std::max(energy, 1e-30);
    std::printf("%s NMSE=%.9g max_abs=%.9g\n", name, nmse, maximum);
    CHECK(nmse <= 1e-5); ++cases;
}
struct fixture {
    ggml_context * c = ggml_init({8 * 1024 * 1024, nullptr, true});
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    fixture() { backend = ggml_backend_qairt_init(); CHECK(c && backend); }
    ~fixture() { if (buffer) { ggml_backend_buffer_free(buffer); } if (backend) { ggml_backend_free(backend); } if (c) { ggml_free(c); } }
    bool allocate(ggml_tensor * y, ggml_cgraph * & graph) {
        if (!backend || !c) { return false; }
        graph = ggml_new_graph(c); ggml_build_forward_expand(graph, y);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) { CHECK(ggml_backend_supports_op(backend, ggml_graph_node(graph, i))); }
        buffer = ggml_backend_alloc_ctx_tensors(c, backend); CHECK(buffer); return buffer != nullptr;
    }
    bool run(ggml_cgraph * graph) { const auto status = ggml_backend_graph_compute(backend, graph); CHECK(status == GGML_STATUS_SUCCESS); return status == GGML_STATUS_SUCCESS; }
};
static void connected_partition() {
    fixture f;
    auto * x = ggml_new_tensor_2d(f.c, GGML_TYPE_F32, 128, 4);
    auto * scale = ggml_new_tensor_1d(f.c, GGML_TYPE_F32, 128);
    auto * normalized = ggml_rms_norm(f.c, x, 1e-6f);
    auto * scaled = ggml_mul(f.c, normalized, scale);
    auto * activated = ggml_silu(f.c, scaled);
    auto * y = ggml_add(f.c, activated, scaled);
    ggml_cgraph * graph; if (!f.allocate(y, graph)) { return; }
    auto input = values(512, 0.3f), scales = std::vector<float>(128, 0.5f);
    upload(x, input); upload(scale, scales);
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(f.backend));
    auto stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
    ggml_backend_qairt_stats before {}, after {};
    CHECK(stats && stats(f.backend, &before));
    if (!f.run(graph)) { return; }
    CHECK(stats && stats(f.backend, &after));
    CHECK(after.executions - before.executions == 1);
    CHECK(after.preparations - before.preparations == 1);
    std::vector<float> expected(512);
    for (size_t row = 0; row < 4; ++row) {
        double energy = 0;
        for (size_t c = 0; c < 128; ++c) { energy += double(input[row*128+c])*input[row*128+c]; }
        const double inv = 1.0 / std::sqrt(energy / 128 + 1e-6);
        for (size_t c = 0; c < 128; ++c) {
            const float v = float(input[row*128+c]*inv*0.5);
            expected[row*128+c] = v + v/(1.0f+std::exp(-v));
        }
    }
    compare("connected branched partition", download(y), expected);
    CHECK(stats && stats(f.backend, &before));
    if (f.run(graph)) {
        CHECK(stats && stats(f.backend, &after));
        CHECK(after.executions - before.executions == 1 && after.preparations == before.preparations);
        compare("cached connected partition", download(y), expected);
    }
}
static void singleton_layout_partition() {
    fixture f;
    auto * x = ggml_new_tensor_3d(f.c, GGML_TYPE_F32, 128, 1, 16);
    auto * normalized = ggml_rms_norm(f.c, x, 1e-6f);
    auto * packed = ggml_cont(f.c, ggml_permute(f.c, normalized, 0, 2, 1, 3));
    auto * y = ggml_silu(f.c, ggml_reshape_2d(f.c, packed, 2048, 1));
    ggml_cgraph * graph; if (!f.allocate(y, graph)) { return; }
    auto input = values(2048, 0.3f), expected = input; upload(x, input);
    for (size_t row = 0; row < 16; ++row) {
        double energy = 0;
        for (size_t c = 0; c < 128; ++c) { energy += double(input[row*128+c])*input[row*128+c]; }
        const double inv = 1.0 / std::sqrt(energy/128+1e-6);
        for (size_t c = 0; c < 128; ++c) {
            const float v = float(input[row*128+c]*inv); expected[row*128+c] = v/(1.0f+std::exp(-v));
        }
    }
    if (f.run(graph)) { compare("single-token permute/CONT/reshape partition", download(y), expected); }
}
static void connected_scatter_alias() {
    fixture f;
    auto * cache = ggml_new_tensor_2d(f.c, GGML_TYPE_F16, 128, 32);
    auto * x = ggml_new_tensor_2d(f.c, GGML_TYPE_F32, 128, 2);
    auto * updates = ggml_rms_norm(f.c, x, 1e-6f);
    auto * ids = ggml_new_tensor_1d(f.c, GGML_TYPE_I64, 2);
    auto * selection = ggml_new_tensor_1d(f.c, GGML_TYPE_I32, 2);
    auto * written = ggml_set_rows(f.c, cache, updates, ids);
    auto * y = ggml_get_rows(f.c, cache, selection);
    ggml_cgraph * graph; if (!f.allocate(written, graph)) { return; }
    ggml_build_forward_expand(graph, y);
    auto initial = values(4096, 0.1f), input = values(256, 0.3f), expected = input;
    upload(cache, initial); upload(x, input);
    const int64_t rows[] {3,5}; const int32_t selected[] {3,5};
    ggml_backend_tensor_set(ids, rows, 0, sizeof(rows)); ggml_backend_tensor_set(selection, selected, 0, sizeof(selected));
    for (size_t row = 0; row < 2; ++row) {
        double energy = 0;
        for (size_t c = 0; c < 128; ++c) { energy += double(input[row*128+c])*input[row*128+c]; }
        const double inv = 1.0/std::sqrt(energy/128+1e-6);
        for (size_t c = 0; c < 128; ++c) { expected[row*128+c] = float(input[row*128+c]*inv); }
    }
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(f.backend));
    auto stats = reinterpret_cast<ggml_backend_qairt_get_stats_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_qairt_get_stats"));
    ggml_backend_qairt_stats before {}, after {}; CHECK(stats && stats(f.backend,&before));
    if (f.run(graph)) {
        CHECK(stats && stats(f.backend,&after)); CHECK(after.executions-before.executions == 1);
        compare("connected KV scatter read through original alias",download(y),expected);
        const auto result = download(cache);
        for (size_t row = 0; row < 32; ++row) { if (row != 3 && row != 5) {
            for (size_t c = 0; c < 128; ++c) { CHECK(result[row*128+c] == initial[row*128+c]); }
        } }
    }
}
static void host_tests() {
    auto * c = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 16, 28);
    auto * scale = ggml_new_tensor_1d(c, GGML_TYPE_F32, 128);
    CHECK(primitive_supported(ggml_rms_norm(c, x, 1e-6f)));
    CHECK(primitive_supported(ggml_mul(c, x, scale)));
    CHECK(primitive_supported(ggml_add(c, x, x)));
    CHECK(primitive_supported(ggml_swiglu_split(c, x, x)));
    auto * positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 28);
    CHECK(primitive_supported(ggml_rope_ext(c, x, positions, nullptr, 128, GGML_ROPE_TYPE_NEOX, 32768, 1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f)));
    auto * logits = ggml_new_tensor_4d(c, GGML_TYPE_F32, 256, 28, 16, 1);
    auto * mask = ggml_new_tensor_2d(c, GGML_TYPE_F32, 256, 32);
    CHECK(primitive_supported(ggml_soft_max_ext(c, logits, mask, 0.125f, 0.0f)));
    auto * key = ggml_new_tensor_3d(c, GGML_TYPE_F16, 128, 256, 8);
    auto * query = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, 28, 16);
    CHECK(primitive_supported(ggml_mul_mat(c, key, query)));
    ggml_free(c);
}
static void normalization(int width, int heads, int tokens) {
    fixture f; auto * x = ggml_new_tensor_3d(f.c, GGML_TYPE_F32, width, heads, tokens);
    auto * gamma = ggml_new_tensor_1d(f.c, GGML_TYPE_F32, width);
    auto * y = ggml_mul(f.c, ggml_rms_norm(f.c, x, 1e-6f), gamma);
    ggml_cgraph * graph; if (!f.allocate(y, graph)) { return; }
    auto input = values(size_t(ggml_nelements(x))), scales = values(width, 0.25f);
    for (auto & v : scales) { v = half(1.0f + v); }
    upload(x, input); upload(gamma, scales);
    std::vector<float> expected(input.size());
    for (size_t row = 0; row < input.size() / size_t(width); ++row) {
        double sum = 0; for (int d = 0; d < width; ++d) { const auto v = input[row * width + d]; sum += double(v)*v; }
        const double inv = 1.0 / std::sqrt(sum / width + 1e-6);
        for (int d = 0; d < width; ++d) { expected[row * width + d] = float(input[row * width + d] * inv * scales[d]); }
    }
    if (f.run(graph)) { compare("RMSNorm + learned scale", download(y), expected); }
}
static void swiglu(int width, int tokens) {
    fixture f; auto * gate = ggml_new_tensor_2d(f.c, GGML_TYPE_F32, width, tokens);
    auto * up = ggml_new_tensor_2d(f.c, GGML_TYPE_F32, width, tokens);
    auto * activation = ggml_swiglu_split(f.c, gate, up);
    auto * y = ggml_add(f.c, activation, up);
    ggml_cgraph * graph; if (!f.allocate(y, graph)) { return; }
    auto a = values(size_t(ggml_nelements(gate)), 2.0f), b = values(a.size(), 0.7f), expected = a;
    upload(gate, a); upload(up, b);
    for (size_t i = 0; i < a.size(); ++i) { expected[i] = a[i] / (1.0f + std::exp(-a[i])) * b[i] + b[i]; }
    if (f.run(graph)) { compare("SwiGLU + residual", download(y), expected); }
}
static void rope(int heads, int tokens) {
    fixture f; const int width = 128;
    auto * x = ggml_new_tensor_3d(f.c, GGML_TYPE_F32, width, heads, tokens);
    auto * positions = ggml_new_tensor_1d(f.c, GGML_TYPE_I32, tokens);
    auto * y = ggml_rope_ext(f.c, x, positions, nullptr, width, GGML_ROPE_TYPE_NEOX, 32768, 1000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_cgraph * graph; if (!f.allocate(y, graph)) { return; }
    auto input = values(size_t(ggml_nelements(x))), expected = input; upload(x, input);
    std::vector<int32_t> ids(tokens); for (int i = 0; i < tokens; ++i) { ids[i] = 17 + i; }
    ggml_backend_tensor_set(positions, ids.data(), 0, ids.size() * 4);
    for (int token = 0; token < tokens; ++token) {
        float theta = float(ids[token]), factor = std::pow(1000000.0f, -2.0f / width);
        for (int d = 0; d < width / 2; ++d) {
            const float cosine = std::cos(theta), sine = std::sin(theta); theta *= factor;
            for (int h = 0; h < heads; ++h) {
                const size_t base = size_t(token * heads + h) * width;
                expected[base + d] = input[base + d] * cosine - input[base + d + width/2] * sine;
                expected[base + d + width/2] = input[base + d] * sine + input[base + d + width/2] * cosine;
            }
        }
    }
    if (f.run(graph)) { compare("NEOX RoPE", download(y), expected); }
    std::reverse(ids.begin(), ids.end()); ggml_backend_tensor_set(positions, ids.data(), 0, ids.size() * 4);
    CHECK(f.run(graph));
}
static void softmax(int keys, int queries, int heads) {
    fixture f; auto * x = ggml_new_tensor_3d(f.c, GGML_TYPE_F32, keys, queries, heads);
    auto * mask = ggml_new_tensor_2d(f.c, GGML_TYPE_F32, keys, queries + 3);
    auto * y = ggml_soft_max_ext(f.c, x, mask, 0.125f, 0.0f);
    ggml_cgraph * graph; if (!f.allocate(y, graph)) { return; }
    auto input = values(size_t(ggml_nelements(x)), 4.0f), expected = input;
    std::vector<float> masks(size_t(keys) * (queries + 3), -std::numeric_limits<float>::infinity());
    for (int q = 0; q < queries; ++q) { for (int k = 0; k <= keys - queries + q; ++k) { masks[size_t(q) * keys + k] = 0; } }
    upload(x, input); upload(mask, masks);
    for (int h = 0; h < heads; ++h) { for (int q = 0; q < queries; ++q) {
        double sum = 0; const size_t start = size_t(h * queries + q) * keys;
        for (int k = 0; k < keys; ++k) { expected[start+k] = k <= keys-queries+q ? std::exp(input[start+k] * 0.125f) : 0; sum += expected[start+k]; }
        for (int k = 0; k < keys; ++k) { expected[start+k] = float(expected[start+k]/sum); }
    } }
    if (f.run(graph)) { compare("masked Softmax", download(y), expected); }
}
static void batched_matmul(int queries) {
    fixture f; const int width=128, keys=256, kv_heads=8, heads=16;
    auto * k = ggml_new_tensor_3d(f.c, GGML_TYPE_F16, width, keys, kv_heads);
    auto * q = ggml_new_tensor_3d(f.c, GGML_TYPE_F32, width, queries, heads);
    auto * y = ggml_mul_mat(f.c, k, q);
    ggml_cgraph * graph; if (!f.allocate(y, graph)) { return; }
    auto a=values(size_t(ggml_nelements(k)),0.15f), b=values(size_t(ggml_nelements(q)),0.2f);
    upload(k,a);upload(q,b);std::vector<float> expected(size_t(ggml_nelements(y)));
    for(int h=0;h<heads;++h){for(int j=0;j<queries;++j){for(int i=0;i<keys;++i){
        double sum=0;for(int d=0;d<width;++d){sum+=double(a[(size_t(h/2)*keys+i)*width+d])*b[(size_t(h)*queries+j)*width+d];}
        expected[(size_t(h)*queries+j)*keys+i]=float(sum);
    }}}
    if(f.run(graph)){compare("grouped batched MatMul",download(y),expected);}
}
static void layernorm_and_silu() {
    fixture f;const int width=128,tokens=13;
    auto * x=ggml_new_tensor_2d(f.c,GGML_TYPE_F32,width,tokens);
    auto * normalized=ggml_norm(f.c,x,1e-5f);auto * y=ggml_silu(f.c,normalized);
    ggml_cgraph * graph;if(!f.allocate(y,graph)){return;}
    auto input=values(size_t(width)*tokens),expected=input;upload(x,input);
    for(int row=0;row<tokens;++row){double mean=0,variance=0;for(int d=0;d<width;++d){mean+=input[size_t(row)*width+d];}mean/=width;
        for(int d=0;d<width;++d){const double centered=input[size_t(row)*width+d]-mean;variance+=centered*centered;}variance/=width;
        for(int d=0;d<width;++d){const float norm=float((input[size_t(row)*width+d]-mean)/std::sqrt(variance+1e-5));expected[size_t(row)*width+d]=norm/(1.0f+std::exp(-norm));}}
    if(f.run(graph)){compare("LayerNorm + SiLU",download(y),expected);}
}
static void transactional_failure() {
    fixture f;auto * x=ggml_new_tensor_2d(f.c,GGML_TYPE_F32,32,2);auto * norm=ggml_rms_norm(f.c,x,1e-6f);
    auto * ids=ggml_new_tensor_1d(f.c,GGML_TYPE_I32,1);auto * y=ggml_get_rows(f.c,norm,ids);ggml_cgraph * graph;if(!f.allocate(y,graph)){return;}
    upload(x,values(64));upload(norm,std::vector<float>(64,7));upload(y,std::vector<float>(32,9));const int32_t invalid=-1;ggml_backend_tensor_set(ids,&invalid,0,4);
    CHECK(ggml_backend_graph_compute(f.backend,graph)==GGML_STATUS_FAILED);
    compare("failed partition preserves predecessor",download(norm),std::vector<float>(64,7));
    compare("failed partition preserves destination",download(y),std::vector<float>(32,9));
}
static void kv_scatter(bool scalar) {
    fixture f; const int width = scalar ? 1 : 1024, rows = scalar ? 65536 : 1024, count = scalar ? 8192 : 28;
    auto * cache = ggml_new_tensor_2d(f.c,GGML_TYPE_F16,width,rows);
    auto * updates = ggml_new_tensor_2d(f.c,GGML_TYPE_F32,width,count);
    auto * indices = ggml_new_tensor_1d(f.c,GGML_TYPE_I64,count);
    auto * y = ggml_set_rows(f.c,cache,updates,indices);
    ggml_cgraph * graph; if (!f.allocate(y,graph)) { return; }
    auto input = values(size_t(width)*rows,0.1f), data = values(size_t(width)*count,0.3f), expected = input;
    std::vector<int64_t> ids(count);
    for (int i=0;i<count;++i) { ids[i]=i*2+3; std::copy_n(data.data()+size_t(i)*width,width,expected.data()+size_t(ids[i])*width); }
    upload(cache,input); upload(updates,data); ggml_backend_tensor_set(indices,ids.data(),0,ids.size()*8);
    if (f.run(graph)) { compare(scalar ? "scalar transposed KV ScatterNd" : "row KV ScatterNd",download(y),expected); }
}
static void gather() {
    fixture f;auto * x=ggml_new_tensor_2d(f.c,GGML_TYPE_F32,2048,28);auto * ids=ggml_new_tensor_1d(f.c,GGML_TYPE_I32,3);
    auto * y=ggml_get_rows(f.c,x,ids);ggml_cgraph * graph;if(!f.allocate(y,graph)){return;}
    auto input=values(size_t(ggml_nelements(x)));upload(x,input);int32_t positions[]={27,3,27};ggml_backend_tensor_set(ids,positions,0,sizeof(positions));
    std::vector<float> expected(2048*3);for(size_t i=0;i<3;++i){std::copy_n(input.data()+size_t(positions[i])*2048,2048,expected.data()+i*2048);}
    if(f.run(graph)){compare("dynamic row Gather",download(y),expected);}
    positions[0]=0;positions[1]=27;positions[2]=2;input=values(input.size(),0.5f);upload(x,input);ggml_backend_tensor_set(ids,positions,0,sizeof(positions));
    for(size_t i=0;i<3;++i){std::copy_n(input.data()+size_t(positions[i])*2048,2048,expected.data()+i*2048);}
    if(f.run(graph)){compare("cached Gather refreshes data and indices",download(y),expected);}
}
static void attention() {
    fixture f;const int width=128,keys=256,queries=2,heads=16,kv_heads=8;
    auto * q=ggml_new_tensor_3d(f.c,GGML_TYPE_F32,width,queries,heads);
    auto * k=ggml_new_tensor_3d(f.c,GGML_TYPE_F16,width,keys,kv_heads);
    auto * v=ggml_new_tensor_3d(f.c,GGML_TYPE_F16,width,keys,kv_heads);
    auto * mask=ggml_new_tensor_2d(f.c,GGML_TYPE_F16,keys,queries);
    auto * y=ggml_flash_attn_ext(f.c,q,k,v,mask,0.125f,0.0f,0.0f);ggml_prec_set_acc(y,GGML_PREC_F32);
    ggml_cgraph * graph;if(!f.allocate(y,graph)){return;}
    auto a=values(size_t(ggml_nelements(q)),0.2f),b=values(size_t(ggml_nelements(k)),0.15f),c=values(b.size(),0.1f);
    std::vector<float> masks(size_t(keys)*queries,-std::numeric_limits<float>::infinity());
    const int total=17,past=total-queries;
    for(int j=0;j<queries;++j){for(int i=0;i<=past+j;++i){masks[size_t(j)*keys+i]=0;}}
    upload(q,a);upload(k,b);upload(v,c);upload(mask,masks);
    std::vector<float> expected(size_t(ggml_nelements(y)));
    for(int h=0;h<heads;++h){for(int j=0;j<queries;++j){
        std::vector<double> prob(total,0);double sum=0;
        for(int i=0;i<=past+j;++i){double dot=0;for(int d=0;d<width;++d){dot+=double(a[(size_t(h)*queries+j)*width+d])*b[(size_t(h/2)*keys+i)*width+d];}prob[i]=std::exp(dot*0.125);sum+=prob[i];}
        for(int d=0;d<width;++d){double value=0;for(int i=0;i<total;++i){value+=prob[i]/sum*c[(size_t(h/2)*keys+i)*width+d];}expected[(size_t(j)*heads+h)*width+d]=float(value);}
    }}
    if(f.run(graph)){compare("grouped attention decomposition",download(y),expected);}
}
int main(int argc, char ** argv) {
    ggml_log_set(logger,nullptr);
    if (argc == 2 && std::strcmp(argv[1], "--host") == 0) { host_tests(); }
    else {
        ggml_backend_load_all();
        connected_partition(); singleton_layout_partition(); connected_scatter_alias(); normalization(128,16,28); normalization(2048,1,28); layernorm_and_silu();
        swiglu(6144,13); swiglu(6144,1);
        rope(16,28); rope(8,2);
        softmax(256,28,16); softmax(1024,1,16);
        batched_matmul(28); batched_matmul(1);
        kv_scatter(false); kv_scatter(true); gather(); attention(); transactional_failure();
    }
    std::printf("QAIRT expanded operators: %d numerical cases, %d failures\n", cases, failures);
    return failures ? 1 : 0;
}
