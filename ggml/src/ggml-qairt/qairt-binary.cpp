#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include "qairt-graph.h"
#include "ggml-impl.h"
#include <System/QnnSystemInterface.h>
#include <HTP/QnnHtpSystemContext.h>
#include <fstream>
#include <limits>
#include <cstring>
#include <stdexcept>

namespace ggml::qairt {
namespace {
std::string binary_digest(const std::vector<uint8_t> & bytes) {
    if (bytes.empty() || bytes.size()>std::numeric_limits<ULONG>::max()) { throw std::runtime_error("Invalid binary size"); }
    BCRYPT_ALG_HANDLE algorithm=nullptr; BCRYPT_HASH_HANDLE hash=nullptr;
    if (BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)!=0) { throw std::runtime_error("SHA256 provider failed"); }
    const auto created=BCryptCreateHash(algorithm,&hash,nullptr,0,nullptr,0,0);
    uint8_t digest[32]{};
    const bool success=created==0 && BCryptHashData(hash,const_cast<PUCHAR>(bytes.data()),ULONG(bytes.size()),0)==0 &&
        BCryptFinishHash(hash,digest,sizeof(digest),0)==0;
    if(hash){BCryptDestroyHash(hash);} BCryptCloseAlgorithmProvider(algorithm,0);
    if(!success){throw std::runtime_error("SHA256 failed");}
    const char digits[]="0123456789abcdef";std::string result(64,'0');
    for(size_t i=0;i<32;++i){result[2*i]=digits[digest[i]>>4];result[2*i+1]=digits[digest[i]&15];}
    return result;
}
result<std::vector<binary_graph>> read_metadata(const std::shared_ptr<runtime_owner> & runtime, std::vector<uint8_t> & bytes) {
    using metadata_result=result<std::vector<binary_graph>>;
    auto module=dll_module::load(runtime->config().library_directory/"QnnSystem.dll");
    if(!module.ok()){return metadata_result::failure(module.error());}
    auto query=module.value()->resolve<decltype(&QnnSystemInterface_getProviders)>("QnnSystemInterface_getProviders");
    if(!query.ok()){return metadata_result::failure(query.error());}
    const QnnSystemInterface_t ** providers=nullptr;uint32_t count=0;
    const auto queried=query.value()(&providers,&count);
    if(queried!=QNN_SUCCESS || !count || !providers || !providers[0]) {
        return metadata_result::failure(host_error(error_stage::binary_restore,"System providers","No binary metadata provider"));
    }
    const auto & system=providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    if(!system.systemContextCreate || !system.systemContextGetBinaryInfo || !system.systemContextFree) {
        return metadata_result::failure(host_error(error_stage::binary_restore,"System interface","Missing binary metadata functions"));
    }
    QnnSystemContext_Handle_t handle=nullptr;
    auto code=system.systemContextCreate(&handle);
    if(code!=QNN_SUCCESS){return metadata_result::failure(sdk_error(error_stage::binary_restore,"systemContextCreate",code));}
    Qnn_ErrorHandle_t cleanup_code = QNN_SUCCESS;
    const auto release = [&](void *) {
        cleanup_code = system.systemContextFree(handle);
        if (cleanup_code != QNN_SUCCESS) {
            GGML_LOG_ERROR("qairt: systemContextFree failed (QNN error %llu)\n", (unsigned long long) cleanup_code);
        }
    };
    std::unique_ptr<void,decltype(release)> guard(handle,release);
    const QnnSystemContext_BinaryInfo_t * info=nullptr;Qnn_ContextBinarySize_t info_size=0;
    code=system.systemContextGetBinaryInfo(handle,bytes.data(),bytes.size(),&info,&info_size);
    if(code!=QNN_SUCCESS){return metadata_result::failure(sdk_error(error_stage::binary_restore,"systemContextGetBinaryInfo",code));}
    if(!info || info->version!=QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3 || !info->contextBinaryInfoV3.numGraphs ||
       info->contextBinaryInfoV3.numGraphs>256 || !info->contextBinaryInfoV3.graphs) {
        return metadata_result::failure(host_error(error_stage::binary_restore,"binary metadata","Unsupported binary graph metadata"));
    }
    std::vector<binary_graph> result;
    const auto & details=info->contextBinaryInfoV3;
    for(uint32_t i=0;i<details.numGraphs;++i) {
        const auto & graph=details.graphs[i];
        if(graph.version!=QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3){throw std::runtime_error("Unsupported graph metadata version");}
        const auto & g=graph.graphInfoV3;
        if(!g.graphName || std::strlen(g.graphName)>128 || g.numGraphInputs>256 || g.numGraphOutputs>128 ||
           !g.numGraphInputs || !g.numGraphOutputs || !g.graphInputs || !g.graphOutputs){throw std::runtime_error("Invalid graph I/O metadata");}
        if((runtime->config().verbose_diagnostics || diagnostic_phase()) && g.graphBlobInfo && g.graphBlobInfoSize>=sizeof(QnnHtpSystemContext_GraphBlobInfo_t)+sizeof(QnnHtpSystemContext_GraphBlobInfoV2_t)) {
            QnnHtpSystemContext_GraphBlobInfoV2_t memory{};
            std::memcpy(&memory,static_cast<const uint8_t *>(g.graphBlobInfo)+sizeof(QnnHtpSystemContext_GraphBlobInfo_t),sizeof(memory));
            if(memory.version==QNN_SYSTEM_CONTEXT_HTP_GRAPH_INFO_BLOB_VERSION_V2) {
                GGML_LOG_INFO("qairt: archive metadata graph=%s opdata_bytes=%llu const_bytes=%llu ddr_tensor_bytes=%llu io_bytes=%llu shared_weights_bytes=%llu\n",
                    g.graphName,(unsigned long long)memory.opDataSize,(unsigned long long)memory.constSize,
                    (unsigned long long)memory.ddrTensorSize,(unsigned long long)memory.ioTensorSize,(unsigned long long)memory.sharedWeightsSize);
            }
        }
        binary_graph record;record.name=g.graphName;
        const auto copy=[&](const Qnn_Tensor_t & tensor,Qnn_TensorType_t role) {
            const auto abi=[&](const auto & t) {
                if(!t.name || std::strlen(t.name)>128 || t.type!=role || !t.rank || t.rank>8 || !t.dimensions) {
                    throw std::runtime_error("Invalid binary tensor metadata");
                }
                tensor_spec spec{t.name,t.type,t.dataType,{t.dimensions,t.dimensions+t.rank}};
                if(!tensor_record::checked_bytes(spec).ok()){throw std::runtime_error("Unsupported binary tensor type or shape");}
                record.io.push_back({std::move(spec),{t.id}});
            };
            if(tensor.version==QNN_TENSOR_VERSION_1){abi(tensor.v1);}
            else if(tensor.version==QNN_TENSOR_VERSION_2){abi(tensor.v2);}
            else{throw std::runtime_error("Unsupported binary tensor version");}
        };
        for(uint32_t j=0;j<g.numGraphInputs;++j){copy(g.graphInputs[j],QNN_TENSOR_TYPE_APP_WRITE);}
        for(uint32_t j=0;j<g.numGraphOutputs;++j){copy(g.graphOutputs[j],QNN_TENSOR_TYPE_APP_READ);}
        result.push_back(std::move(record));
    }
    guard.reset();
    if (cleanup_code != QNN_SUCCESS) {
        return metadata_result::failure(sdk_error(error_stage::cleanup, "systemContextFree", cleanup_code));
    }
    return metadata_result::success(std::move(result));
}
}

result<std::string> context_owner::save_binary(const std::filesystem::path & path) {
    using output=result<std::string>;
    if(!handle() || path.empty() || !binary_.empty()){return output::failure(host_error(error_stage::binary_restore,"save binary","No construction context"));}
    Qnn_ContextBinarySize_t size=0,written=0;
    auto code=runtime_->domain()->call([&](const auto & api){return api.contextGetBinarySize ? api.contextGetBinarySize(handle(),&size) : QNN_COMMON_ERROR_NOT_SUPPORTED;});
    if(code!=QNN_SUCCESS){return output::failure(sdk_error(error_stage::binary_restore,"contextGetBinarySize",code));}
    if(!size || size>std::numeric_limits<ULONG>::max()){return output::failure(host_error(error_stage::binary_restore,"save binary","Binary exceeds supported size"));}
    std::vector<uint8_t> bytes(size);
    code=runtime_->domain()->call([&](const auto & api){return api.contextGetBinary ? api.contextGetBinary(handle(),bytes.data(),size,&written) : QNN_COMMON_ERROR_NOT_SUPPORTED;});
    if(code!=QNN_SUCCESS){return output::failure(sdk_error(error_stage::binary_restore,"contextGetBinary",code));}
    if(!written || written>size){return output::failure(host_error(error_stage::binary_restore,"save binary","Invalid written binary size"));}
    bytes.resize(written);const auto digest=binary_digest(bytes);
    const auto temporary=std::filesystem::path(path.wstring()+L".tmp");
    std::ofstream file(temporary,std::ios::binary|std::ios::trunc);file.write(reinterpret_cast<const char *>(bytes.data()),std::streamsize(bytes.size()));file.close();
    if(!file || !MoveFileExW(temporary.c_str(),path.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)) {
        return output::failure(host_error(error_stage::binary_restore,"save binary","Archive write or rename failed"));
    }
    GGML_LOG_INFO("qairt: archive saved bytes=%llu sha256=%s\n",(unsigned long long)written,digest.c_str());
    return output::success(digest);
}

result<std::shared_ptr<context_owner>> context_owner::from_binary(std::shared_ptr<runtime_owner> runtime,
    const std::filesystem::path & path,const std::string & digest) {
    using output=result<std::shared_ptr<context_owner>>;
    if(!runtime || runtime->closed()){return output::failure(host_error(error_stage::binary_restore,"load binary","No live runtime"));}
    auto owner=std::shared_ptr<context_owner>(new context_owner(),context_owner::dispose);owner->runtime_=runtime;
    std::ifstream file(path,std::ios::binary|std::ios::ate);const auto size=file.tellg();
    if(!file || size<=0 || uint64_t(size)>std::numeric_limits<ULONG>::max()){return output::failure(host_error(error_stage::binary_restore,"load binary","Invalid archive file size"));}
    owner->binary_.resize(size_t(size));file.seekg(0);file.read(reinterpret_cast<char *>(owner->binary_.data()),std::streamsize(size));
    if(!file || binary_digest(owner->binary_)!=digest){return output::failure(host_error(error_stage::binary_restore,"load binary","Archive integrity mismatch"));}
    auto metadata=read_metadata(runtime,owner->binary_);
    if(!metadata.ok()){return output::failure(metadata.error());}
    owner->binary_graphs_=std::move(metadata).value();
    Qnn_ContextHandle_t context=nullptr;
    const auto previous=runtime->log_level();
    const char * phase=diagnostic_phase();
    const bool trace=phase && std::strcmp(phase,"load_archives")==0;
    if(trace){const auto level=runtime->set_log_level(QNN_LOG_LEVEL_DEBUG);if(!level.ok()){return output::failure(level.error());}}
    GGML_LOG_INFO("qairt: archive deserialize_begin bytes=%zu\n",owner->binary_.size());
    const auto code=runtime->domain()->call([&](const auto & api){return api.contextCreateFromBinary ?
        api.contextCreateFromBinary(runtime->backend(),runtime->device(),runtime->context_configs(),owner->binary_.data(),owner->binary_.size(),&context,runtime->profile()) : QNN_COMMON_ERROR_NOT_SUPPORTED;});
    if(trace && !runtime->set_log_level(previous).ok()){GGML_LOG_ERROR("qairt: archive failed to restore SDK log level\n");}
    GGML_LOG_INFO("qairt: archive deserialize_end code=%llu\n",(unsigned long long)code);
    if(code!=QNN_SUCCESS){return output::failure(sdk_error(error_stage::binary_restore,"contextCreateFromBinary",code));}
    owner->context_.adopt_into_empty(context,runtime->domain(),{runtime->profile()});++runtime->active_contexts_;owner->admitted_=true;
    GGML_LOG_INFO("qairt: archive loaded context=%p graphs=%zu bytes=%zu\n",context,owner->binary_graphs_.size(),owner->binary_.size());
    return output::success(std::move(owner));
}

status graph_owner::suspend() {
    if(active_bindings_ || state_!=graph_state::finalized || !context_){return status::failure(host_error(error_stage::cleanup,"graph suspend","Graph is not idle and finalized"));}
    --context_->active_graphs_;context_.reset();graph_=nullptr;state_=graph_state::closed;
    return status::success();
}

result<std::vector<std::pair<tensor_id,tensor_id>>> graph_owner::restore(std::shared_ptr<context_owner> context) {
    using output=result<std::vector<std::pair<tensor_id,tensor_id>>>;
    const binary_graph * metadata=nullptr;
    if(!context || !context->handle() || context_ || state_!=graph_state::closed) {
        return output::failure(host_error(error_stage::binary_restore,"graphRetrieve","Invalid restore state"));
    }
    for(const auto & graph:context->binary_graphs()){if(graph.name==name_){metadata=&graph;break;}}
    if(!metadata || metadata->io.size()!=input_ids_.size()+output_ids_.size()) {
        return output::failure(host_error(error_stage::binary_restore,"graphRetrieve","Binary graph I/O count mismatch"));
    }
    std::vector<std::unique_ptr<tensor_record>> records;std::vector<std::pair<tensor_id,tensor_id>> mapping;
    std::vector<tensor_id> inputs,outputs;
    for(const auto & loaded:metadata->io) {
        const tensor_record * original=nullptr;
        for(const auto & tensor:tensors_){if(tensor->spec().name==loaded.spec.name){original=tensor.get();break;}}
        if(!original || original->spec().role!=loaded.spec.role || original->spec().dtype!=loaded.spec.dtype ||
           original->spec().dimensions!=loaded.spec.dimensions){return output::failure(host_error(error_stage::binary_restore,"graphRetrieve","Binary graph I/O layout mismatch"));}
        for(const auto & record:records){if(record->id().value==loaded.id.value || record->spec().name==loaded.spec.name){return output::failure(host_error(error_stage::binary_restore,"graphRetrieve","Duplicate binary tensor"));}}
        auto record=std::unique_ptr<tensor_record>(new tensor_record(loaded.spec));record->id_=loaded.id;
        mapping.push_back({original->id(),loaded.id});
        (loaded.spec.role==QNN_TENSOR_TYPE_APP_WRITE?inputs:outputs).push_back(loaded.id);records.push_back(std::move(record));
    }
    Qnn_GraphHandle_t graph=nullptr;
    const auto code=context->runtime()->domain()->call([&](const auto & api){return api.graphRetrieve ? api.graphRetrieve(context->handle(),name_.c_str(),&graph) : QNN_COMMON_ERROR_NOT_SUPPORTED;});
    if(code!=QNN_SUCCESS){return output::failure(sdk_error(error_stage::binary_restore,"graphRetrieve",code));}
    context_=std::move(context);++context_->active_graphs_;graph_=graph;tensors_=std::move(records);
    input_ids_=std::move(inputs);output_ids_=std::move(outputs);state_=graph_state::finalized;
    return output::success(std::move(mapping));
}
}