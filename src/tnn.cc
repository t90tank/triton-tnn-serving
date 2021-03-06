// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <memory>
#include "triton/backend/backend_common.h"

//TNNDEMO
#include "TNNProcessor.h"
#include <sstream>
#include <algorithm>

namespace triton { namespace backend { namespace TNN_backend {

//
// Simple backend that demonstrates the TRITONBACKEND API for a
// blocking backend. A blocking backend completes execution of the
// inference before returning from TRITONBACKED_ModelInstanceExecute.
//
// This backend supports any model that has exactly 1 input and
// exactly 1 output. The input and output can have any name, datatype
// and shape but the shape and datatype of the input and output must
// match. The backend simply responds with the output tensor equal to
// the input tensor.
//

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }

  // Does this model support batching in the first dimension. This
  // function should not be called until after the model is completely
  // loaded.
  TRITONSERVER_Error* SupportsFirstDimBatching(bool* supports);

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();
  
  //TNNDEMO 暂时弃用Get the nchw from config for TNN，
  // std::shared_ptr<TNN_NS::InputShapesMap> GetInputShape(); 

 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* name, const uint64_t version,
      common::TritonJson::Value&& model_config);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  const uint64_t version_;
  common::TritonJson::Value model_config_;

  bool supports_batching_initialized_;
  bool supports_batching_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  common::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  *state = new ModelState(
      triton_server, triton_model, model_name, model_version,
      std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* name, const uint64_t version,
    common::TritonJson::Value&& model_config)
    : triton_server_(triton_server), triton_model_(triton_model), name_(name),
      version_(version), model_config_(std::move(model_config)),
      supports_batching_initialized_(false), supports_batching_(false)
{
}

TRITONSERVER_Error*
ModelState::SupportsFirstDimBatching(bool* supports)
{
  // We can't determine this during model initialization because
  // TRITONSERVER_ServerModelBatchProperties can't be called until the
  // model is loaded. So we just cache it here.
  if (!supports_batching_initialized_) {
    uint32_t flags = 0;
    RETURN_IF_ERROR(TRITONSERVER_ServerModelBatchProperties(
        triton_server_, name_.c_str(), version_, &flags, nullptr /* voidp */));
    supports_batching_ = ((flags & TRITONSERVER_BATCH_FIRST_DIM) != 0);
    supports_batching_initialized_ = true;
  }

  *supports = supports_batching_;
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  //TNNDEMO 无需判断下述逻辑
  //TODO 验证下input是否满足条件

  // // There must be 1 input and 1 output.
  // RETURN_ERROR_IF_FALSE(
  //     inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
  //     std::string("expected 1 input, got ") +
  //         std::to_string(inputs.ArraySize()));
  // RETURN_ERROR_IF_FALSE(
  //     outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
  //     std::string("expected 1 output, got ") +
  //         std::to_string(outputs.ArraySize()));

  // common::TritonJson::Value input, output;
  // RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  // RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

  // // Input and output must have same datatype
  // std::string input_dtype, output_dtype;
  // RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
  // RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));

  // RETURN_ERROR_IF_FALSE(
  //     input_dtype == output_dtype, TRITONSERVER_ERROR_INVALID_ARG,
  //     std::string("expected input and output datatype to match, got ") +
  //         input_dtype + " and " + output_dtype);

  // // Input and output must have same shape
  // std::vector<int64_t> input_shape, output_shape;
  // RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
  // RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

  // RETURN_ERROR_IF_FALSE(
  //     input_shape == output_shape, TRITONSERVER_ERROR_INVALID_ARG,
  //     std::string("expected input and output shape to match, got ") +
  //         backend::ShapeToString(input_shape) + " and " +
  //         backend::ShapeToString(output_shape));

  return nullptr;  // success
}

//TNNDEMO 暂时弃用 因为model_config_是private类型，返回input_size需要特殊的实现
// std::shared_ptr<TNN_NS::InputShapesMap> ModelState::GetInputShape() {
//   //get input shape from config
//   auto input_shape_map_ = std::make_shared<TNN_NS::InputShapesMap>(); 
//   common::TritonJson::Value inputs, input;
//   model_config_.MemberAsArray("input", &inputs); 

//   for (size_t r = 0; r < inputs.ArraySize(); ++r) {
//     continue; 
//     common::TritonJson::Value input; 
//     inputs.IndexAsObject(r, &input);
//     std::string input_name;
//     input.MemberAsString("name", &input_name);  
//     std::vector<int64_t> input_shape_64; 
//     backend::ParseShape(input, "dims", &input_shape_64); 
//     // 将只有一维-1的去除， 未来将在输入时验证
//     // 此处需修改 by XiGao
//     if (input_shape_64.size() <= 2) continue; 
//     std::cout<<"input_shape_64.size() = "<<input_shape_64.size()<<std::endl; 

//     std::vector<int> input_shape(input_shape_64.begin(), input_shape_64.end()); 

//     //警告! 这里我们还不支持多batch，所以动态添加一维batch_size=1
//     LOG_MESSAGE(
//         TRITONSERVER_LOG_INFO,
//         "Warning : do not support batching, so batch_size is set to 1.");
//     input_shape.push_back(1); 

//     //注意：TNN格式是nchw，这里需要reverse
//     reverse(input_shape.begin(), input_shape.end()); 

//     input_shape_map_->insert(std::pair<std::string, std::vector<int>>(input_name, input_shape)); 
//   }
//   return input_shape_map_; 

//   // 暂时弃用convert std::vector<int64_t> to std::vector<int> 
//   // std::vector<int> nchw = {1}; 
//   // for (auto x : input_shape) nchw.push_back(x); 
//   // reverse(next(nchw.begin()), nchw.end()); 

//   // //输出调试信息，得到的nchw
//   // std::string S_nchw = "["; 
//   // for (auto x : nchw) S_nchw = S_nchw+std::to_string(x)+','; 
//   // S_nchw.pop_back(); 
//   // S_nchw.push_back(']'); 
//   // LOG_MESSAGE(
//   //     TRITONSERVER_LOG_INFO,
//   //     (std::string("TRITONBACKEND_MODEL_NCHW: ") + S_nchw).c_str());
//   // return nchw; 
// }

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance()
  {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }
  //TNNDEMO 返回TNNProcessor处理器
  std::shared_ptr<TNN_FOR_TRITION::TNNProcessor> GetProcessor() const {return tnn_processor_; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id);

  //TNNDEMO 用于加载模型
  TRITONSERVER_Error* CreateTNNProcessor(); 

  ModelState* model_state_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;

  //TNNDEMO 在这里添加一个模型实例的TNN处理器，执行与TNN相关的模型网络加载，运行配置等逻辑
  std::shared_ptr<TNN_FOR_TRITION::TNNProcessor> tnn_processor_; 

};

//TNNDEMO 加载模型的实现，直接采用构造函数并传递path字符串
TRITONSERVER_Error* ModelInstanceState::CreateTNNProcessor() {
  //得到path
  TRITONBACKEND_ArtifactType artifatct_type; 
  const char *path = ""; 
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model_state_->TritonModel(), &artifatct_type, &path)); 

  std::stringstream ss;
  ss<<path<<'/'<<model_state_->Version(); 
  std::string path_version; 
  ss>>path_version; 

  RETURN_ERROR_IF_FALSE(
      TNN_FOR_TRITION::TNNProcessor::Create(tnn_processor_,
                                            name_, 
                                            device_id_, 
                                            std::string(path_version)), 
      TRITONSERVER_ERROR_NOT_FOUND, 
      std::string("Can not create TNNProcessor using path '") + 
      std::string(path_version) +
      std::string("'.")); 
      
  return nullptr; 
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(
      model_state, triton_model_instance, instance_name, instance_kind,
      instance_id);
    
  //TNNDEMO 调用TNNProcessor的创建
  RETURN_IF_ERROR(
    (*state)->CreateTNNProcessor()); 

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(name), kind_(kind), device_id_(device_id)
{
}

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  //TNNDEMO begin{该部分和0比较编译错误}

//   if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
//       (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
//     return TRITONSERVER_ErrorNew(
//         TRITONSERVER_ERROR_UNSUPPORTED,
//         "triton backend API version does not support this backend");
//   }

  //TNNDEMO end{该部分和0比较编译错误}

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // If we have any global backend state we create and set it here. We
  // don't need anything for this backend but for demonstration
  // purposes we just create something...
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Repository location: ") + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  std::string* backend_state = reinterpret_cast<std::string*>(vbackendstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend state is '") + *backend_state + "'").c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
      
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  // Because this backend just copies IN -> OUT and requires that
  // input and output be in CPU memory, we fail if a GPU instances is
  // requested.
  RETURN_ERROR_IF_FALSE(
      instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'tnn' backend only supports CPU instances"));


  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  bool supports_batching = false;
  RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));

  // 'responses' is initialized with the response objects below and
  // if/when an error response is sent the corresponding entry in
  // 'responses' is set to nullptr to indicate that that response has
  // already been sent.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  // Create a single response object for each request. If something
  // goes wrong when attempting to create the response objects just
  // fail all of the requests by returning an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // The way we collect these batch timestamps is not entirely
  // accurate. Normally, in a performant backend you would execute all
  // the requests at the same time, and so there would be a single
  // compute-start / compute-end time-range. But here we execute each
  // request separately so there is no single range. As a result we
  // just show the entire execute time as being the compute time as
  // well.
  uint64_t min_exec_start_ns = std::numeric_limits<uint64_t>::max();
  uint64_t max_exec_end_ns = 0;
  uint64_t total_batch_size = 0;

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  // For simplicity we just process each request separately... in
  // general a backend should try to operate on the entire batch of
  // requests at the same time for improved performance.
  for (uint32_t r = 0; r < request_count; ++r) {
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);
    min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);

    TRITONBACKEND_Request* request = requests[r];

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &request_id));

    uint64_t correlation_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    // Triton ensures that there is only a single input since that is
    // what is specified in the model configuration, so normally there
    // would be no reason to check it but we do here to demonstate the
    // API.
    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    // If an error response was sent for the above then display an
    // error message and move on to next request.
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read request input/output counts, error response sent")
              .c_str());
      continue;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("request ") + std::to_string(r) + ": id = \"" +
         request_id + "\", correlation_id = " + std::to_string(correlation_id) +
         ", input_count = " + std::to_string(input_count) +
         ", requested_output_count = " + std::to_string(requested_output_count))
            .c_str());

    //TNN_DEMO step1. 我们需要拉入所有inputs
    //首先拿到input_name
    //通过input_name拿到input
    //解析input得到大量信息
    //因为input内存可能不连续，将其复制到申请的vector
    //将其加入Processor的Mat中
    for (size_t i = 0; i < input_count; ++i) {

      //按照下标i拉取input_name和input
      const char* input_name;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestInputName(request, i, &input_name));
      TRITONBACKEND_Input* input = nullptr;
      GUARDED_RESPOND_IF_ERROR(
          responses, r, TRITONBACKEND_RequestInput(request, input_name, &input));

      //解析input类得到更多信息
      TRITONSERVER_DataType input_datatype;
      const int64_t* input_shape;
      uint32_t input_dims_count;
      uint64_t input_byte_size;
      uint32_t input_buffer_count;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_InputProperties(
              input, nullptr /* input_name */, &input_datatype, &input_shape,
              &input_dims_count, &input_byte_size, &input_buffer_count));

      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("\tinput ") + input_name +
          ": datatype = " + TRITONSERVER_DataTypeString(input_datatype) +
          ", shape = " + backend::ShapeToString(input_shape, input_dims_count) +
          ", byte_size = " + std::to_string(input_byte_size) +
          ", buffer_count = " + std::to_string(input_buffer_count))
              .c_str());
      
      //将input_buffer复制到连续内存
      std::vector<char> in_buffer(input_byte_size / sizeof(char));
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          ReadInputTensor(
              request, input_name, in_buffer.data(),
              &input_byte_size));

      //通过input_shape和input_dims_count还原nchw，注意这里还没有batch，我们手工添加一个batchsize:n=1
      std::vector<int> nchw; 
      for (size_t i = 0; i < input_dims_count; ++i) 
        nchw.push_back(input_shape[i]); 
      nchw.push_back(1); 
      std::reverse(nchw.begin(), nchw.end()); 
            
      //设置输入的Mat
      RETURN_ERROR_IF_FALSE(
        instance_state->GetProcessor()->SetInputMat(in_buffer.data(), input_name, nchw),
        TRITONSERVER_ERROR_UNKNOWN,
        std::string("instance_state->SetInputMat() unsuccessful!")); 

      //batch 相关，尚未调试
      if (i == 0) {
        if (supports_batching && (input_dims_count > 0)) {
          total_batch_size += input_shape[0];
        } else {
          total_batch_size++;
        }
      }
    }

    // We only need to produce an output if it was requested.

    if (requested_output_count > 0) {

      //TNNDEMO step.2 Reshape 以及前向计算
      //先根据输入Mat大小决定是否重新分配内存
      //目前采用自动调整的逻辑，即，如果不够就重新分配
      //将前一步骤中所有输入做前向计算
      //计算得到的所有output都以Mat格式存储在Processor中
      //之后解析即可
      RETURN_ERROR_IF_FALSE(
          instance_state->GetProcessor()->AutoReshape(), 
          TRITONSERVER_ERROR_UNKNOWN,
          std::string("instance_state->GetProcessor()->AutoReshape() unsuccessful!") ); 

      RETURN_ERROR_IF_FALSE(
          instance_state->GetProcessor()->Forward(), 
          TRITONSERVER_ERROR_UNKNOWN,
          std::string("instance_state->GetProcessor()->Run() unsuccessful!") ); 

      TRITONBACKEND_Response* response = responses[r];

      //TNNDEMO step.3 得到所有outputs并复制到buffer  
      //枚举所有responses[r]的request的output
      //得到名称
      //根据名称在Processor的TNN示例中获取outputMat
      //TNN返回OutputMat同时返回其他信息和数据指针output_TNN
      //申请绑定在request上的output
      //申请绑定在output上的output_buffer
      //将output_TNN复制到output_buffer中
      for (size_t i = 0; i < requested_output_count; ++i) {
        const char* requested_output_name = nullptr;
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONBACKEND_RequestOutputName(
                  request, i /* index */, &requested_output_name));
        
        void * output_TNN; 
        long *output_shape = nullptr; 
        int output_dims_count = 0; 
        int output_byte_size = 0; 
        RETURN_ERROR_IF_FALSE( 
            instance_state->GetProcessor()->GetOutput(&output_TNN, 
                                                      &output_shape,
                                                      &output_dims_count,
                                                      &output_byte_size,
                                                      std::string(requested_output_name)),
            TRITONSERVER_ERROR_UNKNOWN,
            std::string("instance_state->GetProcessor()->GetOutput() unsuccessful!")); 
            
        TRITONSERVER_DataType output_datatype = TRITONSERVER_DataType::TRITONSERVER_TYPE_FP32; 

        //输出output的调试信息
        LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("\toutput ") + requested_output_name +
          ": datatype = " + TRITONSERVER_DataTypeString(output_datatype) +
          ", shape = " + backend::ShapeToString(output_shape, output_dims_count) +
          ", byte_size = " + std::to_string(output_byte_size)) 
              .c_str());

        //创建一个output
        TRITONBACKEND_Output* output;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_ResponseOutput(
                response, &output, requested_output_name, output_datatype,
                output_shape, output_dims_count));

        //创建output_buffer
        void* output_buffer;
        TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t output_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_OutputBuffer(
                output, &output_buffer, output_byte_size, &output_memory_type,
                &output_memory_type_id));
        
        //将output_TNN复制到output_buffer
        memcpy(output_buffer, output_TNN, output_byte_size); 
      }
    }

    // To demonstrate response parameters we attach some here. Most
    // responses do not use parameters but they provide a way for
    // backends to communicate arbitrary information along with the
    // response.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetStringParameter(
            responses[r], "param0", "an example string parameter"),
        "failed setting string parameter");
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetIntParameter(responses[r], "param1", 42),
        "failed setting integer parameter");
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetBoolParameter(responses[r], "param2", false),
        "failed setting boolean parameter");

    // If we get to this point then there hasn't been any error and
    // the response is complete and we can send it. This is the last
    // (and only) response that we are sending for the request so we
    // must mark it FINAL. If there is an error when sending all we
    // can do is log it.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            nullptr /* success */),
        "failed sending response");

    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);

    // Report statistics for the successful request. For an instance
    // using the CPU we don't associate any device with the
    // statistics, otherwise we associate the instance's device.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request, true /* success */,
            exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");
  }

  // Done with requests...

  // There are two types of statistics that we can report... the
  // statistics for the entire batch of requests that we just executed
  // and statistics for each individual request. Statistics for each
  // individual request were reported above inside the loop as each
  // request was processed (or for failed requests we report that
  // failure below). Here we report statistics for the entire batch of
  // requests.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          min_exec_start_ns, min_exec_start_ns, max_exec_end_ns,
          max_exec_end_ns),
      "failed reporting batch request statistics");

  // We could have released each request as soon as we sent the
  // corresponding response. But for clarity we just release them all
  // here. Note that is something goes wrong when releasing a request
  // all we can do is log it... there is no response left to use to
  // report an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // Before releasing, record failed requests as those where
    // responses[r] is nullptr. The timestamps are ignored in this
    // case.
    if (responses[r] == nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ModelInstanceReportStatistics(
              instance_state->TritonModelInstance(), request,
              false /* success */, 0, 0, 0, 0),
          "failed reporting request statistics");
    }

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  return nullptr;  // success
}

}  // extern "C"

}}} // namespace triton::backend::TNN_backend
