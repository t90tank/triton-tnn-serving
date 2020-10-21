#include "TNNProcessor.h"
#include <iostream>
#include <string>
#include <fstream>

namespace TNN_FOR_TRITION {

std::string fdLoadFile(std::string path) {
  std::ifstream file(path, std::ios::in);
  if (file.is_open()) {
      file.seekg(0, file.end);
      int size      = file.tellg();
      char* content = new char[size];

      file.seekg(0, file.beg);
      file.read(content, size);
      std::string fileContent;
      fileContent.assign(content, size);
      delete[] content;
      file.close();
      return fileContent;
  } else {
      return "";
  }
}

bool TNNProcessor::Create(const std::string &name,
                          const int device_id, 
                          const std::string &path,
                          std::shared_ptr<TNN_NS::InputShapesMap> shapeMap,
                          std::shared_ptr<TNNProcessor> &processor) {
  processor = std::make_shared<TNNProcessor>(name, device_id, shapeMap);
  std::string proto_content = fdLoadFile(path + "/proto.tnnproto"); 
  std::string model_content = fdLoadFile(path + "/model.tnnmodel");
  //多线程的init有些不稳定，待解决……
  for (auto pp : *(shapeMap)) {
    std::cout<<pp.first<<std::endl; 
    std::cout<<"["; 
    for (auto i : pp.second) 
      std::cout<<i<<','; 
    std::cout<<"]\n"; 
  }
  std::cout<<name<<" start init!\n";
  auto status = processor->Init(proto_content, 
                                model_content, 
                                "", 
                                TNNComputeUnitsCPU);
  std::cout<<name<<" finish init!\n"; 
  if (status != TNN_NS::TNN_OK) {
    LOGE( "Unable to load following files :\n");
    LOGE( "%s/proto.tnnproto\n", path.c_str()); 
    LOGE( "%s/model.tnnmodel\n", path.c_str()); 
    return false; 
  }
  return true;
}

// bool TNNProcessor::Run(const void *input, void **output) {
//   auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, nchw_, const_cast<void *>(input));
//   if (image_mat == nullptr) {
//     LOGE("TNN load input tensor from triton input failed!\n");
//     return false; 
//   }
//   auto status = Forward(image_mat, output_mat); 
//   if (status != TNN_NS::TNN_OK) {
//     LOGE("Run TNN net failed %d\n", (int)status); 
//     return false; 
//   }
//   if (output_mat == nullptr) {
//     LOGE("Output_mat is empty!\n"); 
//     return false; 
//   }
//   *output = output_mat.get()->GetData(); 
//   return true; 
// }

// bool TNNProcessor::GetOutputShape(long **output_shape, int *output_dims_count) {
//   if (!output_mat) {
//     LOGE("Can not GetOutputShape before sucessfully run!\n"); 
//     return false; 
//   }

//   //将vector<int>转化为vector<long>，建议后面和triton商量改变接口，不然浪费时间和效率
//   //by XiGao
//   output_shape_buffer.clear(); 
//   for (auto i : output_mat->GetDims()) output_shape_buffer.push_back(i);
//   *output_shape = output_shape_buffer.data(); 
//   *output_dims_count = output_mat.get()->GetDims().size(); 
//   return true; 
// }

// bool TNNProcessor::GetOutputSize(int *output_byte_size) const {
//   if (!output_mat) {
//     LOGE("Can not GetOutputSize before sucessfully run!\n"); 
//     return false; 
//   }
//   auto output_shape_ = output_mat->GetDims(); 
//   int total = 1; 
//   for (auto i : output_shape_) total *= i; 
//   //似乎目前Mat里面都是float类型？没有看到如何定义Mat数据类型？
//   *output_byte_size = total * sizeof(float); 
//   return true; 
// }

bool TNNProcessor::SetInputMat(const void *input_buffer, const std::string &input_name) {
  auto input_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, (*shape_map_)[input_name], const_cast<void *>(input_buffer)); 
  auto status = instance_->SetInputMat(input_mat, GetConvertParam(input_name));   
  return status == TNN_NS::TNN_OK; 
}

bool TNNProcessor::GetOutput(void **output_buffer,
                             long **output_shape,
                             int *output_dims_count,
                             int *output_byte_size,
                             const std::string &output_name) {
  auto status = instance_->GetOutputMat(output_mat, TNN_NS::MatConvertParam(), output_name);
  if (output_mat == nullptr) {
    LOGE("Output_mat is empty!\n"); 
    return false; 
  }
  *output_buffer = output_mat->GetData(); 

  output_shape_buffer.clear(); 
  for (auto i : output_mat->GetDims()) output_shape_buffer.push_back(i);
  *output_shape = output_shape_buffer.data(); 
  *output_dims_count = output_mat.get()->GetDims().size(); 
  
  int total = 1; 
  for (auto i : output_shape_buffer) total *= i; 
  //似乎目前Mat里面都是float类型？没有看到如何定义Mat数据类型？
  *output_byte_size = total * sizeof(float); 

  return status == TNN_NS::TNN_OK; 
}

bool TNNProcessor::Forward() {
  auto status = instance_->ForwardAsync(nullptr);
  if (status != TNN_NS::TNN_OK) {
      LOGE("instance forward failed %d\n", (int)status);
      return false; 
  }
  return true; 
}

TNN_NS::Status TNNProcessor::Init(const std::string &proto_content, const std::string &model_content,
                                  const std::string &library_path, TNNComputeUnits units) {
  //网络初始化
  TNN_NS::Status status;
  if (!net_) {
    TNN_NS::ModelConfig config;
#if TNN_SDK_USE_NCNN_MODEL
    config.model_type = TNN_NS::MODEL_TYPE_NCNN;
#else
    config.model_type = TNN_NS::MODEL_TYPE_TNN;
#endif
    config.params = {proto_content, model_content};

    auto net = std::make_shared<TNN_NS::TNN>();
    status = net->Init(config);
    if (status != TNN_NS::TNN_OK) {
      LOGE("instance.net init failed %d\n", (int)status);
      return status;
    }
    net_ = net;
  }

  // network init
  device_type_ = TNN_NS::DEVICE_ARM;
  if (units >= TNNComputeUnitsGPU) {
#if defined(__APPLE__) && TARGET_OS_IPHONE
    device_type_ = TNN_NS::DEVICE_METAL;
#else
    device_type_ = TNN_NS::DEVICE_OPENCL;
#endif
  }

  //创建实例instance
  TNN_NS::NetworkConfig network_config;
  network_config.library_path = {library_path};
  network_config.device_type = device_type_;
  auto instance = net_->CreateInst(network_config, status, *shape_map_);
  if (status != TNN_NS::TNN_OK || !instance)
  {
    // try device_arm
    if (units >= TNNComputeUnitsGPU)
    {
      device_type_ = TNN_NS::DEVICE_ARM;
      network_config.device_type = TNN_NS::DEVICE_ARM;
      instance = net_->CreateInst(network_config, status, *shape_map_);
    }
  }
  instance_ = instance;

  return status;
}

// TNN_NS::Status TNNProcessor::Forward(const std::shared_ptr<TNN_NS::Mat> input,
//                                     std::shared_ptr<TNN_NS::Mat> &output) {

//   // step 1. set input mat
//   TNN_NS::MatConvertParam input_cvt_param;
//   input_cvt_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
//   input_cvt_param.bias = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};

//   auto status = instance_->SetInputMat(input, input_cvt_param);
//   RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

//   // step 2. Forward
//   status = instance_->ForwardAsync(nullptr);
//   RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

//   // step 3. get output mat
//   status = instance_->GetOutputMat(output);
//   RETURN_ON_NEQ(status, TNN_NS::TNN_OK);

//   return TNN_NS::TNN_OK;
// }

} // namespace TNN_FOR_TRITION
