#ifndef TNNProcessor_H
#define TNNProcessor_H

#include <string>
#include <iostream>
#include "tnn/core/tnn.h"
#include "tnn/core/macro.h"

namespace TNN_FOR_TRITION {

typedef enum{
  // run on cpu
  TNNComputeUnitsCPU = 0,
  // run on gpu, if failed run on cpu
  TNNComputeUnitsGPU = 1,
  // run on npu, if failed run on cpu
  TNNComputeUnitsNPU = 2,
} TNNComputeUnits;

class TNNProcessor{
public:
  TNNProcessor(const std::string &name,
                const int device_id) : name_(name), device_id_(device_id) {}
  //创建，backend创建一个服务实例，就会创建一个对应的TNNProcessor
  static bool Create(const std::string &name,
                      const int device_id,
                      const std::string &path,
                      std::shared_ptr<TNNProcessor> &processor);

  //(暂时弃用)根据input的值得到输出，并将*output指向输出。逻辑为转换指针为TNN_NS::Mat格式，前向计算，然后将结果转换为指针
  // bool Run(const void *input, void **output);
  //(暂时弃用)下面两个函数在backend返回一个response的时调用，用于告知respon应该返回的张量形状和占用字节数
  //这里没有考虑多个output的情况，因为TNN一般只支持一个返回float类型tensor
  // bool GetOutputShape(long **output_shape, int *output_dims_count); 
  // bool GetOutputSize(int *output_byte_size) const; 

  //根据input_buffer和input_name设置input_mat
  bool SetInputMat(const void *input_buffer, const std::string &input_name, const std::vector<int> &nchw); 
  //根据output_mat和output_name设置buffer，shape，dims_cout,byte_size
  bool GetOutput(void **output_buffer,
                long **output_shape,
                int *output_dims_count,
                int *output_byte_size,
                const std::string &output_name); 
  //运行，从input_mat得到output_mat
  bool Forward(); 
  //未来工作，输入张量reshape
  bool Reshape(const TNN_NS::InputShapesMap& inputs) {LOGE("TNN_FOR_TRITION::Reshape is not implemented!\n"); return false;}

private:

  //TNN instance的Init，传入proto内容，模型内容，链接库目录，计算单元，在create时调用
  virtual TNN_NS::Status Init(const std::string &proto_content, const std::string &model_content,
                              const std::string &library_path, TNNComputeUnits units,
                              const TNN_NS::InputShapesMap &input_shape = TNN_NS::InputShapesMap());
  //（暂时弃用）TNN instance的前向计算，在Run时调用
  // virtual TNN_NS::Status Forward(const std::shared_ptr<TNN_NS::Mat> input, 
  //                                 std::shared_ptr<TNN_NS::Mat> &output);    
  //根据输入名称得到图形变换的参数，未来将通过配置或其他方式加载 by XiGao
  TNN_NS::MatConvertParam GetConvertParam(std::string input_name) {
    TNN_NS::MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
    input_cvt_param.bias = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
    return input_cvt_param; 
  }

private:
  // 来自triton的数据
  const std::string name_;
  const int device_id_;
  TNN_NS::InputShapesMap instance_input_shape_map_; 
  TNN_NS::InputShapesMap request_input_shape_map_; 

  // TNN模型所需数据
  std::shared_ptr<TNN_NS::TNN> net_ = nullptr;
  std::shared_ptr<TNN_NS::Instance> instance_ = nullptr;
  TNN_NS::DeviceType device_type_ = TNN_NS::DEVICE_ARM;


  // 因为返回给triton_backend的是指针格式，输出指针的内存管理存在processor内部，使用智能指针管理，以降低内存泄漏风险
  std::shared_ptr<TNN_NS::Mat> output_mat; 
  std::vector<long> output_shape_buffer; 

};

} // namespace TNN_FOR_TRITION
#endif //TNNProcessor_H