#ifndef TNNProcessor_H
#define TNNProcessor_H

#include <string>
#include <iostream>
#include "tnn/core/tnn.h"
#include "tnn/core/macro.h"

namespace TNN_DEMO {

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
                const int device_id,
                std::vector<int> nchw) : name_(name), device_id_(device_id), nchw_(nchw) {}
  //创建，backend创建一个服务实例，就会创建一个对应的TNNProcessor
  static bool Create(const std::string &name,
                      const int device_id,
                      const std::string &path,
                      std::shared_ptr<TNNProcessor> &processor, 
                      std::vector<int> nchw);

  //根据input的值得到输出，并将*output指向输出
  bool Run(const void *input, void **output);

  //下面两个函数在backend返回一个response的时调用，用于告知respon应该返回的张量形状和占用字节数
  //应该与config.pbtxt里面的shape和size对齐
  //这里没有考虑多个output的情况，因为TNN一般只支持一个返回float类型tensor
  bool GetOutputShape(long **output_shape, int *output_dims_count); 
  bool GetOutputSize(int *output_byte_size) const; 

private:
                
  virtual TNN_NS::Status Init(const std::string &proto_content, const std::string &model_content,
                              const std::string &library_path, TNNComputeUnits units);
  virtual TNN_NS::Status Forward(std::shared_ptr<TNN_NS::Mat> input, 
                                  std::shared_ptr<TNN_NS::Mat> &output);

private:
  const std::string name_;
  const int device_id_;
  std::vector<int> nchw_;

  std::shared_ptr<TNN_NS::TNN> net_ = nullptr;
  std::shared_ptr<TNN_NS::Instance> instance_ = nullptr;
  TNN_NS::DeviceType device_type_ = TNN_NS::DEVICE_ARM;

  std::shared_ptr<TNN_NS::Mat> output_mat; 
  std::vector<long> output_shape_buffer; 

};

} // namespace TNN_DEMO
#endif //TNNProcessor_H