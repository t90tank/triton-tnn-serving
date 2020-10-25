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
               const int device_id,
               const std::string &path,
               const std::string &proto_content,
               const std::string &model_content, 
               const std::string &library_path) : 
                name_(name), device_id_(device_id), path_(path),
                proto_content_(proto_content),
                model_content_(model_content),
                library_path_(library_path) {}
  ~TNNProcessor();
  //创建，backend创建一个服务实例，就会创建一个对应的TNNProcessor
  static bool Create(const std::string &name,
                     const int device_id,
                     const std::string &path,
                     std::shared_ptr<TNNProcessor> &processor);

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
  //自动reshape，逻辑：
  //如果当前request_input_shape_map_ > instance_input_shape_map_，调整instance_input_shape_map_并重新init
  //如果当前request_input_shape_map_ <= instance_input_shape_map_ 则只调用instance_自带的reshape
  bool AutoReshape(); 

  //未来接口，手动reshape，输入张量reshape，将其reshpe
  bool ManualReshape(const TNN_NS::InputShapesMap& inputs); 

private:

  //TNN instance的Init，传入proto内容，模型内容，链接库目录，计算单元，在create时调用
  virtual TNN_NS::Status Init(TNNComputeUnits units,
                              const TNN_NS::InputShapesMap &input_shape = TNN_NS::InputShapesMap()); 

  //根据输入名称得到图形变换的参数，未来将通过配置或其他方式加载 by XiGao
  //现在根据TNN开源demo提供的参数转换，通过input名称来分辨
  TNN_NS::MatConvertParam GetConvertParam(std::string input_name) {
    TNN_NS::MatConvertParam input_cvt_param;
    input_cvt_param.scale = {1.0 / (255 * 0.229), 1.0 / (255 * 0.224), 1.0 / (255 * 0.225), 0.0};
    input_cvt_param.bias = {-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225, 0.0};
    
    if (input_name == "input") {
      input_cvt_param.scale = {1.0 / 128, 1.0 / 128, 1.0 / 128, 0.0};
      input_cvt_param.bias  = {-127.0 / 128, -127.0 / 128, -127.0 / 128, 0.0};
    }
    return input_cvt_param; 
  }

private:
  // 来自triton的数据
  const std::string name_;
  const int device_id_;
  const std::string path_; 

  //加载path得到的内容
  const std::string proto_content_;
  const std::string model_content_; 
  const std::string library_path_;

  //当前网络实例的shape，每次init时重置
  TNN_NS::InputShapesMap instance_input_shape_map_; 
  //当前request的shape，每次SetInputMat时修改其中的值
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