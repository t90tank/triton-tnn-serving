# 简介

本项目为基于triton-inference-server（https://github.com/triton-inference-server）开发的tnn-backend例子，阅读https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html 以了解更多关于triton-inference-server的内容以及backend的概念。阅读https://github.com/Tencent/TNN 以了解更多关于NN。本项目暂时支持使用CPU运行服务。

triton-inference-server版本:20.09
TNN版本，见src/tnn/version.h

# 快速开始

### 通过镜像运行服务端

如果还没有tritonserver镜像，先拉取tritonserver镜像
```
docker pull nvcr.io/nvidia/tritonserver:20.09-py3
```

进入triton-tnn-serving文件夹，并利用镜像运行服务，将准备好的模型文件夹my_models挂载到models目录，my_lib挂载到默认的lib目录，tritonserver将自动加载
```
docker run -p8000:8000 -p8001:8001 -p8002:8002 -it -v $(pwd)/my_models:/models -v $(pwd)/my_libs:/opt/tritonserver/backends/tnn nvcr.io/nvidia/tritonserver:20.09-py3 tritonserver --model-store=/models
```
目前多模型有些不稳定，可能要多跑几次

### python客户端（未完善功能）：

运行客户端
```
cd tnnserving_client
python3 image_client_classification.py pics/dog.png
python3 image_client_face_detection.py pics/test_face.jpg
```

由于classification使用squeezeNet目前只支持224X224的图片大小
而face_detection支持特定几种图片大小
model和proto文件来源：https://github.com/Tencent/TNN/tree/master/model
该demo只会将模型init成目前出现过的最大的shape，并且将input_shape设置为当前图片的shape，然后强行运行网络，成功与否由网络结构决定

多模型加载同时init曾经出错过，但目前已经无法复现该BUG

# 编译tnn-backend

```
./build.sh
```
编译结果为build/install/backends/tnn/libtriton_tnn.so
目前smartbuild.sh会默认将新生成的so复制到./my_libs下覆盖原本的库

# 设置模型文件夹

### 请先阅读tritoon-inference-server手册了解模型文件夹
阅读https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_repository.html 以了解更多关于triton-inference-server模型文件夹的架构

### 手动设置模型文件夹
本项目中./my_models为一个模型文件夹的例子。目前，暂定tnn-backend通过proto.tnnproto，model.tnnproto两个文件来加载TNN格式的网络结构，因此一个模型应该包含proto.tnnproto，model.tnnmodel两个文件。构建好文件目录后，调整config.pbtxt，配置backend为tnn，尚未支持多batch的功能，所以将max_batch_size设置为0。

总结必需的文件：
- config.pbtxt 配置文件
- proto.tnnproto TNN模型文件
- model.tnnmodel TNN模型文件


# 设置动态链接库
阅读https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend.html#backend-shared-library 以了解动态链接库libtriton_tnn.so被加载的逻辑

本项目中./my_lib为一个动态库的例子，里面包含本项目的动态链接库和TNN的动态链接库，在运行镜像时挂载

libtriton_tnn.so tnn-backend编译得到的动态链接库，将其挂载到对应位置即可，默认该so会在/opt/tritonserver/backends/tnn/下寻找TNN编译得到的动来链接库libTNN.so，阅读CMakeLists.txt以修改此逻辑
### TODO：
TNN-serving正式上线后，TNN编译得到得动态链接库libTNN.so和本项目的libtriton_tnn.so应该存储在镜像固定路径下/opt/tritonserver/backends/tnn/，无需挂载，但目前没有制作TNN-serving的镜像

# 源码解析

TODO

# BUGS

- 某些时候同时init会报错，无法复现
- 执行命令 python3 image_client_face_detection.py pics/face_216.jpg 时，会没有任何提示地崩溃，目前以确定错误在TNNProcessor.cc的92行中的auto status = instance_->SetInputMat(input_mat, GetConvertParam(input_name), input_name);  TNN内部崩溃，怀疑和TNN版本太旧有关
- 在设置输入张量size后输出size并没有变化，可能和TNN某些机制相关

# 其他功能

- **全新feature（待定）** 网络将根据输入张量的改变动态调整，具体逻辑可参考TNNProcessor.cc里的AutoReshape
- build.sh将使用默认方法编译（一般需要手动修改编译错误），comp.sh不删除build目录直接编译
- 在tnnserving_client中有auto-run.sh会不断运行image_client.py以确定是否有内存泄漏的情况
- 在运行后在http://localhost:8002/metrics 可以查看其他指标（尚未测试过）