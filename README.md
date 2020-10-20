# 简介

本项目为基于triton-inference-server（https://github.com/triton-inference-server）开发的tnn-backend例子，阅读https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html 以了解更多关于triton-inference-server的内容以及backend的概念。阅读https://github.com/Tencent/TNN 以了解更多关于NN。本项目暂时支持使用CPU运行服务。

triton-inference-server版本:20.09
TNN版本，见src/tnn/version.h

# 快速开始

如果还没有tritonserver镜像，先拉取tritonserver镜像
```
docker pull nvcr.io/nvidia/tritonserver:20.09-py3
```

进入triton-tnn-serving文件夹，并利用镜像运行服务，将准备好的模型文件夹my_models挂载到models目录，tritonserver将自动加载
```
docker run -p8000:8000 -p8001:8001 -p8002:8002 -it -v $(pwd)/my_models:/models nvcr.io/nvidia/tritonserver:20.09-py3 tritonserver --model-store=/models
```

客户端为未完善功能：

运行客户端，将已经转换的图片数据data.txt发送给triton的http接口，

```
cd tnnserving_client
python3 image_client.py
```

如果想要查看其他图片分类结果，可以将其他图片通过jpgtodata.py转换为data.txt，修改jpgtodata源码，并运行：
```
python3 jpgtodata.py 
```
目前只支持224X224的图片大小，其他图片需要手动调整大小后再放松

# 编译tnn-backend

由于只支持CPU的triton-inference-server backend源码编译上存在问题，建议使用smartbuild.sh，该脚本会自动将能编译通过的backend_common.cc复制到cmake拉取的文件夹中以修正编译错误。

```
smartbuild.sh
```
编译结果为build/install/backends/tnn/libtriton_tnn.so

#设置模型文件夹

本项目中./my_models为一个模型文件夹的例子
阅读https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_repository.html 以了解更多关于triton-inference-server模型文件夹的架构
阅读https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/backend.html#backend-shared-library 以了解动态链接库libtriton_tnn.so被加载的逻辑

注意因为TNN动态链接库libTNN.so的某些原因，现在不能改变模型名称，请勿修改文件夹结构或模型名称

目前，暂定tnn-backend通过proto.tnnproto，model.tnnproto两个文件来加载TNN格式的网络结构，因此一个模型应该包含proto.tnnproto，model.tnnproto两个文件夹。
将编译好的libtriton_tnn.so也放在对应位置。
构建好文件目录后，调整config.pbtxt，配置backend为tnn，，尚未支持多batch的功能，所以将max_batch_size设置为0。

未确定功能：
TNN-serving正式上线后，TNN编译得到得动态链接库libTNN.so应该存储在镜像固定路径下（其余triton-inference-server的动态链接库有指定目录），但目前没有制作TNN-serving的镜像，所以TNN的动态链接库暂时直接暴露在模型文件夹下。导致模型结构不能随意改变。

# 源码解析

TODO

# 其他功能

在tnnserving_client中有auto-run.sh会不断运行image_client.py以确定是否有内存泄漏的情况
在运行后在http://localhost:8002/metrics 可以查看其他指标（尚未测试过）