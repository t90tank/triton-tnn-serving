

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Triton Inference TNN Backend

An simple TNN backend used for testing. You can learn more about
backends in the [backend repo](https://github.com/triton-inference-server/backend). Ask questions or report problems in the main Triton [issues page](https://github.com/triton-inference-server/server/issues).

Use cmake to build and install in a local directory.

```
./build.sh
```

You need to change some codes to make it work, see here for more information.

https://iwiki.woa.com/pages/viewpage.action?pageId=370600113



server
```
docker run -p8000:8000 -p8001:8001 -p8002:8002 -it -v $(pwd)/my_models:/models nvcr.io/nvidia/tritonserver:20.09-py3 tritonserver --model-store=/models
```

client
```
cd tnnserving_client
python3 image_client.py
```