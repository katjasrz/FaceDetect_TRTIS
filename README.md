# Face detection application using the FaceDetect purpose built model from NVIDIA GPU Cloud (NGC) and Triton Inference Server

[FaceDetect](https://ngc.nvidia.com/catalog/models/nvidia:tlt_facenet) is one of the purpose built models from NGC. In this project, we demonstrate how it can be deployed and utilized using Triton Inference Server.

## Acquiring the FaceDetect model and preparing a model repository

### Downloading the FaceDetect model

You can download a deployable model from the NGC using `wget` command

```shell
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/tlt_facenet/versions/deployable_v1.0/zip -O tlt_facenet_deployable_v1.0.zip
```

Extract the model from the downloaded archive

```shell
unzip  tlt_facenet_deployable_v1.0.zip -d models/  
rm  tlt_facenet_deployable_v1.0.zip 
```

A directory `models` will be created in your project folder, it will contain the extracted deployable FaceDetect model in `.etlt` format.

### Converting the model to TensorRT

As of the time of writing (March 2021), a model in `.etlt` format can not be directly deployed to Triton Inference Server. It needs to be converted into TensorRT format.

Download the `tlt-converter` utility from the [Transfer Learning Toolkit 3.0 page](https://developer.nvidia.com/tlt-get-started). In this demo, we are using [the vesrion](https://developer.nvidia.com/cuda111-cudnn80-trt72) corresponding to 

* CUDA 11.1
* cuDNN 8.0
* TensorRT 7.2

Select one which corresponds to your deployment platform.

Having downloaded the utility, unzip it, do the setup as described in the corresponding [README](trtis_model_repo/facedetect_tlt/1/README.md), navigate to the `tlt-converter`. Before executing it for thr first time you may need to change permissions for the file

```shell
chmod 777 tlt-converter 
``` 

Then run the following command (make sure to replace `PATH_TO_YOUR_PROJECT` with your correct path):

```shell
./tlt-converter /PATH_TO_YOUR_PROJECT/models/model.etlt \
               -k nvidia_tlt \
               -o output_cov/Sigmoid,output_bbox/BiasAdd \
               -d 3,416,736 \
               -i nchw \
               -m 64 \
               -t fp16 \
               -e /PATH_TO_YOUR_PROJECT/trtis_model_repo/facedetect_tlt/1/model.plan \
               -b 4
```
Note, the key `nvidia_tlt` corresponds to this particular model. A model key for a purpose build model can be found on the corresponding model card in the [NGC](https://ngc.nvidia.com/).

Upon successfull execution, the resulting `model.plan` should appear in the directory `/PATH_TO_YOUR_PROJECT/trtis_model_repo/facedetect_tlt/1/`.

If you want to learn more about `tlt-converter` parameters, execute

```shell
./tlt-converter -h
```

### Configuring model repository

Triton Inferense Server requires a model repository of a certain structure. For the given model it should be as follows:

```
└── trtis_model_repo
    └── facedetect_tlt
        ├── 1
        │   └── model.plan
        └── config.pbtxt
```

The `config.pbtx` contains the model configuration:

```
name: "facedetect_tlt"
platform: "tensorrt_plan"
max_batch_size: 64
input [
  {
    name: "input_1"
    data_type: TYPE_FP32
    dims: [ 3, 416, 736 ]
  }
]
output [
  {
    name: "output_bbox/BiasAdd"
    data_type: TYPE_FP32
    dims: [ 4, 26, 46 ]
  },
  {
    name: "output_cov/Sigmoid"
    data_type: TYPE_FP32
    dims: [ 1, 26, 46 ]
  }
]
```

## Setting up the server

Refer to the [quick start guide](https://github.com/triton-inference-server/server/blob/master/docs/quickstart.md), in order to learn how to set up Triton Inference Server. For this demo, we are using the [Docker](https://docs.docker.com/engine/install/) way of running the server, the NGC container version 20.11 corresponding to the Triton Inference Server version 2.5.0. You can acquire it by executing 

```shell
docker pull nvcr.io/nvidia/tritonserver:20.11-py3
``` 
Note, that this container requires

* CUDA 11.1
* CUDA driver 455.x

It includes TensoRT version 7.2.1, required for comaptibility with exported FaceDetect model.

## Instantiating the server

Once you have prepared your model repository and installed the server, you should have everything ready for serving the model. Execute the following command to get the server running:

``` shell
docker run --rm -it --net host --gpus all \
    --name image_tlt_cv_server \
    --ipc "host" \
    -v /PATH_TO_YOUR_PROJECT/trtis_model_repo:/model_repository \
    nvcr.io/nvidia/tritonserver:20.11-py3 /opt/tritonserver/bin/tritonserver \
        --model-store /model_repository --log-info=true --exit-on-error=false
```

Make sure to provide the correct path to your project.

The server is ready to use once you see the following on the terminal:

```shell
I0327 17:08:47.117149 1 server.cc:184] 
+----------------+---------+--------+
| Model          | Version | Status |
+----------------+---------+--------+
| facedetect_tlt | 1       | READY  |
+----------------+---------+--------+

I0327 17:08:47.117364 1 tritonserver.cc:1620] 
+----------------------------------+------------------------------------------------------------------------------+
| Option                           | Value                                                                        |
+----------------------------------+------------------------------------------------------------------------------+
| server_id                        | triton                                                                       |
| server_version                   | 2.5.0                                                                        |
| server_extensions                | classification sequence model_repository schedule_policy model_configuration |
|                                  |  system_shared_memory cuda_shared_memory binary_tensor_data statistics       |
| model_repository_path[0]         | /model_repository                                                            |
| model_control_mode               | MODE_NONE                                                                    |
| strict_model_config              | 1                                                                            |
| pinned_memory_pool_byte_size     | 268435456                                                                    |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                     |
| min_supported_compute_capability | 6.0                                                                          |
| strict_readiness                 | 1                                                                            |
| exit_timeout                     | 30                                                                           |
+----------------------------------+------------------------------------------------------------------------------+

I0327 17:08:47.119604 1 grpc_server.cc:3979] Started GRPCInferenceService at 0.0.0.0:8001
I0327 17:08:47.120206 1 http_server.cc:2717] Started HTTPService at 0.0.0.0:8000
I0327 17:08:47.163077 1 http_server.cc:2736] Started Metrics Service at 0.0.0.0:8002
```

If you've done everything correctly, model status will be shown as `READY`.

## Client application

We implemented our detection client in Python as a Jupyter Notebook walk-through. In order to replicate it you should have Python >=3.6.9 installed.

Furthermore, install the following packages in your environment to run the demo:

```shell
pip install notebook
pip install numpy
pip install matplotlib
pip install Pillow==2.2.1
pip install tritonclient[all]
pip install opencv-python
```

Navigate to the `detection_client` directory of this repository and execute

```shell
jupyter notebook --NotebookApp.iopub_data_rate_limit=1e10
```
In the appeared browser window open `detection_client.ipynb` and follow the instructuins in the notebook.

## Metrics visualization

Triton Inference Server provides a metrics interface in Prometheus data format. [These metrics](https://github.com/triton-inference-server/server/blob/master/docs/metrics.md) indicate GPU utilization, server throughput, and server latency. By default, these metrics are available at [http://localhost:8002/metrics](http://localhost:8002/metrics). The metric format is plain text so you can view them directly, for example:

```shell
curl localhost:8002/metrics
```

It's possible to visualize these metrics with an open source tools [Grafana](https://grafana.com/) and [Prometheus](https://prometheus.io). Set up your monitoring dashboard following the steps:

* Download [Prometheus](https://prometheus.io/download/) monitoring tool.
* Configure it with a `yaml` file:

```yaml
global:
  scrape_interval: 1ms 
  external_labels:
    monitor 'codelab-monitor'
scrape_configs:
  - job_name: 'prometheus'
    scrape_interval: 1ms
    static_configs:
      - targets: ['localhost:8002']
```
Execute as

```shell 
./prometheus --config.file="config.yaml"
```

* Install [Grafana](https://grafana.com/docs/grafana/latest/getting-started/getting-started/) 
* In Grafana, add Prometheus on `localhost:9090` as a data source.
* Start Grafana

```shell
sudo systemctl start grafana-server.service 
```

* In the browser, navigate to `http://localhost:3000/`.
* You can setup your own dashboard to visualize the metrics which are important for you. Or you can import the dashboard we prepared for you. You can find it in this project under `metrics/dashboard.json`.
