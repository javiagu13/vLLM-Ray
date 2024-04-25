# There are a few optional containers

Depending of what you need for deployment one or other container should be built

## vLLM + fastAPI

This is a more limited container since it wont allow dynamic autoscaling for large scale serving,
however it can be useful for a few users at the same time in a sever and easier to handle

Codes:

```
vllm_inference.py
vllm-pipeline.dockerfile
```

In order to build it:

```
docker build -f vllm-pipeline.dockerfile -t vllm_pipeline .
docker run -p 33334:8000 --gpus all vllm_pipeline
docker save -o vllm_pipeline.tar vllm_pipeline
```

## vLLM + ray

This is a robust and prepared for scale container, however more tricky to handle.
It allows multiple CPU and GPU handling and dynamic autoscaling in any kind of server (please change hyperparameters after building it)

### Tested in consumer GPU (just one GPU)

Codes:

```
vllm_ray_inference_working_pc.py
vllm-ray-pipeline.dockerfile -> please make sure you copy vllm_ray_inference_working_pc.py and not other file into the container, modify the dockerfile accordingly
```

In order to build it:

```
docker build -f vllm-ray-pipeline.dockerfile -t vllm_ray_pipeline .
docker run -it --rm -p 8000:8000 -p 8265:8265 --gpus all --name vllm_ray_container vllm_ray_pipeline:latest
docker save -o vllm_ray_pipeline.tar vllm_ray_pipeline
```

The previous code will launch the ray cluster. The following one will allow you to go in a second console into the container. Once inside run the .py file, in this case `python vllm_ray_inference_working_pc.py`
For running with it:

```
docker exec -it vllm_ray_pipeline /bin/bash
python vllm_ray_inference_working_pc.py
```

### Tested in server GPU (multiple GPU)

Codes:

```
vllm_ray_inference_working_server.py
vllm-ray-pipeline.dockerfile -> please make sure you copy vllm_ray_inference_working_server.py and not other file into the container, modify the dockerfile accordingly
```

In order to build it:

```
docker build -f vllm-ray-pipeline.dockerfile -t vllm_ray_pipeline .
docker run -it --rm -p 8000:8000 -p 8265:8265 --gpus all --name vllm_ray_container vllm_ray_pipeline:latest
docker save -o vllm_ray_pipeline.tar vllm_ray_pipeline
```

The previous code will launch the ray cluster. The following one will allow you to go in a second console into the container. Once inside run the .py file, in this case `python vllm_ray_inference_working_server.py`
For running with it:

```
docker exec -it vllm_ray_pipeline /bin/bash
python vllm_ray_inference_working_server.py
```

### Testing proper functioning:

1. `test_in_win.py` file allows you to make a simple prompt test
2. `test_in_win_vllm.py` file allows you to make a simple prompt test
3. If you go to localhost:8625 you can see the ray cluster interface and what is happening
