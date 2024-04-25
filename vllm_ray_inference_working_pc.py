import ray
from ray import serve
import os
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.vllm import build_vllm_token_enforcer_tokenizer_data
from typing import Union, List, Optional
from vllm import SamplingParams, LLM
from IPython.display import display, Markdown
import os
os.environ['LD_LIBRARY_PATH'] = os.getenv('LD_LIBRARY_PATH', '') + ':/usr/local/cuda-12.2/compat/'

# Initialize Ray and Ray Serve
ray.init() #ray.init(http_port=8000)
serve.start(http_options={"host":"0.0.0.0", "port": 8000})

def init_model(gpu_id, model_id, gpu_utilization=0.65):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return LLM(model=model_id, gpu_memory_utilization=gpu_utilization)

class LLMModel:
    def __init__(self, gpu_id, model_id, gpu_utilization):
        self.llm = init_model(gpu_id, model_id, gpu_utilization)
        self.tokenizer_data = build_vllm_token_enforcer_tokenizer_data(self.llm)

    async def __call__(self, request):
        prompt = await request.json()
        prompt_text = prompt.get("text", "")
        sampling_params = SamplingParams(max_tokens=4000)
        results = self.llm.generate(prompt_text, sampling_params=sampling_params)
        if isinstance(results, list):
            return results[0].outputs[0].text
        else:
            return [result.outputs[0].text for result in results]

# Define model IDs, GPU utilization, and specific GPU IDs for each model
model_config = {
    "llama2": {"id": './models/Llama-2-7B-Chat-AWQ', "gpu_id": 0, "gpu_utilization": 0.9},
    #"gemma": {"id": './models/Gemma-2B-Chat-GPTQ', "gpu_id": 0, "gpu_utilization": 0.9},
    #"kullm3": {"id": './models/KULLM3-Chat-GPTQ', "gpu_id": 0, "gpu_utilization": 0.9}
}

# Deploy each model with specific GPU settings
for model_name, config in model_config.items():
    model_deployment = serve.deployment(LLMModel, name=model_name, num_replicas=1, ray_actor_options={"num_gpus": 1})
    deployable = model_deployment.bind(config["gpu_id"], config["id"], config["gpu_utilization"])
    serve.run(deployable)

print("All models deployed successfully.")
