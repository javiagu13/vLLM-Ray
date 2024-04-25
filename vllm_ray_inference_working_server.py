import ray
from ray import serve
import os
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.vllm import build_vllm_token_enforcer_tokenizer_data
from vllm import SamplingParams, LLM

# Ensure environment variables are set correctly
os.environ['LD_LIBRARY_PATH'] = os.getenv('LD_LIBRARY_PATH', '') + ':/usr/local/cuda-12.2/compat/'

# Initialize Ray and Ray Serve properly
if not ray.is_initialized():
    ray.init(ignore_reinit_error=True, log_to_driver=True)
serve.start()

class LLMModel:
    def __init__(self, gpu_id, model_id, gpu_utilization):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        self.llm = LLM(model=model_id, gpu_memory_utilization=gpu_utilization)
        self.tokenize_data = build_vllm_token_enforcer_tokenizer_data(self.llm)

    async def __call__(self, request):
        prompt = await request.json()
        prompt_text = prompt.get("text", "")
        sampling_params = SamplingParams(max_tokens=4000)
        results = self.llm.generate(prompt_text, sampling_params=sampling_params)
        return results[0].outputs[0].text if isinstance(results, list) else [result.outputs[0].text for result in results]

# Model configuration
model_config = {
    #"llama2": {"id": './models/Llama-2-7B-Chat-AWQ', "gpu_id": 1, "gpu_utilization": 0.8}
    "gemma1": {"id": './models/Gemma-2B-Chat-GPTQ', "gpu_id": 1, "gpu_utilization": 0.9}
    #"kullm3": {"id": './models/KULLM3-Chat-GPTQ', "gpu_id": 2, "gpu_utilization": 0.9}
}

deployments = []
# Deploy each model with specific GPU settings
for model_name, config in model_config.items():
    model_deployment = serve.deployment(LLMModel, name=model_name, num_replicas=1)
    model_deployment = model_deployment.options(ray_actor_options={
        "num_gpus": 1
    })
    deployable = model_deployment.bind(config["gpu_id"], config["id"], config["gpu_utilization"])
    serve.run(deployable)

print("All models deployed successfully :)")
