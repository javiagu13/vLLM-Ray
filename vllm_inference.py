from fastapi import FastAPI
from pydantic import BaseModel
import vllm
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data
from typing import Union, List, Optional
from vllm import SamplingParams
from IPython.display import display, Markdown

import os
os.environ['LD_LIBRARY_PATH'] = os.getenv('LD_LIBRARY_PATH', '') + ':/usr/local/cuda-12.2/compat/'

app = FastAPI()

class Query(BaseModel):
    text: str

def display_header(text):
    display(Markdown(f'**{text}**'))

def display_content(text):
    display(Markdown(f'```\n{text}\n```'))


global llm
global CURRENT_LOADED_MODEL
CURRENT_LOADED_MODEL="None"
DEFAULT_MAX_NEW_TOKENS = 1000
ListOrStrList = Union[str, List[str]]
### peta aqui, dice que list or str list not defined
def vllm_with_character_level_parser(prompt: ListOrStrList, parser: Optional[CharacterLevelParser] = None) -> ListOrStrList:

    sampling_params = SamplingParams()
    sampling_params.max_tokens = DEFAULT_MAX_NEW_TOKENS
    if parser:
        logits_processor = build_vllm_logits_processor(tokenizer_data, parser)
        sampling_params.logits_processors = [logits_processor]
    # Note on batched generation:
    # For some reason, I achieved better batch performance by manually adding a loop similar to this:
    # https://github.com/vllm-project/vllm/blob/main/examples/llm_engine_example.py,
    # I don't know why this is faster than simply calling llm.generate() with a list of prompts, but it is from my tests.
    # However, this demo focuses on simplicity, so I'm not including that here.
    results = llm.generate(prompt, sampling_params=sampling_params)
    if isinstance(prompt, str):
        return results[0].outputs[0].text
    else:
        return [result.outputs[0].text for result in results]
### LOAD OF vLLM #####################################################################


@app.post('/chat_llama_2')
async def query(query: Query):
    global llm
    global CURRENT_LOADED_MODEL
    ### LOAD OF vLLM #####################################################################
    ### HERE ADD LOGIC OF IF ALREADY LOADED, DONT LOAD ###################################
    if CURRENT_LOADED_MODEL!="llama":
        model_id = './models/Llama-2-7B-Chat-AWQ'
        llm = vllm.LLM(model=model_id, gpu_memory_utilization=0.9, max_model_len=1000)

        tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm)    
        CURRENT_LOADED_MODEL="llama"

    #response = llm.predict(query.text)
    result = vllm_with_character_level_parser(query.text, None)
    display_content(result)
    return {"response": result}

@app.post('/chat_gemma_2b')
async def query(query: Query):
    global llm
    global CURRENT_LOADED_MODEL
    ### LOAD OF vLLM #####################################################################
    ### HERE ADD LOGIC OF IF ALREADY LOADED, DONT LOAD ###################################
    if CURRENT_LOADED_MODEL!="gemma":
        model_id = './models/Gemma-2B-Chat-GPTQ'
        llm = vllm.LLM(model=model_id, gpu_memory_utilization=0.9, max_model_len=1000)

        tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm)
        CURRENT_LOADED_MODEL="gemma"

    #response = llm.predict(query.text)
    result = vllm_with_character_level_parser(query.text, None)
    display_content(result)
    return {"response": result}


@app.post('/chat_kullm3')
async def query(query: Query):
    global llm
    global CURRENT_LOADED_MODEL
    ### LOAD OF vLLM #####################################################################
    ### HERE ADD LOGIC OF IF ALREADY LOADED, DONT LOAD ###################################
    if CURRENT_LOADED_MODEL!="kullm3":
        model_id = './models/KULLM3-Chat-GPTQ'
        llm = vllm.LLM(model=model_id, gpu_memory_utilization=0.9, max_model_len=1000)

        tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm)
        CURRENT_LOADED_MODEL="kullm3"

    #response = llm.predict(query.text)
    result = vllm_with_character_level_parser(query.text, None)
    display_content(result)
    return {"response": result}



