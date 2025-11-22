import os
import re
import time
import uuid

import runpod
from runpod.serverless import log
from runpod.serverless.utils.rp_validator import validate
from vllm import LLM, SamplingParams

from juno.schema import VALIDATIONS

MODEL = os.getenv("MODEL_NAME")
TOKENIZER = os.getenv("MODEL_TOKENIZER")
CONFIG_FORMAT = os.getenv("MODEL_CONFIG_FORMAT")
LOAD_FORMAT = os.getenv("MODEL_LOAD_FORMAT")
QUANTIZATION = os.getenv("MODEL_QUANTIZATION")
MAX_MODEL_LEN = os.getenv("MODEL_MAX_LEN")

DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.15"))
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_SAMPLING_TOKENS", "32768"))
DEFAULT_TOP_P = float(os.getenv("TOP_P", "0.95"))

if not MODEL:
    print("Define a MODEL_NAME...")
    os._exit(-1)

log.info("Loading {}...".format(MODEL))

model = LLM(
    model=MODEL,
    tokenizer_mode=TOKENIZER if TOKENIZER else "auto",
    config_format=CONFIG_FORMAT if CONFIG_FORMAT else "auto",
    load_format=LOAD_FORMAT if LOAD_FORMAT else "auto",
    quantization=QUANTIZATION if QUANTIZATION else None,
    max_model_len=int(MAX_MODEL_LEN) if MAX_MODEL_LEN else None,
    tensor_parallel_size=int(os.getenv("RUNPOD_GPU_COUNT", "1")),
    gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9")),
)


def handler(job):
    input_validation = validate(job["input"], VALIDATIONS)

    if "errors" in input_validation:
        return {"error": input_validation["errors"]}
    job_input = input_validation["validated_input"]

    temperature = job_input.get("temperature")
    max_tokens = job_input.get("max_tokens")
    top_p = job_input.get("top_p")
    
    sampler = SamplingParams(
        temperature=temperature if temperature is not None else DEFAULT_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS,
        top_p=top_p if top_p is not None else DEFAULT_TOP_P,
    )

    model_output = model.chat(
        messages=job_input["messages"],
        sampling_params=sampler,
        use_tqdm=False,
        chat_template_content_format="string",
        tools=job_input.get("tools", None),
    )

    result = model_output[0]
    output = result.outputs[0]
    
    text = output.text
    reasoning_content = None
    
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    message = {
        "role": "assistant",
        "reasoning_content": reasoning_content,
        "content": text,
    }
    
    return {
        "id": os.getenv("RUNPOD_REQUEST_ID") or f"rp-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": output.finish_reason,
        }],
        "usage": {
            "prompt_tokens": len(result.prompt_token_ids),
            "completion_tokens": len(output.token_ids),
            "total_tokens": len(result.prompt_token_ids) + len(output.token_ids),
        }
    }


runpod.serverless.start({"handler": handler})
