import os
import argparse
import gc
import torch  
# import time
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
  
def eval(
    platform: str = "vllm",
    model_path: str = "Llama-2-7b",
    tokenizer_path: str = None,
    max_model_len: int = None,
    quantization: str = None,
    kv_cache_dtype: str = None,
    gpu_memory_utilization: float = 0.6,
    enforce_eager: bool = False,
    dtype = "auto",
    temperature: float = 0.8,
    top_p: float = 0.95,
    data_path = None,
):
    model_args = {
        "model": model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "dtype": dtype,
    }

    if tokenizer_path: model_args["tokenizer"] = tokenizer_path
    if quantization: model_args["quantization"] = quantization
    if max_model_len: model_args["max_model_len"] = max_model_len
    if kv_cache_dtype: model_args["kv_cache_dtype"] = kv_cache_dtype
    model_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = []
    with open(data_path, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        prompts.append(result['question'])

    if platform == "vllm":

        sampling_params = SamplingParams(temperature=temperature, top_p=top_p)
        model = LLM(**model_args)
        outputs = model.generate(prompts, sampling_params)

        for output in outputs:
            _ = output.prompt
            _ = output.outputs[0].text

    # elif platform == "hf":
    #     model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
    #     tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path else model_path, trust_remote_code=True)

    #     for prompt in prompts:
    #         inputs = tokenizer(prompt, return_tensors='pt')  
    #         inputs = inputs.to(model.device)

    #         with torch.no_grad():
    #             outputs = model(**inputs)
    #             outputs = model.generate(**inputs, min_new_tokens=1, max_new_tokens=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of LLM',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', default="Llama-2-7B-GPTQ", type=str, help='Name of the model.')
    parser.add_argument('--model_path', default="/data/llmQuantModels", type=str, help='Path of all the models.')
    parser.add_argument('--platform', default="vllm", type=str, help='Platform.')
    parser.add_argument('--max_model_len', default=2048, type=int, help='Max sequence length of the model.')
    parser.add_argument('--quantization', default="gptq", type=str, help='Quantization method.')
    parser.add_argument('--kv_cache_dtype', default=None, type=str, help='Data type of the KV cache.')
    parser.add_argument('--gpu_memory_utilization', default=0.4, type=float, help='Utilization of the GPU memory.')
    parser.add_argument('--enforce_eager', action='store_true', help='Apply eager-mode PyTorch.')
    parser.add_argument('--dtype', default="float16", type=str, help='Data type during inference.')
    parser.add_argument('--temperature', default=0.8, type=float, help='Temperature during generation.')
    parser.add_argument('--top_p', default=0.95, type=float, help='Top p during generation.')
    parser.add_argument('--data_path', default="/data/llmQuantModels/dataset/open_qa.jsonl", type=str, help='Path of the data.')
    # parser.add_argument('--output', default="log", type=str, help='Output path.')

    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    kwargs = {
        "platform": args.platform,
        "model_path": os.path.join(args.model_path, args.model_name),
        "max_model_len": args.max_model_len,
        "quantization": args.quantization,
        "kv_cache_dtype": args.kv_cache_dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "dtype": args.dtype,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "data_path": args.data_path,
    }

    eval(**kwargs)
