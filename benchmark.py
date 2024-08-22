import os
import argparse
import gc
import torch  
import time
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
  
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
    batch_size = "auto",
    data_path: str = None,
    data_num: int = None,
):
    prompts = []
    with open(data_path, 'r', encoding='utf-8') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        prompts.append(result['question'])

    if data_num:
        prompts = prompts[:data_num]
    
    TTFT_list = []
    ITL_list = []
    output_num = []
    if platform == "vllm":
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

        model = LLM(**model_args)
        outputs = model.generate(prompts)
        for output in outputs:
            arrival_time = output.metrics.arrival_time
            first_token_time = output.metrics.first_token_time
            finished_time = output.metrics.finished_time
            TTFT_list.append(first_token_time - arrival_time)
            ITL_list.append(finished_time - first_token_time)
            output_num.append(len(output.outputs[0].token_ids))

    elif platform == "hf":
        model_args = {
            "pretrained_model_name_or_path": model_path,
            "dtype": dtype,
            "batch_size": batch_size,
        }
        if tokenizer_path: model_args["tokenizer"] = tokenizer_path
        if kv_cache_dtype: model_args["kv_cache_dtype"] = kv_cache_dtype
        model_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path else model_path, trust_remote_code=True)

        for prompt in tqdm(prompts):
            inputs = tokenizer(prompt, return_tensors='pt')
            inputs = inputs.to(model.device)

            start_time = time.time()
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=1, max_length=None)
            end_time = time.time()
            elapsed_time = end_time - start_time
            TTFT_list.append(elapsed_time)

            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, max_length=None)
            end_time = time.time()
            elapsed_time = end_time - start_time
            ITL_list.append(elapsed_time - TTFT_list[-1])
            output_num.append(len(outputs.tolist()[0][len(inputs["input_ids"][0]):]))
    
    TTFT_latency = sum(TTFT_list) / len(TTFT_list)
    ITL_latency = sum(ITL_list) / sum(output_num)
    results = {
        "TTFT_latency": TTFT_latency,
        "ITL_latency": ITL_latency
    }
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eifficiency test of LLM',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default="/data/llmQuantModels/dataset/open_qa.jsonl", type=str, help='Path of the data.')
    parser.add_argument('--data_num', default=None, type=int, help='Number of data used to test.')
    parser.add_argument('--output', default="log", type=str, help='Output path.')

    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    with open(os.path.join(args.output, 'config.json'), 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    kwargs = {
        "platform": config["platform"],
        "model_path":config["model_path"],
        "max_model_len": config["max_model_len"],
        "quantization": config["quantization"],
        "kv_cache_dtype": config["kv_cache_dtype"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "enforce_eager": config["enforce_eager"],
        "dtype": config["dtype"],
        "batch_size": config["batch_size"],
        "data_path": args.data_path,
        "data_num": args.data_num,
    }

    results = eval(**kwargs)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    json_filtered_results = json.dumps(results, indent=4)  
    with open(os.path.join(args.output, "latency_results.json"), "w") as json_file:  
            json_file.write(json_filtered_results)
