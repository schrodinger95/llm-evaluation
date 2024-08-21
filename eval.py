import os
import argparse
import gc
import torch
import json

from lm_eval.api.registry import get_model
from lm_eval import simple_evaluate

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
    task_list: list = ["piqa"],
):
    model_args = {
        "pretrained": model_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
        "dtype": dtype,
        "batch_size": batch_size,
    }

    if tokenizer_path: model_args["tokenizer"] = tokenizer_path
    if quantization: model_args["quantization"] = quantization
    if max_model_len: model_args["max_model_len"] = max_model_len
    if kv_cache_dtype: model_args["kv_cache_dtype"] = kv_cache_dtype
    model_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    LLM = get_model(platform)
    model = LLM(**model_args)  # Replace with the model you want to evaluate

    # Perform the evaluation
    evaluation_results = simple_evaluate(model=model, tasks=task_list)
    results = {}
    for task in task_list:
        results[task] = evaluation_results['results'][task]['acc_norm,none']
    return results

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
    parser.add_argument('--batch_size', default="auto", help='Batch size during inference.')
    parser.add_argument('--task_list', default=["piqa"], nargs='+', type=str, help='Tasks for evaluation.')
    parser.add_argument('--output', default="log", type=str, help='Output path.')

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
        "batch_size": args.batch_size,
        "task_list": args.task_list,
    }

    results = eval(**kwargs)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    json_filtered_results = json.dumps(results, indent=4)  
    with open(os.path.join(args.output, "results.json"), "w") as json_file:  
            json_file.write(json_filtered_results)

    kwargs['model_name'] = args.model_name
    json_config = json.dumps(kwargs, indent=4)
    with open(os.path.join(args.output, "config.json"), "w") as outfile:
        outfile.write(json_config)