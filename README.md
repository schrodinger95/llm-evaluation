# llm-evaluation
Evaluation of LLM model performance and model efficiency.

## Install
Create virutal environment.
```
conda create -n test python=3.10  
```
Install lm-evaluation-harness.
```
git clone https://github.com/EleutherAI/lm-evaluation-harness  
cd lm-evaluation-harness  
pip install -e .
```
Install vLLM (optional).
```
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .  # This may take 5-10 minutes.
```
Or use my existing conda environment.
```
conda activate /home/rugexu/.conda/envs/nvllm
```
Currently only support testing with `Huggingface` and `vLLM`.

## Model Performance
Run the following command line.
```
python eval.py \
    --model_name $MODEL_NAME \
    --model_path $MODEL_PATH \
    --platform $PLATFORM \
    --max_model_len $MAX_MODEL_LEN \
    --quantization $QUANTIZATION \
    --kv_cache_dtype $KV_CACHE_DTYPE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --enforce_eager $ENFORCE_EAGER \
    --dtype $DTYPE \
    --batch_size $BATCH_SIZE \
    --task_list $TASK_LIST \
    --output $OUTPUT_DIRECTORY
```
Set `NCCL_IB_DISABLE=1` and `NCCL_P2P_DISABLE=1` and set `CUDA_VISIBLE_DEVICES` as your desired GPU if you want to use other CUDA devices.
```
NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python eval.py \
    --model_name $MODEL_NAME \
    --platform $PLATFORM \
    --output $OUTPUT_DIRECTORY
```

## Model Efficiency
Open two command line windows. Run the following command line in the first window.
```
python monitor.py benchmark.py $OUTPUT_DIRECTORY
```
Then run the following command line in the second window.
```
NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --data_path $DATA_PATH \
    --data_num $DATA_NUM \
    --output $OUTPUT_DIRECTORY
```

## Wandb Upload
In `api_key.py`, replace the key with your own wandb key. 

Then run the following command line.
```
python log.py $OUTPUT_DIRECTORY
```
