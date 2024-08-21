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
Or use my existing conda environment.
```
conda activate /home/rugexu/.conda/envs/nvllm
```


## Model Evaluation
Run the following command. Switch `NCCL_P2P_DISABLE` and `CUDA_VISIBLE_DEVICES` if you want to use other CUDA devices.
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python eval.py \
    --model_name $MODEL_NAME \
    --output $OUTPUT_DIRECTORY
```

## Model Performance
Open two command windows. Run the following command in the first window.
```
python monitor.py benchmark.py $OUTPUT_DIRECTORY
```
Then run the following command in the second window.
```
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=1 python benchmark.py \
    --model_name $MODEL_NAME
```

## Wandb Upload
In `api_key.py`, replace the key with your own wandb key. 

Then run the following command.
```
python log.py $OUTPUT_DIRECTORY
```
