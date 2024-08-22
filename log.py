import os
import sys
import json
import wandb

from api_key import API_KEY

if __name__ == '__main__':
    output = sys.argv[1]

    with open(os.path.join(output, 'config.json'), 'r', encoding='utf-8') as json_file:
        config = json.load(json_file)

    name = f"{config['model_name']}_on_{config['platform']}_with_{config['quantization']}"
    project = config["model_name"]

    wandb.login(relogin=True, key=API_KEY)
    wandb.init(
        name=name,
        project=project,
        job_type="eval",
        config=config
    )

    with open(os.path.join(output, 'results.json'), 'r', encoding='utf-8') as json_file:
        results = json.load(json_file)

    with open(os.path.join(output, 'memory_results.json'), 'r', encoding='utf-8') as json_file:
        memory_results = json.load(json_file)

    with open(os.path.join(output, 'latency_results.json'), 'r', encoding='utf-8') as json_file:
        latency_results = json.load(json_file)

    wandb.log({
        **results,
        **memory_results,
        **latency_results
    })

    # Finalize wandb session
    wandb.finish()