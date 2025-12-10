# ConvForever

Scaling Convolutional Neural Networks Forever

Julian Herrera @ College of the Desert, Palm Desert

ConvForever aims to explore the fundamental limits of convolutional neural networks using the ConvNeXT architecture. It will also explore low-bandwidth training to enable scalability on multiple consumer GPUs.

Semi Successor to [MLPScaling](https://github.com/BIRD-Laboratories/MLPScaling)

Possible enhancements:
Compute captions in a dictionary ahead of time, but only store that.

## Instructions

### Phase 1: Caption Classification

First, classify image captions using an LLM to prepare training data:

```
export OPENROUTER_API_KEY="<your-api-key>"
hf login
python llm_class.py \
  --attempt 1 \
  --openrouter_key "$OPENROUTER_API_KEY" \
  --org "<your-huggingface-org>" \
  --sync_every 500 \
  --rate_limit_pause 0.5
```

This will create a classified dataset and upload shards to Hugging Face Hub.

### Phase 2: Model Training

After classifying the data, train the ConvNeXt model:

#### Option A: Using the wrapper script (with optional DeepSpeed)

The wrapper script allows you to run training with or without DeepSpeed using the same interface:

- Without DeepSpeed:
```
python script.py \
  --depth 32 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --classified_json "classified_attempt_1.jsonl" \
  --upload_every 500 \
  --org "<your-huggingface-org>"
```

- With DeepSpeed:
```
python script.py \
  --deepspeed \
  --depth 32 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --classified_json "classified_attempt_1.jsonl" \
  --upload_every 500 \
  --org "<your-huggingface-org>"
```

#### Option B: Direct training (original method)

```
python train.py \
  --depth 32 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --classified_json "classified_attempt_1.jsonl" \
  --upload_every 500 \
  --deepspeed ds_config.json \
  --org "<your-huggingface-org>"
```

This will train a ConvNeXt model with the specified depth and upload checkpoints periodically to Hugging Face Hub.

### Requirements

Make sure to install the required dependencies:

```
pip install -r requirements.txt
```

And ensure you have configured DeepSpeed properly with a configuration file (ds_config.json) if using DeepSpeed.

## Steps to take:
GPU Hour estimates using a 1x 4090 rig

Estimates cost as well using the vast ai servers, add padding due to scaling losses

Code base for training,

Trackio in lieu of W&B 

Huggingface framework will be used.

Dispatch will be done with tmux. 

Dataset will be streamed

Snapshots of the model periodically uploaded to Huggingface hub public repos put in an organization.

Reformats huggingface model definition with deriviate scripts for the changes.

Deepspeed optimizers.

Testing scripts to find errors before proper training.

Uses default hyperparameters.

## Acknowledgements
Compute Resources:
College of the Desert

For guiding me through this project I thank:

Patrick Jacobs & Felix Marhuenda-Donate

For the original 2024 project:
Shanghai AI Laboratory

Papers
https://arxiv.org/abs/2306.13575
https://arxiv.org/abs/2201.03545

Code
https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py

Tooling
Huggingface Hub

Dataset
https://huggingface.co/datasets/Spawning/pd-extended
