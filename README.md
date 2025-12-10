# ConvForever

Scaling Convolutional Neural Networks Forever

Julian Herrera @ College of the Desert, Palm Desert

ConvForever aims to explore the fundamental limits of convolutional neural networks using the ConvNeXt architecture. It will also explore low-bandwidth training to enable scalability on multiple consumer GPUs.

Semi Successor to [MLPScaling](https://github.com/BIRD-Laboratories/MLPScaling)

Possible enhancements:
Compute captions in a dictionary ahead of time, but only store that.

## Project Structure

- `llm_class.py` - Classify captions from datasets using Nova 2 Lite API
- `train.py` - Main training script for ConvNeXt models
- `script.py` - Wrapper for train.py with optional DeepSpeed support
- `util-script.py` - Utility functions
- `ds_config.json` - DeepSpeed configuration file
- `requirements.txt` - Python dependencies

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="<your-openrouter-api-key>"
huggingface-cli login
```

## Usage Instructions

### 1. Caption Classification (Optional - if you don't have classified data)

```bash
python llm_class.py \
  --attempt 1 \
  --openrouter_key "$OPENROUTER_API_KEY" \
  --org "<your-hf-org>" \
  --sync_every 500 \
  --rate_limit_pause 0.5
```

### 2. Model Training

#### Without DeepSpeed (single GPU):
```bash
python train.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --classified_json classified_captions.jsonl \
  --upload_every 500 \
  --org "<your-hf-org>"
```

#### With DeepSpeed (multi-GPU):
```bash
deepspeed train.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --classified_json classified_captions.jsonl \
  --upload_every 500 \
  --use_deepspeed \
  --deepspeed_config ds_config.json \
  --org "<your-hf-org>"
```

#### Using the wrapper script (with optional DeepSpeed):
```bash
# Without DeepSpeed
python script.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --classified_json classified_captions.jsonl \
  --upload_every 500 \
  --org "<your-hf-org>"

# With DeepSpeed
python script.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --classified_json classified_captions.jsonl \
  --upload_every 500 \
  --use_deepspeed \
  --deepspeed_config ds_config.json \
  --org "<your-hf-org>"
```

## Features

- Customizable ConvNeXt depth
- Optional DeepSpeed integration for multi-GPU training
- Automatic model checkpoint uploads to Hugging Face Hub
- Gradient accumulation support
- Image preprocessing and augmentation
- Error handling for failed image downloads

## Configuration Options

- `--depth`: Number of layers in the ConvNeXt model (minimum 4)
- `--micro_batch_size`: Batch size per GPU/device
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients
- `--lr`: Learning rate
- `--classified_json`: Path to classified JSONL file
- `--upload_every`: Upload model every N steps
- `--use_deepspeed`: Enable DeepSpeed training (optional)
- `--deepspeed_config`: Path to DeepSpeed configuration file
- `--org`: Hugging Face organization/user for model uploads

## Testing

Run the unit tests to validate functionality:
```bash
python test_train.py
```

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