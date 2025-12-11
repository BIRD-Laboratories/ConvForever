# ConvForever

Scaling Convolutional Neural Networks Forever

Julian Herrera @ College of the Desert, Palm Desert

ConvForever aims to explore the fundamental limits of convolutional neural networks using the ConvNeXt architecture. It will also explore low-bandwidth training to enable scalability on multiple consumer GPUs.

I need to work on better auditing. 

Semi Successor to [MLPScaling](https://github.com/BIRD-Laboratories/MLPScaling)

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="<your-openrouter-api-key>"
huggingface auth login
```

## Usage Instructions

### Training with Different Dataset Formats

The system supports multiple dataset formats:

- `json`/`laion`: LAION-style JSON format with image URLs and labels (downloads images on-the-fly)
- `imagenet`: Standard ImageNet format from HuggingFace
- `pd_extended`: PD Extended format (streamed HuggingFace parquet with pre-classified labels) - **NEW**
- `laion_imagenet`: LAION format for ImageNet data with possible label mapping

### 1. Training with JSON/LAION Format (Original)

```bash
# Without DeepSpeed (single GPU):
python train.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --classified_json classified_captions.jsonl \
  --upload_every 500 \
  --dataset_format json \
  --org "<your-hf-org>"

# With DeepSpeed (multi-GPU):
deepspeed train.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --classified_json classified_captions.jsonl \
  --upload_every 500 \
  --dataset_format json \
  --use_deepspeed \
  --deepspeed_config ds_config.json \
  --org "<your-hf-org>"
```

### 2. Training with PD Extended Format (NEW)

```bash
# Without DeepSpeed (single GPU):
python train.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --dataset_format pd_extended \
  --data_split train \
  --epochs 1 \
  --upload_every 500 \
  --org "<your-hf-org>"

# With DeepSpeed (multi-GPU):
deepspeed train.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --dataset_format pd_extended \
  --data_split train \
  --epochs 1 \
  --upload_every 500 \
  --use_deepspeed \
  --deepspeed_config ds_config.json \
  --org "<your-hf-org>"
```

### 3. Training with ImageNet Format

```bash
# Without DeepSpeed (single GPU):
python train.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --dataset_format imagenet \
  --data_split train \
  --epochs 1 \
  --upload_every 500 \
  --org "<your-hf-org>"
```

### 4. Using the Wrapper Script (Optional DeepSpeed)

```bash
# PD Extended with DeepSpeed
python script.py \
  --depth 16 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lr 3e-4 \
  --dataset_format pd_extended \
  --data_split train \
  --epochs 1 \
  --upload_every 500 \
  --use_deepspeed \
  --deepspeed_config ds_config.json \
  --org "<your-hf-org>"
```

### 5. Caption Classification (Optional - if you don't have classified data)

```bash
python llm_class.py \
  --attempt 1 \
  --openrouter_key "$OPENROUTER_API_KEY" \
  --org "<your-hf-org>" \
  --sync_every 500 \
  --rate_limit_pause 0.5
```

## Configuration Options

- `--depth`: Number of layers in the ConvNeXt model (minimum 4)
- `--micro_batch_size`: Batch size per GPU/device
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients
- `--lr`: Learning rate
- `--classified_json`: Path to classified JSONL file (for JSON format)
- `--dataset_format`: Dataset format ('json', 'laion', 'imagenet', 'pd_extended', 'laion_imagenet')
- `--data_split`: Data split to use ('train', 'validation', 'test' for non-JSON formats)
- `--epochs`: Number of epochs to train (for non-JSON formats)
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

### Disclaimer

Qwen Coder is used heavily in this project. Reasonable guardrails are in place to ensure quality of code.
