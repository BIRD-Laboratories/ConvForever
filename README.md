# ConvForever

Scaling Convolutional Neural Networks Forever

Julian Herrera @ College of the Desert, Palm Desert

ConvForever aims to explore the fundamental limits of convolutional neural networks using the ConvNeXt architecture. It will also explore low-bandwidth training to enable scalability on multiple consumer GPUs.

Changelog:
12/12/25: I am getting closer to running sweep.sh, I intent to write an interim report in latex format. 

Semi Successor to [MLPScaling](https://github.com/BIRD-Laboratories/MLPScaling)

## Setup
It is highly recommended to use a venv, although it is not required.

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="<your-openrouter-api-key>"
huggingface auth login
```

## Usage Instructions

### Training with Different Dataset Formats

The system supports multiple dataset formats:

- `json`/`laion`: LAION-style JSON format with image URLs and labels (downloads images on-the-fly)
- `imagenet`: Standard ImageNet format from HuggingFace, uses imagenet 1k
- `pd_extended`: PD Extended format (streamed HuggingFace parquet with pre-classified labels)

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

###45. Caption Classification (Optional - if you don't have classified data)

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
sh sweep.sh
```

## Acknowledgements

Compute Resources:
College of the Desert

For guiding me through this project I thank:

Patrick Jacobs & Felix Marhuenda-Donate

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

Qwen Coder and Qwen Deep Researcher is used heavily in this project. Reasonable guardrails are in place to ensure quality of code. More proper testing of code will be implemented.
