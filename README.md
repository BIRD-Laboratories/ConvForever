# ConvForever

Scaling Convolutional Neural Networks Forever

Julian Herrera @ College of the Desert, Palm Desert

ConvForever aims to explore the fundemtal limits of convolutional neural networks using the ConvNeXT architecture. It will also explore low-bandwidth trainign to enable scaliblity on mulitple consumer GPUs.

Semi Successor to [MLPScaling](https://github.com/BIRD-Laboratories/MLPScaling)

Possbile enhancements:
Compute captions in a dictonary ahead of time, but only store that.

## Instructions

Caption creation:

```
export OPENROUTER_API_KEY="<key>"
hf auth login
python llm_class.py \
  --attempt 1 \
  --openrouter_key "$OPENROUTER_API_KEY" \
  --org "<org-name>" \
  --sync_every n_steps \
  --rate_limit_pause 0.5
```

12/5 9:41 Estimates plan to be ran. Looking into SSD loading for extremely large models. 

12/9 10:36 working on dataset creation mostly, looking into a small hyperparam sweep.

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
