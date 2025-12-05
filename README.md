# ConvForever

Scaling Convolutional Neural Networks Forever 

ConvForever aims to look into the fundemtal limits of convolutional neural networks using the ConvNeXT architecture. It will also explore low-bandwidth trainign to enable scaliblity on mulitple consumer GPUs.

Semi Successor to [MLPScaling](https://github.com/BIRD-Laboratories/MLPScaling)

Possbile enhancements:
Compute captions in a dictonary ahead of time, but only store that.

```
hf auth login
deepspeed --num_gpus 1 script.py \
  --depth 48 \
  --micro_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --max_examples 1000 \
  --upload_every 200 \
  --deepspeed ds_config.json
```
12/5 9:41 Estimates plan to be ran. Looking into SSD loading for extremely large models. 
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
Papers
https://arxiv.org/abs/2306.13575
https://arxiv.org/abs/2201.03545

Code
https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-classification/run_image_classification.py

Dataset
With real time catagorization
https://huggingface.co/datasets/laion/relaion1B-nolang-research

Model
https://huggingface.co/allenai/Olmo-3-7B-Instruct
