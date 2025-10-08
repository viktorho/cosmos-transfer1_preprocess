# ======== Setup Environment Variables ========
export CUDA_VISIBLE_DEVICES=4
export CHECKPOINT_DIR=./checkpoints

# ======== Run Multi-GPU Inference ========
PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir $CHECKPOINT_DIR \
  --video_save_folder outputs/iss1_edge \
  --controlnet_specs inputs/iss1_edge.json \
  --offload_text_encoder_model \
  --offload_guardrail_models \
  --num_gpus 1