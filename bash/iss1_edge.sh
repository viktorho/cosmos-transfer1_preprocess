# ======== Setup Environment Variables ========
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR=./checkpoints

# ======== Run Multi-GPU Inference ========
PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
  --checkpoint_dir $CHECKPOINT_DIR \
  --video_save_folder outputs/multi_gpu_edge_ct2 \
  --controlnet_specs inputs/iss4single_control_edge_ct2.json \
  --offload_text_encoder_model \
  --offload_guardrail_models \
  --num_gpus 1