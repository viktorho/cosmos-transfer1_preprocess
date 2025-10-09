#!/bin/bash
export CHECKPOINT_DIR="./checkpoints"
export CUDA_VISIBLE_DEVICES=6,7
export NUM_GPU=2
export OUTPUT_DIR="outputs/tuan1_edge"
export CONTROL_SPECS="inputs/tuan_multi.json"
export OMP_NUM_THREADS=4

PYTHONPATH=$(pwd) torchrun \
    --nproc_per_node=$NUM_GPU \
    --nnodes=1 \
    --node_rank=0 \
    cosmos_transfer1/diffusion/inference/transfer.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --video_save_folder "$OUTPUT_DIR" \
        --controlnet_specs "$CONTROL_SPECS" \
        --offload_text_encoder_model \
        --offload_guardrail_models \
        --num_gpus "$NUM_GPU" \
        2>&1 | tee "$OUTPUT_DIR/run_log.txt"