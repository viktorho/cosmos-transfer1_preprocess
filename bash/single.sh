# ======== Setup Environment Variables ========
export CUDA_VISIBLE_DEVICES=0
export CHECKPOINT_DIR=./checkpoints
export OUTPUT_DIR=outputs/example_e
export INPUT_PATH=inputs/example_e.json

# ======== Generate control input video ========

PYTHONPATH=$(pwd) python preprocess_input.py \
  --video_save_folder $OUTPUT_DIR \
  --input_json $INPUT_PATH

# ======== Update json =======

BASENAME=$(basename "$INPUT_PATH" .json)
UPDATED_PATH="inputs/update/${BASENAME}.json"
export INPUT_PATH="$UPDATED_PATH"

# ======== Run Inference ========
# PYTHONPATH=$(pwd) python cosmos_transfer1/diffusion/inference/transfer.py \
#   --checkpoint_dir $CHECKPOINT_DIR \
#   --video_save_folder $OUTPUT_DIR \
#   --controlnet_specs $UPDATED_PATH \
#   --offload_text_encoder_model \
#   --num_gpus 1
  # --offload_guardrail_models \

