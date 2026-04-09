#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"

PROMPT='A character with blond hair, wearing a blue tunic, white pants, and brown boots, stands on a cobblestone path, facing away from the viewer. They hold a large shield with a stylized ""Z"" on it.  A brown horse stands in a wooden and stone stable to the left.  The stable has a wooden roof supported by wooden beams.  A wooden fence runs along the back of the stable.  The path extends towards rolling green hills under a bright blue sky.  A small wooden sign is visible on the stable roof.  The scene is brightly lit, suggesting a daytime setting.'
IMAGE_PATH=./assets/img/1.png # Now we only provide the i2v model, so the path cannot be None

# PROMPT='A realistic indoor museum exhibition hall with warm ambient lighting, a spacious carpeted floor, and a low grid-pattern ceiling with spotlights. In the center stands a white table covered with a cloth and displayed tableware, placed in front of a blue partition wall. Around the hall are multiple glass display cases with historical artifacts, beige walls, and open walkways leading deeper into the gallery. Large standing illustrated panels of people in historical clothing are placed on both sides of the scene. The atmosphere is quiet, elegant, and softly lit, with a natural exhibition layout and rich interior details.'
# IMAGE_PATH=./assets/img/Art_Gallery.png # Now we only provide the i2v model, so the path cannot be None

SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p # Now we only provide the 480p model
OUTPUT_PATH=./outputs/Zelda/48_kv_509_sliding
MODEL_PATH=/share/liujun/.cache/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038
AR_ACTION_MODEL_PATH=/share/liujun/.cache/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_model/diffusion_pytorch_model.safetensors
BI_ACTION_MODEL_PATH=/share/liujun/.cache/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/bidirectional_model/diffusion_pytorch_model.safetensors
AR_DISTILL_ACTION_MODEL_PATH=/share/liujun/.cache/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_distilled_action_model/diffusion_pytorch_model.safetensors

# POSE='w-31'                   # Camera trajectory: pose string (e.g., 'w-31' means generating [1 + 31] latents) or JSON file path
# NUM_FRAMES=125
# POSE="d-8,left-4,right-4,d-4,left-4,right-4,w-2,right-2,up-2,left-2,down-2,a-9"
# NUM_FRAMES=189
POSE="d-47,right-16,a-32,left-32"
# POSE="right-127"
NUM_FRAMES=509 # 32 cycles
# POSE="w-8,right-4,left-2,a-4,s-8,d-4,left-1" 
# NUM_FRAMES=125
WIDTH=832
HEIGHT=480

# Configuration for faster inference
# The maximum number recommended is 8.
N_INFERENCE_GPU=1 # Parallel inference GPU count.

# Configuration for better quality
REWRITE=false   # Enable prompt rewriting. Please ensure rewrite vLLM server is deployed and configured.
ENABLE_SR=false # Enable super resolution. When the NUM_FRAMES == 125, you can set it to true

# KV compression baseline options for AR inference:
# - none (disable, default)
# - h2o
# - rocketkv
# - infinipot_v
KV_COMPRESSION_METHOD=none
KV_MAX_TOKENS=0        # <=0 disables compression. Example: 16384
KV_RECENT_WINDOW=1024  # Recent window reserved by compression.
ROCKET_POOL_KERNEL=31  # RocketKV coarse selection kernel.
ROCKET_PAGE_SIZE=64    # RocketKV page size.
INFINIPOT_ALPHA=0.6    # InfiniPot-V TaR/VaN mixing weight in [0,1].

# inference with bidirectional model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $BI_ACTION_MODEL_PATH \
#   --few_step false \
#   --model_type 'bi'

# inference with autoregressive model
# torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
#   --prompt "$PROMPT" \
#   --image_path $IMAGE_PATH \
#   --resolution $RESOLUTION \
#   --aspect_ratio $ASPECT_RATIO \
#   --video_length $NUM_FRAMES \
#   --seed $SEED \
#   --rewrite $REWRITE \
#   --sr $ENABLE_SR --save_pre_sr_video \
#   --pose "$POSE" \
#   --output_path $OUTPUT_PATH \
#   --model_path $MODEL_PATH \
#   --action_ckpt $AR_ACTION_MODEL_PATH \
#   --few_step false \
#   --width $WIDTH \
#   --height $HEIGHT \
#   --model_type 'ar'

# inference with autoregressive distilled model
torchrun --nproc_per_node=$N_INFERENCE_GPU -m hyvideo.generate \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite $REWRITE \
  --sr $ENABLE_SR --save_pre_sr_video \
  --pose "$POSE" \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_PATH \
  --action_ckpt $AR_DISTILL_ACTION_MODEL_PATH \
  --few_step true \
  --num_inference_steps 4 \
  --model_type 'ar' \
  --kv_compression_method $KV_COMPRESSION_METHOD \
  --kv_max_tokens $KV_MAX_TOKENS \
  --kv_recent_window $KV_RECENT_WINDOW \
  --rocket_pool_kernel $ROCKET_POOL_KERNEL \
  --rocket_page_size $ROCKET_PAGE_SIZE \
  --infinipot_alpha $INFINIPOT_ALPHA \
  --use_vae_parallel false \
  --use_sageattn false \
  --use_fp8_gemm false
