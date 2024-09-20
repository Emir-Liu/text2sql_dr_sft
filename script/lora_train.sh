# train model with lora

echo "start lora train"

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

# # deepspeed success multi host
# deepspeed --hostfile $SCRIPT_DIR/hostfile \
#     --master_addr server1 \
#     $SCRIPT_DIR/../text2sql/training.py \
#     --dataset spider \
#     --base_model /home/ymLiu/model/CodeLlama-7b-Instruct \
#     --deepspeed $SCRIPT_DIR/ds_config.json

# deepspeed success single host
deepspeed --num_gpus=1 \
    $SCRIPT_DIR/../text2sql/training.py \
    --dataset spider \
    --base_model /home/ymLiu/model/CodeLlama-7b-Instruct \
    --deepspeed $SCRIPT_DIR/ds_config.json

# # train lora with trl and accelerate fails
# # multi_gpu.yaml deepspeed_zero1.yaml
# accelerate launch \
#     --config_file=multi_gpu.yaml \
#     --num_processes 1 \
#     $SCRIPT_DIR/../text2sql/training.py \
#     --dataset spider \
#     --base_model /home/ymLiu/model/CodeLlama-7b-Instruct

# train lora with trl success
# python $SCRIPT_DIR/../text2sql/training.py \
#     --dataset spider \
#     --base_model /home/ymLiu/model/CodeLlama-7b-Instruct

echo 'finish lora train'