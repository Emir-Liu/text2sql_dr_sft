# pred with base model or base model with lora
echo "start predict"

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

python $SCRIPT_DIR/../text2sql/predict.py \
    --dataset spider \
    --base_model_name_or_path /home/ymLiu/model/CodeLlama-7b-Instruct \
    --adapter_name CodeLlama-7b-Instruct_spider_240911_082047/checkpoint-10 \
    --pred_file_name spider_dl_10

echo 'finish predict'