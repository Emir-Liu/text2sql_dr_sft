# pred with base model or base model with lora
echo "start predict"

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

python $SCRIPT_DIR/../text2sql/predict.py \
    --dataset CSpider \
    --base_model_name_or_path /home/ymLiu/model/CodeLlama-7b-Instruct \
    --adapter_name cspider_codellama_old/checkpoint-1000 \
    --pred_file_name cspider_dl_1000.sql

echo 'finish predict'