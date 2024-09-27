# pred with base model or base model with lora
echo "start predict"

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

python $SCRIPT_DIR/../text2sql/predict_with_llm_api.py \
    --dataset CSpider \
    --pred_file_name cspider_ERNIE_Speed_128K.sql 

echo 'finish predict'