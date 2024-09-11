# merge base model with lora

echo "start merge model"

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

python $SCRIPT_DIR/../text2sql/merge_model.py \
	--adapter_name CodeLlama-7b-Instruct_spider_240911_085421/checkpoint-10 \
	--base_model_name_or_path /home/ymLiu/model/CodeLlama-7b-Instruct \
	--merged_model_name merged_model_test \

echo 'finish merge model'
