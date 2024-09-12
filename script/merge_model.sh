# merge base model with lora

echo "start merge model"

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

python $SCRIPT_DIR/../text2sql/merge_model.py \
	--adapter_name cspider_codellama_old/checkpoint-1000 \
	--base_model_name_or_path /home/ymLiu/model/CodeLlama-7b-Instruct \
	--merged_model_name cspider_cl_7b_1000 \

echo 'finish merge model'
