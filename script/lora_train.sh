# train model with lora

echo "start lora train"

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

python $SCRIPT_DIR/../text2sql/training.py \
    --dataset spider \
    --base_model /home/ymLiu/model/CodeLlama-7b-Instruct

echo 'finish lora train'