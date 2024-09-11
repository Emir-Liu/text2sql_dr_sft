# merge base model with lora

echo "start merge model"

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)

python $SCRIPT_DIR/../text2sql/merge_model.py

echo 'finish merge model'