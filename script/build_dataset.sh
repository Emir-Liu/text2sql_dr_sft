# build dataset

echo "start build dataset"

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
# echo $SCRIPT_DIR
python $SCRIPT_DIR/../text2sql/data_process.py

echo "build dataset finish"