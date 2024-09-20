# eval pred res

echo "start evaluate pred "

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
# echo $SCRIPT_DIR
python $SCRIPT_DIR/../eval/evaluation.py \
    --plug_value \
    --input $SCRIPT_DIR/../pred/spider_sqlcoder.sql \
    --db $SCRIPT_DIR/../data/spider/spider/database \
    --gold $SCRIPT_DIR/../data/spider/spider/dev_gold.sql \
    --table $SCRIPT_DIR/../data/spider/spider/tables.json

echo "build evaluate pred"