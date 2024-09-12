# eval pred res

echo "start evaluate pred "

SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
# echo $SCRIPT_DIR
python $SCRIPT_DIR/../eval/evaluation.py \
    --plug_value \
    --input $SCRIPT_DIR/../pred/cspider_dl_2000 \
    --db $SCRIPT_DIR/../data/CSpider/CSpider/database \
    --gold $SCRIPT_DIR/../data/CSpider/CSpider/dev_gold.sql \
    --table $SCRIPT_DIR/../data/CSpider/CSpider/tables.json

echo "build evaluate pred"