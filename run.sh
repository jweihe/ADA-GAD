dataset=$1

[ -z "${dataset}" ] && dataset="inj_cora"

python main.py --use_cfg --seeds 0 1 2 3 4 5 6 7 8 9 --dataset $dataset