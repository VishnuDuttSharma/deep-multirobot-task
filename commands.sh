python main.py configs/coverageTask_10rob_2layer.json --mode train --tgt_feat 20 --rbt_feat 10 --nGraphFilterTaps 4 --map_w 20 --num_agents 10 --trained_num_agents 10

python get_results.py configs/coverageTask_10rob_2layer.json --mode test --num_agents 10 --timeid 1618107064

