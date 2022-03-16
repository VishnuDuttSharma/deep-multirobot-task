# PyTorch implementation for Graph Neural Networks for Decentralized Multi-Robot Submodular Action Selection
Code accompanying the paper
[Graph Neural Networks for Decentralized Multi-Robot Submodular Action Selection](https://arxiv.org/abs/2105.08601) 

from Lifeng Zhou<sup>1</sup>, Vishnu D. Sharma<sup>2</sup>, Qingbiao Li<sup>3</sup>, Amanda Prorok<sup>3</sup>, Alejandro Ribeiro<sup>1</sup>, Vijay Kumar<sup>1</sup>
at the (1) University of Pennsylvania, (2) University of Maryland, and (3) University of Cambridge.

The code here is based on the [PyTorch Project for Graph Neural Network based MAPF](https://github.com/proroklab/gnn_pathplanning) by [Qingbiao Li](https://github.com/QingbiaoLi).

### Requirements:
```
easydict>=1.7
matplotlib>=3.1.2
numpy>=1.14.5
Pillow>=5.2.0
scikit-image>=0.14.0
scikit-learn>=0.19.1
scipy>=1.1.0
tensorboardX>=1.2
torch>=1.1.0
torchvision>=0.3.0
```
### How to use this repo:

## Generating the data
1. Update the `constants.py` with the required number of robots, FoV, step size, communication range, etc.
2. Create a folder (say `./robot20_data`) to save the data 
```
python -u gnn_setup.py --batch_size 2000 --save_path ./robot20_data/ --mode rect
```
This command generates 100 files each containing `batch_size` examples each. First 60 files are used for training, next 20 for validation, and the rest for testing.

#### Train a new network
1. Create/update the configuration file to reflect the correct path to the training data 
```
python main.py configs/coverageTask_20rob_large_6FOV_8STEP_10COMM_3layer_2filtertap_GNN2layer_32_128.json --mode train --tgt_feat 20 --rbt_feat 10 --nGraphFilterTaps 2 --num_agents 20 --trained_num_agents 20
```
The command above trains the model with 2 filter taps i.e. 1-hop.


#### Testing the model
Use the configuration and the time ID (from output log) of the trained model as following: 
```
python get_results.py configs/coverageTask_6rob_6FOV_8STEP_10COMM_3layer_2filtertap_GNN2layer_32_128.json --mode test --num_agents 20 --timeid 1635268525
```

For testing the reward coverage performance, generate more datailed data with random configuration:
```
python test_data_generation.py --batch_size 1000 --save_path ./test_data_20robot --mode rect
```
This script also depends on `constants.py` and the number of robots can be changed in it to generate data with difference number of robots.

The results can be obtained over this data by running the following command (also requires config file path and timeid/log_time_trained):
```
python test_coverage.py configs/coverageTask_20rob_FIX_6FOV_20STEP_10COMM_3layer_2filtertap_GNN2layer_32_128.json --log_time_trained 1630851013 --data_path ./test_data_20robot
```

### License:
The project of graph mapf is licensed under MIT License - see the LICENSE file for details

### Citation:
If you use this paper in an academic work, please cite:
```
@article{zhou2021graph,
  title={Graph Neural Networks for Decentralized Multi-Robot Submodular Action Selection},
  author={Zhou, Lifeng and Sharma, Vishnu D and Li, Qingbiao and Prorok, Amanda and Ribeiro, Alejandro and Kumar, Vijay},
  journal={arXiv preprint arXiv:2105.08601},
  year={2021}
}
```
