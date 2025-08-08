# Mixed-Precision Quantization for Federated Learning on Resource-Constrained Heterogeneous Devices

This is the official implementation for the CVPR2024 paper


## Install packages in requrement.txt

`pip install -r requirements.txt`


## Code Structure
- ```main.py```: endpoint for starting experiments
- ```option.py```: hyper-parameters for experiments
- ```trainer.py```: includes three algorithms: "FedMPQ", "AQFL", "FP"
- ```QuantOptimizer.py```: quantization-aware optimizer 
- ```sampling.py```: functions for generating data partitions with Dirichlet distribution
- ```./client```: includes clients implementing different algorithms
- ```./server```: includes server implementing aggregation
- ```./model```: includes ResNet model with bit-level operation
- ```./utils```: utility function for evaluation
- ```./configs```: includes training configurations for different experiments


## Running an experiment
```
python main.py --config ./configs/CIFAR10_FedMPQ_0.5.json 
```

## Acknowledgement
This repository is built on the top of [BSQ](https://github.com/yanghr/BSQ).


### Citeation
Please cite our paper, if you think this is useful:
```
@inproceedings{chen2024mixed,
  title={Mixed-precision quantization for federated learning on resource-constrained heterogeneous devices},
  author={Chen, Huancheng and Vikalo, Haris},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6138--6148},
  year={2024}
}







