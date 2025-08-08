# Mixed-Precision Quantization for Federated Learning on Resource-Constrained Heterogeneous Devices

This is the official implementation for the CVPR2024 paper


## Install packages in requrement.txt

`pip install -r requirements.txt`


## Code instruction
- ```main.py```: endpoint for starting experiments
- ```option.py```: hyper-parameters for experiments
- ```trainer.py```: including three algorithms: "FedMPQ", "AQFL", "FP"
- ```QuantOptimizer.py```: quantization-aware optimizer 
- ```sampling.py```: functions for generating data partitions with Dirichlet distribution


## Running an experiment

- --dataset: CIFAR10, CIFAR100, FMNIST
- --batch_size: size of mini batch
- --num_epochs: total number of global communication rounds
- --num_clients: number of clients
- --sampling_rate: fraction of clients participating local training each global round
- --local_ep: number of local epochs
- --alphas: list of concentration parameters for generating data partitions
- --T: scaling parameter, temperature
- --seed: random seed for generating data partitions
- --alg: random, pow-d, CS, DivFL, HiCS
- --lr: initializing learning rate

We gave an example in `train_script.sh`

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







