# Mixed-Precision Quantization for Federated Learning on Resource-Constrained Heterogeneous Devices

This is the official implementation for the CVPR2024 paper


## Install packages in requrement.txt

`pip install -r requirements.txt`


## Code instruction
- `models.py`: the model's structure used in the experiments
- `utils.py`: utilization functions for computing metrics of the experiments
- `DivFL_utils.py`: utilization functions for DivFL sampling method
- `clustering_utils.py`: utilization function for Clustered sampling method
- `sampling.py`: functions for generating data partitions with Dirichlet distribution
- `HiCS.py`: utilization function for HiCS-FL sampling method
- `train.py`: training main function

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







