import os
import os.path
import sys
import logging
import copy
import time
import torch
from sampling import count_data_partitions,get_dataloaders_Dirichlet

from server import Server
def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    device = device.split(',')
    print(args)
    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)

    myseed = 42069  # set a random seed for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        
        
        
        
def _train(args):
    trainloaders,testloader = get_dataloaders_Dirichlet(n_clients = args["n_clients"], alpha=args["alpha"], rand_seed = 
                                                        args["seed"], dataset = args["dataset"], batch_size = args["batch_size"])
    
    server = Server(args,trainloaders,testloader)
    server.train()
    
