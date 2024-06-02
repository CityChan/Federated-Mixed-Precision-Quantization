import os
import os.path
import sys
import logging
import copy
import time
import torch
from sampling import count_data_partitions,get_dataloaders_Dirichlet

from server.server_fedmpq import Server_FedMPQ
from server.server_aqfl import Server_AQFL
from server.server_fp import Server_FP

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
    if args["alg"] == "FedMPQ":
        server = Server_FedMPQ(args,trainloaders,testloader)
    if args["alg"] == "AQFL":
        server = Server_AQFL(args,trainloaders,testloader)
    if args["alg"] == "FP":
        server = Server_FP(args,trainloaders,testloader)
    server.train()
    
