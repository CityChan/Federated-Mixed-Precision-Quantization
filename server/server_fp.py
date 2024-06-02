import sys
sys.path.append('../')
from client.client_fp import Client_FP as Client
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as opAccuracyim
from utils.misc import AverageMeter,mkdir_p
from utils.eval import accuracy,average_weights, sum_list,global_acc, average_stat
from progress.bar import Bar
from models.bit import BitLinear, BitConv2d
from models.resnet_fp import resnet20
import numpy as np

class Server_FP(object):
    def __init__(self,args,trainloaders, testloader):
        self.args = args
        self.trainloaders = trainloaders
        self.testloader = testloader
        self.device = args["device"]
        self.clients = []
        
        # local modes are quantized
        for idx in range(args["n_clients"]):
            self.clients.append(Client(args, trainloaders[idx], idx))
        
        # global model is full-precision 
        self.global_model = resnet20(num_classes = args["n_classes"]).cuda()
        
        n_samples = np.array([len(client.trainloader.dataset) for client in self.clients])
        # compute client's weight based on number of data samples
        self.client_weights = n_samples / np.sum(n_samples)
        
    def train(self):
        for epoch in range(self.args["epochs"]):
            local_weights = []
            local_delta_bits = []
            self.global_model.train()
            print(f'\n | Global Training Round : {epoch+1} |\n')
            K = int(self.args["sampling_rate"]*self.args["n_clients"])
            sampled_clients = np.random.choice(self.args["n_clients"], K , replace=False)
            
            for idx in sampled_clients:
                self.clients[idx].local_training(epoch)
                local_weights.append(self.clients[idx].model.state_dict())
                
            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            acc_top1, acc_top5 = global_acc(self.global_model, self.testloader)
            print(f'Top 1 accuracy: {acc_top1}, Top 5 accuracy: {acc_top5}  at global round {epoch}.')
            
            for idx in sampled_clients:
                self.clients[idx].load_model(global_weights)
            

