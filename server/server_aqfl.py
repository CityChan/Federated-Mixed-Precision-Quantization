import sys
sys.path.append('../')
from client.client_aqfl import Client_AQFL as Client
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as opAccuracyim
from utils.misc import AverageMeter,mkdir_p
from utils.eval import accuracy,average_weights, sum_list,global_acc, average_stat
from progress.bar import Bar
from models.bit import BitLinear, BitConv2d
from models.resnet import resnet
import numpy as np

class Server_AQFL(object):
    def __init__(self,args,trainloaders, testloader):
        self.args = args
        self.trainloaders = trainloaders
        self.testloader = testloader
        self.device = args["device"]
        self.clients = []
        self.budgets = args["budgets"]
        
        # local modes are quantized
        for idx in range(args["n_clients"]):
            self.clients.append(Client(args, trainloaders[idx], idx, self.budgets[idx], bin = True))
        
        # global model is full-precision 
        self.global_model = resnet(num_classes=args["n_classes"], depth=20, block_name= 'BasicBlock', 
                                              Nbits = 16, act_bit = 16, bin = False).cuda()
        
        
        n_samples = np.array([len(client.trainloader.dataset)*client.budget for client in self.clients])
        # compute client's weight based on number of data samples
        self.client_weights = n_samples / np.sum(n_samples)
        num_params = []
        for name, module in self.global_model.named_modules():   
            if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                num_params.append(module.total_weight)
        # compute layer's weight based on number of parameters
        self.num_params = np.array(num_params)
        self.layer_weights = self.num_params/np.sum(self.num_params)

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
                # send to the server and get float local model
                for name, module in self.clients[idx].model.named_modules():
                    if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                        module.to_float() 

                local_weights.append(self.clients[idx].model.state_dict())
                
            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            acc_top1, acc_top5 = global_acc(self.global_model, self.testloader)
            print(f'Top 1 accuracy: {acc_top1}, Top 5 accuracy: {acc_top5}  at global round {epoch}.')

            for idx in sampled_clients:
                self.clients[idx].load_model(global_weights)
                for name, module in self.clients[idx].model.named_modules():                
                    if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                        N0 = module.Nbits
                        module.Nbits = self.clients[idx].budget
                        module.to_bin() 
                        N = module.Nbits
                        ex = np.arange(N-1, -1, -1)
                        module.exps = torch.Tensor((2**ex)/(2**(N)-1)).float()
