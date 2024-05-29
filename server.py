from client import Client
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as opAccuracyim
from utils.misc import AverageMeter,mkdir_p
from utils.eval import accuracy,average_weights, sum_list,global_acc, average_stat
from progress.bar import Bar
from models.bit import BitLinear, BitConv2d
import numpy as np

class Server(object):
    def __init__(self,args,trainloaders, testloader):
        self.args = args
        self.trainloaders = trainloaders
        self.testloader = testloader
        self.device = args["device"]
        self.clients = []
        self.budgets = args["budgets"]
        for idx in range(args["n_clients"]):
            self.clients.append(Client(args, trainloaders[idx], idx, self.budgets[idx], bin = True))
        self.global_model = resnet(num_classes=args["n_classes"], depth=20, block_name= 'BasicBlock', 
                                              Nbits = 16, act_bit = 16, bin = False).cuda()
        n_samples = np.array([len(client.trainloader.dataset)*client.budget for client in self.clients])
        self.client_weights = n_samples / np.sum(n_samples)
        
        num_params = []
        for name, module in self.global_model.named_modules():   
            if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                num_params.append(module.total_weight)
        num_params = np.array(num_params)
        self.layer_weights = np.sum(num_params)/num_params

    def train(self):
        for epoch in range(args["epochs"]):
            local_weights = []
            local_bit_assignments = []
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
                local_bit_assignments.append(np.array(self.clients[idx].bit_assignment))

            global_weights = average_weights(local_weights)
            average_bit_assignment = average_stat(local_bit_assignments,sampled_clients, self.client_weights)

            global_model.load_state_dict(global_weights)
            acc_top1, acc_top5 = global_acc(self.global_model, self.testloader)
            print(f'Top 1 accuracy: {acc_top1}, Top 5 accuracy: {acc_top5}  at global round {epoch}.')

            for idx in sampled_clients:
                self.clients[idx].load_model(global_weights)
                self.clients[idx].bit_assignment = self.pruning_or_growing(copy.deepcopy(average_bit_assignment),                                                                 layer_weights, num_params, self.clients[idx].budget)
                
                # re-binarize the local model with the obtained bitwidth assignment
                count = 0
                for name, module in self.clients[idx].model.named_modules():                
                    if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                        N0 = module.Nbits
                        module.Nbits = int(Clients[idx].bit_assignment[count])
                        module.to_bin() 
                        N = module.Nbits
                        ex = np.arange(N-1, -1, -1)
                        module.exps = torch.Tensor((2**ex)/(2**(N)-1)).float()
                        count += 1


    def pruning_or_growing(self,average_bit_assignment, layer_weights, num_params, budget):
        average_bit = 0
        n = len(average_bit_assignment)
        for i in range(n):
            average_bit += layers_weights[i]*average_bit_assignment[i]

        priority = copy.deepcopy(layer_weights)
        cursor = np.argmin(priority)    
        count = 0
        while average_bit > budget and count < n:
            if average_bit_assignment[cursor] > 1:
                average_bit_assignment[cursor] -= 1
                average_bit -= layers_weights[cursor]*1
            else:
                priority[cursor] = float('inf')
                cursor = np.argmin(priority)
                count += 1
        priority = copy.deepcopy(layer_weights)
        cursor = np.argmax(priority)
        count = 0
        while average_bit < budget and count < n:
            if average_bit_assignment[cursor] < 8:
                average_bit_assignment[cursor] += 1
                average_bit += layers_weights[cursor]*1
            else:
                priority[cursor] = -float('inf')
                cursor = np.argmax(priority)
                count += 1
        print(average_bit_assignment)
        return average_bit_assignment