import torch.nn as nn
import torch.optim as optim
from utils.misc import AverageMeter,mkdir_p
from utils.eval import accuracy,average_weights,average_weights_weighted, sum_list,global_acc
from progress.bar import Bar
from models.bit import BitLinear, BitConv2d
from models.resnet import resnet
import time
import torch
import copy
class Client(object):
    def __init__(self, args, trainloader,idx, budget, bin = True):
        self.args = args
        self.trainloader = trainloader
        self.idx = idx
        self.budget = budget
                
        self.model  = resnet(num_classes=args["n_classes"], depth=20, block_name= 'BasicBlock', Nbits = self.budget,
                      act_bit = 4, bin = bin).cuda()
        
        print(f'Total params of client_{idx}: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))
        self.criterion = nn.CrossEntropyLoss()
        self.Nbit_dict = self.model.pruning(threshold=0.0, drop=True)
        self.bit_assignment = []
        
        # getting original bit width assignment 
        for key in self.Nbit_dict.keys():
            self.bit_assignment.append(self.Nbit_dict[key][0])
        self.TP = self.model.total_param()
        self.TB = self.model.total_bit()
        self.Comp = (self.TP *32)/self.TB
        self.optimizer = optim.SGD(self.model.parameters(), lr=args["lr"], momentum=args["momentum"], 
                                   weight_decay=args["weight_decay"])
        self.delta_bit = [0]*len(self.bit_assignment)
        state = copy.deepcopy(args)
    def train(self,epoch):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Processing', max=len(self.trainloader))
        
                
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # sparsity-promoting regularization term
            reg=0.
            if self.args["lambda"]:
                for name, module in self.model.named_modules():
                    if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                        reg = module.L1reg(reg)
            total_loss = loss+self.args["lambda"]*reg/self.TP

            # measure accuracy
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # plot progress
            bar.suffix  = '(Client: {idx} | CP: {Comp:.2f}X | Epoch:{epoch} | top1: {top1: .4f}'.format(
                        idx = self.idx,
                        Comp=self.Comp,
                        epoch=epoch,
                        top1=top1.avg,
                        )
            bar.next()
        bar.finish()
                
        return top1.avg


    def local_training(self, global_epoch):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args["lr"], momentum=self.args["momentum"],                                                             weight_decay=self.args["weight_decay"])
        
        for epoch in range(self.args["local_epochs"]):
            self.adjust_learning_rate(epoch + global_epoch*self.args["local_epochs"])
            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
            train_acc = self.train(epoch)
            
            for name, module in self.model.named_modules():
                if isinstance(module, BitConv2d) or isinstance(module, BitLinear):
                    module.quant(maxbit = 8)
                    
        print(f'Client {self.idx} Training Top 1 Acc at global round {global_epoch} : {train_acc}') 
            
        
        # update bitwidth assignment 
        self.Nbit_dict = self.model.pruning(threshold=self.args["thre"], drop=True)
        
        bit_assignment = []
        delta_bit = []
        for key in self.Nbit_dict.keys():
            bit_assignment.append(self.Nbit_dict[key][0])
        
        for i in range(len(bit_assignment)):
            delta_bit.append(self.bit_assignment[i] - bit_assignment[i])
        
        self.delta_bit = delta_bit
        self.bit_assignment = bit_assignment
        
        del self.optimizer
        self.TP = self.model.total_param()
        self.TB = self.model.total_bit()
        self.Comp = (self.TP*32)/self.TB 
        print(' Compression rate after pruning [%d / %d]:  %.2f X' % (self.TP*32, self.TB, self.Comp))
        
        
    def adjust_learning_rate(self, epoch):
        global state
        if epoch in self.args["schedule"]:
            state['lr'] *= self.args["gamma"]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = state['lr']
                
    def load_model(self,global_weights):
        self.model.load_state_dict(global_weights)
        