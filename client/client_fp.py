import torch.nn as nn
import torch.optim as optim
from utils.misc import AverageMeter,mkdir_p
from utils.eval import accuracy,average_weights,average_weights_weighted, sum_list,global_acc
from progress.bar import Bar
from models.bit import BitLinear, BitConv2d
from models.resnet_fp import resnet20
import time
import torch
import copy
from QuantOptimizer import QuantOptimizer
class Client_FP(object):
    def __init__(self, args, trainloader,idx):
        self.args = args
        self.trainloader = trainloader
        self.idx = idx
        self.model  = resnet20(num_classes = args["n_classes"]).cuda()
        
        print(f'Total params of client_{idx}: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=args["lr"], momentum=args["momentum"], 
                                   weight_decay=args["weight_decay"])
        self.state = copy.deepcopy(args)
        
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

            # measure accuracy
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # plot progress
            bar.suffix  = '(Client: {idx} | Epoch:{epoch} | top1: {top1: .4f}'.format(
                        idx = self.idx,
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

        print(f'Client {self.idx} Training Top 1 Acc at global round {global_epoch} : {train_acc}')      
        del self.optimizer
        
    def adjust_learning_rate(self, epoch):
        if epoch in self.args["schedule"]:
            self.state['lr'] *= self.args["gamma"]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']
                
    def load_model(self,global_weights):
        self.model.load_state_dict(global_weights)
        