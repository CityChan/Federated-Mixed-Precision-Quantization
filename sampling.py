import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os


cache_path = "./data/data_partitions"

if not os.path.exists(cache_path):
    os.makedirs(cache_path)



def count_data_partitions(list_loaders, dataset):
    n_clients = len(list_loaders)
    Counts = []
    if dataset == 'MNIST' or 'CIFAR10' or 'SVHN':
        n_classes = 10
    if dataset == 'CIFAR100':
        n_classes = 100
    if dataset == 'TinyImageNet':
        n_classes = 200
    for idx in range(n_clients):
        counts = [0]*n_classes
        for batch_idx,(X,y) in enumerate(list_loaders[idx]):
            batch_size = len(y)
            y = np.array(y)
            for i in range(batch_size):
                counts[int(y[i])] += 1
        Counts.append(counts)
    return Counts


class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y
    
def LocalDataloaders(dataset, dict_users, batch_size, Shuffle = True):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,dict_users[i]),
                        batch_size=batch_size,
                        shuffle = Shuffle,
                        drop_last=True)
        loaders.append(loader)
    return loaders


def get_dataloaders_Dirichlet(n_clients, alpha=0.5,rand_seed = 0, dataset = 'CIFAR10', batch_size = 64):
    if dataset == 'CIFAR10':
        K = 10
        data_dir = './data/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        
        file_name_train = f"data_partitions/CIFAR10_train_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_train = data_dir + file_name_train

        
        file_name_labels = f"data_partitions/CIFAR10_train_{n_clients}_alpha{alpha}_seed{rand_seed}_labels.pkl"
        path_labels = data_dir + file_name_labels
                
        
    if dataset == 'CIFAR100':
        K = 100
        data_dir = './data/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408),  (0.2675, 0.2565, 0.2761))])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
  
        file_name_train = f"data_partitions/CIFAR100_train_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_train = data_dir + file_name_train
        
        
        file_name_labels = f"data_partitions/CIFAR100_train_{n_clients}_alpha{alpha}_seed{rand_seed}_labels.pkl"
        path_labels = data_dir + file_name_labels
        
    if dataset == 'TinyImageNet':
        K = 200
        data_dir = './data/'
        train_data_path = '../data/tiny-imagenet-200/train'
        test_data_path = '../data/tiny-imagenet-200/val'

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_dataset = ImageFolder(root=train_data_path, transform=transform)
        
        y_train = np.array(train_dataset.targets)
        
        file_name_train = f"data_partitions/TinyImageNet_train_{n_clients}_alpha{alpha}_seed{rand_seed}.pkl"
        path_train = data_dir + file_name_train

        file_name_labels = f"data_partitions/TinyImageNet_train_{n_clients}_alpha{alpha}_seed{rand_seed}_labels.pkl"
        path_labels = data_dir + file_name_labels
        
        
    if not os.path.isfile(path_train):
        
        min_size = 0
        N = len(train_dataset)
        net_dataidx_map = {}
        np.random.seed(rand_seed)

        while min_size < 0.1*(N/n_clients):
            idx_batch = [[] for _ in range(n_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
                ## Balance
                proportions_train = np.array([p*(len(idx_j)<N/n_clients) for p,idx_j in zip(proportions,idx_batch)])
                proportions_train = proportions_train/proportions_train.sum()
                proportions_train = (np.cumsum(proportions_train)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions_train))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
   
    
        with open(path_train, "wb") as output:
            pickle.dump(net_dataidx_map, output)

            
    with open(path_train, "rb") as r:
        dict_users = pickle.load(r)
        trainloaders = LocalDataloaders(train_dataset,dict_users,batch_size,Shuffle = True)
        

    testloader = DataLoader(test_dataset,batch_size,shuffle = True)
        
    if not os.path.isfile(path_labels):
        Counts = count_data_partitions(trainloaders, dataset)
        with open(path_labels, "wb") as output:
            pickle.dump(Counts, output)
            
    with open(path_labels, "rb") as r:
        Counts = pickle.load(r)
        print('Print out data label distributions for each clients:')
        for idx in range(len(Counts)):
            print(Counts[idx])
            
    return trainloaders, testloader