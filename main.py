import json
import argparse
import ipdb
from trainer import train
def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  
    args.update(param)  
    train(args)

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of FedMPQ algorthm.')
    parser.add_argument('--config', type=str, default='./configs/CIFAR100_FedMPQ.json',
                        help='Json file of settings.')
    parser.add_argument('--device', type=str, default='0')
    return parser


if __name__ == '__main__':
    main()