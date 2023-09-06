import argparse
from xmlrpc.client import boolean

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = 'new_BeiBei', type = str,
                        help = 'Dataset to be used. (BeiBei or BeiDian)')
    parser.add_argument('--seed', default = 1008, type = int,
                        help = 'Random Seed')
    parser.add_argument('--model', default = 'MF', type = str,
                        help = 'Model')
    parser.add_argument('--embed_size', default = 64, type = int,
                        help = 'embedding size for all layer')
    parser.add_argument('--lr', default = 0.01, type = float,
                        help = 'learning rate') 
    parser.add_argument('--epochs', default = 300, type = int,
                        help = 'epoch number')
    parser.add_argument('--num_negatives', default = 1, type = int,
                        help = 'number of negative')
    parser.add_argument('--early_stop', default = 10, type = int,
                        help = 'early_stop validation')
    parser.add_argument('--batch_size', default = 512, type = int,
                        help = 'batch size')
    parser.add_argument('--drop_ratio', default = 0.2, type = float,
                        help = 'drop_ratio')
    parser.add_argument('--layer_num', default = 3, type = int,
                        help = 'number of layers')         
    parser.add_argument('--cuda', default = 0, type = int,
                        help = 'cuda')
    parser.add_argument('--weight_decay', default = 1e-4, type = float,
                        help = 'weight_decay')
    parser.add_argument('--loss_reg', default = 0.1, type = float,
                        help = 'loss_reg')
    args = parser.parse_args()
    return args