import os
import logging
import argparse
from pathlib import Path
import torch.multiprocessing as mp

from run import train, test, infer
from utils import setuplogging, str2bool, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", choices=['train', 'test', 'infer'])
parser.add_argument("--data_path", type=str)
parser.add_argument("--model_dir", type=str, default='ckpt/OAG_Venue', choices=['ckpt/OAG_Venue', 'ckpt/GoodReads', 'ckpt/Patent'])  # path to save
parser.add_argument("--data_mode", default="text", type=str, choices=['text'])
parser.add_argument("--pretrain_embed", type=str2bool, default=False) # use pretrained textless node embedding or not
parser.add_argument("--pretrain_dir", default="data/pretrain_embed", type=str) # pretrain author/venue embedding dir

# turing
parser.add_argument("--pretrain_LM", type=str2bool, default=True)
parser.add_argument("--heter_embed_size", type=int, default=100)
parser.add_argument("--attr_embed_size", type=int, default=768)
parser.add_argument("--num_hidden_layers", type=int, default=4)
parser.add_argument("--attr_vec", type=str, default='tfidf', choices=['cnt', 'tfidf'])

# some parameters fixed depend on dataset
parser.add_argument("--max_length", type=int, default=20) # this parameter should be the same for all models for one particular dataset
parser.add_argument("--train_batch_size", type=int, default=72)
parser.add_argument("--val_batch_size", type=int, default=12)
parser.add_argument("--test_batch_size", type=int, default=12)

parser.add_argument("--author_neighbour", type=int, default=5)
parser.add_argument("--paper_neighbour", type=int, default=3)

# distribution
parser.add_argument("--local-rank", type=int, default=-1)

# model training (these parameters can be fixed)
parser.add_argument("--lr", type=float, default=6e-5)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--early_stop", type=int, default=3)
parser.add_argument("--log_steps", type=int, default=1624)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--load", type=str2bool, default=False)
parser.add_argument("--max_grad_norm", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--adam_epsilon", type=float, default=1e-8)
parser.add_argument("--enable_gpu", type=str2bool, default=True)


# load checkpoint or test
parser.add_argument("--model_name_or_path", default="../data/bert-base-cased", type=str,
                    help="Path to pre-trained model or shortcut name. ")
parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default='ckpt/book/xxx.pt',
        help="choose which ckpt to load and test"
    )

# half float
parser.add_argument("--fp16", type=str2bool, default=True)

# MAG-CS
parser.add_argument("--author_num", type=int, default=703184)
parser.add_argument("--venue_num", type=int, default=105)
parser.add_argument("--paper_num", type=int, default=564340)
parser.add_argument("--feats", type=int, default=4)


args = parser.parse_args()

if args.local_rank in [-1,0]:
    logging.info(args)
    print(args)

if __name__ == "__main__":

    set_seed(args.random_seed)
    setuplogging()

    if args.local_rank in [-1,0]:
        print(os.getcwd())
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == 'train':
        if args.local_rank in [-1,0]:
            print('-----------train------------')
        train(args)

    if args.mode == 'test':
        print('-------------test--------------')
        ################## You should use single GPU for testing. ####################
        assert args.local_rank == -1
        test(args)

    if args.mode == 'infer':
        print('-------------infer--------------')
        ################## You should use single GPU for infering. ####################
        assert args.local_rank == -1
        infer(args)

    # if args.mode == 'author_embed':
    #     print('-------------author embedding--------------')
    #     ################## You should use single GPU for infering. ####################
    #     assert args.local_rank == -1
    #     author_embed(args)
