import os
import pickle
import logging
from tqdm import tqdm
import numpy as np

import torch

from torch.utils.data.dataset import Dataset, TensorDataset
from transformers.data.data_collator import DataCollatorForLanguageModeling
from torch.nn.utils.rnn import pad_sequence




logger = logging.getLogger(__name__)


def load_dataset_text(args, tokenizer, infer=False, evaluate=False, test=False, mask=True):
    '''
    features : (token_query_and_neighbors, attention_query_and_neighbors, mask_query_and_neighbors), (token_key_and_neighbors, attention_key_and_neighbors, mask_key_and_neighbors)
    '''

    # block for processes which are not the core process
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    # load data features from cache or dataset file
    evaluation_set_name = 'infer' if infer else 'train'
    if evaluate:
        if test:
            evaluation_set_name = 'test'
        else:
            evaluation_set_name = 'valid'
    cached_features_file = os.path.join(args.data_path, 'cached_{}_{}_{}_{}_{}_{}_{}'.format(
        args.data_mode,
        evaluation_set_name,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_length),
        args.paper_neighbour,
        args.author_neighbour,
        mask))
    # exist or not
    if os.path.exists(cached_features_file):
        if args.local_rank in [-1, 0]:
            logger.info(f"Loading features from cached file {cached_features_file}")
        features = pickle.load(open(cached_features_file,'rb'))
    else:
        if args.local_rank in [-1, 0]:
            logger.info("Creating features from dataset file at %s",
                    cached_features_file)

        read_file = evaluation_set_name + '_pp.tsv'
        if infer:
            features = read_process_data_infer(os.path.join(args.data_path, read_file), tokenizer, args.max_length)
        else:
            features = read_process_data_heter(os.path.join(args.data_path, read_file), tokenizer, args.max_length, mask)
        logger.info(f"Saving features into cached file {cached_features_file}")
        pickle.dump(features, open(cached_features_file, 'wb'))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()
    # convert to Tensors and build dataset
    
    if infer:
        token_query_neighbors = torch.LongTensor(features[0][0])
        attention_query_neighbors = torch.LongTensor(features[0][1])
        graph_node = torch.LongTensor(features[0][2])
        dataset = TensorDataset(token_query_neighbors, attention_query_neighbors, graph_node)
    else:
        token_query_neighbors = torch.LongTensor(features[0][0])
        attention_query_neighbors = torch.LongTensor(features[0][1])

        token_key_neighbors = torch.LongTensor(features[1][0])
        attention_key_neighbors = torch.LongTensor(features[1][1])
        graph_node = torch.LongTensor(features[2])
        if evaluate == False and test == False:
            query_label = torch.LongTensor(features[0][2])
            key_label = torch.LongTensor(features[1][2])
        
            dataset = TensorDataset(token_query_neighbors, attention_query_neighbors, query_label,\
                                    token_key_neighbors, attention_key_neighbors, key_label, graph_node)
        else:
            dataset = TensorDataset(token_query_neighbors, attention_query_neighbors,\
                                token_key_neighbors, attention_key_neighbors, graph_node)

    return dataset

def read_process_data_heter(dir_text, tokenizer, max_length, mask=True):
    '''
    Each line is a tweet/POI node pair. Each node is made up of [itself, tweet_neighbour * neighbour tweet, mention_neighbour * neighbour mention, tag_neighbour * neighbour tag, author].
    '''
    token_query = []
    attention_query = []
    query_label_list = []

    token_key = []
    attention_key = []
    key_label_list = []

    iidx = []

    data_collecter = DataCollatorForLanguageModeling(tokenizer)
    # cnt = 0
    with open(dir_text) as f:
        data = f.readlines()
        for line in tqdm(data):
            # cnt +=1
            # if cnt == 1000:
            #     break
            a = line.strip().split('\$\$')
            if len(a) == 3:
                iid, query_all, key_all = a
            else:
                print(a)
                raise ValueError('stop')

            query_all = [list(map(int, s.split())) for s in [query_all]]
            key_all = [list(map(int, s.split())) for s in [key_all]]
            def padding_to_max_length(seqs, max_length):
                processed_sequences = []
                for seq in seqs:
                    if len(seq) >= max_length:
                        processed_sequences.append(seq[:max_length])
                    else:
                        processed_sequences.append(seq + [0] * (max_length - len(seq)))
                return torch.tensor(processed_sequences)
            encoded_query = padding_to_max_length(query_all, max_length)
            encoded_query = pad_sequence(encoded_query, batch_first=True, padding_value=0)


            if mask:
                query_ret = data_collecter([torch.tensor(i) for i in encoded_query])
                query_label = query_ret['labels'].tolist()
                query_label_list.append(query_label)

            token_query.append(encoded_query[0].tolist())
            attention_query.append((encoded_query!= 0).float()[0].tolist())

            encoded_key = padding_to_max_length(key_all, max_length)
            encoded_key = pad_sequence(encoded_key, batch_first=True, padding_value=0)


            if mask:
                key_ret = data_collecter([torch.tensor(i) for i in encoded_key])
                key_label = key_ret['labels'].tolist()
                key_label_list.append(key_label)

            token_key.append(encoded_key[0].tolist())
            attention_key.append((encoded_key!= 0).float()[0].tolist())

            iidx.append([int(i) for i in iid.split()])
                   

    if mask:
        return (token_query, attention_query, query_label_list), \
            (token_key, attention_key, key_label_list), \
            (iidx)
    else:
        return (token_query, attention_query), \
            (token_key, attention_key), \
            (iidx)

def read_process_data_infer(dir_text, tokenizer, max_length):
    '''
    Each line is a tweet/POI node pair. Each node is made up of [itself, tweet_neighbour * neighbour tweet, mention_neighbour * neighbour mention, tag_neighbour * neighbour tag, author].
    '''
    token_query = []
    attention_query = []
    iidx = []

    with open(dir_text) as f:
        data = f.readlines()
        for line in tqdm(data):
            a = line.strip().split('\$\$')
            if len(a) == 2:
                iid, query_all = a
            else:
                print(a)
                raise ValueError('stop')
            
            query_all = [list(map(int, s.split())) for s in [query_all]]
            def padding_to_max_length(seqs, max_length):
                processed_sequences = []
                for seq in seqs:
                    if len(seq) >= max_length:
                        processed_sequences.append(seq[:max_length])
                    else:
                        processed_sequences.append(seq + [0] * (max_length - len(seq)))
                return torch.tensor(processed_sequences)
            encoded_query = padding_to_max_length(query_all, max_length)
            encoded_query = pad_sequence(encoded_query, batch_first=True, padding_value=0)

            token_query.append(encoded_query[0].tolist())
            attention_query.append((encoded_query!= 0).float()[0].tolist())

            iidx.append([int(i) for i in iid.split()])
                   

        return (token_query, attention_query, iidx),
