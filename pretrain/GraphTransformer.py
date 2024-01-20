import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from transformers.models.bert.modeling_bert import BertSelfAttention, BertLayer, BertEmbeddings, BertPreTrainedModel
from transformers import AutoModelForMaskedLM

from utils import roc_auc_score, mrr_score, ndcg_score


############################ Take care that we assume the last one is venue #########################
############################ Take care that here we assume that there are three relations #########################
class GraphAggregation(BertSelfAttention):
    def __init__(self, config):
        super(GraphAggregation, self).__init__(config)
        self.output_attentions = False

    def forward(self, hidden_states, attention_mask):
        
        query = self.query(hidden_states)  # B 1 D
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        station_embed = self.multi_head_attention(query=query,
                                                    key=key,
                                                    value=value,
                                                    attention_mask=attention_mask)[0]  # B 1 D
        
        station_embed = station_embed.squeeze(1)

        return station_embed

    def multi_head_attention(self, query, key, value, attention_mask):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)


class GraphBertEncoder(nn.Module):
    def __init__(self, config):
        super(GraphBertEncoder, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

        self.graph_attention = GraphAggregation(config=config)

    def forward(self,
                hidden_states,
                attention_mask,
                feats):

        all_hidden_states = ()
        all_attentions = ()

        all_nodes_num, seq_length, emb_dim = hidden_states.shape

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > 0:
                # prepare for the graph aggregation
                cls_emb = hidden_states[:, :0].clone()  # B SN D
                cls_emb = torch.cat((cls_emb, feats),dim=1)
                station_emb_paper = self.graph_attention(hidden_states=cls_emb, attention_mask=None)  # B SN D
                # update the station in the query/key
                hidden_states[:, 0] = station_emb_paper[:,0]
                hidden_states = hidden_states.view(all_nodes_num, seq_length, emb_dim)
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask)

            else:
                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[:, :, :, :1] = -10000.0
                layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask)

            hidden_states = layer_outputs[0]

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class GraphTransformer(BertPreTrainedModel):
    def __init__(self, config):
        super(GraphTransformer, self).__init__(config=config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.embeddings = BertEmbeddings(config=config)
        self.encoder = GraphBertEncoder(config=config)
        self.paper_neighbour = 3

    def forward(self,
                input_ids,
                attention_mask,
                feats):
        all_nodes_num, seq_length = input_ids.shape
        # batch_size, subgraph_node_num = neighbor_mask.shape
        embedding_output = self.embeddings(input_ids=input_ids)

        # Add station attention mask
        ############################## we add two new position, one for text neighbour, one for non-text neighbour ####################################
        station_mask = torch.ones((all_nodes_num, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 2+L

        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # Add station_placeholder
        ############################# we add two new position, one for text neighbour, one for non-text neighbour ######################################
        station_placeholder = torch.zeros(all_nodes_num, 1, embedding_output.size(-1)).type(
            embedding_output.dtype).to(embedding_output.device)
        embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # N 2+L D

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            feats=feats)

        return encoder_outputs

class LinearPerMetapath(nn.Module):
    '''
        Linear projection per metapath for feature projection in SeHGNN.
    '''
    def __init__(self, cin, cout, num_metapaths):
        super(LinearPerMetapath, self).__init__()
        self.cin = cin
        self.cout = cout
        self.num_metapaths = num_metapaths

        self.W = nn.Parameter(torch.randn(self.num_metapaths, self.cin, self.cout))
        self.bias = nn.Parameter(torch.zeros(self.num_metapaths, self.cout))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias.unsqueeze(0)
    

class GraphTransformerForNeighborPredict(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = GraphTransformer(config)
        self.hidden_size = config.hidden_size
        self.init_weights()
        self.mlm_head = AutoModelForMaskedLM.from_config(config).cls
        self.mlm_weight = 1
        self.vocab_size = config.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.edge_coef = 10


    def init_mta_embed(self, args, feat_id):
        feats_path = os.path.join(args.data_path, 'feats_'+str(feat_id)+'.npy')
        print(feat_id)
        self.feats = np.load(feats_path, allow_pickle=True).item()
        self.embeding_node = nn.ParameterDict({})
        for k, v in self.feats.items():
            self.embeding_node[str(k)] = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size).uniform_(-0.5,0.5))
            # nn.init.xavier_normal_(self.embeding_node[str(k)])
        dropout = 0.5
        n_fp_layers = 2
        unfold_nested_list = lambda x: sum(x, [])
        self.input_drop = nn.Dropout(dropout)
        self.feats_len = len(self.feats)
        self.feature_projection = nn.Sequential(
            *([LinearPerMetapath(self.hidden_size, self.hidden_size, self.feats_len),
               nn.LayerNorm([self.feats_len, self.hidden_size]),
               nn.PReLU(),
               nn.Dropout(dropout),]
            + unfold_nested_list([[
               LinearPerMetapath(self.hidden_size, self.hidden_size, self.feats_len),
               nn.LayerNorm([self.feats_len, self.hidden_size]),
               nn.PReLU(),
               nn.Dropout(dropout),] for _ in range(n_fp_layers - 1)])
            )
        )

        self.out_projection = nn.Sequential(
            *([nn.Linear(self.hidden_size * (self.feats_len), self.hidden_size),
               nn.PReLU(),
               nn.Dropout(dropout),]
            + unfold_nested_list([[
               nn.Linear(self.hidden_size, self.hidden_size),
               nn.PReLU(),
               nn.Dropout(dropout),] for _ in range(n_fp_layers - 1)])
            )
        )


        

    def infer(self, input_ids_node_batch, attention_mask_node_batch, batch_feats, mask_predict=False, infer=False):
        '''
        B: batch size, N: 1 + neighbour_num, L: max_token_len, D: hidden dimmension
        '''
        if infer:
            device = input_ids_node_batch.device
            graph_node = batch_feats[:,0]
            graph_node = graph_node.flatten().tolist()
            batch_feats = {k: x[graph_node].to(device) for k, x in self.feats.items()}
            batch_feats = [x for k, x in batch_feats.items()]
            batch_feats = torch.stack(batch_feats, dim=1)

        # batch_feats = self.feature_projection(batch_feats)
        batch_feats = self.feature_projection(batch_feats)

        hidden_states = self.bert(input_ids_node_batch, attention_mask_node_batch, batch_feats)
        
        last_hidden_states = hidden_states[0]
        if mask_predict:
            return last_hidden_states
        cls_embeddings = last_hidden_states[:, 0]

        return cls_embeddings

    def forward(self, input_ids_query_batch, attention_mask_query_batch,\
                query_label_batch, input_ids_key_batch, attention_mask_key_batch,\
                key_label_batch, graph_node_id, **kwargs):
        device = input_ids_query_batch.device
        query_node_id = graph_node_id[:,0]
        key_node_id = graph_node_id[:,1]

        query_node_id = query_node_id.flatten().tolist()
        query_batch_feats = {k: x[query_node_id].to(device) for k, x in self.feats.items()}
        query_batch_feats = [self.input_drop(x @ self.embeding_node[k]) for k, x in query_batch_feats.items()]
        query_batch_feats = torch.stack(query_batch_feats, dim=1)

        key_node_id = key_node_id.flatten().tolist()
        key_batch_feats = {k: x[key_node_id].to(device) for k, x in self.feats.items()}
        key_batch_feats = [self.input_drop(x @ self.embeding_node[k]) for k, x in key_batch_feats.items()]
        key_batch_feats = torch.stack(key_batch_feats, dim=1)

        query_embeddings = self.infer(input_ids_query_batch, attention_mask_query_batch, query_batch_feats, mask_predict=True)
        key_embeddings = self.infer(input_ids_key_batch, attention_mask_key_batch, key_batch_feats, mask_predict=True)

        query_embedding_cls = query_embeddings[:, 0]
        key_embeddings_cls = key_embeddings[:, 0]
       

        scores = torch.matmul(query_embedding_cls, key_embeddings_cls.transpose(0, 1))
        labels = torch.arange(start=0, end=scores.shape[0], dtype=torch.long, device=scores.device)
        tt_loss = F.cross_entropy(scores, labels)


        # mlm loss
        q_prediction_scores = self.mlm_head(query_embeddings)[:,1:]
        k_prediction_scores = self.mlm_head(key_embeddings)[:,1:]
        q_mlm_loss = F.cross_entropy(q_prediction_scores.contiguous().view(-1, self.vocab_size), query_label_batch.contiguous().view(-1))
        k_mlm_loss = F.cross_entropy(k_prediction_scores.contiguous().view(-1, self.vocab_size), key_label_batch.contiguous().view(-1))
        loss = self.mlm_weight * (q_mlm_loss + k_mlm_loss) + tt_loss
        return loss
