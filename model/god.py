import copy
from tqdm import tqdm
import torch
import torch.nn as nn

from model.utils import create_optimizer, accuracy

import sys

import importlib
from pygod.utils import load_data
from pygod.metrics import eval_roc_auc,eval_average_precision,eval_ndcg,eval_precision_at_k,eval_recall_at_k
from pygod.models import ADANET
from torch_geometric.utils import to_dense_adj
import numpy as np

anomaly_num_dict={'weibo':868,'reddit':366,'disney':6,'books':28,'enron':5,'inj_cora':138,'inj_amazon':694,'inj_flickr':4414}

def god_evaluation(data_name,model_name,attr_encoder_name,struct_encoder_name,topology_encoder_name,
attr_decoder_name,struct_decoder_name,attr_ssl_model,struct_ssl_model,topology_ssl_model,graph, x, 
aggr_f,lr_f, max_epoch_f, alpha_f,dropout_f,loss_f,loss_weight_f,T_f,num_hidden,node_encoder_num_layers,edge_encoder_num_layers,subgraph_encoder_num_layers,
attr_decoder_num_layers=1,struct_decoder_num_layers=1,use_ssl=False,use_encoder_num=1,attention=None,sparse_attention_weight=0.001,
theta=1.001,eta=1.001):
    
    if use_encoder_num==1:
        attr_ssl_model.eval()
    if use_encoder_num==2:
        attr_ssl_model.eval()
        struct_ssl_model.eval()
    if use_encoder_num==3:
        attr_ssl_model.eval()
        struct_ssl_model.eval()
        topology_ssl_model.eval()
        
    
    model= eval(model_name)(epoch=max_epoch_f,aggr=aggr_f,hid_dim=num_hidden,alpha=alpha_f,dropout=dropout_f,\
        lr=lr_f,loss_name=loss_f,loss_weight=loss_weight_f,T=T_f,use_encoder_num=use_encoder_num,attention=attention,\
            attr_encoder_name=attr_encoder_name,struct_encoder_name=struct_encoder_name,topology_encoder_name=topology_encoder_name,attr_decoder_name=attr_decoder_name,struct_decoder_name=struct_decoder_name,\
            node_encoder_num_layers=node_encoder_num_layers,edge_encoder_num_layers=edge_encoder_num_layers,subgraph_encoder_num_layers=subgraph_encoder_num_layers,\
                attr_decoder_num_layers=attr_decoder_num_layers,\
                struct_decoder_num_layers=struct_decoder_num_layers,sparse_attention_weight=sparse_attention_weight,theta=theta,eta=eta)

    if use_ssl and use_encoder_num>0:
        if use_encoder_num==1:
            model.fit(graph,pretrain_attr_encoder=attr_ssl_model.encoder,pretrain_struct_encoder=None,pretrain_topology_encoder=None)
        elif use_encoder_num==2:
            model.fit(graph,pretrain_attr_encoder=attr_ssl_model.encoder,pretrain_struct_encoder=struct_ssl_model.encoder,pretrain_topology_encoder=None) 
        elif use_encoder_num==3: 
            model.fit(graph,pretrain_attr_encoder=attr_ssl_model.encoder,pretrain_struct_encoder=struct_ssl_model.encoder,pretrain_topology_encoder=topology_ssl_model.encoder)       
        else:
            assert(f'wrong encoder num: {use_encoder_num}')
    else:
        model.fit(graph)
    labels = model.predict(graph)

    outlier_scores= model.decision_function(graph)
    edge_outlier_scores=model.decision_struct_function(graph)

    auc_score = eval_roc_auc(graph.y.bool().cpu().numpy(), outlier_scores)

    print(f'auc_score: {auc_score:.4f}',)

    return auc_score,ap_score,ndcg_score,pk_score,rk_score,outlier_scores,edge_outlier_scores
