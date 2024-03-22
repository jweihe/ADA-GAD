# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import NeighborLoader
from sklearn.utils.validation import check_is_fitted

from . import BaseDetector
from .basic_nn import GCN
from .gat import GAT

from ..utils import validate_device
from ..metrics import eval_roc_auc

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity

def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead, nhead_out, attn_drop, negative_slope=0.2, concat_out=True,aggr='sum') -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation="relu",
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=int(in_dim),
            num_hidden=int(num_hidden),
            out_dim=int(out_dim),
            num_layers=num_layers,
            dropout=dropout,
            activation="relu",
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod= GCN(
                in_channels=int(in_dim),
                hidden_channels=int(num_hidden),
                num_layers=num_layers,
                out_channels=int(out_dim),
                dropout=dropout,
                act=activation,
                aggr=aggr)
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod

class ADANET(BaseDetector):
    def __init__(self,
                 hid_dim=64,
                 dropout=0.3,
                 weight_decay=0.,
                 act=F.relu,
                 alpha=None,
                 contamination=0.1,
                 lr=5e-3,
                 epoch=5,
                 gpu=0,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=False,
                 max_batch_size=20000,
                 aggr='add',
                 loss_name='rec',
                 loss_weight=0.1,
                 T=100,
                 use_encoder_num=1,
                 attention=0,
                 attr_encoder_name='gcn',
                 struct_encoder_name='gcn',
                 topology_encoder_name='gcn',
                 attr_decoder_name='mlp',
                 struct_decoder_name='mlp',
                 node_encoder_num_layers=2,
                 edge_encoder_num_layers=2,
                 subgraph_encoder_num_layers=2,
                 attr_decoder_num_layers=1,
                 struct_decoder_num_layers=1,
                 sparse_attention_weight=0.001,
                 theta=1.00,
                 eta=1.00):
        super(ADANET, self).__init__(contamination=contamination)

        # model param
        self.hid_dim = hid_dim
        self.num_layers=4
        self.weight_decay = weight_decay
        self.act = act
        self.alpha = alpha

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = validate_device(gpu)
        self.batch_size = batch_size
        self.num_neigh = num_neigh

        # other param
        self.verbose = verbose
        self.model = None

        self.max_batch_size=max_batch_size

        self.aggr=aggr
        self.loss_name=loss_name
        self.loss_weight=loss_weight
        self.T=T

        self.use_encoder_num=use_encoder_num
        self.attention_value=attention
        self.add_encoder_abs_loss=False

        self.attr_encoder_name=attr_encoder_name
        self.struct_encoder_name=struct_encoder_name
        self.topology_encoder_name=topology_encoder_name

        self.attr_decoder_name=attr_decoder_name
        self.struct_decoder_name=struct_decoder_name

        self.node_encoder_num_layers=node_encoder_num_layers
        self.edge_encoder_num_layers=edge_encoder_num_layers
        self.subgraph_encoder_num_layers=subgraph_encoder_num_layers

        # self.decoder_num_layers=decoder_num_layers
        self.attr_decoder_num_layers=attr_decoder_num_layers
        self.struct_decoder_num_layers=struct_decoder_num_layers

        self.dropout=dropout

        self.sparse_attention_weight=sparse_attention_weight

        self.eta=eta
        self.theta=theta
    def fit(self, G,y_true=None,pretrain_attr_encoder=None,pretrain_struct_encoder=None,pretrain_topology_encoder=None,attr_remask=None,struct_remask=None):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : torch_geometric.data.Data
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]


        if self.alpha is None:
            self.alpha = torch.std(G.s).detach() / \
                         (torch.std(G.x).detach() + torch.std(G.s).detach())

        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]
   
        self.num_node=self.batch_size

        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model = ADANET_Base(in_dim=G.x.shape[1],
                                   hid_dim=self.hid_dim,
                                   dropout=self.dropout,
                                   act=self.act,
                                   aggr=self.aggr,
                                   use_encoder_num=self.use_encoder_num,
                                   attention=self.attention_value,
                                   attr_remask=attr_remask,
                                   struct_remask=struct_remask,
                                   attr_encoder_name=self.attr_encoder_name,
                                   struct_encoder_name=self.struct_encoder_name,
                                   topology_encoder_name=self.topology_encoder_name,
                                   attr_decoder_name=self.attr_decoder_name,
                                   struct_decoder_name=self.struct_decoder_name,
                                   node_encoder_num_layers=self.node_encoder_num_layers,
                                   edge_encoder_num_layers=self.edge_encoder_num_layers,
                                   subgraph_encoder_num_layers=self.subgraph_encoder_num_layers,
                                   attr_decoder_num_layers=self.attr_decoder_num_layers,
                                   struct_decoder_num_layers=self.struct_decoder_num_layers,
                                   num_node=self.num_node).to(self.device)

        if self.use_encoder_num==1 and pretrain_attr_encoder!=None:
            self.model.attr_encoder.load_state_dict(pretrain_attr_encoder.state_dict())

            num_finetune_params = [p.numel() for p in self.model.parameters() if  p.requires_grad]
            print(f"num parameters first: {sum(num_finetune_params)}") 

            for k,v in self.model.named_parameters():
                if k.split('.')[0]=='attr_encoder':
                    v.requires_grad=False  

            num_finetune_params = [p.numel() for p in self.model.parameters() if  p.requires_grad]
            print(f"num parameters after: {sum(num_finetune_params)}") 
        elif self.use_encoder_num==2:
            self.model.attr_encoder.load_state_dict(pretrain_attr_encoder.state_dict())
            self.model.struct_encoder.load_state_dict(pretrain_struct_encoder.state_dict())

            for k,v in self.model.named_parameters():
                if k.split('.')[0]=='attr_encoder' or k.split('.')[0]=='struct_encoder':
                    v.requires_grad=False 

        elif self.use_encoder_num==3:
            self.model.attr_encoder.load_state_dict(pretrain_attr_encoder.state_dict())
            self.model.struct_encoder.load_state_dict(pretrain_struct_encoder.state_dict())
            self.model.topology_encoder.load_state_dict(pretrain_topology_encoder.state_dict())

            for k,v in self.model.named_parameters():
                if k.split('.')[0]=='attr_encoder' or k.split('.')[0]=='struct_encoder' or k.split('.')[0]=='topology_encoder':
                    v.requires_grad=False 

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        decision_scores = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.node_idx
                x, s, edge_index = self.process_graph(sampled_data)
                if self.attention_value >=2 and self.attention_value<=4:
                    x_, s_,soft_attention= self.model(x, edge_index)
                else:
                    x_, s_,_= self.model(x, edge_index)

                rank_score,score = self.loss_func(x[:batch_size],
                                       x_[:batch_size],
                                       s[:batch_size, node_idx],
                                       s_[:batch_size])
                decision_scores[node_idx[:batch_size]] = rank_score.detach() \
                    .cpu().numpy()
                loss = torch.mean(score)


                if self.attention_value >=2 and self.sparse_attention_weight>0 and self.use_encoder_num>1:
                    if self.use_encoder_num==2:
                        soft_attention_loss =self.entropy_sigmoid_loss(soft_attention)
                    elif self.use_encoder_num==3:
                        soft_attention_loss =self.entropy_softmax_loss(soft_attention)
                    loss+= soft_attention_loss*self.sparse_attention_weight

                epoch_loss += loss.item() * batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.verbose:
                print("Epoch {:04d}: Loss {:.4f}"
                      .format(epoch, epoch_loss / G.x.shape[0]), end='')
                if y_true is not None:
                    auc = eval_roc_auc(y_true, decision_scores)
                    print(" | AUC {:.4f}".format(auc), end='')
                print()

        self.decision_scores_ = decision_scores
        self._process_decision_scores()
        return self
        
    def decision_function(self, G):
        """
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]

        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = np.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            x, s, edge_index = self.process_graph(sampled_data)
            if self.attention_value >=2 and self.attention_value<=4:
                x_, s_,_ = self.model(x, edge_index)
            else:
                x_, s_,_= self.model(x, edge_index)
            rank_score,score = self.loss_func(x[:batch_size],
                                   x_[:batch_size],
                                   s[:batch_size, node_idx],
                                   s_[:batch_size])

            outlier_scores[node_idx[:batch_size]] = rank_score.detach() \
                .cpu().numpy()
        return outlier_scores

    def decision_struct_function(self, G):
        """
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]

        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_edge_scores = np.zeros((G.x.shape[0],G.x.shape[0]))
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx
            x, s, edge_index = self.process_graph(sampled_data)

            if self.attention_value >=2 and self.attention_value<=4:
                x_, s_,_= self.model(x, edge_index)
            else:
                x_, s_,_= self.model(x, edge_index)

            s_score = self.s_loss_func(s[:batch_size, node_idx],s_[:batch_size])

            s_score[G.s==0]=0
            outlier_edge_scores[:batch_size, node_idx] = s_score.detach() \
                .cpu().numpy()
        return outlier_edge_scores

    def process_graph(self, G):
        """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        s : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
        """
        s = G.s.to(self.device)
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)

        return x, s, edge_index

    def s_loss_func(self,s,s_):
        diff_structure=torch.abs(s-s_)
        return diff_structure

    def single_sce_loss(self,x,y,alpha=3):        
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss

    def sce_loss(self,x,x_,s,s_,alpha=3):
        diff_attribute = self.single_sce_loss(x,x_,alpha)
        diff_structure = self.single_sce_loss(s,s_, alpha)

        score = self.alpha * diff_attribute + (1 - self.alpha) * diff_structure
        return score


    def loss_func(self, x, x_, s, s_):

        if self.loss_name=='rec':
            score=self.rec_loss(x,x_,s,s_)
            return score,score

        elif self.loss_name=='add_log_t_entropy':
            score=self.rec_loss(x,x_,s,s_)
            entropy_loss=self.log_t_entropy_loss(x,x_,s,s_,score)

            rank_score=score+self.loss_weight*entropy_loss
            return rank_score,rank_score
        else:
            assert(False,'wrong loss func')

    def entropy_loss(self,x,x_,s,s_,score):
        diag_s=torch.eye(s.size()[0]).to(s.device)+s
        all_score=score.repeat(score.size()[0],1)

        all_score=torch.where(diag_s>0.1,all_score,0)+1e-6

        all_score=all_score/torch.sum(all_score,1)
        all_log_score=-torch.log(all_score)
        all_log_score=torch.sum(all_log_score,1)

        return all_log_score

    def log_t_entropy_loss(self,x,x_,s,s_,score):

        diag_s=torch.eye(s.size()[0]).to(s.device)+s
        all_score=score.repeat(score.size()[0],1).float()

        all_score=torch.where(diag_s.float()>0.1,all_score,torch.tensor(0.0, dtype=torch.float).to(s.device))+1e-6
        log_all_score=torch.log(all_score)/self.T

        all_score=F.softmax(log_all_score,dim =1)

        all_log_score=-torch.log(all_score)*all_score
        all_log_score=torch.sum(all_log_score,1)
        
        return all_log_score

    def rec_loss(self, x, x_, s, s_):
        reversed_adj = 1 - s
        thetas = torch.where(reversed_adj.float() > 0, reversed_adj.float(),
                             torch.full(s.shape, self.theta).float().to(self.device))
        reversed_attr = 1 - x
        etas = torch.where(reversed_attr == 1.0, reversed_attr.float(),
                           torch.full(x.shape, self.eta).float().to(self.device))
        diff_attribute = torch.pow(x_ - x, 2) * etas
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        diff_structure = torch.pow(s_ - s, 2) * thetas
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors \
                + (1 - self.alpha) * structure_errors
        return score

class ADANET_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 dropout,
                 act,
                 aggr,
                 use_encoder_num=1,
                 attention=None,
                 attr_remask=None,
                 struct_remask=None,
                 attr_encoder_name='gcn',
                 struct_encoder_name='gcn',
                 topology_encoder_name='gcn',
                 attr_decoder_name='mlp',
                 struct_decoder_name='mlp',
                 node_encoder_num_layers=2,
                 edge_encoder_num_layers=2,
                 subgraph_encoder_num_layers=2,
                 attr_decoder_num_layers=1,
                 struct_decoder_num_layers=1,
                 num_node=1
                 ):
        super(ADANET_Base, self).__init__()

        attr_decoder_layers = attr_decoder_num_layers
        struct_decoder_layers = struct_decoder_num_layers
        self.attr_remask=attr_remask
        self.struct_remask=struct_remask
        self.attention_value=attention
        if self.attr_remask is not None:
            self.attr_remask=self.attr_remask.unsqueeze(1)
        
        if self.struct_remask is not None:
            self.struct_remask=self.struct_remask.unsqueeze(1)

        self.use_encoder_num=use_encoder_num

        self.attr_encoder_name=attr_encoder_name
        self.struct_encoder_name=struct_encoder_name
        self.topology_encoder_name=topology_encoder_name

        self.attr_decoder_name=attr_decoder_name
        self.struct_decoder_name=struct_decoder_name

        self.num_node=num_node

        self.aggr=aggr

        if self.use_encoder_num>=1:
            self.attr_encoder = setup_module(
                m_type=attr_encoder_name,
                enc_dec="encoding",
                in_dim=in_dim,
                num_hidden=hid_dim,
                out_dim=hid_dim,
                num_layers= node_encoder_num_layers,
                nhead=1,
                nhead_out=1,
                concat_out=False,
                activation=act,
                dropout=0,
                attn_drop=0,
                negative_slope=0.2,
                residual=False,
                norm=None,
                aggr=aggr,
            )

        if self.use_encoder_num>=2:
            self.struct_encoder = setup_module(
                m_type=struct_encoder_name,
                enc_dec="encoding",
                in_dim=in_dim,
                num_hidden=hid_dim,
                out_dim=hid_dim,
                num_layers= edge_encoder_num_layers,
                nhead=1,
                nhead_out=1,
                concat_out=False,
                activation=act,
                dropout=0,
                attn_drop=0,
                negative_slope=0.2,
                residual=False,
                norm=None,
                aggr=aggr,
            )
        if self.use_encoder_num>=3:
            self.topology_encoder = setup_module(
                m_type=topology_encoder_name,
                enc_dec="encoding",
                in_dim=in_dim,
                num_hidden=hid_dim,
                out_dim=hid_dim,
                num_layers= subgraph_encoder_num_layers,
                nhead=1,
                nhead_out=1,
                concat_out=False,
                activation=act,
                dropout=0,
                attn_drop=0,
                negative_slope=0.2,
                residual=False,
                norm=None,
                aggr=aggr,
            )
        if self.use_encoder_num>=2:
            if self.attention_value <0:
                self.attention = torch.nn.Parameter(torch.FloatTensor([-attention]), requires_grad=True)
            elif self.attention_value>=0 and self.attention_value<=1:
                self.attention = torch.nn.Parameter(torch.FloatTensor([attention]), requires_grad=False)
            elif self.attention_value==2:
                if self.use_encoder_num==2:
                    self.attention_layer = torch.nn.Sequential(torch.nn.Linear(hid_dim*2, hid_dim),
                    torch.nn.Sigmoid())
                else:
                    self.attention_layer1 =torch.nn.Linear(hid_dim*3, hid_dim*3)
                    self.attention_layer2=torch.nn.Softmax(dim=2)
            else:
                assert(f"use wrong attention {attention}")
        self.use_attrae=False

        if self.use_attrae:
            decoder_in_dim=hid_dim+in_dim
        else: 
            decoder_in_dim=hid_dim

        self.attr_decoder = setup_module(
                m_type=attr_decoder_name,
                enc_dec="decoding",
                in_dim=decoder_in_dim,
                num_hidden=hid_dim,
                out_dim=in_dim,
                num_layers=attr_decoder_num_layers,
                nhead=1,
                nhead_out=1,
                concat_out=False,
                activation=act,
                dropout=dropout,
                attn_drop=0,
                negative_slope=0.2,
                residual=False,
                norm=None,
                aggr=aggr,
            )

        self.struct_decoder = setup_module(
                m_type=struct_decoder_name,
                enc_dec="decoding",
                in_dim=hid_dim,
                num_hidden=hid_dim,
                out_dim=in_dim,
                num_layers=struct_decoder_num_layers,
                nhead=1,
                nhead_out=1,
                concat_out=False,
                activation=act,
                dropout=dropout,
                attn_drop=0,
                negative_slope=0.2,
                residual=False,
                norm=None,
                aggr=aggr,
            )

        self.attributAE=AttributeAE(in_dim=self.num_node,
                                    embed_dim=hid_dim,
                                    out_dim=hid_dim,
                                    dropout=dropout,
                                    act=F.relu)
  
    def forward(self, x, edge_index):
        if  self.use_encoder_num==1:   
            h = self.attr_encoder(x, edge_index)
        elif self.use_encoder_num==2:
            h_attr=self.attr_encoder(x, edge_index)
            h_struct=self.struct_encoder(x,edge_index)

            if self.attention_value <=1:
                h=h_attr*self.attention+h_struct*(1-self.attention)
            elif self.attention_value==2:
                self.attention=self.attention_layer(torch.cat([h_attr,h_struct],dim=1))
                h=h_attr*self.attention+h_struct*(1-self.attention)
        elif self.use_encoder_num==3:
            h_attr=self.attr_encoder(x, edge_index)
            h_struct=self.struct_encoder(x,edge_index)
            h_topology=self.topology_encoder(x,edge_index)

            if self.attention_value <=1:
                h=h_attr*self.attention+h_struct*((1-self.attention)/2)+h_topology*((1-self.attention)/2)
            elif self.attention_value==2:
                self.attention=self.attention_layer1(torch.cat([h_attr,h_struct,h_topology],dim=1))

                self.attention=self.attention_layer2(torch.reshape(self.attention,(-1,h_attr.size()[-1],3)))

                h=h_attr*self.attention[:,:,0]+h_struct*self.attention[:,:,1]+h_topology*self.attention[:,:,2]
        h=h.to(torch.float32)

        if self.use_attrae:
            x_1 = self.attributAE(x, h)
            x_ = self.attr_decoder(torch.cat([x_1,h],1), edge_index)
        else: 
            if self.attr_decoder_name=='mlp' or self.attr_decoder_name=='linear':
                x_ = self.attr_decoder(h)
            else:
                x_ = self.attr_decoder(h, edge_index)
        if self.struct_decoder_name=='mlp' or self.struct_decoder_name=='linear':
            h_ = self.struct_decoder(h)
        else:
            h_ = self.struct_decoder(h, edge_index)
        s_ = h_ @ h_.T

        if self.attention_value>=2 and self.use_encoder_num>1:
            return x_, s_,self.attention
        else:
            return x_, s_,1

class AttributeAE(nn.Module):
    """
    Attribute Autoencoder in AnomalyDAE model: the encoder
    employs two non-linear feature transform to the node attribute
    x. The decoder takes both the node embeddings from the structure
    autoencoder and the reduced attribute representation to
    reconstruct the original node attribute.

    Parameters
    ----------
    in_dim:  int
        dimension of the input number of nodes
    embed_dim: int
        the latent representation dimension of node
        (after the first linear layer)
    out_dim:  int
        the output dim after two linear layers
    dropout: float
        dropout probability for the linear layer
    act: F, optional
         Choice of activation function

    Returns
    -------
    x : torch.Tensor
        Reconstructed attribute (feature) of nodes.
    """

    def __init__(self,
                 in_dim,
                 embed_dim,
                 out_dim,
                 dropout,
                 act):
        super(AttributeAE, self).__init__()
        self.dense1 = nn.Linear(in_dim, embed_dim)
        self.dense2 = nn.Linear(embed_dim, out_dim)
        self.dropout = dropout
        self.act = act

    def forward(self, x, h):
        x = self.act(self.dense1(x.T))
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)
        x = F.dropout(x, self.dropout)
        # decoder
        x = h @ x.T
        return x
