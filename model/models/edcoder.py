from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn

from .gat import GAT
from .gin import GIN
from model.utils import create_norm
from torch_geometric.utils import dropout_edge,dropout_path
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.utils import to_dense_adj,dense_to_sparse

import sys
from pygod.models import GCN

from model.models.drop_graph import dropout_subgraph
from E_high import compute_E_high,compute_G_ano

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
            activation=activation,
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
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "mlp":
        # * just for decoder 
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
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

    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    
    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            encoder_num_layers: int,
            attr_decoder_num_layers: int,
            struct_decoder_num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            attr_decoder_type: str = "gat",
            struct_decoder_type:str='gat',
            loss_fn: str = "sce",
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            drop_edge_rate: float = 0.0,
            drop_path_rate: float=0.0,
            predict_all_node: bool=False,
            predict_all_edge: float=0,
            drop_path_length:int=3,
            walks_per_node:int=3,
            select_gano_num:int=0,
            sano_weight:float=0.0
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._attr_decoder_type = attr_decoder_type
        self._struct_decoder_type=struct_decoder_type
        self._output_hidden_size = num_hidden
        self._concat_hidden = concat_hidden
        
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate

        self._drop_edge_rate = drop_edge_rate
        self._drop_path_rate = drop_path_rate
        self.predict_all_node=predict_all_node
        self.predict_all_edge=predict_all_edge
        self.drop_path_length=drop_path_length

        self.walks_per_node=walks_per_node
        self.neg_s=None

        self.select_gano_num=select_gano_num
        self.sano_weight=sano_weight

        if self.sano_weight==-1:
            self.sano_weight=1
            self.aano_weight=0
        else:
            self.aano_weight=1

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        attr_dec_num_hidden = num_hidden // nhead_out if attr_decoder_type in ("gat", "dotgat") else num_hidden 
        struct_dec_num_hidden = num_hidden // nhead_out if struct_decoder_type in ("gat", "dotgat") else num_hidden 


        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=encoder_num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        # build decoder for attribute prediction
        self.attr_decoder = setup_module(
            m_type=attr_decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=attr_dec_num_hidden,
            out_dim=in_dim,
            num_layers=attr_decoder_num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=False,
        )

        self.struct_decoder = setup_module(
            m_type=struct_decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=struct_dec_num_hidden,
            out_dim=in_dim,
            num_layers=struct_decoder_num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=False,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        keep_nodes = perm[num_mask_nodes: ]

        # print('_replace_rate',self._replace_rate)
        if self._replace_rate > 0 and int(self._replace_rate * num_mask_nodes)>0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def forward(self, x, edge_index):
        # ---- attribute and edge reconstruction ----
        loss,final_edge_mask_rate = self.mask_attr_prediction(x, edge_index)
        loss_item = {"loss": loss.item()}
        return loss, loss_item,final_edge_mask_rate

    def intersection_edge(self,edge_index_1, edge_index_2,max_num_nodes): 
        s1=to_dense_adj(edge_index_1,max_num_nodes=max_num_nodes)[0]
        s2=to_dense_adj(edge_index_2,max_num_nodes=max_num_nodes)[0]
        intersection_s=torch.min(s1,s2)

        intersection_edge_index,_=dense_to_sparse(intersection_s)
        # print('intersection_edge_index',intersection_edge_index)

        return intersection_edge_index,intersection_s,


    def mask_attr_prediction(self, x, edge_index):
        num_nodes=x.size()[0]

        # mask edge to reduce struct uncertainty
        _mask_rate=self._mask_rate
        dence_edge_index=to_dense_adj(edge_index,max_num_nodes=num_nodes)[0]
        # use_e_high=False

        if self.select_gano_num:

            G_ano_init=float('inf')
            for j in range(self.select_gano_num):
                use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, _mask_rate)
                a_ano,s_ano=compute_G_ano(dence_edge_index,use_x)
                G_ano=a_ano*self.aano_weight+s_ano*self.sano_weight
              #print('G_ano',G_ano)    
                if G_ano<G_ano_init:
                    use_x_select=use_x
                    mask_nodes_select=mask_nodes
                    keep_nodes_select=keep_nodes
                    G_ano_init=G_ano
          #print('final G_ano',G_ano_init)
            use_x=use_x_select
            mask_nodes=mask_nodes_select
            keep_nodes=keep_nodes_select
        else:
            use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(x, _mask_rate)

        use_x=use_x.to(torch.float32)
        # print('use_x',use_x.size())
        _drop_path_rate=self._drop_path_rate
        _drop_edge_rate=self._drop_edge_rate

        # mask edge for struct reconstruction
        if _drop_edge_rate > 0:
            # use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)
            
            if self.select_gano_num:
                G_ano_init=float('inf')
                for j in range(self.select_gano_num):
                    use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)
                    # to_dense_adj(edge_index)[0]
                    a_ano,s_ano=compute_G_ano(to_dense_adj(use_mask_edge_edge_index,max_num_nodes=num_nodes)[0],use_x)
                    G_ano=a_ano*self.aano_weight+s_ano*self.sano_weight
                  #print('G_ano',G_ano)    
                    if G_ano<G_ano_init:
                        use_mask_edge_edge_index_select=use_mask_edge_edge_index
                        masked_edge_edges_select=masked_edge_edges
                        G_ano_init=G_ano
              #print('final G_ano',G_ano_init)
                use_mask_edge_edge_index=use_mask_edge_edge_index_select
                masked_edge_edges=masked_edge_edges_select
            else:
                use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)

            use_mask_edge_edge_index = add_self_loops(use_mask_edge_edge_index)[0]
        else:
            use_mask_edge_edge_index = edge_index

        # mask path for struct reconstruction
        if _drop_path_rate > 0:
            if self.select_gano_num:
                G_ano_init=float('inf')
                for j in range(self.select_gano_num):
                    use_mask_path_edge_index, masked_path_edges,_= dropout_subgraph(edge_index, p=_drop_path_rate,walk_length=self.drop_path_length,walks_per_node=self.walks_per_node,return_subgraph=False)
                  #print('to_dense_adj(use_mask_path_edge_index)[0],use_x',to_dense_adj(use_mask_path_edge_index,max_num_nodes=num_nodes)[0].size(),use_x.size())
                    a_ano,s_ano=compute_G_ano(to_dense_adj(use_mask_path_edge_index,max_num_nodes=num_nodes)[0],use_x)
                    G_ano=a_ano*self.aano_weight+s_ano*self.sano_weight
                  #print('G_ano',G_ano)    
                    if G_ano<G_ano_init:
                        use_mask_path_edge_index_select=use_mask_path_edge_index
                        masked_path_edges_select=masked_path_edges
                        G_ano_init=G_ano
              #print('final G_ano',G_ano_init)
                use_mask_path_edge_index=use_mask_path_edge_index_select
                masked_path_edges=masked_path_edges_select
            else:
                use_mask_path_edge_index, masked_path_edges,_= dropout_subgraph(edge_index, p=_drop_path_rate,walk_length=self.drop_path_length,walks_per_node=self.walks_per_node,return_subgraph=False)
            
            use_mask_path_edge_index = add_self_loops(use_mask_path_edge_index)[0]
        else:
            use_mask_path_edge_index = edge_index
            
        # mask edge and path
        use_edge_index,use_s=self.intersection_edge(use_mask_edge_edge_index,use_mask_path_edge_index,num_nodes)

        self.return_hidden=False

        if self.return_hidden==False:
            enc_rep= self.encoder(use_x, use_edge_index, return_hidden=self.return_hidden)
        else:
            enc_rep, all_hidden = self.encoder(use_x, use_edge_index, return_hidden=self.return_hidden)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute and edge reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        loss=0
        final_mask_rate=0
        # ---- attribute reconstruction ----
        if _mask_rate>0:
            # if self._attr_decoder_type not in ("mlp", "linear"):
            #     # * remask, re-mask
            #     rep[mask_nodes] = 0
            if self._attr_decoder_type in ("mlp", "linear") :
                attr_recon = self.attr_decoder(rep)
            else:
                attr_recon = self.attr_decoder(rep, use_edge_index)

            use_mask_loss=False
            if use_mask_loss:
                x_init = x[mask_nodes]
                x_rec = attr_recon[mask_nodes]
            else:
                x_init = x
                x_rec = attr_recon

            loss += self.criterion(x_rec, x_init)
        # ---- edge reconstruction ----
        if _drop_edge_rate>0 or _drop_path_rate>0 :
            if self._struct_decoder_type in ("mlp", "linear") :
                h_recon = self.struct_decoder(rep)
            else:
                h_recon = self.struct_decoder(rep, use_edge_index)      
        
            struct_recon = h_recon @ h_recon.T

            s_init = to_dense_adj(edge_index,max_num_nodes=num_nodes)[0]

            mask_edge_num=(use_s==0) & (s_init==1)
            final_mask_rate=mask_edge_num.sum()/edge_index.size()[1]

            if self.predict_all_edge:
                if  self.neg_s==None:
                    neg_rate=edge_index.size()[1]/(s_init.size()[0]**2)*self.predict_all_edge
                    self.neg_s=torch.rand(s_init.size()) <neg_rate
                   
                s_rec = torch.where((((use_s==0) & (s_init==1))|(self.neg_s).to(use_s.device)),struct_recon,s_init)
            else:
                s_rec = torch.where((use_s==0) & (s_init==1),struct_recon,s_init)

            loss += self.criterion(s_rec, s_init)

        
        return loss,final_mask_rate

    def embed(self, x, edge_index):
        rep = self.encoder(x, edge_index)
        return rep

    @property
    def enc_params(self):
        return self.encoder.parameters()
    
    @property
    def dec_params(self):
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
