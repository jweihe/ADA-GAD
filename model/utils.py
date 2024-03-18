import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch import optim as optim
from tensorboardX import SummaryWriter



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--dataset", type=str, default="inj_cora")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=40,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")

    parser.add_argument("--node_encoder_num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--edge_encoder_num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--subgraph_encoder_num_layers", type=int, default=2,
                        help="number of hidden layers")    
    parser.add_argument("--attr_decoder_num_layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--struct_decoder_num_layers", type=int, default=1,
help="number of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=256,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu for GAT")
    parser.add_argument("--activation", type=str, default="prelu")

    parser.add_argument("--replace_rate", type=float, default=0.0)

    parser.add_argument("--attr_encoder", type=str, default="gat")
    parser.add_argument("--struct_encoder", type=str, default="gcn")
    parser.add_argument("--topology_encoder", type=str, default="gcn")

    parser.add_argument("--attr_decoder", type=str, default="gcn")
    parser.add_argument("--struct_decoder", type=str, default="gcn")
    parser.add_argument("--loss_fn", type=str, default="mse")
    parser.add_argument("--alpha_l", type=float, default=3, help="`pow`coefficient for `sce` loss")
    parser.add_argument("--optimizer", type=str, default="adam")
    
    #for GAD finetune    
    parser.add_argument("--model_name", type=str, default="ADANET")
    parser.add_argument("--aggr_f", type=str, default='add')
    parser.add_argument("--max_epoch_f", type=int, default=30)
    parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    parser.add_argument("--alpha_f", type=float, default=0.5)
    parser.add_argument("--dropout_f", type=float, default=0.0)
    parser.add_argument("--loss_f", type=str, default='rec')
    parser.add_argument("--loss_weight_f", type=float, default=-0.0001)
    parser.add_argument("--T_f", type=float, default=1.0)

    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--save_model", action="store_true")

    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--use_nni", action="store_true")
    parser.add_argument("--logging", action="store_true")

    parser.add_argument("--scheduler", type=int, default=1)
    parser.add_argument("--concat_hidden", action="store_true", default=False)

    # for graph classification
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    parser.add_argument("--batch_size", type=int, default=32)

    # for graph anomaly detection
    parser.add_argument("--use_ssl", type=int, default=1)
    parser.add_argument("--use_encoder_num", type=int, default=3)

    parser.add_argument("--attention", type=int, default=2,help="-1~0 learning attention, 0~1 unlearning attention, \
        2 hard attention, 3 soft attention")


    # for attr prediction
    parser.add_argument("--mask_rate1", type=float, default=0.1)
    
    parser.add_argument("--drop_edge_rate1", type=float, default=0)
    parser.add_argument("--predict_all_node1", type=bool, default=False)
    parser.add_argument("--predict_all_edge1", type=float, default=0)

    parser.add_argument("--drop_path_rate1", type=float, default=0)
    parser.add_argument("--drop_path_length1", type=int, default=0)
    parser.add_argument("--walks_per_node1", type=int, default=1)

    # for edge prediction
    parser.add_argument("--mask_rate2", type=float, default=0)

    parser.add_argument("--drop_edge_rate2", type=float, default=0.1)
    parser.add_argument("--predict_all_node2", type=bool, default=False)
    parser.add_argument("--predict_all_edge2", type=float, default=0)

    parser.add_argument("--drop_path_rate2", type=float, default=0)
    parser.add_argument("--drop_path_length2", type=int, default=0)
    parser.add_argument("--walks_per_node2", type=int, default=1)

    # for subgraph prediction
    parser.add_argument("--mask_rate3", type=float, default=0)
    
    parser.add_argument("--drop_edge_rate3", type=float, default=0)
    parser.add_argument("--predict_all_node3", type=bool, default=False)
    parser.add_argument("--predict_all_edge3", type=float, default=0)

    parser.add_argument("--drop_path_rate3", type=float, default=0.1)
    parser.add_argument("--drop_path_length3", type=int, default=3)
    parser.add_argument("--walks_per_node3", type=int, default=3)

    parser.add_argument("--sparse_attention_weight", type=float, default=0)

    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)

    parser.add_argument("--all_encoder_layers", type=int, default=0)

    parser.add_argument("--max_pu_epoch", type=int, default=45)
    parser.add_argument("--each_pu_epoch", type=int, default=15)

    parser.add_argument("--select_gano_num", type=int, default=30,help="select smallest G_ano")
    parser.add_argument("--sano_weight", type=int, default=1)

    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


# ------ logging ------

class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias
