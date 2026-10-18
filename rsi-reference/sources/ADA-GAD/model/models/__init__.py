from .edcoder import PreModel
from .edcoder_cotrain import PreModelCotrain

def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden

    node_encoder_num_layers=args.node_encoder_num_layers
    edge_encoder_num_layers=args.edge_encoder_num_layers
    subgraph_encoder_num_layers=args.subgraph_encoder_num_layers

    attr_decoder_num_layers=args.attr_decoder_num_layers
    struct_decoder_num_layers=args.struct_decoder_num_layers

    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    
    attr_encoder_type = args.attr_encoder
    struct_encoder_type = args.struct_encoder
    topology_encoder_type=args.topology_encoder

    attr_decoder_type = args.attr_decoder
    struct_decoder_type=args.struct_decoder

    replace_rate = args.replace_rate


    activation = args.activation
    loss_fn = args.loss_fn
    alpha_l = args.alpha_l
    concat_hidden = args.concat_hidden
    num_features = args.num_features

    mask_rate1 = args.mask_rate1
    drop_edge_rate1=args.drop_edge_rate1
    drop_path_rate1=args.drop_path_rate1
    predict_all_node1=args.predict_all_node1
    predict_all_edge1=args.predict_all_edge1
    drop_path_length1=args.drop_path_length1
    walks_per_node1=args.walks_per_node1

    mask_rate2 = args.mask_rate2
    drop_edge_rate2=args.drop_edge_rate2
    drop_path_rate2=args.drop_path_rate2
    predict_all_node2=args.predict_all_node2
    predict_all_edge2=args.predict_all_edge2
    drop_path_length2=args.drop_path_length2
    walks_per_node2=args.walks_per_node2

    mask_rate3 = args.mask_rate3
    drop_edge_rate3=args.drop_edge_rate3
    drop_path_rate3=args.drop_path_rate3
    predict_all_node3=args.predict_all_node3
    predict_all_edge3=args.predict_all_edge3
    drop_path_length3=args.drop_path_length3
    walks_per_node3=args.walks_per_node3

    select_gano_num=args.select_gano_num
    sano_weight=args.sano_weight

    attr_model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        encoder_num_layers=node_encoder_num_layers,
        attr_decoder_num_layers=attr_decoder_num_layers,
        struct_decoder_num_layers=struct_decoder_num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=attr_encoder_type,
        attr_decoder_type=attr_decoder_type,
        struct_decoder_type=struct_decoder_type,
        mask_rate=mask_rate1,
        norm=norm,
        loss_fn=loss_fn,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        drop_edge_rate=drop_edge_rate1,
        drop_path_rate=drop_path_rate1,
        predict_all_node=predict_all_node1,
        predict_all_edge=predict_all_edge1,
        drop_path_length=drop_path_length1,
        walks_per_node=walks_per_node1,
        select_gano_num=select_gano_num,
        sano_weight=sano_weight
    )

    struct_model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        encoder_num_layers=edge_encoder_num_layers,
        attr_decoder_num_layers=attr_decoder_num_layers,
        struct_decoder_num_layers=struct_decoder_num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=struct_encoder_type,
        attr_decoder_type=attr_decoder_type,
        struct_decoder_type=struct_decoder_type,
        mask_rate=mask_rate2,
        norm=norm,
        loss_fn=loss_fn,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        drop_edge_rate=drop_edge_rate2,
        drop_path_rate=drop_path_rate2,
        predict_all_node=predict_all_node2,
        predict_all_edge=predict_all_edge2,
        drop_path_length=drop_path_length2,
        walks_per_node=walks_per_node2,
        select_gano_num=select_gano_num,
        sano_weight=sano_weight
    )

    topology_model = PreModel(
        in_dim=int(num_features),
        num_hidden=int(num_hidden),
        encoder_num_layers=subgraph_encoder_num_layers,
        attr_decoder_num_layers=attr_decoder_num_layers,
        struct_decoder_num_layers=struct_decoder_num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=topology_encoder_type,
        attr_decoder_type=attr_decoder_type,
        struct_decoder_type=struct_decoder_type,
        mask_rate=mask_rate3,
        norm=norm,
        loss_fn=loss_fn,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=concat_hidden,
        drop_edge_rate=drop_edge_rate3,
        drop_path_rate=drop_path_rate3,
        predict_all_node=predict_all_node3,
        predict_all_edge=predict_all_edge3,
        drop_path_length=drop_path_length3,
        walks_per_node=walks_per_node3,
        select_gano_num=select_gano_num,
        sano_weight=sano_weight
    )
    
    return attr_model,struct_model,topology_model
