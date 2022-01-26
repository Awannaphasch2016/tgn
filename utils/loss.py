#!/usr/bin/env python3

import torch
from utils.utils import get_neighbor_finder, MLP, MLP_multiple_class

def select_decoder_and_loss(args,device,feat_dim, n_unique_labels):
    raise NotImplementedError()
    ## use with pre-training model to substitute prediction head
    if n_unique_labels == 2:
        decoder = MLP(feat_dim, drop=args.drop_out)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        decoder = decoder.to(device)
        # decoder_loss_criterion = torch.nn.BCELoss()
        decoder_loss_criterion = torch.nn.BCELoss
    else:
        decoder = MLP_multiple_class(feat_dim, n_unique_labels ,drop=args.drop_out)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
        decoder = decoder.to(device)
        # decoder_loss_criterion = torch.nn.CrossEntropyLoss()
        decoder_loss_criterion = torch.nn.CrossEntropyLoss
    return decoder_optimizer, decoder, decoder_loss_criterion


def time_based_similarity(edge, pred_edge, edge_t, pred_edge_t):
    """"
    calculate one edges at at time
    """
    from torch import nn
    edges = None
    predicted_edges = None
    n_node_features = 5 # randomly chosen.

    time_encoder = TimeEncoder(dimension=n_node_features)

    time_diff = 5 # randomly chosen
    time_diff_encoding = time_encoder(time_diff)
    assert time_diff_encoding.shape[0] == time_diff_encoding.reshape(-1).shape[0]

    time_param = np.array(range(time_diff_encoding.shape[0]))

    return nn.ReLU(time_param * time_diff_encoding) * edge * predicted_edges
