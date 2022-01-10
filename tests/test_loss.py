#!/usr/bin/env python3

import utils.loss
from utils.loss import select_decoder_and_loss
from train_self_supervised import args, device
from utils.utils import MLP, MLP_multiple_class
import torch

def test_select_decoder_and_loss_when_uniq_labels_are_two(mocker):
    # test 1: when n_unique_labels == 2
    feat_dim = 10 # value of feat_dim is randomly chosen.
    n_unique_labels = 2
    decoder_optimizer, decoder, decoder_loss_criterion = select_decoder_and_loss(args,device,feat_dim, n_unique_labels)

    assert isinstance(decoder_optimizer, torch.optim.Adam)
    assert isinstance(decoder, MLP)
    assert isinstance(decoder_loss_criterion(), torch.nn.BCELoss)


def test_select_decoder_and_loss_when_uniq_labels_are_four(mocker):
    # test 1: when n_unique_labels == 4
    feat_dim = 10 # value of feat_dim is randomly chosen.
    n_unique_labels = 4
    decoder_optimizer, decoder, decoder_loss_criterion = select_decoder_and_loss(args,device,feat_dim, n_unique_labels)

    assert isinstance(decoder_optimizer, torch.optim.Adam)
    assert isinstance(decoder, MLP_multiple_class)
    assert isinstance(decoder_loss_criterion(), torch.nn.CrossEntropyLoss)

def test_time_based_similarity(mocker):
    pass
