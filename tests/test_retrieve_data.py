#!/usr/bin/env python3

import utils.loss
from scripts.retrieve_data_from_link_prediction import LinkPredictionCrawler, ValueCollection, return_list_of_test_data_on_the_same_period, get_list_of_window_idx_with_same_period, get_data_size


def test_fair_evaluation_for_model_with_varied_batch_size():

    ## test 1: test with log of model with different batch_size
    vc = ValueCollection()
    # log_file = '1640232708.151559'
    log_file = '1640243549.1885495' # original; use_weight = False ; batch size = 200
    c2 = LinkPredictionCrawler(log_file)
    # c2.plot()
    vc.append_value(c2)

    log_file = '1642416734.1726859' # original + epoch 5 + batch size 1000
    c16 = LinkPredictionCrawler(log_file)
    vc.append_value(c16)

    log_file = '1642416474.9904623' # original + epoch 5 + batch size 2000
    c17 = LinkPredictionCrawler(log_file)
    # c17.plot()
    vc.append_value(c17)


    header_dicts = vc.header_dicts
    data_size = 10000

    assert list_of_list_of_test_instances_end_idx[-1] == data_size

    list_of_list_of_test_instances_end_idx = return_list_of_test_data_on_the_same_period(header_dicts)
    list_of_window_begin_idxs_with_the_same_period, list_of_window_end_idxs_with_the_same_period  = get_list_of_window_idx_with_same_period(vc.header_dicts, list_of_list_of_test_instances_end_idx)


    for i in list_of_window_begin_idxs_with_the_same_period:
        assert i[0] > 0

    for idx, i in enumerate(header_dicts):
        assert 10000 >= i["batch_size"] * list_of_window_end_idxs_with_the_same_period[idx][-1]

    ## test 2: test with log of model with same batch_size
    vc = ValueCollection()

    log_file = '1642763194.1611047' # inverse_ef-iwf * 1 + epoch 5 + batch size 1000
    c23 = LinkPredictionCrawler(log_file)
    # c17.plot()
    vc.append_value(c23)

    log_file = '1642763750.673035' # inverse_ef-iwf * 50 + epoch 5 + batch size 1000
    c24 = LinkPredictionCrawler(log_file)
    # c17.plot()
    vc.append_value(c24)

    log_file = '1642764868.0464556' # inverse_ef-iwf * 500 + epoch 5 + batch size 1000
    c25 = LinkPredictionCrawler(log_file)
    # c17.plot()
    vc.append_value(c25)

    log_file = '1642766439.4182317' # inverse_ef-iwf * 0.1 + epoch 5 + batch size 1000
    c26 = LinkPredictionCrawler(log_file)
    # c17.plot()
    vc.append_value(c26)

    header_dicts = vc.header_dicts
    data_size = 10000

    list_of_list_of_test_instances_end_idx = return_list_of_test_data_on_the_same_period(header_dicts)
    list_of_window_begin_idxs_with_the_same_period, list_of_window_end_idxs_with_the_same_period  = get_list_of_window_idx_with_same_period(vc.header_dicts, list_of_list_of_test_instances_end_idx)


    assert list_of_list_of_test_instances_end_idx[-1] == data_size

    for i in list_of_window_begin_idxs_with_the_same_period:
        assert i[0] > 0

    for idx, i in enumerate(header_dicts):
        assert data_size >= i["batch_size"] * list_of_window_end_idxs_with_the_same_period[idx][-1]
