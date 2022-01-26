#!/usr/bin/env python3

from utils.utils import convert_ind_to_n_instances

def get_all_ensemble_training_data_inds(begin_idx_of_data, end_idx_of_data, window_size, offset_begin_ind = 0, fix_begin_ind=False):
    # offset_begin_ind = apply_off_set_ind([offset_ind], off_set_ind)[0]
    # n_instances = end_train_idx + 1
    n_instances = convert_ind_to_n_instances(end_idx_of_data)

    assert n_instances/window_size == int(n_instances/window_size)
    n_ensembles = int(n_instances/window_size)
    begin_inds = []
    end_inds = []
    for i in range(n_ensembles):
        added_length = ((i + 1) * window_size) - 1
        # added_length = ((i + 1) * window_size)
        if fix_begin_ind:
            begin_inds.append(begin_idx_of_data)
            end_ind = begin_idx_of_data + added_length
            end_inds.append(end_ind)
            assert end_ind <= end_idx_of_data
        else:
            end_inds.append(end_idx_of_data)
            begin_ind = end_idx_of_data - added_length
            # if begin_ind < 0:
            #     begin_ind = 0
            begin_inds.append(begin_ind)
            assert begin_ind >= offset_begin_ind

    assert begin_inds[-1] == 0
    return begin_inds, end_inds
