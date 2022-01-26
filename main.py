#!/usr/bin/env python3
from tuning import run_tuning
from train import run_model, get_param_config

if __name__ == "__main__":
  if args.run_tuning:
    raise NotImplementedError()
    run_tuning()
  else:
    param_config                  = get_param_config(NUM_EPOCH, BATCH_SIZE)
    run_model(param_config,
              args                =args,
              # logger
              logger              = logger,
              logger_2            = logger_2,
              # data
              full_data           = full_data,
              # code params
              node_features       = node_features,
              edge_features       = edge_features,
              mean_time_shift_src = mean_time_shift_src,
              std_time_shift_src  = std_time_shift_src,
              mean_time_shift_dst = mean_time_shift_dst,
              std_time_shift_dst  = std_time_shift_dst,
              # config params
              device              = device,
              MODEL_SAVE_PATH     = MODEL_SAVE_PATH)
