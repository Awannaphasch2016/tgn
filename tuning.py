#!/usr/bin/env python3

from pathlib import Path

def print_hi(config, tmp, checkpoint_dir=None):
    print(config)

def run_tuning():

  data_dir = Path.cwd()/'data/{}.npy'.format(args.data)
  num_samples = args.n_tuning_samples

  config = get_param_config_for_tuning(args)

  # config = {
  #     "n_epoch": tune.choice([5, 10, 50]),
  #     "batch_size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
  #     # "lr": tune.loguniform(1e-4, 1e-1),
  #     "batch_size": tune.choice([2, 4, 8, 16]),
  #     # parameter from
  # }

  scheduler = ASHAScheduler(
      metric="loss",
      mode="min",
      max_t=args.n_epoch,
      grace_period=1,
      reduction_factor=2)
  reporter = CLIReporter(
      # parameter_columns=["l1", "l2", "lr", "batch_size"],
      metric_columns=["loss", "accuracy", "training_iteration"])

  # result = tune.run(
  #     partial(print_hi, 5),
  #     # resources_per_trial={"cpu": 2, "gpu": args.n_gpu},
  #     name="print_hi",
  #     resources_per_trial={"cpu": 2},
  #     config=config,
  #     num_samples=num_samples,
  #     scheduler=scheduler,
  #     progress_reporter=reporter)

  result = tune.run(
      partial(run_model,
              args                =args,
              # logger
              logger              =logger,
              logger_2            =logger_2,
              # data
              full_data           =full_data,
              # code params
              node_features       =node_features,
              edge_features       =edge_features,
              mean_time_shift_src =mean_time_shift_src,
              std_time_shift_src  =std_time_shift_src,
              mean_time_shift_dst =mean_time_shift_dst,
              std_time_shift_dst  =std_time_shift_dst,
              # config params
              device              =device,
              MODEL_SAVE_PATH     =MODEL_SAVE_PATH),

      name="train_self_supervised",
      # resources_per_trial={"cpu": 2, "gpu": args.n_gpu},
      resources_per_trial={"cpu": 1},
      # resources_per_trial={"cpu": 6},
      config=config,
      num_samples=num_samples,
      scheduler=scheduler,
      progress_reporter=reporter)

  # best_trial = result.get_best_trial("loss", "min", "last")

  # print("Best trial config: {}".format(best_trial.config))
  # print("Best trial final validation loss: {}".format(
  #     best_trial.last_result["loss"]))
  # print("Best trial final validation accuracy: {}".format(
  #     best_trial.last_result["accuracy"]))

  # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
  # device = "cpu"
  # if torch.cuda.is_available():
  #     device = "cuda:0"
  #     if gpus_per_trial > 1:
  #         best_trained_model = nn.DataParallel(best_trained_model)
  # best_trained_model.to(device)

  # best_checkpoint_dir = best_trial.checkpoint.value
  # model_state, optimizer_state = torch.load(os.path.join(
  #     best_checkpoint_dir, "checkpoint"))
  # best_trained_model.load_state_dict(model_state)

  # test_acc = test_accuracy(best_trained_model, device)
  # print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    raise NotImplementedError()
