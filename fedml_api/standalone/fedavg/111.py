import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.Mydata.data_loader import load_partition_data_Mydata
from fedml_api.data_preprocessing.Experiment.data_loader import load_partition_data_Experiment
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.standalone.fedavg.fedavg_trainer import FedAvgTrainer


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='lr', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='Experiment', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/Experiment_4/14_truncat',
                        help='test directory')

    parser.add_argument('--partition_method', type=str, default='hetero-fix', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0_truncat.9)')

    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0_truncat.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=3, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=3, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=600,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=2,
                        help='the frequency of the algorithms')

    parser.add_argument('--local_fine_tuning', type=bool, default=False,
                        help='the frequency of the algorithms')

    parser.add_argument('--epochs_of_local_fine_tuning', type=int, default=0,
                        help='the frequency of the algorithms')

    parser.add_argument('--if_join_midway', type=bool, default=True,
                        help='if client C join midway')

    parser.add_argument('--join_epoch', type=bool, default=300)

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    return parser


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128 # temporary batch size
    else:
        full_batch = False

    if dataset_name == "Experiment":
        data_loader = load_partition_data_Experiment
    else:
        data_loader = load_partition_data_Mydata
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                            args.partition_alpha, args.client_num_in_total, args.batch_size)

    if centralized:
        train_data_local_num_dict = {0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset

def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]

def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    else:
        logging.info("LogisticRegression + Experiment")
        model = LogisticRegression( 1, output_dim)
    return model


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="test",
        #project="fedml",
        name="实验4 (微调100)- A:-1/1.5_B:0/1.5_C:3/1.5_" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )
    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load test
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)

    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    logging.info(model)
    rou = 0
    shift = 0
    if args.if_join_midway:
        rou = args.join_epoch
        shift = 1
    trainer = FedAvgTrainer(dataset, model, device, args, rou, shift)
    trainer.train(args.data_dir)