import logging
import os
import sys
from pathlib import Path

import click
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory, abort

import wandb
from FedML.fedml_api.distributed.fedavg.FedAVGAggregator import FedAVGAggregator
from FedML.fedml_api.distributed.fedavg.FedAvgServerManager import FedAVGServerManager
from FedML.fedml_api.distributed.fedavg.MyModelTrainer import MyModelTrainer
from FedML.fedml_api.model import mobilenet, resnet56, LogisticRegression, RNNOriginalFedAvg
from fedml_api.data_preprocessing import MNISTDataLoader, ShakespeareDataLoader, Cifar10DatasetLoader, \
    Cifar100DatasetLoader, Cinic10DatasetLoader
from fedml_core import RunConfig

# HTTP server
app = Flask(__name__)
app.config['MOBILE_PREPROCESSED_DATASETS'] = './preprocessed_dataset/'
app.config['device_id_to_client_id_dict'] = dict()


@click.command()
@click.option('--model', 'model_name', type=str, default='lr', help='Neural network used in training')
@click.option('--dataset', 'dataset_name', type=str, default='mnist', help='Dataset used for training')
@click.option('--data_dir', type=click.Path(file_okay=False, path_type=Path), default='./../../FedML/data/MNIST/',
              help='Data directory (has to exist in clients)')
@click.option('--partition_method', type=str, default='hetero', help='How to partition the dataset on local workers')
@click.option('--partition_alpha', type=float, default=0.5, help='Partition alpha')
@click.option('--client_num_in_total', type=int, default=1000, help='Number of workers in a distributed cluster')
@click.option('--client_num_per_round', type=int, default=2, help='Number of workers')
@click.option('--batch_size', type=int, default=10, help='Input batch size for training')
@click.option('--client_optimizer', type=str, default='adam', help='SGD with momentum; adam')
@click.option('--lr', type=float, default=0.03, help='Learning rate')
@click.option('--wd', help='Weight decay parameter', type=float, default=0.001)
@click.option('--epochs', type=int, default=10, help='How many epochs will be trained locally')
@click.option('--comm_round', type=int, default=5, help='how many round of communications we shoud use')
@click.option('--is_mobile', type=bool, default=False, is_flag=True,
              help='Whether the program is running on the FedML-Mobile server side')
@click.option('--frequency_of_the_test', type=int, default=1, help='The frequency of the algorithms')
@click.option('--gpu_server_num', type=int, default=1, help='gpu_server_num')
@click.option('--gpu_num_per_server', type=int, default=4, help='gpu_num_per_server')
@click.option('--ci', type=bool, default=False, is_flag=True, help='continuous integration')
@click.option('--is_preprocessed', type=bool, default=False, is_flag=True, help='True if data has been preprocessed')
@click.option('--grpc_ipconfig_path', type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=Path),
              default=Path("../executor/grpc_ipconfig.csv"), help='config table containing ipv4 address of grpc server')
def main(model_name, dataset_name, data_dir: Path, partition_method, partition_alpha, client_num_in_total,
         client_num_per_round, batch_size, client_optimizer, lr, wd, epochs, comm_round, is_mobile,
         frequency_of_the_test, gpu_server_num, gpu_num_per_server, ci, is_preprocessed, grpc_ipconfig_path: Path):
    # MQTT client connection

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    wandb.init(project="fedml", name=f"mobile(mqtt){partition_method}r{comm_round}-e{epochs}-lr{lr}",
               config=click.get_current_context().params)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(0)
    torch.manual_seed(10)

    # GPU 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    dataset = load_data(dataset_name, batch_size, data_dir, partition_method, partition_alpha, client_num_in_total)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(model_name, dataset_name, output_dim=dataset.output_len)

    config = RunConfig(dataset_name, partition_alpha, client_num_in_total, client_num_per_round, None, is_mobile,
                       gpu_server_num, gpu_num_per_server, client_optimizer, lr, wd, batch_size, epochs, comm_round,
                       frequency_of_the_test, ci)

    model_trainer = MyModelTrainer(model, dataset_name, client_optimizer, lr, wd, epochs)

    aggregator = FedAVGAggregator(device, client_num_in_total, config, dataset, model_trainer)
    size = client_num_per_round + 1
    server_manager = FedAVGServerManager(aggregator, config, rank=0, size=size, backend="MQTT",
                                         is_preprocessed=is_preprocessed, grpc_ipconfig_path=grpc_ipconfig_path)
    server_manager.run()

    # if run in debug mode, process will be single threaded by default
    app.config['config'] = config
    app.config['data_dir'] = str(data_dir)
    app.config['model'] = model_name
    app.config['partition_method'] = partition_method
    app.config['is_preprocessed'] = is_preprocessed
    app.config['grpc_ipconfig_path'] = str(grpc_ipconfig_path)
    app.run(host='0.0.0.0', port=5000)


@app.route('/', methods=['GET'])
def index():
    return 'backend service for Fed_mobile'


@app.route('/get-preprocessed-data/<dataset_name>', methods=['GET'])
def get_preprocessed_data(dataset_name):
    directory = f'{app.config["MOBILE_PREPROCESSED_DATASETS"]}{dataset_name.upper()}_mobile_zip/'
    try:
        return send_from_directory(directory, filename=dataset_name + '.zip', as_attachment=True)
    except FileNotFoundError:
        abort(404)


@app.route('/api/register', methods=['POST'])
def register_device():
    device_id_to_client_id_dict = app.config['device_id_to_client_id_dict']
    config: RunConfig = app.config['config']
    # __log.info("register_device()")
    device_id = request.args['device_id']
    registered_client_num = len(device_id_to_client_id_dict)
    if device_id in device_id_to_client_id_dict:
        client_id = device_id_to_client_id_dict[device_id]
    else:
        client_id = registered_client_num + 1
        device_id_to_client_id_dict[device_id] = client_id

    training_task_args = dict(
        dataset_name=config.dataset_name,
        data_dir=app.config['data_dir'],
        partition_method=app.config['partition_method'],
        partition_alpha=config.partition_alpha,
        model=app.config['model'],
        client_num_in_total=config.client_num_in_total,
        client_num_per_round=config.client_num_per_round,
        comm_round=config.comm_round,
        epochs=config.epochs,
        lr=config.lr,
        wd=config.wd,
        batch_size=config.batch_size,
        frequency_of_the_test=config.frequency_of_the_test,
        is_mobile=config.is_mobile,
        dataset_url=f'{request.url_root}/get-preprocessed-data/{client_id - 1}',
        is_preprocessed=app.config['is_preprocessed'],
        grpc_ipconfig_path=app.config['grpc_ipconfig_path']
    )

    return jsonify(dict(errno=0, executorId="executorId", executorTopic="executorTopic", client_id=client_id,
                        training_task_args=training_task_args))


def load_data(dataset_name, batch_size, data_dir, partition_method, partition_alpha, client_num_in_total):
    dl_cls = dict(
        mnist=MNISTDataLoader,
        shakespeare=ShakespeareDataLoader,
        cifar10=Cifar10DatasetLoader,
        cifar100=Cifar100DatasetLoader,
        cinic10=Cinic10DatasetLoader,
    ).get(dataset_name, Cifar10DatasetLoader)

    if dl_cls is None:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    logging.info(f"load_data. dataset_name = {dataset_name}")

    dl_args = dict(batch_size=batch_size)

    if dataset_name == 'mnist':
        dl_args.update(data_dir="./../FedML/data/MNIST")
    elif dataset_name != 'shakespeare':
        dl_args.update(data_dir=data_dir, partition_method=partition_method, partition_alpha=partition_alpha,
                       client_num_in_total=client_num_in_total)

    dl = dl_cls(**dl_args)

    return dl.load_partition_data()


def create_model(model_name, dataset_name, output_dim):
    logging.info(f"create_model. model_name = {model_name}, output_dim = {output_dim}")
    if model_name == "lr" and dataset_name == "mnist":
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "rnn" and dataset_name == "shakespeare":
        model = RNNOriginalFedAvg(28 * 28, output_dim)
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    else:
        model = None
    return model


if __name__ == '__main__':
    main()
