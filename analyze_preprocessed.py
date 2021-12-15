import json
from collections import Counter, defaultdict
from pathlib import Path
from pprint import pprint

base_path = Path() / 'executor' / 'preprocessed_dataset' / 'MNIST_mobile'
n_clients = 2
client_dict = defaultdict(dict)

for client in range(n_clients):
    for phase in ('train', 'test'):
        json_path = base_path / str(client) / phase / f'{phase}.json'
        with json_path.open() as fp:
            json_dict = json.load(fp)
        counter = Counter()
        for username in json_dict['user_data']:
            counter.update(json_dict['user_data'][username]['y'])
        client_dict[client][phase] = counter

pprint(client_dict)
