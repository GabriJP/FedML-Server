# Installation
http://doc.fedml.ai/#/installation

After the clone of this repository, please run the following command to get `FedML` submodule to your local.
```shell
cd FedML
git submodule init
git submodule update
```

# Update FedML Submodule
```shell
cd FedML
git checkout master && git pull
cd ..
git add FedML
git commit -m "updating submodule FedML to latest"
git push
```

# Run mosquitto docker
```shell
docker run -it --name mosquitto -p 1883:1883 --rm -v $(pwd)/mosquitto.conf:/mosquitto/config/mosquitto.conf eclipse-mosquitto
```