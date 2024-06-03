#!/usr/bin/env bash

sudo apt update
sudo apt install software-properties-common
sudo apt install python3.10

sudo apt-get install python3-pip
python3 -m pip install -U pip

python3 -m pip install -r requirements.txt

# start a local head node
ray start --head