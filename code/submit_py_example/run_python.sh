#!/bin/bash

tar -xzf python38.tar.gz
tar -xzf packages.tar.gz

export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD


python3 my_script.py $1 $2 $3
