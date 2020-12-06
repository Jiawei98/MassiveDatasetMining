#!/bin/bash

# untar your Python installation. Make sure you are using the right version!
tar -xzf python38.tar.gz
# untar packages
tar -xzf packages.tar.gz

# make sure the script will use your Python installation, 
# and the working directory as its home location
export PATH=$PWD/python/bin:$PATH
export PYTHONPATH=$PWD/packages
export HOME=$PWD

tar -xzf dataset.tgz
find dataset -name "._*" -delete

# run your script
python3 --version
python3 CNN.py $1

