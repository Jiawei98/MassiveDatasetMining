#!/bin/bash

tar -xzf R402.tar.gz
export PATH=$PWD/R/bin:$PATH
export RHOME=$PWD/R

tar -xzf packages.tar.gz
export R_LIBS=$PWD/packages

tar -xzf $1
fn=$(echo $1 | cut -d. -f1)
mkdir ${fn}_gray

Rscript convertImages.R ${fn}

tar -czf ${fn}_gray.tgz ${fn}_gray

