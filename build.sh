#!/bin/bash


set -o errexit
set -o xtrace

# require image name as argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <image name>"
    exit 1
fi
img=$1

# if there isn't a weights directory, tell user to unzip weights to it
if [ ! -d weights ]; then
    echo "Please download/unzip weights to weights directory"
    exit 1
fi

docker build -t $img .

docker push $img