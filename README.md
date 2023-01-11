# dreambooth-builder

This is a prototype to build a replicate model using existing replicate models as the base.

These concepts might be incorporated into cog/replicate's [dreambooth api](https://replicate.com/blog/dreambooth-api)

## motivation

using `cog push` to build/push a model many times results in a lot of work - downloading / building / ... But each model you build technically just has a difference of the weights.

if we could take an existing popular stable diffusion model, and throw our weights and any customization to predict.py - our image should be much smaller/faster builds/... 

we only need to upload our changes (weights & predict.py)

## requirements

1. docker must be installed
2. install `cog` and authenticate

## usage

0. make sure cog is authenticated

1. run dreambooth trainer on replicate.com - not dreambooth api

2. Download the output.zip and put it into a directory called weights

3. ./build.sh r8.im/username/modelname

## todo

- [ ] parse the sha of the pushed model to let you the replicate verison numer
- [ ] is there a way to skip downloading the existing layers, ... just create new layers and push - since this isn't for running locally