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
3. the modelname must already exist for your username (go [here](https://replicate.com/create) to create a model on replicate, then use it with `build.sh`)

## usage

0. make sure cog is authenticated
1. run dreambooth trainer on replicate.com - not dreambooth api
2. Download the output.zip and put it into a directory called weights
3. ./build.sh r8.im/username/modelname

## todo / open questions

- [ ] parse the sha of the pushed model to let you the replicate verison / image
- [ ] speed!!! downloading layers doesn't seem useful... is there a way to skip downloading the existing r8.im layers, ... just create new layer(s) and push - as this isn't for running locally
- [ ] efficiency?!? is sharing the base image of cog-stable-diffusion helpful? does having the SD2.1 weights in an unused but present layer make things better or worse?
- [ ] build these into cog?
