#!/bin/bash
# For training
python main.py --voiceName nick --xFilesList ref_files.list --yFilesList gen_files.list --outf models --manualSeed 9999 --batchSize 1 --mode train --mgcDim 41 \
    --netD models/netD_epoch_1.pth --netG models/netG_epoch_1.pth --workers 1  --cuda

#or

#python main.py --voiceName nick --xFilesList ref_files.list --yFilesList gen_files.list --outf models --manualSeed 9999 --mode train --cuda --mgcDim 41

