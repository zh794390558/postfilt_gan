#!/bin/bash



if [[ $1 == 'test' ]];then
	# for test
	python main.py --cuda --netG model/netG_epoch_3.pth --testdata_dir
else
	# For training
	python main.py --voiceName nick --xFilesList ref_files.list --yFilesList gen_files.list --outf models --manualSeed 9 --batchSize 1 --mode train --mgcDim 41 \
	    --netD models/netD_epoch_3.pth --netG models/netG_epoch_3.pth --workers 6  --cuda
fi

