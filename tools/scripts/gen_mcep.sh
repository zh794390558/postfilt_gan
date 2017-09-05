#!/bin/bash

usage(){
    echo "usage: $0 f0file wavfile mcepfile"
    exit 1
}

if [[ $# != 3 ]];then
    usage
fi

# 40 is for 41 dims feature
# extract features
./straight_mceplsf -f 22050 -shift 5 -mcep -pow -order 40 -f0file $1 -ana $2 $3
