#!/bin/bash

if [[ $# == 0 ]];then
	echo "usage: $0 filename"
	exit 1
fi

rm -r ./demo || true
mkdir ./demo

cp ../test/out/$1.*  ./demo/$1.enforce.mcep

cp /lustre/atlas/zhanghui/StandFemale_22K/MCEP/nature_par/scep_cut/$1.*  ./demo/$1.nat.mcep
cp /lustre/atlas/zhanghui/StandFemale_22K/MCEP/nature_par/postf0_cut/$1.*  ./demo/$1.nat.f0

cp /lustre/atlas/zhanghui/StandFemale_22K/MCEP/gen_par/f0_cut/$1.*  ./demo/$1.syn.f0
cp /lustre/atlas/zhanghui/StandFemale_22K/MCEP/gen_par/mcep_cut/$1.*  ./demo/$1.syn.mcep

# syn f0 for wav
bash ./gen.sh ./demo/$1.syn.f0 ./demo/$1.enforce.mcep ./demo/$1.enforce.wav
bash ./gen.sh ./demo/$1.syn.f0 ./demo/$1.syn.mcep ./demo/$1.syn.wav
bash ./gen.sh ./demo/$1.syn.f0 ./demo/$1.nat.mcep ./demo/$1.nat.wav
bash ./gen.sh ./demo/$1.nat.f0 ./demo/$1.nat.mcep ./demo/$1.natf0.nat.wav

# text
./x2x +f +a41 ./demo/$1.syn.mcep > ./demo/$1.syn.mcep.txt
./x2x +f +a41 ./demo/$1.nat.mcep > ./demo/$1.nat.mcep.txt
./x2x +f +a41 ./demo/$1.enforce.mcep > ./demo/$1.enforce.mcep.txt

# use GV
cp ./demo/$1.syn.mcep ./demo/$1.syn.gv.mcep
python mcep_gv.py --mceppath ./demo/$1.syn.gv.mcep
bash ./gen.sh ./demo/$1.syn.f0 ./demo/$1.syn.gv.mcep ./demo/$1.syn.gv.wav

