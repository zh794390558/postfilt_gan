#/binb/ash



if [[ $# == 0 ]]; then
	echo "usage: $0 gen.f0 mcepfile wavfile"
	exit 1
fi

#./straight_mceplsf -f 22050 -mcep -pow -order 40 -shift 5 -f0file 140079.f0 -syn 140079.mcep.cut.mcep 140079.wav 

./straight_mceplsf -f 22050 -mcep -pow -order 40 -shift 5 -f0file $1  -syn $2 $3 
