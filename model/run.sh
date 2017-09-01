pip install matplotlib --user
# For training 
python main_1.py --voiceName nick --xFilesList ref_files.list --yFilesList gen_files.list --outf models --manualSeed 9999 --mode train --cuda --mgcDim 41

#or

#python main.py --voiceName nick --xFilesList ref_files.list --yFilesList gen_files.list --outf models --manualSeed 9999 --mode train --cuda --mgcDim 41

