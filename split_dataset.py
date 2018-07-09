import numpy as np
import os 
from shutil import copyfile
import argparse

parser = argparse.ArgumentParser(description="reorganize training and testing dataset")
parser.add_argument('--dir_in', default='./image',help="the dirctory where the pairs are in")
parser.add_argument('--dir_out',default='./images',help="the dirctory where the train and test sets are in")
parser.add_argument('--tr_te_split', default=0.8,help="training set and testing set ratio")
parser.add_argument('--random_seed',default=1,type=int,help="random seed for splitting dataset")
#parser.add_argument('--split_training_data',default=0,type=int,help="random seed for splitting dataset")

args = parser.parse_args()
random_seed=args.random_seed
np.random.seed(random_seed)
dir_in=args.dir_in
dir_out=args.dir_out
tr_te_split=args.tr_te_split
split_training_data=args.split_training_data

def init_reorganize(dir_in):
    count = 0
    folders = ["sharp","blur"]
    folder_paths = [os.path.join(dir_in, folder) for folder in folders]
    for folder_path in folder_paths:
        for file_ in os.listdir(folder_path):
            current_file_path=os.path.join(folder_path, file_)
            if file_== '.DS_Store':
                count=count+1
                os.remove(current_file_path)
        else:
            pass
    print str(count)+' files deleted'
def reorganize_dataset(dir_in,dir_out,tr_te_split):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    tr_dir=os.path.join(dir_out,'train')
    te_dir=os.path.join(dir_out,'test')
    if not os.path.exists(tr_dir):
        os.makedirs(tr_dir)
    if not os.path.exists(te_dir):
        os.makedirs(te_dir)
    total_training=0
    total_testing=0
    for folder in os.listdir(dir_in):
        current_folder_path = os.path.join(dir_in, folder)
        if folder=='.DS_Store':
            os.remove(current_folder_path)
        else:
            print 'processing '+folder
            print str(len(os.listdir(current_folder_path)))+' samples in total'
            length=len(os.listdir(current_folder_path))
            n_tr=int(tr_te_split*length)
            n_te=length-n_tr
            tr_inds=np.random.choice(length-1,n_tr,replace=False)
            tr_folder_dir=os.path.join(tr_dir,folder)
            #print tr_folder_dir
            if not os.path.exists(tr_folder_dir):
                os.makedirs(tr_folder_dir)
            te_folder_dir=os.path.join(te_dir,folder)
            #print te_folder_dir
            if not os.path.exists(te_folder_dir):
                os.makedirs(te_folder_dir)
            ind=0
            for files in os.listdir(current_folder_path):
                file_path=os.path.join(current_folder_path,files)
                if ind in tr_inds:
                    #this is a training sample
                    output_file_path=os.path.join(tr_folder_dir,files)
                    copyfile(file_path, output_file_path)
                else:
                    #this is a testing sample 
                    output_file_path=os.path.join(te_folder_dir,files)
                    copyfile(file_path, output_file_path)
                ind=ind+1
            print str(n_tr)+' training samples copied'
            print str(n_te)+' testing samples copied'
            total_training=total_training+n_tr
            total_testing=total_testing+n_te
    print "number of training samples: "+str(total_training)
    print "number of testing samples: "+str(total_testing) 

init_reorganize(dir_in)
reorganize_dataset(dir_in,dir_out,tr_te_split)