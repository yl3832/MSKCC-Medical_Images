import numpy as np
import os 
from shutil import copyfile,move
import argparse

parser = argparse.ArgumentParser(description="reorganize training and testing dataset")
parser.add_argument('--dir_in', default='./image',help="the dirctory where the pairs are in")
parser.add_argument('--dir_out',default='./images',help="the dirctory where the train and test sets are in")
parser.add_argument('--tr_te_split', default=0.9,help="training set and testing set ratio")
parser.add_argument('--random_seed',default=1,type=int,help="random seed for splitting dataset")
parser.add_argument('--split_training_data',default=1,type=int,help="shuffle and split training data")
parser.add_argument('--in_',default="./images/train",help="dir in of original training dataset")
parser.add_argument('--out_',default="./images",help="dir out of split training dataset")
parser.add_argument('--n',default=7,type=int,help="how many sub-sets")
parser.add_argument('--split_data',default=0,type=int,help="shuffle and split data")

args = parser.parse_args()
random_seed=args.random_seed
np.random.seed(random_seed)
dir_in=args.dir_in
dir_out=args.dir_out
tr_te_split=args.tr_te_split
split_training_data=args.split_training_data
in_=args.in_
out_=args.out_
n=args.n
split_data=args.split_data

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
    print(str(count)+' files deleted')

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
    length=len(os.listdir(os.path.join(dir_in, "sharp")))
    #print str(length)+' sample pairs in total'
    print(str(length)+' sample pairs in total')
    n_tr=int(tr_te_split*length)
    n_te=length-n_tr
    tr_inds=np.random.choice(length-1,n_tr,replace=False)
    print(tr_inds)
    for folder in os.listdir(dir_in):
        current_folder_path = os.path.join(dir_in, folder)
        if folder=='.DS_Store':
            os.remove(current_folder_path)
        else:
            #print 'processing '+folder
            #print str(len(os.listdir(current_folder_path)))+' samples in total'
            print('processing '+folder)
            print(str(len(os.listdir(current_folder_path)))+' samples in total')
            tr_folder_dir=os.path.join(tr_dir,folder)
            if not os.path.exists(tr_folder_dir):
                os.makedirs(tr_folder_dir)
            te_folder_dir=os.path.join(te_dir,folder)
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
            #print str(n_tr)+' training samples copied'
            #print str(n_te)+' testing samples copied'
            print(str(n_tr)+' training samples copied')
            print(str(n_te)+' testing samples copied')
            total_training=total_training+n_tr
            total_testing=total_testing+n_te
    #print "number of training samples: "+str(total_training)
    #print "number of testing samples: "+str(total_testing) 
    print ("number of training samples: "+str(total_training))
    print ("number of testing samples: "+str(total_testing))

def rename(in_):
    for folder in os.listdir(in_):
        current_folder_path=os.path.join(in_,folder)
        if folder=='.DS_Store':
            os.remove(current_folder_path)
        if folder=='blur':
            new_path_A=os.path.join(in_,"A")
            os.rename(current_folder_path,new_path_A)
            for files in os.listdir(new_path_A):
                if files=='.DS_Store':
                    current_file_path=os.path.join(new_path_A,files)
                    os.remove(current_file_path)
        if folder=='sharp':
            new_path_B=os.path.join(in_,"B")
            os.rename(current_folder_path,new_path_B)
            for files in os.listdir(new_path_B):
                if files=='.DS_Store':
                    current_file_path=os.path.join(new_path_B,files)
                    os.remove(current_file_path)

def split(in_,out_,n):
    length=len(os.listdir(os.path.join(in_,"A")))
    folder_indis=np.random.choice(n,length,replace=True)
    for folder in os.listdir(in_):
        current_folder_path=os.path.join(in_,folder)
        ind=0
        for files in os.listdir(current_folder_path):
            current_file_path=os.path.join(current_folder_path,files)
            folder_name="train_"+str(folder_indis[ind])
            ind=ind+1
            folder_path=os.path.join(out_,folder_name,folder)
            out_file_path=os.path.join(folder_path,files)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            move(current_file_path,out_file_path)
    for num in range(n):
        folders_name="train_"+str(num)
        if folders_name in os.listdir(out_):
            folders_path_A=os.path.join(out_,folders_name,"A")
            folders_path_B=os.path.join(out_,folders_name,"B")
            # print (folders_name)
            # print ("A:"+str(len(os.listdir(folders_path_B))))
            # print ("B:"+str(len(os.listdir(folders_path_B))))
            #print folders_name
            #print "A:"+str(len(os.listdir(folders_path_A)))
            #print "B:"+str(len(os.listdir(folders_path_B)))
            print(folders_name)
            print("A:"+str(len(os.listdir(folders_path_A))))
            print("B:"+str(len(os.listdir(folders_path_B))))
        else: pass
    
if split_data==1:
    init_reorganize(dir_in)
    reorganize_dataset(dir_in,dir_out,tr_te_split)
    rename(in_)
else: pass    
if split_training_data==1:
    split(in_,out_,n)