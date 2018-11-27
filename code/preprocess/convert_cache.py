import os
import h5py
from utils import load_images
import argparse

parser = argparse.ArgumentParser(description="preprocess and convert")
parser.add_argument('--train_in', default='./images',help="the dirctory where the pairs are in")
param = parser.parse_args()

def convert_cache(train_in):
    print('Loading data')
    for folder in os.listdir(train_in)[1:]:
        print(folder)
        train_data = load_images(os.path.join(param.train_in, folder),n_images=-1)
        cache_file = folder+'.hdf5'
        h5f = h5py.File(cache_file, 'w')
        h5f.create_dataset('A', data=train_data['A'])
        h5f.create_dataset('B', data=train_data['B'])
        h5f.close()
    
convert_cache(param.train_in)    