import tensorflow as tf
import numpy as np
from model_tf import deblur_model
import argparse
from utils import load_images, load_own_images
import os
import h5py

if __name__ == '__main__':
    
    # add argument
    parser = argparse.ArgumentParser(description="deblur train")
    parser.add_argument("--is_train", help="train or generate", default=0,type=int)
    parser.add_argument("--g_input_size", help="Generator input size of the image", default=256,type=int)
    #parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--n_downsampling', type=int, default=2, help='# of downsampling in generator')
    parser.add_argument('--n_blocks_gen', type=int, default=9, help='# of res block in generator')
    parser.add_argument('--d_input_size', type=int, default=256, help='Generator input size')
    parser.add_argument('--kernel_size', type=int, default=4, help='kernel size factor in discriminator')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size when training')
    parser.add_argument('--model_name',default=None, help='The pre-trained model name')
    parser.add_argument('--save_freq',default=300, type=int, help='Model save frequency')
    parser.add_argument('--total_epoch_num',default=5,type=int, help='Total number of epoch for training')
    parser.add_argument('--epoch_num',default=1, type=int, help='Number of epoch for training')
    parser.add_argument('--generate_image_freq', default=100, type=int, help='Number of iteration to generate image for checking')
    parser.add_argument('--LAMBDA_A', default=100000, type=int, help='The lambda for preceptual loss')
    parser.add_argument('--g_train_num', default=0, type=int, help='Train the generator for x epoch before adding discriminator')
    parser.add_argument('--L1_content_loss',default=0, type=int, help='use L1 loss for content loss')
    
    param = parser.parse_args()
    print('Building model')
    model = deblur_model(param)
    
    for i in range(param.total_epoch_num):
        for j in range(7):
            cache_file = 'train_'+str(j)+'.hdf5'
            print (cache_file) 
            h5f = h5py.File(cache_file,'r')
            train_data = {'A':h5f['A'][:], 'B':h5f['B'][:]}
            print('Training model')
            if i==0 and j==0:
                 cur_model_name=model.train(train_data, 
                    batch_size=param.batch_size, 
                    pre_trained_model=param.model_name, 
                    save_freq = param.save_freq,
                    epoch_num = param.epoch_num,
                    generate_image_freq = param.generate_image_freq)
            else:    
                cur_model_name=model.train(train_data, 
                        batch_size=param.batch_size, 
                        pre_trained_model=cur_model_name, 
                        save_freq = param.save_freq,
                        epoch_num = param.epoch_num,
                        generate_image_freq = param.generate_image_freq)
            
