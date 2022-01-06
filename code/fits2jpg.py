import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2
import os
'''
def norm_image(image):
    img = image.copy()
    img -= img.min()
    img /= img.max()
    img *= 255.
    img = np.uint8(img)
    
    return img

# relevant path
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

data_path_c = basedir + '/data/test_input/continuum/'
save_path_c = basedir + '/user_data/tmp_data/test_input/continuum/'
file_names_c = os.listdir(data_path_c)
file_list_c = [os.path.join(data_path_c, file) for file in file_names_c]

data_path_m = basedir + '/data/test_input/magnetogram/'
save_path_m = basedir + '/user_data/tmp_data/test_input/magnetogram/'
file_names_m = os.listdir(data_path_m)
file_list_m = [os.path.join(data_path_m, file) for file in file_names_m]

# fits2jpg function
def fits2jpg_fun(file_list, save_path):
    for file in file_list:
        hdu_list = fits.open(file)
        hdu_list.verify('fix')
        img = np.array(hdu_list[1].data)
        img = norm_image(img)
    
        hdu_list.close()
        file_name_base = os.path.basename(file)
        file_name = save_path + os.path.splitext(file_name_base)[0] + '.jpg'
        cv2.imwrite(file_name, img)

# convertion for continuum & magnetogram
fits2jpg_fun(file_list=file_list_c, save_path= save_path_c)
fits2jpg_fun(file_list=file_list_m, save_path= save_path_m)
'''
def norm_image(image):
    img = image.copy()
    img -= img.min()
    img /= img.max()
    img *= 255.
    img = np.uint8(img)
    
    return img
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
path_list = ['continuum/alpha','continuum/beta','continuum/betax','magnetogram/alpha','magnetogram/beta','magnetogram/betax']
for path in path_list:
    path_o = basedir + '/data/trainset/' + path
    file_names = os.listdir(path_o)
    file_list = [os.path.join(path_o, file) for file in file_names]

    for file in file_list:
        print(file)
        hdu_list = fits.open(file)
        hdu_list.verify('fix')
        img = np.array(hdu_list[1].data)
        img = norm_image(img)
        
        hdu_list.close()
        file_name_base = os.path.basename(file)
        file_name = basedir + '/user_data/tmp_data/train_input/' + path + '/' + \
            os.path.splitext(file_name_base)[0] + '.jpg'
        print(file_name)
        cv2.imwrite(file_name, img)

