########################################################################################
#                            Autorzy skryptu: Michał Tomaszewski                       #
########################################################################################

import os
import cv2
import sys
from support_modules.croppack import multichannel_cropmod
from concurrent.futures import ThreadPoolExecutor

# make the directory
MT_dir = 'MT'

if not os.path.isdir(MT_dir):
    os.mkdir(MT_dir)

for MASTER_DIR in [MT_dir]:
    crop_op_path = os.path.join(MASTER_DIR, 'crop_op')
    if not os.path.isdir(crop_op_path):
        os.mkdir(crop_op_path)

# MT cropping helper function
def crop_and_save_MT(img_path):

    DESTINATION = os.path.join(MT_dir, 'crop_op')

    cropped_img = multichannel_cropmod(img_path, membership_margin = 100, cropx = 500, cropy = 500)
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    new_name = "cropped" + img_path.split('\\')[-1]
    save_path = os.path.join(DESTINATION, new_name)
    cv2.imwrite(save_path, cropped_img)

# source directory (where the images sit)
DIR = r''

# make file paths for all files to be cropped
file_paths = [os.path.join(DIR, fname) for fname in os.listdir(DIR)]

# crop
with ThreadPoolExecutor() as executor:
    executor.map(crop_and_save_MT, file_paths)