########################################################################################
#                       Autorzy skryptu: Michał Tomaszewski                            #
########################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from PIL import Image
import time
import cv2

def get_nonzeros(img):
    # turn 2D array (image) into a vector
    img = img.ravel()
    # get all non-zero values from the vector
    nonzeros = img[img != 0]
    
    return nonzeros, len(nonzeros), np.mean(nonzeros)  


def maxswarm(img):
    # get coordinates of max intensity pixels
    maxpointlist = np.argwhere(img == img.max())
    # prepare and fill lists with appropriate coordinate values
    x = []
    y = []
    for i in range(len(maxpointlist)):
        y.append(maxpointlist[i][0])
        x.append(maxpointlist[i][1])

    # find center of points (using mean)
    xmean = np.mean(np.array(x), dtype = 'uint32')
    ymean = np.mean(np.array(y), dtype = 'uint32')
    
    return xmean, ymean


def crop_center(img,x,y,cropx,cropy):
    # according to what point should the image be cropped?
    startx = x
    starty = y
    
    return img[starty-cropy:starty+cropy,startx-cropx:startx+cropx]


def crop_by_max(img, cropx, cropy):
    # get coordinates of max intensity pixels
    maxpointlist = np.argwhere(img == img.max())
    # prepare and fill lists with appropriate coordinate values
    x = []
    y = []
    for i in range(len(maxpointlist)):
        y.append(maxpointlist[i][0])
        x.append(maxpointlist[i][1])

    # find center of points
    xmean = np.mean(np.array(x), dtype = 'uint32')
    ymean = np.mean(np.array(y), dtype = 'uint32')

    # call simple cropping function
    croppedimggray = crop_center_mod(img, xmean, ymean, 300, 300)
    #croppedimgrgb = crop_center(imgrgb, xmean, ymean, 300, 300)
        
    return croppedimggray


def crop_center_mod(img,x,y,cropx,cropy):
    # start point according to which cropping will be done
    startx = x
    starty = y
    # get image shape
    imgshape = np.shape(img)
    # check if cropping request exceeds VERTICAL image borders
    # if it does trim cropping request accordingly
    if ((starty + cropy) > imgshape[0]):
        cropy = imgshape[0] - starty 
    elif ((starty - cropy) < 0):
        cropy = starty 
    else:
        cropy = cropy
    # check if cropping request exceeds HORIZONTAL image borders
    # if it does trim cropping request accordingly
    if ((startx + cropx) > imgshape[1]):
        cropx = imgshape[1] - startx 
    elif ((startx - cropx) < 0):
        cropx = startx 
    else:
        cropx = cropx
        
    return img[starty-cropy:starty+cropy,startx-cropx:startx+cropx]


def check_points(point_list, img_length):
    # save all points that find themselves in the first half of the picture (from the left)
    accepted = point_list[point_list[:, 0] < round(img_length/2)]  
    
    return accepted

def euclid_dist(p1, p2):
    # cast points' coordinates to higher int and calculate the distance between
    x1, y1 = np.int64(p1[0]), np.int64(p1[1])
    x2, y2 = np.int64(p2[0]), np.int64(p2[1])
    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2, dtype = 'float64')
    
    return distance


def find_closest(point_list):
    # create a matrix to hold distance to every point from every point
    distances = np.zeros((len(point_list), len(point_list)))
    for i in range(0, len(point_list)):
        # current point
        current_point = point_list[i, :]
        # delete current point from initial list to obtain remaining ones
        other_points = np.delete(point_list, (i), axis = 0) 
        for j in range(0, len(point_list)):
            if i == j:
                # fill zeros with high values (so that the function doesnt say that point x is closest to point x)
                distances[i,j] = 10**6
            else:
                distances[i,j] = euclid_dist(current_point, point_list[j,:])
    # find points closest to each other by looking for minimal distance
    closest_points = np.argwhere(distances == np.min(distances))
    # return their indices
    closest_points_idx = np.unique(closest_points[:, 0])
    
    return closest_points_idx 

def calc_membership(all_points, start_points, margin):    
    '''This function checks if points found by maximum intensity value in each channel lay in acceptable range(margin)
    to be included in cropping operation.'''
    # cast start_points to higher int
    start_points = np.int64(start_points)
    # calculate starter point by taking mean of start_points coordinates
    start_mean_pt = np.array([0,0])    
    start_mean_pt[0] = np.mean(all_points[start_points,[0]], dtype = 'uint32')
    start_mean_pt[1] = np.mean(all_points[start_points,[1]], dtype = 'uint32')
    # get remaining points
    remaining_points = np.delete(all_points, start_points, axis = 0)
    
    # this whole block is responsible for accepting points
    # new_points holds all the accepted_points
    new_points = all_points[start_points, :]
    # new_mean holds the "cropping point" --> mean of coordinates of all accepted points
    new_mean = start_mean_pt
    # update the above variables accordingly
    while len(remaining_points) >= 1:
        next_pt_dist = euclid_dist(start_mean_pt, remaining_points[0])
        if next_pt_dist <= margin:
            new_points = np.vstack((new_points,remaining_points[0]))
            new_mean[0] = np.mean(new_points[:,0], dtype = 'uint32')
            new_mean[1] = np.mean(new_points[:,1], dtype = 'uint32')
            remaining_points = np.delete(remaining_points, [0], axis = 0)
        else:
            remaining_points = np.delete(remaining_points, [0], axis = 0)
            
    return new_mean
    
    
def multi_channel_crop(image):
    # Load the image and intially throw away the right side of the matrix
    img = Image.open(image)
    img_array = np.asarray(img) 
    half_array = img_array[:,:img_array.shape[1]//2+200]
    # Do the same for CMYK colorspace
    img_cmyk = img.convert('CMYK')
    cmyk_array = np.asarray(img_cmyk)
    cmyk_array = cmyk_array[:,:cmyk_array.shape[1]//2+200]
    # Create gray/green versions
    img_gray = cv2.cvtColor(half_array, cv2.COLOR_RGB2GRAY)
    img_green = half_array.copy()[:,:,1]
    
    # Get what you need from CMYK
    m = cmyk_array.copy()[:,:,1]
    y = cmyk_array.copy()[:,:,2]     
    inv_m = 255*np.ones(m.shape, 'uint8') - m
    inv_y = 255*np.ones(y.shape, 'uint8') - y  
    
    # Find maxpoints in all of these spaces
    m_x, m_y = maxswarm(inv_m)
    y_x, y_y = maxswarm(inv_y)
    gray_x, gray_y = maxswarm(img_gray)
    green_x, green_y = maxswarm(img_green)
    
    # Aggregate, find two closest points and calculate mean coords
    all_points = np.array([[m_x, m_y], [y_x, y_y], [gray_x, gray_y], [green_x, green_y]])
    closest = find_closest(all_points)
    membership_mean = calc_membership(all_points, closest, 100) 
    
    # Crop & return
    cropped_img = crop_center_mod(img_array, membership_mean[0], membership_mean[1], 400, 400)
    
    return cropped_img

def multichannel_cropmod(image_path, membership_margin, cropx, cropy):
    # This function does the same as the last one but is modified for more params
    img = Image.open(image_path)
    img_array = np.asarray(img) 
    half_array = img_array[:,:img_array.shape[1]//2+200]
    
    img_cmyk = img.convert('CMYK')
    cmyk_array = np.asarray(img_cmyk)
    cmyk_array = cmyk_array[:,:cmyk_array.shape[1]//2+200]
    
    img_gray = cv2.cvtColor(half_array, cv2.COLOR_RGB2GRAY)
    img_green = half_array.copy()[:,:,1]
    
    m = cmyk_array.copy()[:,:,1]
    y = cmyk_array.copy()[:,:,2]     
    inv_m = 255*np.ones(m.shape, 'uint8') - m
    inv_y = 255*np.ones(y.shape, 'uint8') - y  
    
    m_x, m_y = maxswarm(inv_m)
    y_x, y_y = maxswarm(inv_y)
    gray_x, gray_y = maxswarm(img_gray)
    green_x, green_y = maxswarm(img_green)
    
    all_points = np.array([[m_x, m_y], [y_x, y_y], [gray_x, gray_y], [green_x, green_y]])
    
    closest = find_closest(all_points)
    
    membership_mean = calc_membership(all_points, closest, membership_margin) 
    
    cropped_img = crop_center_mod(img_array, membership_mean[0], membership_mean[1], cropx, cropy)
    #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    return cropped_img

def localize_xy(image):
    # This is just a helper I think (UNUSED)
    img = Image.fromarray(np.uint8(image))
    half_array = image[:,:image.shape[1]//2+200]
    
    img_cmyk = img.convert('CMYK')
    cmyk_array = np.asarray(img_cmyk)
    cmyk_array = cmyk_array[:,:cmyk_array.shape[1]//2+200]
    
    img_gray = cv2.cvtColor(half_array, cv2.COLOR_RGB2GRAY)
    img_green = half_array.copy()[:,:,1]
    
    m = cmyk_array.copy()[:,:,1]
    y = cmyk_array.copy()[:,:,2]     
    inv_m = 255*np.ones(m.shape, 'uint8') - m
    inv_y = 255*np.ones(y.shape, 'uint8') - y  
    
    m_x, m_y = maxswarm(inv_m)
    y_x, y_y = maxswarm(inv_y)
    gray_x, gray_y = maxswarm(img_gray)
    green_x, green_y = maxswarm(img_green)
    
    all_points = np.array([[m_x, m_y], [y_x, y_y], [gray_x, gray_y], [green_x, green_y]])
    
    closest = find_closest(all_points)
    
    membership_mean = calc_membership(all_points, closest, 100) 

    return membership_mean


