import numpy as np

import cv2

from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion, binary_opening, binary_closing
from skimage.morphology import disk
from skimage import measure

import skimage.measure

from math import ceil

import random

class FunctionBundle(object):
    def __init__(self, min_function_index, max_function_index):
        self.min_function_index = min_function_index
        self.max_function_index = max_function_index

    def get_random_function_index(self):
        return random.randrange(self.min_function_index, self.max_function_index+1)

    def execute(self, function_index):
        raise NotImplementedError('"execute" method must be implemented')

'''
Scipy interface with all the computer vision functions
'''


def fill_holes(mask):
    return binary_fill_holes(mask)

def dilate(mask, disk_size, iterations):
    return binary_dilation(mask, disk(disk_size), iterations=iterations)

def erode(mask, disk_size, iterations):
    return binary_erosion(mask, disk(disk_size), iterations=iterations)

def open(mask, disk_size, iterations):
    return binary_opening(mask, disk(disk_size), iterations=iterations)

def close(mask, disk_size, iterations):
    padding_value = disk_size
    expended_mask = np.pad(mask, ((padding_value, padding_value), (padding_value, padding_value)), mode='constant')
    closed_mask = binary_closing(expended_mask, disk(disk_size), iterations=iterations)
    return closed_mask[padding_value:-padding_value, padding_value:-padding_value]

def open_then_close(mask, disk_size, iterations):
    return close(open(mask, disk_size, iterations), disk_size, iterations)

def mask_and(mask_1, mask_2):
    return np.bitwise_and(mask_1, mask_2)

def mask_or(mask_1, mask_2):
    return np.bitwise_or(mask_1, mask_2)

def mask_and_not(mask_1, mask_2):
    return np.bitwise_and(mask_1, np.bitwise_not(mask_2))

def count_true(mask):
    return np.count_nonzero(mask)

def find_contours(mask, level):
    return measure.find_contours(mask, level,  positive_orientation='high')

import numpy as np
from numpy.lib.stride_tricks import as_strided

def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size[0])//stride + 1,
                    (A.shape[1] - kernel_size[1])//stride + 1)
    #kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'min':
        return A_w.min(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg' or pool_mode == 'norm':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'norm':
        #(255*(connection0[i,j] - np.min(connection0[yc:yd,xa:xb]))/max(1,np.max(connection0[yc:yd,xa:xb]))).astype("uint8")
        maximums = A_w.max(axis=(1,2))
        #return TODO

class SuppFunctions(FunctionBundle):

    MIN_FUNCTION_INDEX = 51
    MAX_FUNCTION_INDEX = 68

    def __init__(self):
        super(SuppFunctions, self).__init__(SuppFunctions.MIN_FUNCTION_INDEX, SuppFunctions.MAX_FUNCTION_INDEX)
    # Function Int
    # Connection 0 Int
    # Connection 1 Int
    # Parameter 0 Real no limitation
    # Parameter 1 Int [−16, +16]
    # Parameter 2 Int [−16, +16]
    # Gabor Filter Frequ. Int [0, 16]
    # Gabor Filter Orient. Int [−8, +8]

    def execute(self, function_index, connection0, connection1, parameter0, parameter1, parameter2, gabor_filter_frequency, gabor_filter_orientation):

        # DILATE
        if function_index == self.min_function_index:
            return dilate(connection0, parameter0, 1)

        # ERODE
        if function_index == self.min_function_index+1:
            return erode(connection0, parameter0, 1)

        # OPEN
        if function_index == self.min_function_index+2:
            return open(connection0, parameter0, 1)

        # CLOSE
        if function_index == self.min_function_index+3:
            return close(connection0, parameter0, 1)

        # OPEN_THEN_CLOSE
        if function_index == self.min_function_index+4:
            return open_then_close(connection0, parameter0, 1)
            
        # FILL_HOLES
        if function_index == self.min_function_index+5:
            return fill_holes(connection0)

        # AND
        if function_index == self.min_function_index+6:
            return mask_and(connection0, connection1)

        # OR
        if function_index == self.min_function_index+7:
            return mask_or(connection0, connection1)

        # AND_NOT
        if function_index == self.min_function_index+8:
            return mask_and_not(connection0, connection1)

        # CONTOURS
        if function_index == self.min_function_index+9:
            return connection0
            #return find_contours(connection0, parameter0)

        # LOCALMIN parameter1 parameter2 61
        if function_index == self.min_function_index+10:
            retval = np.ndarray(connection0.shape,dtype="uint8")
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            xa = 0
            xb = 0
            yc = 0
            yd = 0
            if x==0:
                x = 1
            if y==0:
                y = 1
            for i in range(0,height):
                yc = i - y
                yd = i + y
                if yc<0:
                    yc = 0
                if yd>=height:
                    yd = height-1
                for j in range(0,width):
                    xa = j - x
                    xb = j + x
                    if xa<0:
                        xa = 0
                    if xb>=width:
                        xb = width-1
                    retval[i,j] = connection0[yc:yd,xa:xb].min()
            return retval
        # LOCALMAX parameter1 parameter2 62
        if function_index == self.min_function_index+11:
            retval = np.ndarray(connection0.shape,dtype="uint8")
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            xa = 0
            xb = 0
            yc = 0
            yd = 0
            if x==0:
                x = 1
            if y==0:
                y = 1
            for i in range(0,height):
                yc = i - y
                yd = i + y
                if yc<0:
                    yc = 0
                if yd>=height:
                    yd = height-1
                for j in range(0,width):
                    xa = j - x
                    xb = j + x
                    if xa<0:
                        xa = 0
                    if xb>=width:
                        xb = width-1
                    retval[i,j] = connection0[yc:yd,xa:xb].max()
            return retval
        # LOCALAVG parameter1 parameter2 63
        if function_index == self.min_function_index+12:
            retval = np.ndarray(connection0.shape,dtype="uint8")
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            xa = 0
            xb = 0
            yc = 0
            yd = 0
            if x==0:
                x = 1
            if y==0:
                y = 1
            for i in range(0,height):
                yc = i - y
                yd = i + y
                if yc<0:
                    yc = 0
                if yd>=height:
                    yd = height-1
                for j in range(0,width):
                    xa = j - x
                    xb = j + x
                    if xa<0:
                        xa = 0
                    if xb>=width:
                        xb = width-1
                    retval[i,j] = connection0[yc:yd,xa:xb].mean()
            return retval
        # LOCALNORMALIZE parameter1 parameter2 64
        if function_index == self.min_function_index+13:
            retval = np.ndarray(connection0.shape,dtype="uint8")
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            xa = 0
            xb = 0
            yc = 0
            yd = 0
            if x==0:
                x = 1
            if y==0:
                y = 1
            for i in range(0,height):
                yc = i - y
                yd = i + y
                if yc<0:
                    yc = 0
                if yd>=height:
                    yd = height-1
                for j in range(0,width):
                    xa = j - x
                    xb = j + x
                    if xa<0:
                        xa = 0
                    if xb>=width:
                        xb = width-1
                    retval[i,j] = (255*(connection0[i,j] - np.min(connection0[yc:yd,xa:xb]))/max(1,np.max(connection0[yc:yd,xa:xb]))).astype("uint8")
            return retval

        # LOCAL_MIN
        if function_index == self.min_function_index+14:
            #maximums = skimage.measure.block_reduce(connection0, (abs(parameter1), abs(parameter2)), np.min)
            #return np.kron(maximums, np.ones((abs(parameter1), abs(parameter2))))
            return pool2d(connection0, kernel_size=(abs(parameter1)*2, abs(parameter2)*2), stride=1, padding=abs(parameter1), pool_mode='min')

        # LOCAL_MAX
        if function_index == self.min_function_index+15:
            #maximums = skimage.measure.block_reduce(connection0, (abs(parameter1), abs(parameter2)), np.max)
            #return np.kron(maximums, np.ones((abs(parameter1), abs(parameter2))))
            return pool2d(connection0, kernel_size=(abs(parameter1)*2, abs(parameter2)*2), stride=1, padding=abs(parameter1), pool_mode='max')

        # LOCAL_AVG
        if function_index == self.min_function_index+16:
            #maximums = skimage.measure.block_reduce(connection0, (abs(parameter1), abs(parameter2)), np.mean)
            #return np.kron(maximums, np.ones((abs(parameter1), abs(parameter2))))
            return pool2d(connection0, kernel_size=(abs(parameter1)*2, abs(parameter2)*2), stride=1, padding=abs(parameter1), pool_mode='avg')

        # LOCAL_NORM
        if function_index == self.min_function_index+17:
            return pool2d(connection0, kernel_size=(abs(parameter1)*2, abs(parameter2)*2), stride=1, padding=abs(parameter1), pool_mode='norm')