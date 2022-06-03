import random
import numpy as np
import cv2
import sys
import math
from multiprocessing import Pool
from numpy.lib.stride_tricks import as_strided

class EXT_UINT8:

    # Function Int
    # Connection 0 Int
    # Connection 1 Int
    # Parameter 0 Real no limitation
    # Parameter 1 Int [−16, +16]
    # Parameter 2 Int [−16, +16]
    # Gabor Filter Frequ. Int [0, 16]
    # Gabor Filter Orient. Int [−8, +8]

    num_functions = 2
    ksize = (3,3)
    nb_processes = 16

    two_arguments = [False,False,False]

    @classmethod
    def getNbFunction(cls):
        return cls.num_functions

    @classmethod
    def getRandomFunction(cls):
        return random.randrange(1,1+cls.num_functions)

    @classmethod
    def needSecondArgument(cls,function):
        return cls.two_arguments[function]

    @classmethod
    def execute(cls, func, connection0, connection1, parameter0, parameter1, parameter2, gabor_filter_frequency, gabor_filter_orientation):
        # DISTANCE TRANSFORM
        if func==1:
            return cv2.distanceTransform(connection0, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, cv2.uint8)
        # WATERSHED
        elif func==2:
            lbl = cv2.watershed(connection0)

            return lbl.astype(np.uint8)

