import random
import numpy as np
import cv2
import sys
import math
from multiprocessing import Pool
from numpy.lib.stride_tricks import as_strided

data = None
data_x = None
data_y = None

def init_pool_mmmn(data_arg,data_y_arg,data_x_arg):
    global data
    global data_x
    global data_y
    data = data_arg
    data_x = data_x_arg
    data_y = data_y_arg

def localmin(a):
    y,x = a
    return data[y-data_y:y+data_y,x-data_x:x+data_x].min()

def localmax(a):
    y,x = a
    return data[y-data_y:y+data_y,x-data_x:x+data_x].max()

def localmean(a):
    y,x = a
    return data[y-data_y:y+data_y,x-data_x:x+data_x].mean()

def localnormalize(a):
    y,x = a
    tmp = (data[y,x] - np.min(data[y-data_y:y+data_y,x-data_x:x+data_x]))
    tmp = tmp/max(1,np.max(data[y-data_y:y+data_y,x-data_x:x+data_x]))
    return (255*(tmp)).astype("uint8")

class STD_UINT8_MP:

    # Function Int
    # Connection 0 Int
    # Connection 1 Int
    # Parameter 0 Real no limitation
    # Parameter 1 Int [−16, +16]
    # Parameter 2 Int [−16, +16]
    # Gabor Filter Frequ. Int [0, 16]
    # Gabor Filter Orient. Int [−8, +8]

    num_functions = 50
    ksize = (3,3)
    nb_processes = 16

    two_arguments = [False,False,False,False,False,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]

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
        # CONST
        if func==1:
            return np.full(connection0.shape,fill_value=parameter0,dtype="uint8")
        # NOP
        elif func==2:
            return connection0
        # ADD connection0 connection1
        elif func==3:
            if True:
                tmp = np.add(connection0,connection1,dtype="int16")
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.add(connection0,connection1) # with overflow modulo
        # SUB connection0 connection1
        elif func==4:
            if True:
                tmp = np.subtract(connection0,connection1,dtype="int16")
                tmp[tmp<0] = 0
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.subtract(connection0,connection1) # with overflow modulo
        # MUL connection0 connection1
        elif func==5:
            if True:
                tmp = np.multiply(connection0,connection1,dtype="int16")
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.multiply(connection0,connection1) # with overflow modulo
        # LOG connection0
        elif func==6:
            if True:
                tmp = np.array(connection0, dtype="uint8")
                tmp[tmp==0] = 1
                return np.asarray(np.log(tmp),dtype="uint8")
            else:
                return np.asarray(np.log(connection0), dtype="uint8") # with 0 warnings
        # EXP connection0
        elif func==7:
            if True:
                tmp = np.exp(connection0, dtype="float64")
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.asarray(np.exp(connection0), dtype="uint8") # inf => 0
        # SQRT connection0
        elif func==8:
            return np.asarray(np.sqrt(connection0), dtype="uint8")
        # ADDC connection0 parameter0
        elif func==9:
            if True:
                tmp = np.int16(connection0) + parameter0
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.asarray(connection0 + parameter0, dtype="uint8") # with overflow modulo
        # SUBC connection0 parameter0
        elif func==10:
            if True:
                tmp = np.int16(connection0) - parameter0
                tmp[tmp<0] = 0
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.asarray(connection0 - parameter0, dtype="uint8") # with overflow modulo
        # MULLC connection0 parameter0
        elif func==11:
            if True:
                tmp = np.int16(connection0) * parameter0
                tmp[tmp>255] = 255
                return np.asarray(tmp,dtype="uint8")
            else:
                return np.asarray(connection0 * parameter0, dtype="uint8") # with overflow modulo
        # DILATE connection0
        elif func==12:
            return cv2.dilate(connection0,cls.ksize)
        # ERODE connection0
        elif func==13:
            return cv2.erode(connection0,cls.ksize)
        # LAPLACE connection0
        elif func==14:
            return cv2.Laplacian(connection0,cv2.CV_8U,cls.ksize)
        # CANNY connection0
        elif func==15:
            return cv2.Canny(connection0,100,200) # FIXED MIN MAX VALUES ?
        # GAUSS
        elif func==16:
            return cv2.GaussianBlur(connection0,cls.ksize,0)
        # GAUSS2 parameter1 parameter2
        elif func==17:
            x = abs(parameter1)
            y = abs(parameter2)
            if x%2==0:
                x = x + 1
            if y%2==0:
                y = y + 1
            return cv2.GaussianBlur(connection0,(abs(x),abs(y)),0)
        # MIN connection0 connection1
        elif func==18:
            return np.minimum(connection0,connection1)
        # MAX connection0 connection1
        elif func==19:
            return np.maximum(connection0,connection1)
        # AVG connection0 connection1
        elif func==20:
            return np.asarray(connection0/2+connection1/2, dtype="uint8")
        # ABSDIFFERENCE connection0 connection1
        elif func==21:
            return np.abs(connection0-connection1)
        # MINC connection0 parameter0
        elif func==22:
            return np.minimum(connection0,np.full(connection0.shape,fill_value=abs(parameter0),dtype="uint8"))
        # MAXC connection0 parameter0
        elif func==23:
            return np.maximum(connection0,np.full(connection0.shape,fill_value=abs(parameter0),dtype="uint8"))
        # NORMALIZE
        elif func==24:
            # Normalised [0,255] as integer
            return (255*(connection0 - np.min(connection0))/max(1,np.max(connection0))).astype("uint8")
        # SOBEL
        elif func==25:
            return cv2.Sobel(connection0,cv2.CV_8U,1,1,cls.ksize)
        # SOBELX
        elif func==26:
            return cv2.Sobel(connection0,cv2.CV_8U,1,0,cls.ksize)
        # SOBELY
        elif func==27:
            return cv2.Sobel(connection0,cv2.CV_8U,0,1,cls.ksize)
        # THRESHOLD connection0 parameter0
        elif func==28:
            retval, dst = cv2.threshold(connection0,abs(parameter0),0,cv2.THRESH_TRUNC)
            return dst
        # SMOOTHMEDIAN
        elif func==29:
            return cv2.medianBlur(connection0,cls.ksize[0])
        # SMOOTHBILATERAL
        elif func==30:
            return cv2.bilateralFilter(connection0,parameter1,parameter2,parameter2)
        # SMOOTHBLUR
        elif func==31:
            return cv2.blur(connection0,cls.ksize)
        # UNSHARPEN
        elif func==32:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(connection0, -1, kernel)
        # SHIFT
        elif func==33:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,parameter1], [0,1,parameter2] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # SHIFTUP
        elif func==34:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,0], [0,1,-1] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # SHIFTDOWN
        elif func==35:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,0], [0,1,1] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # SHIFTLEFT
        elif func==36:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,-1], [0,1,0] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # SHIFTRIGHT
        elif func==37:
            num_rows, num_cols = connection0.shape[:2]
            translation_matrix = np.float32([ [1,0,1], [0,1,0] ])
            return cv2.warpAffine(connection0, translation_matrix, (num_cols, num_rows))
        # RESCALE parameter0
        elif func==38:
            height, width = connection0.shape[:2] # TO CHECK +1 to only reduce
            return cv2.resize(cv2.resize(connection0, (int(width/(abs(parameter0)+1)), int(height/(abs(parameter0)+1))), interpolation = cv2.INTER_AREA) , (width, height), interpolation = cv2.INTER_AREA) 
        # GABOR
        elif func==39:
            g_kernel = cv2.getGaborKernel(cls.ksize, gabor_filter_frequency, gabor_filter_orientation, 10.0, 0.5) # CHECK 10 and 0.5 VALUES
            return cv2.filter2D(connection0, -1, g_kernel)
        # RESIZETHENGABOR
        elif func==40:
            height, width = connection0.shape[:2] # TO CHECK +1 to only reduce
            tmp = cv2.resize(cv2.resize(connection0, (int(width/(abs(parameter0)+1)), int(height/(abs(parameter0)+1))), interpolation = cv2.INTER_AREA) , (width, height), interpolation = cv2.INTER_AREA) 
            g_kernel = cv2.getGaborKernel(cls.ksize, gabor_filter_frequency, gabor_filter_orientation, 10.0, 0.5) # CHECK 10 and 0.5 VALUES
            return cv2.filter2D(tmp, -1, g_kernel)
        # MINVALUE
        elif func==41:
            return np.full(connection0.shape,fill_value=connection0.min(),dtype="uint8")
        # MAXVALUE
        elif func==42:
            return np.full(connection0.shape,fill_value=connection0.max(),dtype="uint8")
        # AVGVALUE
        elif func==43:
            return np.full(connection0.shape,fill_value=connection0.mean(),dtype="uint8")
        # LOCALMIN parameter1 parameter2
        elif func==44:
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            if x==0:
                x = 1
            if y==0:
                y = 1

            tmp = np.full((height+2*y,width+2*x),fill_value=255,dtype="uint8")
            tmp[y:height+y,x:width+x] = connection0

            pool = Pool(processes=cls.nb_processes,initializer=init_pool_mmmn, initargs=(tmp,y,x))

            indexes = []

            for i in range(y,height+y):
                for j in range(x,width+x):
                    indexes.append((i,j))

            res = np.array(pool.map(localmin,indexes))

            pool.close()

            return res.reshape((height,width))
        # LOCALMAX parameter1 parameter2
        elif func==45:
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            if x==0:
                x = 1
            if y==0:
                y = 1

            tmp = np.full((height+2*y,width+2*x),fill_value=0,dtype="uint8")
            tmp[y:height+y,x:width+x] = connection0

            pool = Pool(processes=cls.nb_processes,initializer=init_pool_mmmn, initargs=(tmp,y,x))

            indexes = []

            for i in range(y,height+y):
                for j in range(x,width+x):
                    indexes.append((i,j))

            res = np.array(pool.map(localmax,indexes))

            pool.close()

            return res.reshape((height,width))
        # LOCALAVG parameter1 parameter2
        elif func==46:
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            if x==0:
                x = 1
            if y==0:
                y = 1

            tmp = np.copy(connection0)

            pool = Pool(processes=cls.nb_processes,initializer=init_pool_mmmn, initargs=(connection0,y,x))

            indexes = []

            for i in range(y,height-y):
                for j in range(x,width-x):
                    indexes.append((i,j))

            tmp[y:height-y,x:width-x] = np.array(pool.map(localmean,indexes)).reshape((height-2*y,width-2*x))

            pool.close()

            return tmp
        # LOCALNORMALIZE parameter1 parameter2
        elif func==47:
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            if x==0:
                x = 1
            if y==0:
                y = 1

            tmp = np.copy(connection0)

            pool = Pool(processes=cls.nb_processes,initializer=init_pool_mmmn, initargs=(connection0,y,x))

            indexes = []

            for i in range(y,height-y):
                for j in range(x,width-x):
                    indexes.append((i,j))

            tmp[y:height-y,x:width-x] = np.array(pool.map(localnormalize,indexes)).reshape((height-2*y,width-2*x))

            pool.close()

            return tmp
