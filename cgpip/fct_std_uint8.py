import random
import numpy as np
import cv2
import sys
import math
from multiprocessing import Pool
from numpy.lib.stride_tricks import as_strided

def np_localnormalize(a,y,x):
    tmp = (a[y,x] - a.min())
    tmp = tmp/max(1,a.max())
    return (255*(tmp)).astype("uint8")

class STD_UINT8:

    # Function Int
    # Connection 0 Int
    # Connection 1 Int
    # Parameter 0 Real no limitation
    # Parameter 1 Int [−16, +16]
    # Parameter 2 Int [−16, +16]
    # Gabor Filter Frequ. Int [0, 16]
    # Gabor Filter Orient. Int [−8, +8]

    num_functions = 47
    ksize = (3,3)
    nb_processes = 16

    two_arguments = [False,False,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False]

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
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                return cv2.dilate(connection0,(size,size))
            else:
                return cv2.dilate(connection0,cls.ksize)
        # ERODE connection0
        elif func==13:
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                return cv2.erode(connection0,(size,size))
            else:
                return cv2.erode(connection0,cls.ksize)
        # LAPLACE connection0
        elif func==14:
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                return cv2.Laplacian(connection0,cv2.CV_8U,(size,size))
            else:
                return cv2.Laplacian(connection0,cv2.CV_8U,cls.ksize)
        # CANNY connection0
        elif func==15:
            return cv2.Canny(connection0,100,200) # FIXED MIN MAX VALUES ?
        # GAUSS
        elif func==16:
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                return cv2.GaussianBlur(connection0,(size,size),0)
            else:
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
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                return cv2.Sobel(connection0,cv2.CV_8U,1,1,(size,size))
            else:
                return cv2.Sobel(connection0,cv2.CV_8U,1,1,cls.ksize)
        # SOBELX
        elif func==26:
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                return cv2.Sobel(connection0,cv2.CV_8U,1,0,(size,size))
            else:
                return cv2.Sobel(connection0,cv2.CV_8U,1,0,cls.ksize)
        # SOBELY
        elif func==27:
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                return cv2.Sobel(connection0,cv2.CV_8U,0,1,(size,size))
            else:
                return cv2.Sobel(connection0,cv2.CV_8U,0,1,cls.ksize)
        # THRESHOLD connection0 parameter0
        elif func==28:
            retval, dst = cv2.threshold(connection0,abs(parameter0),255,cv2.THRESH_BINARY)
            return dst
        # SMOOTHMEDIAN
        elif func==29:
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                return cv2.medianBlur(connection0,size)
            else:
                return cv2.medianBlur(connection0,cls.ksize[0])
        # SMOOTHBILATERAL
        elif func==30:
            return cv2.bilateralFilter(connection0,parameter1,parameter2,parameter2)
        # SMOOTHBLUR
        elif func==31:
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                return cv2.blur(connection0,(size,size))
            else:            
                return cv2.blur(connection0,cls.ksize)
        # UNSHARPEN
        elif func==32:
            if True:
                size = abs(parameter1)
                if size%2==0:
                    size = size + 1
                gauss = cv2.GaussianBlur(connection0,(size,size),0)
                return cv2.addWeighted(connection0, 2, gauss, -1, 0)
            else:
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

            stride = 1

            kernel_size = (2*y+1,2*x+1)

            # Padding
            A = np.pad(connection0, ((y,y),(x,x)), mode='constant', constant_values=255)

            # Window view of A
            output_shape = ((A.shape[0] - kernel_size[0])//stride+1,
                            (A.shape[1] - kernel_size[1])//stride+1)

            A_w = as_strided(A, shape = output_shape + kernel_size, 
                                strides = (stride*A.strides[0],
                                        stride*A.strides[1]) + A.strides)

            A_w = A_w.reshape(-1, *kernel_size)

            return A_w.min(axis=(1,2)).reshape(connection0.shape)
        # LOCALMAX parameter1 parameter2
        elif func==45:
            height, width = connection0.shape[:2]
            x = abs(parameter1)
            y = abs(parameter2)
            if x==0:
                x = 1
            if y==0:
                y = 1

            stride = 1

            kernel_size = (2*y+1,2*x+1)

            # Padding
            A = np.pad(connection0, ((y,y),(x,x)), mode='constant', constant_values=0)

            # Window view of A
            output_shape = ((A.shape[0] - kernel_size[0])//stride+1,
                            (A.shape[1] - kernel_size[1])//stride+1)

            A_w = as_strided(A, shape = output_shape + kernel_size, 
                                strides = (stride*A.strides[0],
                                        stride*A.strides[1]) + A.strides)

            A_w = A_w.reshape(-1, *kernel_size)

            return A_w.max(axis=(1,2)).reshape(connection0.shape)
        # LOCALAVG parameter1 parameter2
        elif func==46:
                height, width = connection0.shape[:2]
                x = abs(parameter1)
                y = abs(parameter2)
                if x==0:
                    x = 1
                if y==0:
                    y = 1

                stride = 1

                kernel_size = (2*y+1,2*x+1)

                tmp = np.copy(connection0)

                # Window view of A
                output_shape = ((connection0.shape[0] - kernel_size[0])//stride+1,
                                (connection0.shape[1] - kernel_size[1])//stride+1)

                A_w = as_strided(connection0, shape = output_shape + kernel_size, 
                                    strides = (stride*connection0.strides[0],
                                            stride*connection0.strides[1]) + connection0.strides)

                A_w = A_w.reshape(-1, *kernel_size)

                tmp[y:-y,x:-x] = A_w.mean(axis=(1,2)).reshape(output_shape)

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

            stride = 1

            kernel_size = (2*y+1,2*x+1)

            tmp = np.copy(connection0)

            # Window view of A
            output_shape = ((connection0.shape[0] - kernel_size[0])//stride+1,
                            (connection0.shape[1] - kernel_size[1])//stride+1)

            A_w = as_strided(connection0, shape = output_shape + kernel_size, 
                                strides = (stride*connection0.strides[0],
                                        stride*connection0.strides[1]) + connection0.strides)

            A_w = A_w.reshape(-1, *kernel_size)

            tmp[y:-y,x:-x] = np.array([np_localnormalize(xi,2*y,2*x) for xi in A_w]).reshape(output_shape)

            return tmp
