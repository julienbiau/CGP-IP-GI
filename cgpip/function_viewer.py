import matplotlib.pyplot as plt

from functions import Functions
from fct_std_uint8 import STD_UINT8
#from supp_functions import SuppFunctions

from skimage import data
import time

class FunctionViewer(object):
    def __init__(self,functions):
        self.function_bundle = functions
        self.num_functions = Functions.getNbFunction()
        
    def run(self, img0, img1, p0, p1, p2, p3, p4):
        fig, axs = plt.subplots(self.num_functions, 3, figsize=(15, self.num_functions*3))
        for i, function in enumerate(range(1,self.num_functions)):
            start_time = time.time()
            img_out = self.function_bundle.execute(function, img0, img1, p0, p1, p2, p3, p4)
            print(function)
            print(img_out.shape)
            print(time.time()-start_time)
            axs[i, 0].imshow(img0, cmap=plt.get_cmap('gray'))
            axs[i, 1].imshow(img1, cmap=plt.get_cmap('gray'))
            axs[i, 2].imshow(img_out, cmap=plt.get_cmap('gray'))
            axs[i, 1].set_title('Function ' + str(function) + ', p0=' + str(p0)+ ', p1=' + str(p1)+ ', p2=' + str(p2)+ ', p0=' + str(p2)+ ', p3=' + str(p3))
        print('done')
        plt.axis('off')
        plt.savefig('test.png', dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    Functions.add(STD_UINT8)
    src_img = data.astronaut()
    img0 = src_img[:, :, 0]
    img1 = src_img[:, :, 1]
    FunctionViewer(Functions).run(img0, img1, 8, 8, 8, 0, 0)