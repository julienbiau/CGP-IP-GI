
from cgpip import CGPIP
from cgpip import Chromosome
from cgpip import Functions
from cgpip import STD_UINT8
from cgpip import EXT_UINT8
import sys
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import time
from numpy.lib.stride_tricks import as_strided
import random
from math import ceil, sqrt, floor

warnings.filterwarnings("error")

if __name__ == '__main__':

    max_iterations = 300
    size_mutation = 2
    num_islands = 8
    num_indiv = 5
    graph_length = 50
    mutation_rate = 0.05
    insertion_rate = 0.05
    deletion_rate = 0.05
    insertion_nop = False
    sync_interval_island = 100
    batch_size = 10
    max_element = 200

    Functions.add(STD_UINT8)
    Functions.add(EXT_UINT8)
    Functions.setOutsideNbFunctions(3)

    random.seed(5)

    cgp = CGPIP(Functions,graph_length,mutation_rate,insertion_rate,deletion_rate,insertion_nop,size_mutation,num_islands,num_indiv,sync_interval_island,max_iterations,True,False,Chromosome.FITNESS_MCC,Chromosome.MUTATE,batch_size)

    if os.path.exists('./chromo.txt'):
        cgp.load_chromosome('./chromo.txt')

    cgp.load_input_data(cgp.COLOR_RGBHSV,max_element,'../CGP-IP-DATA/lunar/images/render')
    cgp.load_output_data(cgp.COLOR_GRAYSCALE,max_element,'../CGP-IP-DATA/lunar/images/clean')

