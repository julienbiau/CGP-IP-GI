
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

    if len(sys.argv)==1:
        exit()
    elif sys.argv[1]=='display':
        if os.path.exists(sys.argv[2]):
            chromosome = Chromosome(0,0,0,Chromosome.FITNESS_MCC,Functions)
            chromosome.fromFile(sys.argv[2])

            chromosome.print()
            chromosome.printGraph()

            cgp = CGPIP(Functions,graph_length,mutation_rate,insertion_rate,deletion_rate,insertion_nop,size_mutation,num_islands,num_indiv,sync_interval_island,max_iterations,True,False,Chromosome.FITNESS_MCC,Chromosome.GOLDMAN_MUTATE,batch_size)

            cgp.load_input_data(cgp.COLOR_RGBHSV,max_element,'../CGP-IP-DATA/lunar/images/render')
            cgp.load_output_data(cgp.COLOR_GRAYSCALE,max_element,'../CGP-IP-DATA/lunar/images/clean')

            inputs = cgp.getAllInputs()
            outputs = cgp.getAllOutputs()

            for i in range(0,len(inputs)):
                print(chromosome.calculateFitness([inputs[i]],[outputs[i]],True))

                width, height = cgp.getInputShape()

                if cgp.getInputType()==cgp.COLOR_GRAYSCALE:
                    for j in range(0,cgp.getNumInputs()):
                        image = cv2.cvtColor(inputs[i][j], cv2.COLOR_GRAY2BGR )
                        cv2.imshow("input "+str(j), image)
                elif cgp.getInputType()==cgp.COLOR_RGB:
                    for j in range(0,int(cgp.getNumInputs()/3)):
                        image = np.zeros([width,height,3],dtype=np.uint8)
                        image[:,:,0] = np.int8(inputs[i][3*j+0])
                        image[:,:,1] = np.int8(inputs[i][3*j+1])
                        image[:,:,2] = np.int8(inputs[i][3*j+2])
                        cv2.imshow("input "+str(j), image)
                elif cgp.getInputType()==cgp.COLOR_RGBHSV:
                    for j in range(0,int(cgp.getNumInputs()/6)):
                        image = np.zeros([width,height,3],dtype=np.uint8)
                        image[:,:,0] = np.int8(inputs[i][3*j+0])
                        image[:,:,1] = np.int8(inputs[i][3*j+1])
                        image[:,:,2] = np.int8(inputs[i][3*j+2])
                        cv2.imshow("input "+str(j), image)
                elif cgp.getInputType()==cgp.COLOR_HSV:
                    for j in range(0,int(cgp.getNumInputs()/3)):
                        image = np.zeros([width,height,3],dtype=np.uint8)
                        image[:,:,0] = np.int8(inputs[i][3*j+0])
                        image[:,:,1] = np.int8(inputs[i][3*j+1])
                        image[:,:,2] = np.int8(inputs[i][3*j+2])
                        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR )
                        cv2.imshow("input "+str(j), image)

                if cgp.getOutputType()==cgp.COLOR_GRAYSCALE:
                    for j in range(0,cgp.getNumOutputs()):
                        image = cv2.cvtColor(outputs[i][j], cv2.COLOR_GRAY2BGR )
                        cv2.imshow("output "+str(j), image)

                    output_values = chromosome.getOutputValues()
                    for j in range(0,cgp.getNumOutputs()):
                        image = cv2.cvtColor(output_values[j], cv2.COLOR_GRAY2BGR )
                        cv2.imshow("result "+str(j), image)

                c = cv2.waitKey(0) & 0xff
                if c == ord('q'):
                    break
    elif sys.argv[1]=='explain':
        if os.path.exists(sys.argv[2]):
            chromosome = Chromosome(0,0,0,Chromosome.FITNESS_MCC,Functions)
            chromosome.fromFile(sys.argv[2])

            chromosome.print()
            chromosome.printGraph()

            cgp = CGPIP(Functions,graph_length,mutation_rate,insertion_rate,deletion_rate,insertion_nop,size_mutation,num_islands,num_indiv,sync_interval_island,max_iterations,True,False,Chromosome.FITNESS_MCC,Chromosome.GOLDMAN_MUTATE,batch_size)

            cgp.load_input_data(cgp.COLOR_RGBHSV,max_element,'./tata_blob_min/input1','./tata_blob_min/input2')
            cgp.load_output_data(cgp.COLOR_GRAYSCALE,max_element,'./tata_blob_min/output')

            inputs = cgp.getAllInputs()
            outputs = cgp.getAllOutputs()

            for i in range(0,len(inputs)):
                print(chromosome.calculateFitness([inputs[i]],[outputs[i]],True))

                width, height = cgp.getInputShape()

                if cgp.getInputType()==cgp.COLOR_GRAYSCALE:
                    for j in range(0,cgp.getNumInputs()):
                        image = cv2.cvtColor(inputs[i][j], cv2.COLOR_GRAY2BGR )
                        cv2.imshow("input "+str(j), image)
                elif cgp.getInputType()==cgp.COLOR_RGB:
                    for j in range(0,int(cgp.getNumInputs()/3)):
                        image = np.zeros([width,height,3],dtype=np.uint8)
                        image[:,:,0] = np.int8(inputs[i][3*j+0])
                        image[:,:,1] = np.int8(inputs[i][3*j+1])
                        image[:,:,2] = np.int8(inputs[i][3*j+2])
                        cv2.imshow("input "+str(j), image)
                elif cgp.getInputType()==cgp.COLOR_RGBHSV:
                    for j in range(0,int(cgp.getNumInputs()/6)):
                        image = np.zeros([width,height,3],dtype=np.uint8)
                        image[:,:,0] = np.int8(inputs[i][3*j+0])
                        image[:,:,1] = np.int8(inputs[i][3*j+1])
                        image[:,:,2] = np.int8(inputs[i][3*j+2])
                        cv2.imshow("input "+str(j), image)
                elif cgp.getInputType()==cgp.COLOR_HSV:
                    for j in range(0,int(cgp.getNumInputs()/3)):
                        image = np.zeros([width,height,3],dtype=np.uint8)
                        image[:,:,0] = np.int8(inputs[i][3*j+0])
                        image[:,:,1] = np.int8(inputs[i][3*j+1])
                        image[:,:,2] = np.int8(inputs[i][3*j+2])
                        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR )
                        cv2.imshow("input "+str(j), image)

                if cgp.getOutputType()==cgp.COLOR_GRAYSCALE:
                    for j in range(0,cgp.getNumOutputs()):
                        image = cv2.cvtColor(outputs[i][j], cv2.COLOR_GRAY2BGR )
                        cv2.imshow("output "+str(j), image)

                    output_values = chromosome.getOutputValues()
                    for j in range(0,cgp.getNumOutputs()):
                        image = cv2.cvtColor(output_values[j], cv2.COLOR_GRAY2BGR )
                        cv2.imshow("result "+str(j), image)

                nodes = chromosome.getActiveNodesValues()
                if False:
                    for j in range(0,cgp.getNumOutputs()):
                        for k in nodes[j]:
                            image = cv2.cvtColor(nodes[j][k], cv2.COLOR_GRAY2BGR )
                            cv2.imshow("node "+str(k), image)

                    c = cv2.waitKey(0) & 0xff
                    if c == ord('q'):
                        break
                else:
                    cv2.waitKey(205)

                    for j in range(0,cgp.getNumOutputs()):
                        nb = len(nodes[j])
                        cols = ceil(sqrt(nb))
                        rows = ceil(nb/cols)
                        print(str(cols)+" "+str(rows))
                        fig, axs = plt.subplots(rows, cols)
                        x = 0
                        y = 0
                        for k in nodes[j]:
                            image = cv2.cvtColor(nodes[j][k], cv2.COLOR_GRAY2BGR )
                            axs[y, x].imshow(image, cmap=plt.get_cmap('gray'))
                            axs[y, x].set_title('Node '+str(k))
                            axs[y, x].axis('off')
                            x = x + 1
                            if x >= cols:
                                y = y + 1
                                x = 0
                        #fig.tight_layout()
                        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0.14, hspace=0.14)
                        plt.show(True)

    elif sys.argv[1]=='test':
        chromosome = Chromosome(1,1,50,Chromosome.FITNESS_MCC,Functions)
        chromosome.random()

        input = np.ndarray((512,512),dtype="uint8")
        inputs = [input]

        for i in range(0,50):
            chromosome.setFunctionForNode(i,i+1)
            chromosome.setConnectionsForNode(i,1,1)

        chromosome.print()
        
        chromosome.setOutputNodes([1])

        chromosome.executeChromosome(inputs,True)

    elif sys.argv[1]=='more':
        chromosome = Chromosome(6,3,20,Chromosome.FITNESS_MCC,Functions)
        chromosome.random()

        input = np.ndarray((512,512),dtype="uint8")
        inputs = [input,input,input,input,input,input]

        while True:
            chromosome.print()

            chromosome.insertNode(insertion_nop)

            chromosome.print()

            chromosome.deleteNode()

            chromosome.print()

            chromosome.executeChromosome(inputs,True)

