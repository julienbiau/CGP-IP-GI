from .island import Island, IslandProcess
from .chromosome import Chromosome
import random
import os
import cv2
import numpy as np
import copy
from tqdm import tqdm
from multiprocessing import Queue, Process

class CGPIP:

    COLOR_RGB = 0
    COLOR_GRAYSCALE = 1
    COLOR_HSV = 2
    COLOR_RGBHSV = 3

    def __init__(self, functions, graph_length, mutation_rate, insertion_rate, deletion_rate, insertion_nop, size_of_mutations, num_islands, num_indiv_island, sync_interval_island, max_iterations, chromosomeOptimization, islandOptimization, fitnessFunction, mutationFunction, batch_size):
        self.functions = functions
        self.graph_length = graph_length
        self.mutation_rate = mutation_rate
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self.insertion_nop = insertion_nop
        self.size_of_mutations = size_of_mutations
        self.num_islands = num_islands
        self.islands = []
        self.num_indiv_island = num_indiv_island
        self.sync_interval_island = sync_interval_island
        self.max_iterations = max_iterations
        self.num_run = 0
        self.inputs = None
        self.outputs = None
        self.num_inputs = 0
        self.num_outputs = 0
        self.chromosome = None
        self.input_shape = None
        self.output_shape = None
        self.batch_size = batch_size
        self.nb_batch = 0
        self.chromosomeOptimization = chromosomeOptimization
        self.islandOptimization = islandOptimization
        self.fitnessFunction = fitnessFunction
        self.mutationFunction = mutationFunction
        self.best_chromosome = None
        self.logs = []
        #np.seterr(all='ignore')

    def load_data(self,input_data, output_data, num_inputs, num_outputs):
        self.inputs = input_data

        self.outputs = output_data

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.data_loaded = True

        self.nb_batch = len(self.inputs) / self.batch_size

        assert self.nb_batch * self.batch_size == len(self.inputs), 'incompatible batch_size'

    def load_input_data(self, input_type, max_el, *args):
        nb_inputs = len(args)

        self.input_type = input_type

        inputs = []
        filenames = []
        
        image_type = cv2.IMREAD_COLOR
        if input_type==self.COLOR_GRAYSCALE:
            image_type = cv2.IMREAD_GRAYSCALE

        for i in range(0,nb_inputs):
            filenames.append(sorted(os.listdir(args[i])))
        
        for i in tqdm(range(max_el)):
            tmp = []
            for j in range(0,nb_inputs):
                input = cv2.imread(args[j]+"/"+filenames[j][i], image_type)

                if (input_type==self.COLOR_RGB or input_type==self.COLOR_RGBHSV):
                    tmp.append(np.asarray(input[:,:,0],dtype="uint8"))
                    tmp.append(np.asarray(input[:,:,1],dtype="uint8"))
                    tmp.append(np.asarray(input[:,:,2],dtype="uint8"))
                    if self.input_shape==None:
                        self.input_shape = input[:,:,0].shape
                    else:
                        assert self.input_shape == input[:,:,0].shape
                elif (input_type==self.COLOR_GRAYSCALE):
                    tmp.append(np.asarray(input,dtype="uint8"))
                    if self.input_shape==None:
                        self.input_shape = input.shape
                    else:
                        assert self.input_shape == input.shape

                if (input_type==self.COLOR_HSV or input_type==self.COLOR_RGBHSV):
                    input = cv2.cvtColor(input,cv2.COLOR_RGB2HSV)
                    tmp.append(np.asarray(input[:,:,0],dtype="uint8"))
                    tmp.append(np.asarray(input[:,:,1],dtype="uint8"))
                    tmp.append(np.asarray(input[:,:,2],dtype="uint8"))
                    if self.input_shape==None:
                        self.input_shape = input[:,:,0].shape
                    else:
                        assert self.input_shape == input[:,:,0].shape

            inputs.append(tmp)

        self.inputs = inputs

        if input_type==self.COLOR_GRAYSCALE:
            self.num_inputs = nb_inputs
        elif input_type==self.COLOR_RGB or input_type==self.COLOR_HSV:
            self.num_inputs = 3 * nb_inputs
        else:
            self.num_inputs = 6 * nb_inputs

        self.nb_batch = len(self.inputs) / self.batch_size

        assert self.nb_batch * self.batch_size == len(self.inputs), 'incompatible batch_size'

        return len(self.inputs)

    def load_output_data(self, output_type, max_el, *args):
        assert output_type==self.COLOR_GRAYSCALE, 'invalid output type'

        nb_outputs = len(args)

        assert nb_outputs==1, 'only 1 output supported'

        self.output_type = output_type

        outputs = []
        filenames = []
        
        image_type = cv2.IMREAD_COLOR
        if output_type==self.COLOR_GRAYSCALE:
            image_type = cv2.IMREAD_GRAYSCALE

        for i in range(0,nb_outputs):
            filenames.append(sorted(os.listdir(args[i])))
        
        for i in tqdm(range(max_el)):
            tmp = []
            for j in range(0,nb_outputs):
                output = cv2.imread(args[j]+"/"+filenames[j][i], image_type)

                if (output_type==self.COLOR_RGB or output_type==self.COLOR_RGBHSV):
                    tmp.append(np.asarray(output[:,:,0],dtype="uint8"))
                    tmp.append(np.asarray(output[:,:,1],dtype="uint8"))
                    tmp.append(np.asarray(output[:,:,2],dtype="uint8"))
                    if self.output_shape==None:
                        self.output_shape = output[:,:,0].shape
                    else:
                        assert self.input_shape == output[:,:,0].shape

                elif (output_type==self.COLOR_GRAYSCALE):
                    tmp.append(np.asarray(output,dtype="uint8"))
                    if self.output_shape==None:
                        self.output_shape = output.shape
                    else:
                        assert self.input_shape == output.shape

                if (output_type==self.COLOR_HSV or output_type==self.COLOR_RGBHSV):
                    output = cv2.cvtColor(output,cv2.COLOR_RGB2HSV)
                    tmp.append(np.asarray(output[:,:,0],dtype="uint8"))
                    tmp.append(np.asarray(output[:,:,1],dtype="uint8"))
                    tmp.append(np.asarray(output[:,:,2],dtype="uint8"))
                    if self.output_shape==None:
                        self.output_shape = output[:,:,0].shape
                    else:
                        assert self.input_shape == output[:,:,0].shape

            outputs.append(tmp)

        self.outputs = outputs

        if output_type==self.COLOR_GRAYSCALE:
            self.num_outputs = nb_outputs
        elif output_type==self.COLOR_RGB or output_type==self.COLOR_HSV:
            self.num_outputs = 3 * nb_outputs
        else:
            self.num_outputs = 6 * nb_outputs

        self.data_loaded = True

        return len(self.outputs)

    def getInputs(self):
        nb = int(self.num_run % self.nb_batch)

        return self.inputs[nb*self.batch_size:(nb+1)*self.batch_size]

    def getOutputs(self):
        nb = int(self.num_run % self.nb_batch)

        return self.outputs[nb*self.batch_size:(nb+1)*self.batch_size]

    def getAllInputs(self):
        return self.inputs

    def getAllOutputs(self):
        return self.outputs

    def getNumInputs(self):
        return self.num_inputs

    def getNumOutputs(self):
        return self.num_outputs

    def getInputShape(self):
        return self.input_shape

    def getOutputShape(self):
        return self.output_shape

    def getInputType(self):
        return self.input_type

    def getOutputType(self):
        return self.output_type

    def load_chromosome(self,filename):
        self.chromosome = Chromosome(0,0,0,self.fitnessFunction,self.functions)
        self.chromosome.fromFile(filename)

    def set_parent_chromosome(self,chromosome):
        self.chromosome = chromosome

    def get_best_chromosome(self):
        return self.best_chromosome

    def get_logs(self):
        return self.logs

    def run(self):
        if not self.data_loaded:
            # load data
            print("Load data first")

        if self.islandOptimization==True:
            print("Island optimization")
        elif self.chromosomeOptimization==True:
            print("Chromosome optimization")
        
        if self.chromosome!=None:
            print("Chromosome fitness: "+str(self.chromosome.calculateFitness(self.inputs,self.outputs,True)))

        for i in range(0,self.num_islands):
            # create island
            island = Island(self.functions,self.chromosome,self.num_inputs,self.num_outputs,self.graph_length,self.mutation_rate,self.insertion_rate,self.deletion_rate,self.insertion_nop,self.num_indiv_island,self.fitnessFunction,self.mutationFunction,self.nb_batch>1)
            self.islands.append(island)
            if self.nb_batch==1:
                island.updateParentFitness(self.inputs,self.outputs)

        print("islands created")

        for i in range(0, self.max_iterations):

            if self.islandOptimization==True:
                for j in range(0,self.num_islands):
                    self.islands[j].updateFitnessIsland(self.getInputs(),self.getOutputs())

                for j in range(0,self.num_islands):
                    self.islands[j].waitForUpdateFitnessIsland()

                    if self.num_run % 5 == 0:
                        print("Island "+str(j)+" iterations "+str(self.num_run)+" fitness: "+str(self.islands[j].getBestChromosome().getFitness())+" active nodes: "+str(self.islands[j].getBestChromosome().getNbActiveNodes()))
            elif self.chromosomeOptimization==True:
                for j in range(0,self.num_islands):
                    self.islands[j].updateFitnessChromosome(self.getInputs(),self.getOutputs())

                for j in range(0,self.num_islands):
                    self.islands[j].waitForUpdateFitnessChromosome()

                    if self.num_run % 5 == 0:
                        print("Island "+str(j)+" iterations "+str(self.num_run)+" fitness: "+str(self.islands[j].getBestChromosome().getFitness())+" active nodes: "+str(self.islands[j].getBestChromosome().getNbActiveNodes())+" duration: "+str(int(1000*self.islands[j].getBestChromosome().getDuration()/self.batch_size))+" ms")
            else:
                for j in range(0,self.num_islands):
                    self.islands[j].updateFitness(self.getInputs(),self.getOutputs())

                    if self.num_run % 5 == 0:
                        print("Island "+str(j)+" iterations "+str(self.num_run)+" fitness: "+str(self.islands[j].getBestChromosome().getFitness())+" active nodes: "+str(self.islands[j].getBestChromosome().getNbActiveNodes()))

            self.num_run = self.num_run + 1

            if self.sync_interval_island>0 and self.num_run % self.sync_interval_island == 0:
                islands_best = []
                # update all island with best chromosome

                if self.nb_batch!=1:
                    if self.islandOptimization==True:
                        for j in range(0,self.num_islands):
                            self.islands[j].updateFitnessIsland(self.getAllInputs(),self.getAllOutputs())

                        for j in range(0,self.num_islands):
                            self.islands[j].waitForUpdateFitnessIsland()

                    elif self.chromosomeOptimization==True:
                        for j in range(0,self.num_islands):
                            self.islands[j].updateFitnessChromosome(self.getAllInputs(),self.getAllOutputs())

                        for j in range(0,self.num_islands):
                            self.islands[j].waitForUpdateFitnessChromosome()

                    else:
                        for j in range(0,self.num_islands):
                            self.islands[j].updateFitness(self.getAllInputs(),self.getAllOutputs())

                for j in range(0,self.num_islands):
                    islands_best.append(self.islands[j].getBestChromosome())

                self.best_chromosome = islands_best[0]

                for j in range(1,self.num_islands):
                    if self.best_chromosome.getFitness()>islands_best[j].getFitness():
                        self.best_chromosome = islands_best[j]

                print("Fitness: "+str(self.best_chromosome.getFitness()))

                self.logs.append({"iteration":self.num_run,"accuracy":self.best_chromosome.getFitness(),"nb_actives_nodes":self.best_chromosome.getNbActiveNodes()})

                for j in range(0,self.num_islands):
                    self.islands[j].updateBestChromosome(self.best_chromosome)

            for j in range(0,self.num_islands):
                self.islands[j].doEvolution()

        islands_best = []
        # update all island with best chromosome
        for j in range(0,self.num_islands):
            islands_best.append(self.islands[j].getBestChromosome())

        self.best_chromosome = islands_best[0]
        
        for j in range(1,self.num_islands):
            if self.best_chromosome.getFitness()>islands_best[j].getFitness():
                self.best_chromosome = islands_best[j]

        print("Chromosome fitness: "+str(self.best_chromosome.calculateFitness(self.inputs,self.outputs,True)))
