from .chromosome import Chromosome, ChromosomeProcess
import copy
import random
from multiprocessing import Queue, Process

class IslandProcess(Process):
    def __init__(self,island,input_data,output_data,queue):
        super(IslandProcess, self).__init__()
        self.island = island
        self.input_data = input_data
        self.output_data = output_data
        self.queue = queue

    def run(self):
        self.queue.put(self.island.updateFitness(self.input_data,self.output_data))

class Island:

    def __init__(self,functions,chromosome,num_inputs,num_outputs,graph_length,mutation_rate,insertion_rate,deletion_rate,insertion_nop,num_indiv,fitnessFunction,mutationFunction,calculate_parent_fitness):
        self.functions = functions
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph_length = graph_length
        self.mutation_rate = mutation_rate
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self.insertion_nop = insertion_nop
        self.emu = 1
        self.elambda = num_indiv - self.emu
        self.fitnessFunction = fitnessFunction
        self.mutationFunction = mutationFunction
        self.calculate_parent_fitness = calculate_parent_fitness
        self.parent = Chromosome(self.num_inputs,self.num_outputs,self.graph_length,self.fitnessFunction,self.functions)
        self.childs = []
        self.best_chromosome = None
        self.processes = []
        self.queues = []
        if chromosome==None:
            self.parent.random()
        else:
            self.parent = copy.deepcopy(chromosome)

        for i in range(0,self.elambda):
            self.childs.append(Chromosome(self.num_inputs,self.num_outputs,self.graph_length,self.fitnessFunction,self.functions))

            if chromosome==None:
                self.childs[i].random()
            else:
                self.childs[i] = copy.deepcopy(chromosome)

    def updateParentFitness(self,input_data,output_data):
        self.parent.calculateFitness(input_data,output_data,False)

    def updateFitness(self,input_data,output_data):
        if self.calculate_parent_fitness:
            self.updateParentFitness(input_data,output_data)

        for i in range(0,self.elambda):
            self.childs[i].calculateFitness(input_data,output_data,False)

        return self.setBestChromosome()

    def updateFitnessChromosome(self,input_data,output_data):
        if self.calculate_parent_fitness:
            q = Queue()
            c = ChromosomeProcess(self.parent,input_data,output_data,q)
            c.start()
            self.processes.append(c)
            self.queues.append(q)

        for i in range(0,self.elambda):
            q = Queue()
            c = ChromosomeProcess(self.childs[i],input_data,output_data,q)
            c.start()
            self.processes.append(c)
            self.queues.append(q)

    def waitForUpdateFitnessChromosome(self):
        for i in range(0,len(self.processes)):
            self.processes[i].join()

        if self.calculate_parent_fitness:
            fitness, duration = self.queues[0].get()
            self.parent.setFitness(fitness)
            self.parent.setDuration(duration)

            for i in range(1,len(self.processes)):
                fitness, duration = self.queues[i].get()
                self.childs[i-1].setFitness(fitness)
                self.childs[i-1].setDuration(duration)
        else:
            for i in range(0,len(self.processes)):
                fitness, duration = self.queues[i].get()
                self.childs[i].setFitness(fitness)
                self.childs[i].setDuration(duration)

        self.setBestChromosome()

        self.processes = []
        self.queues = []

    def updateFitnessIsland(self,input_data,output_data):
        q = Queue()
        i = IslandProcess(self,input_data,output_data,q)
        i.start()

        self.process = i
        self.queue = q

    def waitForUpdateFitnessIsland(self):
        self.process.join()

        index, fitness = self.queue.get()
        if index>=0:
            self.updateBestChromosomeWithChild(index, fitness)
        else:
            self.updateBestChromosome(self.parent)

        self.process = None
        self.queue = None

    def setBestChromosome(self):
        self.best_chromosome = self.parent
        best = -1
        fitness = self.parent.getFitness()
        parent = True
        for i in range(0,self.elambda):
            if (parent and self.childs[i].getFitness()<self.best_chromosome.getFitness()) or (self.childs[i].getFitness()<self.best_chromosome.getFitness()) or (self.childs[i].getFitness()==self.best_chromosome.getFitness() and (self.childs[i].getDuration()<self.best_chromosome.getDuration())):
                self.best_chromosome = self.childs[i]
                best = i
                fitness = self.childs[i].getFitness()
                parent = False

        return best, fitness

    def doEvolution(self):
        self.parent = copy.deepcopy(self.best_chromosome)

        for i in range(0,self.elambda):
            self.childs[i] = copy.deepcopy(self.parent)
        
        for i in range(0,self.elambda):
            self.childs[i].mutateFunction(self.mutationFunction,self.mutation_rate,self.insertion_rate,self.deletion_rate,self.insertion_nop)
    
    def getBestChromosome(self):
        return self.best_chromosome

    def updateBestChromosome(self,chromosome):
        self.best_chromosome = chromosome

    def updateBestChromosomeWithChild(self,index,fitness):
        self.best_chromosome = self.childs[index]
        self.best_chromosome.setFitness(fitness)
