import random
import math
import copy
import numpy as np
from multiprocessing import Process, Queue
import time
from tqdm import tqdm

def mutateOrNot(rate):
    if rate!=None and random.random()<rate:
        return True
    else:
        return False

class ChromosomeProcess(Process):
    def __init__(self,chromosome,input_data,output_data,queue):
        super(ChromosomeProcess, self).__init__()
        self.chromosome = chromosome
        self.input_data = input_data
        self.output_data = output_data
        self.queue = queue

    def run(self):
        start_time = time.time()
        fitness = self.chromosome.calculateFitness(self.input_data,self.output_data,False)
        duration = time.time() - start_time
        #print("--- %s seconds ---" % duration)
        self.queue.put((fitness, duration))

class Node:
    # Function Int
    # Connection 0 Int
    # Connection 1 Int
    # Parameter 0 Real no limitation
    # Parameter 1 Int [−16, +16]
    # Parameter 2 Int [−16, +16]
    # Gabor Filter Frequ. Int [0, 16]
    # Gabor Filter Orient. Int [−8, +8]

    def __init__(self,nb_functions,num_inputs,num_outputs,node_index,random_init=True):
        self.nb_functions = nb_functions
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        if random_init:
            self.getRandomFunction()
            self.getRandomConnection0(node_index)
            self.getRandomConnection1(node_index)

            self.getRandomParameter0()
            self.getRandomParameter1()
            self.getRandomParameter2()

            self.getRandomGaborFilterFrequence()
            self.getRandomGaborFilterOrientation()

    def setValues(self,function,connection0,connection1,parameter0,parameter1,parameter2,gaborFilterFrequence,gaborFilterOrientation):
        self.function = function
        self.connection0 = connection0
        self.connection1 = connection1
        self.parameter0 = parameter0
        self.parameter1 = parameter1
        self.parameter2 = parameter2
        self.gaborFilterFrequence = gaborFilterFrequence
        self.gaborFilterOrientation = gaborFilterOrientation

    def setFunction(self,function):
        self.function = function

    def setConnections(self,connection0,connection1):
        self.connection0 = connection0
        self.connection1 = connection1

    def getRandomFunction(self):
        self.function = random.randrange(1,1+self.nb_functions)

    def getRandomConnection0(self,node_index):
        try:
            self.connection0 = random.randrange(1,node_index+self.num_inputs)
        except ValueError:
            self.connection0 = 1

    def getRandomConnection1(self,node_index):
        try:
            self.connection1 = random.randrange(1,node_index+self.num_inputs)
        except ValueError:
            self.connection1 = 1

    def getRandomParameter0(self):
        self.parameter0 = random.uniform(-2^32,2^32)

    def getRandomParameter1(self):
        self.parameter1 = random.randrange(-16, 16, 1)

    def getRandomParameter2(self):
        self.parameter2 = random.randrange(-16, 16, 1)

    def getRandomGaborFilterFrequence(self):
        self.gaborFilterFrequence = random.randrange(0, 16, 1)

    def getRandomGaborFilterOrientation(self):
        self.gaborFilterOrientation = random.randrange(-8, 8, 1)

    def getFunction(self):
        return self.function

    def getConnection0(self):
        return self.connection0

    def getConnection1(self):
        return self.connection1

    def getParameter0(self):
        return self.parameter0

    def getParameter1(self):
        return self.parameter1

    def getParameter2(self):
        return self.parameter2

    def getGaborFilterFrequence(self):
        return self.gaborFilterFrequence

    def getGaborFilterOrientation(self):
        return self.gaborFilterOrientation

    def execute(self,functions,value0,value1):
        return functions.execute(self.function,value0,value1,self.parameter0,self.parameter1,self.parameter2,self.gaborFilterFrequence,self.gaborFilterOrientation)

class Chromosome:

    FITNESS_MEAN_ERROR = 0
    FITNESS_MCC = 1

    MUTATE = 2
    GOLDMAN_MUTATE = 3
    MUTATE_ONLY_PARAMETERS = 4
    GOLDMAN_ONLY_PARAMETERS = 5

    def __init__(self,num_inputs,num_outputs,graph_length,fitnessFunction,functions):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.graph_length = graph_length
        self.output_nodes = []
        self.output_values = []
        self.inputs_index = 0
        self.nodes = []
        self.active_nodes = []
        self.active_nodes_by_output = [None] * num_outputs
        self.nodes_value = []
        self.fitness = None
        self.duration = 0.0
        self.fitnessFunction = fitnessFunction
        self.functions = functions

    def fromFile(self,filename):
        file = open(filename,"r")

        lines = file.readlines()

        line = lines[0].strip('\n')
        self.num_inputs = int(line.split(' ')[0])
        self.num_outputs = int(line.split(' ')[1])
        self.graph_length = int(line.split(' ')[2])
        self.active_nodes_by_output = [None] * self.num_outputs

        for i in range(1,len(lines)):
            line = lines[i].strip('\n')

            data = line.split(' ')
            if len(data)==1:
                self.output_nodes.append(int(data[0]))
            else:
                node = Node(self.functions.getNbFunction(),self.num_inputs,self.num_outputs,i-1,False)
                node.setValues(int(data[0]),int(data[1]),int(data[2]),float(data[3]),int(data[4]),int(data[5]),int(data[6]),int(data[7]))
                self.nodes.append(node)

        file.close()

        self.updateActiveNodes()

    def random(self):
        for i in range(0, self.graph_length):
            self.nodes.append(Node(self.functions.getNbFunction(),self.num_inputs,self.num_outputs,i,True))

        for i in range(0,self.num_outputs):
            self.output_nodes.append(random.randrange(1,self.graph_length))

        self.updateActiveNodes()

    def getOutputValues(self):
        return self.output_values

    def mutateFunction(self,mutate_function,mutation_rate,insertion_rate,deletion_rate,insertion_nop):
        if mutateOrNot(insertion_rate):
            self.insertNode(insertion_nop)
        elif mutateOrNot(deletion_rate):
            self.deleteNode()
        elif mutate_function==self.MUTATE:
            self.mutate(mutation_rate)
        elif mutate_function==self.GOLDMAN_MUTATE:
            self.goldman_mutate(mutation_rate)
        elif mutate_function==self.MUTATE_ONLY_PARAMETERS:
            self.mutate_only_parameters(mutation_rate)
        elif mutate_function==self.GOLDMAN_ONLY_PARAMETERS:
            self.goldman_mutate_only_parameters(mutation_rate)
        
    def mutate(self,mutation_rate):
        nb_mutations = math.floor((self.graph_length*8 + self.num_outputs)*mutation_rate)

        for i in range(0,nb_mutations):
            index = random.randrange(0, self.graph_length*8 + self.num_outputs, 1)

            if index < self.graph_length*8:
                parameter = index % 8
                node_index = int((index - parameter)/8)

                # nodes
                if parameter == 0:
                    self.nodes[node_index].getRandomFunction()
                elif parameter == 1:
                    self.nodes[node_index].getRandomConnection0(node_index)
                elif parameter == 2:
                    self.nodes[node_index].getRandomConnection1(node_index)
                elif parameter == 3:
                    self.nodes[node_index].getRandomParameter0()
                elif parameter == 4:
                    self.nodes[node_index].getRandomParameter1()
                elif parameter == 5:
                    self.nodes[node_index].getRandomParameter2()
                elif parameter == 6:
                    self.nodes[node_index].getRandomGaborFilterFrequence()
                elif parameter == 7:
                    self.nodes[node_index].getRandomGaborFilterOrientation()
            else:
                # outputs
                self.output_nodes[index-self.graph_length*8] = random.randrange(1,self.graph_length,1)

        self.updateActiveNodes()

    def goldman_mutate(self,mutation_rate):
        nb_mutations = math.floor((self.graph_length*8 + self.num_outputs)*mutation_rate)

        while nb_mutations>0:
            mutation = False

            index = random.randrange(0, self.graph_length*8 + self.num_outputs, 1)

            if index < self.graph_length*8:
                parameter = index % 8
                node_index = int((index - parameter)/8)

                try:
                    if self.active_nodes.index(node_index+self.num_inputs):
                        mutation = True
                except:
                    pass

                # nodes
                if parameter == 0:
                    self.nodes[node_index].getRandomFunction()
                elif parameter == 1:
                    self.nodes[node_index].getRandomConnection0(node_index)
                elif parameter == 2:
                    self.nodes[node_index].getRandomConnection1(node_index)
                elif parameter == 3:
                    self.nodes[node_index].getRandomParameter0()
                elif parameter == 4:
                    self.nodes[node_index].getRandomParameter1()
                elif parameter == 5:
                    self.nodes[node_index].getRandomParameter2()
                elif parameter == 6:
                    self.nodes[node_index].getRandomGaborFilterFrequence()
                elif parameter == 7:
                    self.nodes[node_index].getRandomGaborFilterOrientation()
            else:
                # outputs
                self.output_nodes[index-self.graph_length*8] = random.randrange(1,self.graph_length,1)
                mutation = True

            current_active_nodes = list(self.active_nodes)

            self.updateActiveNodes()

            if not mutation and not (current_active_nodes==self.active_nodes):
                mutation = True

            if mutation:
                nb_mutations = nb_mutations - 1

    def mutate_only_parameters(self,mutation_rate):
        nb_mutations = math.floor((self.graph_length*7)*mutation_rate)

        for i in range(0,nb_mutations):
            index = random.randrange(0, self.graph_length*7, 1)

            parameter = index % 7
            node_index = int((index - parameter)/7)

            # nodes
            if parameter == 0:
                self.nodes[node_index].getRandomConnection0(node_index)
            elif parameter == 1:
                self.nodes[node_index].getRandomConnection1(node_index)
            elif parameter == 2:
                self.nodes[node_index].getRandomParameter0()
            elif parameter == 3:
                self.nodes[node_index].getRandomParameter1()
            elif parameter == 4:
                self.nodes[node_index].getRandomParameter2()
            elif parameter == 5:
                self.nodes[node_index].getRandomGaborFilterFrequence()
            elif parameter == 6:
                self.nodes[node_index].getRandomGaborFilterOrientation()

        self.updateActiveNodes()

    def goldman_mutate_only_parameters(self,mutation_rate):
        nb_mutations = math.floor((self.graph_length*7)*mutation_rate)

        while nb_mutations>0:
            mutation = False

            index = random.randrange(0, self.graph_length*7, 1)

            parameter = index % 7
            node_index = int((index - parameter)/7)

            try:
                if self.active_nodes.index(node_index+self.num_inputs):
                    mutation = True
            except:
                pass

            # nodes
            if parameter == 0:
                self.nodes[node_index].getRandomConnection0(node_index)
            elif parameter == 1:
                self.nodes[node_index].getRandomConnection1(node_index)
            elif parameter == 2:
                self.nodes[node_index].getRandomParameter0()
            elif parameter == 3:
                self.nodes[node_index].getRandomParameter1()
            elif parameter == 4:
                self.nodes[node_index].getRandomParameter2()
            elif parameter == 5:
                self.nodes[node_index].getRandomGaborFilterFrequence()
            elif parameter == 6:
                self.nodes[node_index].getRandomGaborFilterOrientation()

            current_active_nodes = list(self.active_nodes)

            self.updateActiveNodes()

            if not mutation and not (current_active_nodes==self.active_nodes):
                mutation = True

            if mutation:
                nb_mutations = nb_mutations - 1

    def calculateFitness(self,input_data,output_data,verbose=False):
        if self.fitnessFunction==self.FITNESS_MEAN_ERROR:
            return self.calculateFitnessMeanError(input_data,output_data,verbose)
        elif self.fitnessFunction==self.FITNESS_MCC:
            return self.calculateFitnessMCC(input_data,output_data,verbose)

    def calculateFitnessMCC(self,input_data,output_data,verbose=False):
        mean = 0.0

        if verbose:
            for i in tqdm(range(0, len(input_data))):
                self.executeChromosome(input_data[i])

                width, height = output_data[i][0].shape

                n = np.zeros(output_data[i][0].shape)
                nr = np.zeros(output_data[i][0].shape)

                for j in range(0,len(output_data[i])):
                    n = n + output_data[i][j]
                    nr = nr + self.output_values[j]

                n_mask = np.ma.masked_equal(n,0).mask
                p_mask = np.logical_not(n_mask)


                tmp = nr * n_mask
                tmp[tmp>0] = 1
                fp = tmp.sum()
                #print("FP: "+str(fp))

                tn = int(n_mask.sum()-fp)
                #print("TN: "+str(tn))

                tmp = np.ma.masked_equal(nr,0).mask

                fn = (tmp * p_mask).sum()
                #print("FN: "+str(fn))

                tmp = np.int16(self.output_values) - np.int16(output_data[i])

                tmp2 = np.ma.masked_equal(tmp[0],0).mask

                #for j in range(1,len(tmp)):
                    #tmp2 = tmp2 * np.ma.masked_equal(tmp[j],0).mask

                tp = (tmp2 * p_mask).sum()
                #print("TP: "+str(tp))

                fp = width*height - tn - fn - tp
                #print("FP: "+str(fp))

                mean = mean + self.MCC(tp,tn,fp,fn)
        else:
            for i in range(0, len(input_data)):
                self.executeChromosome(input_data[i])

                width, height = output_data[i][0].shape

                n = np.zeros(output_data[i][0].shape)
                nr = np.zeros(output_data[i][0].shape)

                for j in range(0,len(output_data[i])):
                    n = n + output_data[i][j]
                    nr = nr + self.output_values[j]

                n_mask = np.ma.masked_equal(n,0).mask
                p_mask = np.logical_not(n_mask)

                tmp = nr * n_mask
                tmp[tmp>0] = 1
                fp = tmp.sum()

                tn = int(n_mask.sum()-fp)

                tmp = np.ma.masked_equal(nr,0).mask

                fn = (tmp * p_mask).sum()

                tmp = np.int16(self.output_values) - np.int16(output_data[i])

                tmp2 = np.ma.masked_equal(tmp[0],0).mask

                #for j in range(1,len(tmp)):
                    #tmp2 = tmp2 * np.ma.masked_equal(tmp[j],0).mask

                tp = (tmp2 * p_mask).sum()

                fp = width*height - tn - fn - tp

                mean = mean + self.MCC(tp,tn,fp,fn)

        mean = mean / len(input_data)

        self.fitness = mean

        return mean

    def calculateFitnessMeanError(self,input_data,output_data,verbose=False):
        mean = 0

        if verbose:
            for i in tqdm(range(0, len(input_data))):
                self.executeChromosome(input_data[i])

                for j in range(0, len(output_data[i])):
                    mean = mean + abs(np.int16(output_data[i][j])-np.int16(self.output_values[j])).sum()/output_data[i][j].size
        else:
            for i in range(0, len(input_data)):
                self.executeChromosome(input_data[i])

                for j in range(0, len(output_data[i])):
                    mean = mean + abs(np.int16(output_data[i][j])-np.int16(self.output_values[j])).sum()/output_data[i][j].size

        mean = mean / len(input_data)

        self.fitness = mean

        return mean

    def updateActiveNodes(self):
        nodes_to_check = []
        self.active_nodes = []

        for i in range(0,len(self.output_nodes)):
            nodes_to_check.append(self.graph_length+self.num_inputs-self.output_nodes[i])

            self.active_nodes_by_output[i] = []

            while len(nodes_to_check)>0:
                node_to_check = nodes_to_check.pop(0)

                self.active_nodes.append(node_to_check)
                self.active_nodes_by_output[i].append(node_to_check)

                if node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection0()>=self.num_inputs and nodes_to_check.count(node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection0())==0:
                    nodes_to_check.append(node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection0())

                if self.nodes[node_to_check-self.num_inputs].getFunction()>3 and self.functions.needSecondArgument(self.nodes[node_to_check-self.num_inputs].getFunction()) and node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection1()>=self.num_inputs and nodes_to_check.count(node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection1())==0:
                    nodes_to_check.append(node_to_check-self.nodes[node_to_check-self.num_inputs].getConnection1())

            self.active_nodes_by_output[i] = list(set(self.active_nodes_by_output[i]))
            self.active_nodes_by_output[i].sort()

        self.active_nodes = list(set(self.active_nodes))
        self.active_nodes.sort()
        
    def MCC(self,tp,tn,fp,fn):
        mcc = float((tp * tn) - (fp * fn))

        if mcc!=0.0:
            mcc = mcc / (math.sqrt(tp+fp)*math.sqrt(tp+fn)*math.sqrt(tn+fp)*math.sqrt(tn+fn))

        assert mcc <= 1.1, "mcc > 1"
        assert mcc >= -1.1, "mcc < -1"

        return 1 - mcc

    def executeChromosome(self,input_data,verbose=False):
        try:
            self.nodes_value = [None] * (self.num_inputs+self.graph_length)
            self.output_values = []

            for i in range(0, len(input_data)):
                self.nodes_value[i] = input_data[i]

            for i in self.active_nodes:
                if verbose:
                    start_time = time.time()

                # INP
                if self.nodes[i-self.num_inputs].getFunction()==1:
                    self.inputs_index = (self.inputs_index+1)%self.num_inputs
                    self.nodes_value[i] = self.nodes_value[self.inputs_index]
                # INPP
                elif self.nodes[i-self.num_inputs].getFunction()==2:
                    self.inputs_index = (self.inputs_index-1)%self.num_inputs
                    self.nodes_value[i] = self.nodes_value[self.inputs_index]
                # SKIP
                elif self.nodes[i-self.num_inputs].getFunction()==3:
                    self.inputs_index = (self.inputs_index+math.floor(self.nodes[i-self.num_inputs].getParameter0()))%self.num_inputs
                    self.nodes_value[i] = self.nodes_value[self.inputs_index]
                else:
                    self.nodes_value[i] = self.nodes[i-self.num_inputs].execute(self.functions,self.nodes_value[i-self.nodes[i-self.num_inputs].getConnection0()],self.nodes_value[i-self.nodes[i-self.num_inputs].getConnection1()])

                if verbose:
                    print("Function "+str(self.nodes[i-self.num_inputs].getFunction())+" - %s seconds" % (time.time() - start_time))

            for i in range(0,len(self.output_nodes)):
                self.output_values.append(self.nodes_value[self.graph_length+self.num_inputs-self.output_nodes[i]])
        except:
            print("Exception node "+str(i))
            print(str(self.nodes[i-self.num_inputs].getFunction()))
            print(str(self.nodes[i-self.num_inputs].getParameter0()))
            print(str(self.nodes[i-self.num_inputs].getParameter1()))
            print(str(self.nodes[i-self.num_inputs].getParameter2()))
            print(self.nodes_value[i-self.nodes[i-self.num_inputs].getConnection0()])
            print(self.nodes_value[i-self.nodes[i-self.num_inputs].getConnection1()])
            self.saveFile("debug.txt")
            raise

        # self.graph_length = graph_length
        # self.output_nodes = []
        # self.output_values = []
        # self.nodes = []
        # self.active_nodes = []
        # self.active_nodes_by_output = [None] * num_outputs
        # self.nodes_value = []

    def insertNode(self,insertion_nop):
        nb = len(self.active_nodes)
        print("insert len "+str(nb))

        index = random.randrange(0,nb,1)

        print("index "+str(index))
        print(self.active_nodes)

        node_index = self.active_nodes[index]

        print("node_index "+str(node_index))

        #self.nodes.insert(node_index-self.num_inputs,copy.deepcopy(self.nodes[node_index-1-self.num_inputs]))
        self.nodes.insert(node_index-self.num_inputs,copy.deepcopy(self.nodes[node_index-self.num_inputs]))

        if insertion_nop:
            #self.nodes[node_index-1-self.num_inputs].setFunction(5)
            self.nodes[node_index-self.num_inputs].setFunction(5)
        else:
            self.nodes[node_index-self.num_inputs].getRandomFunction()

        # version one NOP before connection 0
        #self.nodes[node_index-self.num_inputs].setConnections(1,self.nodes[node_index-self.num_inputs].getConnection1()+1)
        self.nodes[node_index-self.num_inputs+1].setConnections(1,self.nodes[node_index-self.num_inputs].getConnection1()+1)

        # update outputs
        for i in range(0,self.num_outputs):
            #if self.graph_length + 1 - self.output_nodes[i] < node_index:
            if self.graph_length - self.output_nodes[i] < node_index - self.num_inputs:
                self.output_nodes[i] = self.output_nodes[i] + 1

        for i in range(node_index+2,self.graph_length+1+self.num_inputs):
            if i-self.nodes[i-self.num_inputs].getConnection0() <= node_index:
                self.nodes[i-self.num_inputs].setConnections(self.nodes[i-self.num_inputs].getConnection0()+1,self.nodes[i-self.num_inputs].getConnection1())
            if i-self.nodes[i-self.num_inputs].getConnection1() <= node_index:
                self.nodes[i-self.num_inputs].setConnections(self.nodes[i-self.num_inputs].getConnection0(),self.nodes[i-self.num_inputs].getConnection1()+1)

        self.graph_length = self.graph_length + 1

        self.updateActiveNodes()

        print(self.active_nodes)

    def deleteNode(self):
        nb = len(self.active_nodes)
        print("delete len "+str(nb)+" graph len "+str(self.graph_length))

        index = random.randrange(0,nb,1)

        print("index "+str(index))
        print(self.active_nodes)

        node_index = self.active_nodes[index]

        print("node_index "+str(node_index))

        for i in range(0,self.num_outputs):
            print(self.active_nodes_by_output[i])
            for j in range(0,len(self.active_nodes_by_output[i])):
                if self.active_nodes_by_output[i][j]==node_index and len(self.active_nodes_by_output[i])==1:
                    return
                if j==len(self.active_nodes_by_output[i])-1 and node_index - self.nodes[node_index-self.num_inputs].getConnection0() < self.num_inputs:
                    return

        # update outputs
        for i in range(0,self.num_outputs):
            #if self.graph_length + 1 - self.output_nodes[i] < node_index:
            if self.graph_length - self.output_nodes[i] < node_index - self.num_inputs:
                self.output_nodes[i] = self.output_nodes[i] - 1
            # elif self.graph_length + 1 - self.output_nodes[i] == node_index:
            elif self.graph_length - self.output_nodes[i] == node_index - self.num_inputs:
                self.output_nodes[i] = self.output_nodes[i] + self.nodes[node_index-self.num_inputs].getConnection0() - 1

        for i in range(node_index+1,self.graph_length+self.num_inputs):
            if i-self.nodes[i-self.num_inputs].getConnection0() < node_index:
                self.nodes[i-self.num_inputs].setConnections(self.nodes[i-self.num_inputs].getConnection0()-1,self.nodes[i-self.num_inputs].getConnection1())
            elif i-self.nodes[i-self.num_inputs].getConnection0() == node_index:
                self.nodes[i-self.num_inputs].setConnections(self.nodes[i-self.num_inputs].getConnection0()+self.nodes[node_index-self.num_inputs].getConnection0()-1,self.nodes[i-self.num_inputs].getConnection1())
            if i-self.nodes[i-self.num_inputs].getConnection1() < node_index:
                self.nodes[i-self.num_inputs].setConnections(self.nodes[i-self.num_inputs].getConnection0(),self.nodes[i-self.num_inputs].getConnection1()-1)
            elif i-self.nodes[i-self.num_inputs].getConnection1() == node_index:
                self.nodes[i-self.num_inputs].setConnections(self.nodes[i-self.num_inputs].getConnection0(),self.nodes[i-self.num_inputs].getConnection1()+self.nodes[node_index-self.num_inputs].getConnection1()-1)

        deleted = self.nodes.pop(node_index-self.num_inputs)

        self.graph_length = self.graph_length - 1

        self.updateActiveNodes()

    def getFitness(self):
        return self.fitness

    def getNbActiveNodes(self):
        return len(self.active_nodes)
        
    def setFitness(self,fitness):
        self.fitness = fitness

    def setDuration(self,duration):
        self.duration = duration

    def getDuration(self):
        return self.duration

    def setFunctionForNode(self,node,function):
        self.nodes[node].setFunction(function)

    def setConnectionsForNode(self,node,connection0,connection1):
        self.nodes[node].setConnections(connection0,connection1)

    def setOutputNodes(self,outputs):
        for i in range(0,self.num_outputs):
            self.output_nodes[i] = outputs[i]

        self.updateActiveNodes()

    def print(self):
        print("Nb inputs: "+str(self.num_inputs))
        for i in range(0, self.graph_length):
            print("Node "+str(i+self.num_inputs)+" "+str(self.nodes[i].getFunction())+" "+str(self.nodes[i].getConnection0())+" "+str(self.nodes[i].getConnection1())+" "+str(self.nodes[i].getParameter0())+" "+str(self.nodes[i].getParameter1())+" "+str(self.nodes[i].getParameter2()))

        print("Nb outputs: "+str(self.num_outputs))
        for i in range(0, self.num_outputs):
            print(str(self.output_nodes[i]))
    
    def printGraph(self):
        for i in range(0,self.num_outputs):
            for j in range(0,len(self.active_nodes_by_output[i])):
                print(str(i)+" "+str(self.active_nodes_by_output[i][j]))

    def getActiveNodesValues(self):
        values = []
        for i in range(0,self.num_outputs):
            tmp = {}
            for j in range(0,len(self.active_nodes_by_output[i])):
                #print(str(i)+" "+str(self.active_nodes_by_output[i][j]))
                tmp[self.active_nodes_by_output[i][j]]=self.nodes_value[self.active_nodes_by_output[i][j]]
            values.append(tmp)

        return values

    def saveFile(self,filename):
        file = open(filename,"w")

        file.write(str(self.num_inputs)+" "+str(self.num_outputs)+" "+str(self.graph_length)+"\n")

        for i in range(0,self.graph_length):
            file.write(str(self.nodes[i].getFunction())+" "+str(self.nodes[i].getConnection0())+" "+str(self.nodes[i].getConnection1())
                +" "+str(self.nodes[i].getParameter0())+" "+str(self.nodes[i].getParameter1())+" "+str(self.nodes[i].getParameter2())
                +" "+str(self.nodes[i].getGaborFilterFrequence())+" "+str(self.nodes[i].getGaborFilterOrientation())+"\n")

        for i in range(0,self.num_outputs):
            file.write(str(self.output_nodes[i])+"\n")

        file.close()
