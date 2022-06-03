import random

class Functions:

    # Function Int
    # Connection 0 Int
    # Connection 1 Int
    # Parameter 0 Real no limitation
    # Parameter 1 Int [−16, +16]
    # Parameter 2 Int [−16, +16]
    # Gabor Filter Frequ. Int [0, 16]
    # Gabor Filter Orient. Int [−8, +8]

    num_functions = 0

    outside_nb_functions = 0

    classes = []

    indexes = []

    @classmethod
    def setOutsideNbFunctions(cls,nb):
        cls.outside_nb_functions = nb

    @classmethod
    def add(cls,item):
        cls.classes.append(item)

        cls.indexes.append(cls.num_functions+item.getNbFunction())
        cls.num_functions += item.getNbFunction()

    @classmethod
    def getNbFunction(cls):
        return cls.num_functions

    @classmethod
    def getRandomFunction(cls):
        return random.randrange(1,1+cls.num_functions)

    @classmethod
    def needSecondArgument(cls,function):
        last = 0
        function -= cls.outside_nb_functions

        for i in range(0,len(cls.indexes)):
            if function<=cls.indexes[i]:
                return cls.classes[i].needSecondArgument(function-last)
            last = cls.indexes[i]

    @classmethod
    def execute(cls, function, connection0, connection1, parameter0, parameter1, parameter2, gabor_filter_frequency, gabor_filter_orientation):
        last = 0
        function -= cls.outside_nb_functions

        for i in range(0,len(cls.indexes)):
            if function<=cls.indexes[i]:
                return cls.classes[i].execute(function-last,connection0,connection1,parameter0,parameter1,parameter2,gabor_filter_frequency,gabor_filter_orientation)
            last = cls.indexes[i]
