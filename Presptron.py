#we use numpy for matrix manipulation
import numpy as np
#we use pandas to read csv files
import pandas as pd

#first we converted the dataset (text file) to (csv file)
#now we read the csv file using pd (pandas) as we mentioned before
dataframe= pd.read_csv("NN.csv")
#convert the to a list
data=dataframe[:].values.tolist()

#represent every "Iris-setosa" with (1) and every "Iris-virginica" with (-1)
for x in  range(len(data)):
    if data[x][4]=="Iris-setosa":
        data[x][4] = 1
    else :
        data[x][4] = -1

trainInput=[[]]
trainClass=[]
testInput=[[]]
testClass=[]

#devide the dataset to testing and training
for x in  range(len(data)):
    if x<=29 or (x>=50 and x<=79):
        trainInput.append(data[x][0:4])#X1,X2,X3,X4
        trainClass.append(data[x][4])#1 or -1
    else:
        testInput.append(data[x][0:4])#X1,X2,X3,X4
        testClass.append(data[x][4])#1 or -1

trainInput.pop(0)
testInput.pop(0)


class Perceptron(object):
    #constructor of the perceptron class
    def __init__(self, no_of_inputs=4, epoch=50, learning_rate=0.01):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self,Input):
        #summation= (w1.x1 + w2.x2 + w3.x3 + w4.x4) + b
        #where (x1,x2...) = testInput
        #where (W1,W2....) = weights
        #perceptron uses a signum activation function
        summation = np.dot(Input, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activiation = 1
        else:
            activiation = -1
        return activiation

    def train(self, trainInput, trainClass):
        #trainInput contain 60 row of features, 30 of Iris-setosa and 30 of Iris-virginica
        #trainclass contain 60 label, each label = 1 (Iris-setosa) or = -1 (Iris-virginica)
        for _ in range(self.epoch):
            for features,label  in zip(trainInput, trainClass):
                prediction = self.predict(features)
                for n in range(len(features)):
                  self.weights[1:] += self.learning_rate *(label - prediction) * features[n]
                  self.weights[0] += self.learning_rate * (label - prediction)

    def testing(self, testInput, testClass):
        # testInput contain 40 row of features, 20 of Iris-setosa and 20 of Iris-virginica
        # testclass contain 40 label, each label = 1 (Iris-setosa) or = -1 (Iris-virginica)
        c=0
        f=0
        for n in testInput:
            prediction = self.predict(n)
            if prediction == testClass[f]:
                c+=1
            f+=1
        #calculate the accuracy
        return (c/len(testInput))*100

perc=Perceptron()
perc.train(trainInput,trainClass)
print(perc.testing(testInput,testClass))
