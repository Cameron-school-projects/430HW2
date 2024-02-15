import numpy as np
import re
import statistics
allVals =[]
with open('BostonNumbers.txt',encoding='utf-8') as f:
    idx =0 
    for line in f:
        #see if we split the first or second row
        #remove extra whitespace
        line=line.strip()
        #split based on any amount of whitespace
        lineVals=re.split(r"[ \t\n]+",line)
        #check length of split
        length = len(lineVals)
        #convert values to floats
        for index,number in enumerate(lineVals):
            lineVals[index]=float(number)
        if(length>3):
            allVals.append(lineVals)
        else:
            allVals[idx].extend(lineVals)
            idx+=1

#calculate mean and stdev for each set 
validationSet = allVals[(len(allVals)-50):]
trainingSet = allVals[:(len(allVals)-50)]
validationSet = np.array(validationSet)
trainingSet = np.array(trainingSet)

#normalize data
for i in range(0,12):
    mean = statistics.mean(trainingSet[:,i])
    standardDeviation = statistics.stdev(trainingSet[:,i])
    for index, number in enumerate(trainingSet[:,i]):
        trainingSet[index,i] = abs(number-mean)/standardDeviation

def calculateCost(m, X, y):
    total = 0
    for i in range(m):
        squared_error = (y[i] - X[i]) ** 2
        total += squared_error
    
    return total * (1 / (2*m))

def sumDJ0(hTheta,y,m):
    sum=0
    for idx,yi in enumerate(y):
        sum+=(hTheta-yi)
    return sum 


def sumDJN(htheta,y,x,m):
    sum=0
    for idx,yi in enumerate(y):
        sum+=(htheta-yi)*x[idx]

    return sum


def calculateHtheta(allThetas,x):
    allThetas = allThetas.transpose()
    newThetas = allThetas*x
    return newThetas

def calculateGradientDescent(x,y,theta0=0,theta1=0,alpha=0.01):
    #define x as a numpy array
    x= np.array(x)
    m=len(y)
    #calculating h theta by multiplying all x values by theta1,then adding theta0 to all x values
    hTheta = calculateHtheta(theta0,theta1,x)
    cost = calculateCost(m,hTheta,y)
    while True:
        oldCost = cost
        #calculate new thetas
        sum1 = sumDJ0(hTheta,y,m)
        sum2 = sumDJN(hTheta,y,x,m)
        sum3 = sumDJN(hTheta,y,z,m)
        temp0 = theta0 - (1/m) * alpha * sum1
        temp1= theta1 - (1/m) * alpha * sum2
        temp2 = theta2 - (1/m) * alpha * sum2
        #recalculate h theta
        #update together
        theta0=temp0
        theta1=temp1
        theta2=temp2
        hTheta = calculateHtheta(theta0,theta1,x)
        cost = calculateCost(m,hTheta,y)
        #break if we've reached acceptable accuracy
        if(abs(cost-oldCost)<.01):
            break
    return theta0,theta1





