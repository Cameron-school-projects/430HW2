import numpy as np
import re
import statistics
allVals =[]
#read in values
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
#split sets
validationSet = allVals[(len(allVals)-50):]
trainingSet = allVals[:(len(allVals)-50)]
validationSet = np.array(validationSet)
trainingSet = np.array(trainingSet)
allVals = np.array(allVals)
#normalize data
for i in range(0,12):
    mean = statistics.mean(trainingSet[:,i])
    meanVal = statistics.mean(validationSet[:,i])
    standardDeviation = statistics.stdev(trainingSet[:,i])
    standardDeviationVal = statistics.stdev(validationSet[:,i])
    for index, number in enumerate(trainingSet[:,i]):
        #if number is 0, we dont need to divide, as it could give us NAN
        if(trainingSet[index,i]!=0):
            trainingSet[index,i] = abs(number-mean)/standardDeviation
    for index2,number2 in enumerate(validationSet[:,i]):
        #if number is 0, we dont need to divide, as it could give us NAN
        if(validationSet[index2,i]!=0):
            validationSet[index2,i]=abs(number2-meanVal)/standardDeviationVal

def calculateCost(m, X, y):
    total = 0
    for i in range(m-1):
        squared_error = (y[i] - X[i]) ** 2
        total += squared_error
    
    return total * (1 / (2*m))

def sumDJ0(hTheta,y,m):
    sum=0
    for idx,yi in enumerate(y):
        sum+=(hTheta[idx]-yi)
    return sum 


def sumDJN(hTheta,y,x,m):
    sum=0
    for idx,yi in enumerate(y):
        sum+=(hTheta[idx]-yi)*x[idx]

    return sum


def calculateHtheta(allThetas,x,theta0):
    allThetas = allThetas.transpose()
    #calculate htheta without theta0, so we can use the transpose and multiply method
    newThetas =np.dot(allThetas,x)
    #add theta0 back in
    newThetas = newThetas+theta0
    return newThetas

def getAllHthetas(thetas,inputs,numOfInputs,m):
    hThetas = []
    newInputs = []
    thetasMinusTheta0=thetas[1:]
    #calculate start value of htheta
    #prevents us from going out of range when we calculate inputs[i][i+x]
    for i in range(0,m):
        #pull single row for h(theta)
        for x in range(0,numOfInputs):
            newInputs.append(inputs[x][i])
        hThetas.append(calculateHtheta(thetasMinusTheta0,newInputs,thetas[0]))
        #clear new inputs
        newInputs=[]
    return hThetas
def calculateGradientDescent(inputs,output,thetas,numOfInputs,alpha=0.001,):
    #using one array for input, array for thetas, loop to calculate new vals, loop to assign new vals
    #define as a numpy array
    m=len(inputs[0])
    #get starting cost and hThetas
    hThetas = getAllHthetas(thetas,inputs,numOfInputs,m)
    cost = calculateCost(m,hThetas,output)
    while True:
        sums =[]
        #calculate new predicted values
        DJ0=sumDJ0(hThetas,output,m)
        sums.append(thetas[0] - (1/m) * alpha * DJ0)
        #calculate theta 1 - theta n+1
        for i in range(1,numOfInputs+1):
            #inputs does not account for theta0, so we subtract one from its index
            DJN = sumDJN(hThetas,inputs[i-1],output,m)
            sums.append(thetas[i] - (1/m) * alpha * DJN)
        oldCost = cost
        #assign values together 
        sums = np.array(sums)
        for i in range(0,numOfInputs+1):
            temp = sums[i]
            thetas[i] = temp
        hThetas = getAllHthetas(thetas,inputs,numOfInputs,m)
        cost = calculateCost(m,hThetas,output)
        #break if we've reached acceptable accuracy
        if(abs(cost-oldCost)<.01):
            break
    return thetas

def predictValue1(newTheats,validationSet,numOfInputs):
    sum=newTheats[0]+newTheats[1]*validationSet[6]+newTheats[2]*validationSet[9]
    return sum

#first calculation using two inputs
inputs = np.array([trainingSet[:,6],trainingSet[:,9]])
#add the two columns of inputs
#output column
output = np.array(trainingSet[:,13])
thetas = np.array([0.0,0.0,0.0])
newThetas = calculateGradientDescent(inputs,output,thetas,2)
print(newThetas)
# test = newThetas[0] + newThetas[1]*validationSet[0][0]+newTvalidationSet[0][1]
# print("predicted values for validation set")
# for i in range(0,49):
#     print(f"predicted value for {i}: {predictValue1(newThetas,validationSet[i],2)}, actual value: {validationSet[i][13]}")
# #calculation using all columns
# inputs = np.array(trainingSet[:,:12])
# output = np.array(trainingSet[:,13])
# thetas = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# newThetas = calculateGradientDescent(inputs,output,thetas,13)
# for i in range(0,49):
#     print(f"predicted value for {i}: {newThetas[i]}, actual value: {validationSet[i]}")

