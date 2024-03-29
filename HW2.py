import numpy as np
import re
import statistics
allVals =[]
#read in values
with open('BostonNumbers.txt',encoding='utf-8') as f:
    idx=0 
    #used for skipping header data
    lineNum = 0
    #skipping lines of header and non numerical data (first 22 rows)
    for line in f:
        if(lineNum<22):
            next(f)
            lineNum=lineNum+1
            continue
        
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

allVals = np.array(allVals)
#normalize data
for i in range(0,14):
    mean = statistics.mean(allVals[:,i])
    standardDeviation = statistics.stdev(allVals[:,i])
    for index, number in enumerate(allVals[:,i]):
        #if number is 0, we dont need to divide, as it could give us NAN
        if(standardDeviation!=0 and mean!=0):
            allVals[index,i] = abs(number-mean)/standardDeviation

validationSet = allVals[(len(allVals)-50):]
trainingSet = allVals[:(len(allVals)-50)]
validationSet = np.array(validationSet)
trainingSet = np.array(trainingSet)

def calculateCost(m, X, y):
    total = 0
    for i in range(m):
        squared_error = (X[i] - y[i]) ** 2
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
    newThetas = np.matmul(allThetas,x)
    #add theta0 back in
    newThetas = newThetas+theta0
    return newThetas

def getAllHthetas(thetas,inputs,numOfInputs,m):
    hThetas = []
    thetasMinusTheta0=thetas[1:]
    #calculate start value of htheta
    #prevents us from going out of range when we calculate inputs[i][i+x]
    for i in range(0,m):
        hThetas.append(calculateHtheta(thetasMinusTheta0,inputs[i],thetas[0]))
    return hThetas
def calculateGradientDescent(inputs,output,thetas,numOfInputs,alpha=0.1):
    #using one array for input, array for thetas, loop to calculate new vals, loop to assign new vals
    m=len(inputs)
    #get starting cost and hThetas
    hThetas = getAllHthetas(thetas,inputs,numOfInputs,m)
    cost = calculateCost(m,hThetas,output)
    while True:
        sums =[]
        #calculate new predicted values
        DJ0=sumDJ0(hThetas,output,m)
        sums.append(thetas[0] - alpha * (1/m) * DJ0)
        #calculate theta 1 - theta n+1
        for i in range(1,numOfInputs+1):
            #inputs does not account for theta0, so we subtract one from its index
            #get each column, which represents a different variable of the calculation
            DJN = sumDJN(hThetas,output,inputs[:,i-1],m)
            sums.append(thetas[i] - alpha * (1/m) * DJN)
        oldCost = cost
        sums = np.array(sums)
        #assign values together 
        for i in range(0,numOfInputs+1):
            temp = sums[i]
            thetas[i] = temp

        hThetas = getAllHthetas(thetas,inputs,numOfInputs,m)
        cost = calculateCost(m,hThetas,output)
        costDifference = cost-oldCost
        #break if we've reached acceptable accuracy
        if(abs(costDifference)<.01):
            break
    return thetas

def predictValue1(newThetas,validationSet):
    sum=newThetas[0]+newThetas[1]*validationSet[6]+newThetas[2]*validationSet[9]
    return sum

def predictValue2(newThetas,validationSet,numOfInputs):
    sum=newThetas[0]
    for i in range(0,numOfInputs):
        sum = sum + newThetas[i]*validationSet[i]
    return sum

def buildInputs(appendedInputs,numOfInputs):
    builtInput=[]
    for i in range(0,len(appendedInputs[0])):
        newInputs = []
        for x in range(0,numOfInputs):
            newInputs.append(appendedInputs[x][i])
        builtInput.append(newInputs)
    builtInput = np.array(builtInput)
    return builtInput
#first calculation using two inputs
#add the two columns of inputs
inputs = buildInputs([trainingSet[:,6],trainingSet[:,9]],2)
#output column
output = np.array(trainingSet[:,13])
thetas = np.array([0.0,0.0,0.0])
newThetas = calculateGradientDescent(inputs,output,thetas,2)
print("predicted values for validation set")
for i in range(0,49):
    print(f"predicted value for {i+1}: {predictValue1(newThetas,validationSet[i])}, actual value: {validationSet[i][13]}")
# #calculation using all columns
#the : operator needs to be +1 the index, looks like it takes it at 1 based? not sure
inputs = inputs = buildInputs([trainingSet[:,0],trainingSet[:,1],trainingSet[:,2],trainingSet[:,3],trainingSet[:,4],trainingSet[:,5],trainingSet[:,6],trainingSet[:,7],trainingSet[:,8],trainingSet[:,9],trainingSet[:,10],trainingSet[:,11],trainingSet[:,12]],13)
output = np.array(trainingSet[:,13])
thetas = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
newThetas = calculateGradientDescent(inputs,output,thetas,13)
print("predicted values for validation set with all columns")
for i in range(0,49):
    print(f"predicted value for {i}: {predictValue2(newThetas,validationSet[i],2)}, actual value: {validationSet[i][13]}")

