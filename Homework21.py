import numpy as np
import re
import statistics
allVals =[]
#read in values
with open('boston.txt',encoding='utf-8') as f:
    idx=0 
    #used for skipping header data
    lineNum = 0
    #skipping lines of header and non numerical data (first 22 rows)
    for line in f:
        if(lineNum<11):
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
print(allVals)
#Save non-normalized data for part 2
o_validationSet = allVals[(len(allVals)-50):]
o_trainingSet = allVals[:(len(allVals)-50)]
o_validationSet = np.array(o_validationSet)
o_trainingSet = np.array(o_trainingSet)

validationSet = allVals[(len(allVals)-50):]
trainingSet = allVals[:(len(allVals)-50)]
validationSet = np.array(validationSet)
trainingSet = np.array(trainingSet)

#normalize data
for i in range(0,14):
    mean = statistics.mean(trainingSet[:,i])
    meanVal = statistics.mean(validationSet[:,i])
    standardDeviation = statistics.stdev(trainingSet[:,i])
    standardDeviationVal = statistics.stdev(validationSet[:,i])
    for index, number in enumerate(trainingSet[:,i]):
        #if number is 0, we dont need to divide, as it could give us NAN
        if(standardDeviation!=0 and mean!=0):
            trainingSet[index,i] = abs(number-mean)/standardDeviation
    for index2,number2 in enumerate(validationSet[:,i]):
        #if number is 0, we dont need to divide, as it could give us NAN
        if(standardDeviationVal!=0 and meanVal!=0):
            validationSet[index2,i]=abs(number2-meanVal)/standardDeviationVal    


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
def calculateGradientDescent(inputs,output,thetas,numOfInputs,alpha=0.01):
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
# outFile = open("output.txt","w")
# f.write("predicted values for validation set")
#print("predicted values for validation set")
#for i in range(0,50):
    #print(f"predicted value for {i+1}: {predictValue1(newThetas,validationSet[i])}, actual value: {validationSet[i][13]}")
    #f.write(f"predicted value for {i+1}: {predictValue1(newThetas,validationSet[i])}, actual value: {validationSet[i][13]}")
#calculate sum of square and mean square error for the data
sum_square_error = 0
for i in range(0, len(validationSet)):
    sum_square_error = sum_square_error + (predictValue1(newThetas,validationSet[i]) - validationSet[i][13])**2
mean_square_error = sum_square_error / len(validationSet)
#print("Sum of Square Error for MEDV calculated using AGE and TAX:", sum_square_error)
#f.write("Sum of Square Error for MEDV calculated using AGE and TAX:", sum_square_error)
#print("Mean Square Error for MEDV calculated using AGE and TAX:", mean_square_error)
#f.write("Mean Square Error for MEDV calculated using AGE and TAX:", mean_square_error)
# #calculation using all columns
#the : operator needs to be +1 the index, looks like it takes it at 1 based? not sure
inputs = inputs = buildInputs([trainingSet[:,0],trainingSet[:,1],trainingSet[:,2],trainingSet[:,3],trainingSet[:,4],trainingSet[:,5],trainingSet[:,6],trainingSet[:,7],trainingSet[:,8],trainingSet[:,9],trainingSet[:,10],trainingSet[:,11],trainingSet[:,12]],13)
output = np.array(trainingSet[:,13])
thetas = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
newThetas = calculateGradientDescent(inputs,output,thetas,13)
#print("predicted values for validation set with all columns")
#f.write("predicted values for validation set with all columns")
#for i in range(0,50):
    #print(f"predicted value for {i+1}: {predictValue2(newThetas,validationSet[i],2)}, actual value: {validationSet[i][13]}")
    #f.write(f"predicted value for {i+1}: {predictValue2(newThetas,validationSet[i],2)}, actual value: {validationSet[i][13]}")

#calculate sum of square and mean square error for the data
sum_square_error = 0
for i in range(0, len(validationSet)):
    sum_square_error = sum_square_error + (predictValue2(newThetas,validationSet[i],2) - validationSet[i][13])**2
mean_square_error = sum_square_error / len(validationSet)
#print("Sum of Square Error for MEDV calculated using all variables:", sum_square_error)
#f.write("Sum of Square Error for MEDV calculated using all variables:", sum_square_error)
#print("Mean Square Error for MEDV calculated using all variables:", mean_square_error)
#f.write("Mean Square Error for MEDV calculated using all variables:", mean_square_error)
###############################################################################
# Part 2

#manipulate data format to simplify
validationSet_t = validationSet.transpose()
actual_y = validationSet_t[(len(validationSet_t)-1):][0]
arrayof1 = np.full((len(trainingSet), 1), 1)
t_set = trainingSet.transpose()
allInputs = t_set[:(len(t_set)-1)]
arrayofAge = t_set[6:7].transpose()
arrayofTax = t_set[9:10].transpose()
set_1 = np.append(arrayof1, arrayofAge, axis = 1)
set_1 = np.append(set_1, arrayofTax, axis = 1)
print(set_1)
training_y = t_set[(len(t_set)-1):]
#thetas = closed form solution Theta = (X^T*X)^-1 * X^T * y
#split up variables for testing
set_x_t = set_1.transpose()
x_t_x= np.matmul(set_x_t, set_1)
inv_x_t_x = np.linalg.inv(x_t_x)
inv_x_t_x_x_t = np.matmul(inv_x_t_x, set_x_t)
thetas= np.matmul(inv_x_t_x_x_t, training_y.transpose())
#calculate in one line
thetas2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(set_1.transpose(), set_1)), set_1.transpose()), training_y.transpose())
#print(thetas)
#print(thetas2)
predicted_y = []

#calculate predicted y for the validation set
for i in range(0, len(validationSet)):
    predicted_y.append(thetas[0][0] + (thetas[1][0] * validationSet[i][6]) + (thetas[2][0] * validationSet[i][9]))
    #print(o_validationSet[i][6], o_validationSet[i][9])
    #f.write(o_validationSet[i][6], o_validationSet[i][9])
  
#calculate sum of square and mean square error for the data
sum_square_error = 0
for i in range(0, len(predicted_y)):
    sum_square_error += (predicted_y[i] - actual_y[i])**2
mean_square_error = sum_square_error / len(predicted_y)
#print("predicted values for validation set with AGE and TAX with the Closed Form Solution")
#f.write("predicted values for validation set with AGE and TAX with the Closed Form Solution")
#for i in range(0,50):
    #print(f"predicted value for {i+1}: {predicted_y[i]}, actual value: {actual_y[i]}")
    #f.write(f"predicted value for {i+1}: {predicted_y[i]}, actual value: {actual_y[i]}")
#print("Sum of Square Error for MEDV calculated using AGE and TAX with the Closed Form Solution:", sum_square_error)
#f.write("Sum of Square Error for MEDV calculated using AGE and TAX with the Closed Form Solution:", sum_square_error)
#print("Mean Square Error for MEDV calculated using AGE and TAX with the Closed Form Solution:", mean_square_error)
#f.write("Mean Square Error for MEDV calculated using AGE and TAX with the Closed Form Solution:", mean_square_error)
#f.close()
