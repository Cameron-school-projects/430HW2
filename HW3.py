import numpy as np
from scipy.optimize import minimize


class1Data = []
allData = []
class2And3Data = []
#stating values for params/thetas
params = np.hstack([0.0,0.0,0.0,0.0,0.0])
#read in data
with open('iris.data',encoding='utf-8') as f:
    lineNum = 0
    #skipping lines of header and non numerical data (first 22 rows)
    for line in f:
        #remove extra whitespace
        line=line.strip()
        lineData = line.split(",")
        tempData = [float(lineData[0]),float(lineData[1]),float(lineData[2]),float(lineData[3])]
        #first 50 entries are class 1, rest are class 2
        if(lineNum<50):
            tempData.append(1.0)
            class1Data.append(tempData)
            allData.append(tempData)
        else:
            tempData.append(0.0)
            class2And3Data.append(tempData)
            allData.append(tempData)
            
        lineNum = lineNum+1
#add in column of 1's for param 0
class1Data = np.hstack((np.ones((len(class1Data), 1)), class1Data))
class2And3Data = np.hstack((np.ones((len(class2And3Data), 1)), class2And3Data))
allData = np.hstack((np.ones((len(allData), 1)), allData))
#convert to numpy arrays
class1Data = np.array(class1Data)
class2And3Data = np.array(class2And3Data)
allData = np.array(allData)
#shuffle and get training and validation sets
rng = np.random.default_rng()
rng.shuffle(class1Data)
rng.shuffle(class2And3Data)
rng.shuffle(allData)
#20 percent of 50(for class 1) is 10, and 20 percent of 100 (class 2 ) is 20

trainingSet = allData[:(len(allData)-30)]
validationSet = allData[(len(allData)-30):]
class1Validation = class1Data[(len(class1Data)-10):]
class2And3Validation = class2And3Data[len(class2And3Data)-20:]
class1Training = class1Data[:(len(class1Data)-10)]
class2And3Training = class2And3Data[:len(class2And3Data)-20]
#grab class attribute from training sets
class1TrainingOutput = class1Training[:,5]
class2And3TrainingOutput = class2And3Training[:,5]
trainingOutput = trainingSet[:,5]
#remove class labels from training sets
class1Training = np.delete(class1Training,5,1)
class2And3Training = np.delete(class2And3Training,5,1)
trainingSet = np.delete(trainingSet,5,1)
#grab class attribute for validation
class1ValidationOutput = class1Validation[:,5]
class2And3ValidationOutput = class2And3Validation[:,5]
validationOutput = validationSet[:,5]
#remove class labels from validation sets
class1Validation = np.delete(class1Validation,5,1)
class2And3Validation = np.delete(class2And3Validation,5,1)
validationSet = np.delete(validationSet,5,1)
def sigmoid(x, w, threshold=0.5):
    # @ uses matrix multiplication
    p = 1 / (1 + np.exp(-x @ w))
    return np.where(p > threshold, 1, 0)

def sigmoid2(x, w, threshold=0.5):
    # @ uses matrix multiplication
    p = 1 / (1 + np.exp(-x @ w))
    return p

def logistic_cost(w, X, y):
    return -(y * np.log(sigmoid2(X, w)) + (1 - y) * np.log(1 - sigmoid2(X, w))).mean()



def logistic_cost_grad(w, X, y):
    # @ uses matrix multiplication
    return (X.T @ (sigmoid2(X, w) - y)) / len(X)

def accuracy(y, y_hat):
    print(y)
    print(y_hat)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for idx,num in enumerate(y):
        if(num == y_hat[idx]):
            if(num==1):
                TP = TP+1
            else:
                TN = TN+1
        else:
            if(num==0):  #should be == 0, but that causes a divide by zero since our predictions are wrong
                FP = FP +1
            else:
                FN = FN + 1
        totalAccuracy = (y_hat == y).sum() / len(y)
    return [totalAccuracy,TP,TN,FP,FN]

def precision(TP,FP):
    return TP/(TP+FP)
#this gets us a new set of W's which form the w array we pass into the sigmoid
print(trainingOutput)

optimized_params = minimize(logistic_cost,params, jac=logistic_cost_grad, args=(trainingSet, trainingOutput)).x
#optimized_params = minimize(logistic_cost,optimized_params, jac=logistic_cost_grad, args=(class2And3Training, class2And3TrainingOutput)).x
print(optimized_params)

testingAccuracy = sigmoid(validationSet,optimized_params)
accuracyInfo=accuracy(validationOutput,testingAccuracy)
print(f"Accuracy: {accuracyInfo[0]}")
print("Confusion Matrix:")
print(f"|True Positive: {accuracyInfo[1]}  |  False Positive: {accuracyInfo[3]}|")
print(f"|False Negative: {accuracyInfo[4]} |  True Negative: {accuracyInfo[2]} |")
print(f"\nPrecision: {precision(accuracyInfo[1],accuracyInfo[3])}")
