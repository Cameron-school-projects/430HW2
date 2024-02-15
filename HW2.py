import numpy as np
import re
import statistics
crim = []
zn = []
indus = []
chas = []
nox = []
rm = []
age = []
dis = []
rad = []
tax = []
ptratio = []
b = []
lstat = []
medv = []
with open('BostonNumbers.txt',encoding='utf-8') as f:
    for idx,line in f:
        #see if we split the first or second row
        #remove extra whitespace
        line=line.strip()

        #split based on any amount of whitespace
        lineVals=re.split(r"[ \t\n]+",line)
        #check length of split
        length = len(lineVals)
        if(length>3):
            crim.append(float(lineVals[0]))
            zn.append(float(lineVals[1]))
            indus.append(float(lineVals[2]))
            chas.append(float(lineVals[3]))
            nox.append(float(lineVals[4]))
            rm.append(float(lineVals[5]))
            age.append(float(lineVals[6]))
            dis.append(float(lineVals[7]))
            rad.append(float(lineVals[8]))
            tax.append(float(lineVals[9]))
            ptratio.append(float(lineVals[10]))
        else:
            b.append(float(lineVals[0]))
            lstat.append(float(lineVals[1]))
            medv.append(float(lineVals[2]))

#calculate mean and stdev for each set 


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



plt.scatter(xVals,yVals,marker='x',c='red')
# naming the x axis
plt.xlabel('Population of City in 10,000s')
# naming the y axis
plt.ylabel('Profit in 10,000s')
 
# function to show the plot
plt.show()



# function to show the plot
plt.show()



