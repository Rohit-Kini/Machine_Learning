#!/usr/bin/env python3
import monkdata as m
import dtree as d
import numpy as np
import drawtree_qt5 as dt
import matplotlib.pyplot as plt
import random

####### A1 #######
E_M1 = d.entropy(m.monk1)
E_M2 = d.entropy(m.monk2)
E_M3 = d.entropy(m.monk3)
print('Entropy of MONK-1 is ', E_M1)
print('Entropy of MONK-2 is ', E_M2)
print('Entropy of MONK-3 is ', E_M3)

######## A3 #######
Gain_M1 = np.empty([6,1], dtype = float)
Gain_M2 = np.empty([6,1], dtype = float)
Gain_M3 = np.empty([6,1], dtype = float)
for i in range(0,6):
    Gain_M1[i] = d.averageGain(m.monk1, m.attributes[i])
    Gain_M2[i] = d.averageGain(m.monk2, m.attributes[i])
    Gain_M3[i] = d.averageGain(m.monk3, m.attributes[i])
print('Information Gain of all attributes of MONK-1 is \n',Gain_M1)
print('Information Gain of all attributes of MONK-2 is \n',Gain_M2)
print('Information Gain of all attributes of MONK-3 is \n',Gain_M3)

####### A5 #######
t_M1 = d.buildTree(m.monk1, m.attributes)
t_M2 = d.buildTree(m.monk2, m.attributes)
t_M3 = d.buildTree(m.monk3, m.attributes)
print('MONK-1')
print('Train Set Error: ',(1-d.check(t_M1,m.monk1)))
print('Test Set Error: ',(1-d.check(t_M1,m.monk1test)))
print('MONK-2')
print('Train Set Error: ',(1-d.check(t_M2,m.monk2)))
print('Test Set Error: ',(1-d.check(t_M2,m.monk2test)))
print('MONK-3')
print('Train Set Error: ',(1-d.check(t_M3,m.monk3)))
print('Test Set Error: ',(1-d.check(t_M3,m.monk3test)))
"Visualization of the decision trees"
#dt.drawTree(t_M1)
#dt.drawTree(t_M2)
#dt.drawTree(t_M3)

####### A7 #######
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata)*fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruning(traindata, testdata, attributes, fraction):
    Mean = np.empty([6,1], dtype = float)
    Variance = np.empty([6,1], dtype = float)
    m = 0
    for i in fraction:
        Error = list()
        "Several Runs due to Shuffling"
        for j in range(0,500):
            train, val = partition(traindata, i)
            t = d.buildTree(train, attributes)
            "All combination of pruned trees"
            pruned_t = d.allPruned(t)
            eff = 0
            max_eff = 0
            index = 0
            "Run the loop till no trees left to be pruned"
            for k in range(0, len(pruned_t)):
                eff = d.check(pruned_t[k], val)
                if (eff>max_eff):
                    max_eff = eff
                    index = k
            Error.append(1 - d.check(pruned_t[index], testdata))
        Mean[m] = np.mean(Error)
        Variance[m] = np.var(Error)
        m = m+1
    return Mean, Variance

fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
Error_M1, Variance_M1 = pruning(m.monk1, m.monk1test, m.attributes, fraction)
print('Error in MONK-1: \n', Error_M1)
print('Variance in MONK-1: \n', Variance_M1)
Error_M3, Variance_M3 = pruning(m.monk3, m.monk3test, m.attributes, fraction)
print('Error in MONK-3: \n', Error_M3)
print('Variance in MONK-3: \n', Variance_M3)
"Plotting the Mean & Variance"
plt.figure(1)
plt.subplot(2,1,1)
plt.title('Mean of errors vs Fractions')
plt.xlabel('Fractions')
plt.ylabel('Mean of Errors')
line1 = plt.plot(fraction, Error_M1, 'bo--', label = 'MONK-1')
line2 = plt.plot(fraction, Error_M3, 'go--', label = 'MONK-3')
plt.legend(loc = 'upper right')

plt.subplot(2,1,2)
plt.title('Variance of errors vs Fractions')
plt.xlabel('Fractions')
plt.ylabel('Variance of Errors')
line1 = plt.plot(fraction, Variance_M1, 'bo--', label = 'MONK-1')
line2 = plt.plot(fraction, Variance_M3, 'go--', label = 'MONK-3')
plt.legend(loc = 'upper right')
plt.show()
