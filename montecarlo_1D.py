import numpy as np
from matplotlib import pyplot as plt

size = 10
step = 10000

initialstate = np.random.randint(0, (2**size))
s = np.zeros(size)
for i in range(size):
    x = int(initialstate/(2**(i)))%2
    s[i] = 2*x-1

def p(i, beta):
    if i ==0:
        return np.exp(-beta*s[i]*s[i+1])/(np.exp(beta*s[i]*s[i+1])+np.exp(-beta*s[i]*s[i+1]))
    elif i == size-1:
        return np.exp(-beta*s[i-1]*s[i])/(np.exp(beta*s[i-1]*s[i])+np.exp(-beta*s[i-1]*s[i]))
    else:
        return np.exp(-beta*(s[i-1]*s[i]+s[i]*s[i+1]))\
               /(np.exp(beta*(s[i-1]*s[i]+s[i]*s[i+1]))+np.exp(-beta*(s[i-1]*s[i]+s[i]*s[i+1])))

def spin_sum():
    spin_sum = 0
    for i in range(size-1):
        spin_sum += s[i]*s[i+1]
    return spin_sum

def E_sample(T):
    E_sample = np.zeros(step)
    beta = 1/T
    for j in range(step):
        for k in range(size):
            i = np.random.randint(0, size)
            r = np.random.random()
            if r <=p(i, beta):
                s[i] = -s[i]
            else:
                pass
        E_sample[j] += -spin_sum()
    return E_sample

def E_ave(T):
    return np.mean(E_sample(T))

def C(T):
    return np.var(E_sample(T))/(T**2)

T_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0])
E_array = np.zeros(len(T_array))
C_array = np.zeros(len(T_array))

for i in range(len(T_array)):
    E_array[i] = E_ave(T_array[i])
    C_array[i] = C(T_array[i])

graph1 = plt.figure(1)
plt.plot(T_array, E_array)
plt.xlabel('T')
plt.ylabel('E(T)')
plt.show()

graph2 = plt.figure(2)
plt.plot(T_array, C_array)
plt.xlabel('T')
plt.ylabel('C(T)')
plt.show()