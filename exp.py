import pandas as pd
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import style
import scipy
style.use('ggplot')

#def filter(y,freq, n=10000):
#    for index, val in enumerate(y):
#        if abs(freq[index]) > n:
#            y[index] = 0.
#    return y


def filter(y,n = 50):
    fft1 = rfft(y)
    bp = fft1[:]
    for i in range(len(bp)):
        if i>=n:bp[i]=0
    ibp = irfft(bp)
    return ibp

def adj_data(y):
    a = min(y[0:3500,1])
    point =list(y[:,1]).index(a)
    y[:,1] = y[:,1] - a
    for i in range(point):
        y[i,1] = 0

    return y, point

def read_imp(name):
   fil = pd.read_csv(name,index_col=False)
   fil_arr = fil.values
   fil_arr = np.delete(fil_arr,0,1)
   a = [complex(i) for i in fil_arr[:,1]]
   fil_arr[:,1] = a
   return fil_arr

#in[]
def input(imp, op, time):
    x = ifft(fft(op)/(fft(imp)*fft(time)))
    
    return x

def getting_file(name, num_sheet):
    
    x1 = pd.ExcelFile(name+".xlsx")
    #print(x1.sheet_names)
    sheet = x1.sheet_names
    df = pd.DataFrame()
    df = df.append(pd.read_excel(name+".xlsx",sheet[num_sheet]))
    df = df[8:-1]
    df = df.drop(df.columns[[-1]], axis = 'columns')
    a = df.values
    #print(data_tim)
    #data_tim = filter(a)
    return a



#02052019_1500 02052019_1825 02052019_1910 06052019_1605 07052019_1245 07052019_1450
#26042019_1508 30042019_1500 30042019_1545 x30042019_1653 x30042019_1825
#30042019_1920

imp = read_imp('blunt_impulseResfun.csv')
#imp = read_imp('cup_impulseResfun.csv')
#imp = read_imp('flat_impulseResfun.csv')

data1 = getting_file("30042019_1500",0)
data2 = getting_file("30042019_1500",2)
data2[:,1] = filter(data2[:,1])
data3,n = adj_data(data2)
dt = data1[1,0] - data1[0,0]
freque = fftfreq(data1[:,0].size, dt)

#data3[:,1] = filter(data3[:,1], freque, 100000)
#data1[:,1] = filter(data1[:,1], freque, 100000)
#data1[:,1] = filter(data1[:,1])

#freque = fftfreq(data3[n:7014+n,0].size, 1e-06)
i_p = input(imp[:,1], data3[n:7014+n,1],data3[n:7014+n,0])
i_p = filter(i_p)
#plt.figure()
#print(data1[:,0])
#plt.plot(data3[:,0],data3[:,1])
#plt.plot(data1[:,0],data1[:,1])
#plt.show()

plt.figure()
plt.plot(data3[n:7014+n,0],i_p)
plt.plot(data2[n:7014+n,0],data2[n:7014+n,1])
plt.show()
#


