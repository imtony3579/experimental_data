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

def force(arr,time, low, larg):
	a = []
	for index, val in enumerate(time):
		if time[index]>low and time[index]<larg:
			a.append(arr[index])
	return sum(a)/float(len(a))

def filter(y,n = 50):
    fft1 = rfft(y)
    bp = fft1[:]
    for i in range(len(bp)):
        if i>=n:bp[i]=0
    ibp = irfft(bp)
    return ibp

def adj_d(y, n):
    for i in range(n):
        y[i] = 0

    return y

def adj_data(y):
    a = min(y[0:3500,1])
    point =list(y[:,1]).index(a)
    if point <= 1000:
        point = 1810
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
def input(imp, op, time, old='no'):
	if old == 'flat':
		x = ifft(fft(op)/(fft(imp)*fft(time)))	
		return x
	else:
		x = ifft((fft(op)/(fft(imp*fft(time)))))
		return x

def drag_calc(name,fil,force,time):
	if name == 'blunt':
		low, larg = 0.003, 0.0032
		a = []
		for index, val in enumerate(time):
			if time[index]>low and time[index]<larg:
				a.append(fil[index])
		temp = max(a)
	if name == 'cup':
		low, larg = 0.003, 0.0032
		a = []
		for index, val in enumerate(time):
			if time[index]>low and time[index]<larg:
				a.append(fil[index])
		temp = max(a)
	if name == 'flat':
		low, larg = 0.003, 0.0032
		a = []
		for index, val in enumerate(time):
			if time[index]>low and time[index]<larg:
				a.append(fil[index])
		temp = max(a)
	area = 0.002376
	mach = 8.8
	gama = 1.4
	s = 0.145e-06
	p_ratio = (1 + (0.2*(mach*mach)))**(gama/(gama-1))
	p_0 = temp / s
	p_inf = p_0 / p_ratio
	c_d = force/(gama*0.5*mach*mach*p_inf*area)
	print("pinf= {},p_0={}, c_d={}, press_ratio={}".format(p_inf, p_0, c_d, p_ratio))


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

#cup 02052019_1825 06052019_1605 07052019_1245  
#blunt 30042019_1500 30042019_1920 
#flat 07052019_1450

#imp = read_imp('blunt_impulseResfun.csv')
#imp = read_imp('cup_impulseResfun.csv')
imp = read_imp('flat_impulseResfun.csv')

data1 = getting_file("07052019_1450",4)
data2 = getting_file("07052019_1450",6)
data2[:,1] = filter(data2[:,1])

data3, n = adj_data(data2)
dt = data1[1,0] - data1[0,0]
freque = fftfreq(data1[:,0].size, dt)

#data3[:,1] = filter(data3[:,1], freque, 100000)
#data1[:,1] = filter(data1[:,1], freque, 100000)
#data1[:,1] = filter(data1[:,1])

#freque = fftfreq(data3[n:7014+n,0].size, 1e-06)
#i_p = input(imp[:,1], data3[0:7014,1],data3[0:7014,0], 'flat')

i_p = input(imp[:,1], data3[0:7014,1],data3[0:7014,0], 'flat')
i_p = adj_d(filter(i_p),n)


forc = force(i_p,data3[0:7014,0], 0.0037, 0.005)
print(forc)
#plt.figure()
##plt.plot(data2[:,0],data2[:,1])
#plt.plot(data1[:,0],data1[:,1])
#plt.show()

plt.figure()
plt.plot(data3[0:7014,0],np.real(i_p))
plt.title('Drag force N')
plt.xlabel('Time (s)')
plt.grid(True)
plt.ylabel('Applied load, F(t) (N)')
#plt.plot(data2[0:7014,0],data2[0:7014,1])
plt.show()


drag_calc('blunt',data1[:,1],forc,data1[:,0])
