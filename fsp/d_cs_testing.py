import scipy.io as sio
import d_cs

data = sio.loadmat('mvnrnd01.mat')
#print(data['X'])

A = data['X'][:350,:]
B = data['X'][350:,:]
#print(A.shape)
#print(B.shape)

Divergence_Measure_Case = 2
print(d_cs.Divergence_Measure(A,B,Divergence_Measure_Case))