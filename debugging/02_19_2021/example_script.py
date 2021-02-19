import numpy as np
import matplotlib.pyplot as plt
from random import sample
import scipy.stats as sps
from surmise.calibration import calibrator 
from surmise.emulation import emulator

real_data = np.loadtxt('real_observations.csv', delimiter=',')
description = np.loadtxt('observation_description.csv', delimiter=',',dtype='object')
param_values = 1/np.loadtxt('param_values.csv', delimiter=',')
func_eval = np.loadtxt('func_eval.csv', delimiter=',')
param_values_test = 1/np.loadtxt('param_values_test.csv', delimiter=',')
func_eval_test = np.loadtxt('func_eval_test.csv', delimiter=',')

keepinds = np.squeeze(np.where(description[:,0].astype('float') > 30))
real_data = real_data[keepinds]
description = description[keepinds, :]
func_eval = func_eval[:,keepinds]
func_eval_test = func_eval_test[:, keepinds]

print('N:', func_eval.shape[0])
print('D:', param_values.shape[1])
print('M:', real_data.shape[0])
print('P:', description.shape[1])

# Get the random sample of 500
rndsample = sample(range(0, 2000), 500)
func_eval_rnd = func_eval[rndsample, :]
param_values_rnd = param_values[rndsample, :]


x = np.hstack((np.reshape(np.tile(range(134), 3), (402, 1)),
              np.reshape(np.tile(np.array(('tothosp','totadmiss','icu')),134), (402, 1))))
x =  np.array(x, dtype='object')

# (No filter) Fit an emulator via 'PCGP'
emulator_1 = emulator(x=x, theta=param_values_rnd, f=func_eval_rnd.T, method='PCGP')
pred_test = emulator_1(theta = param_values_test)

print('RMSE : (as small as possible)' )
print(np.sqrt(np.mean((func_eval_test.T -pred_test.mean())**2)))
print('R2: (as close to one as possible)' )
print(1-np.mean((func_eval_test.T -pred_test.mean())**2)/np.mean((func_eval_test-np.mean(func_eval_test))**2))
print('mean((f-fhat)/sqrt(var)) (should be close to 0):' )
print(np.mean((func_eval_test.T -pred_test.mean())/np.sqrt(pred_test.var())))
print('mean((f-fhat)**2/var)(should be close to 1):' )
print(np.mean((func_eval_test.T -pred_test.mean())**2/pred_test.var()))
print('average normalized value (should be close to 1)):' )
#please make this faster, it is too slow
residstand = np.empty([100, pred_test.covxhalf().shape[2]])
for k in range(0,100):
    residstand[k,:] = np.linalg.pinv(pred_test.covxhalf()[:,k,:]) @ (func_eval_test.T[:,k] -pred_test.mean()[:,k])
print(np.mean(residstand ** 2))


emulator_2 = emulator(x=x, theta=param_values_rnd, f=func_eval_rnd.T, method='PCGPwM')
pred_test = emulator_2(theta = param_values_test)

print('RMSE : (as small as possible)' )
print(np.sqrt(np.mean((func_eval_test.T -pred_test.mean())**2)))
print('R2: (as close to one as possible)' )
print(1-np.mean((func_eval_test.T -pred_test.mean())**2)/np.mean((func_eval_test-np.mean(func_eval_test))**2))
print('mean((f-fhat)/sqrt(var)) (should be close to 0):' )
print(np.mean((func_eval_test.T -pred_test.mean())/np.sqrt(pred_test.var())))
print('mean((f-fhat)**2/var)(should be close to 1):' )
print(np.mean((func_eval_test.T -pred_test.mean())**2/pred_test.var()))
print('average normalized value (should be close to 1)):' )
#please make this faster, it is too slow
residstand = np.empty([100, pred_test.covxhalf().shape[2]])
for k in range(0,100):
    residstand[k,:] = np.linalg.pinv(pred_test.covxhalf()[:,k,:]) @ (func_eval_test.T[:,k] -pred_test.mean()[:,k])
print(np.mean(residstand ** 2))

