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

def plot_pred_interval(cal):
    pr = cal.predict(x)
    rndm_m = pr.rnd(s = 1000)
    plt.rcParams["font.size"] = "10"
    fig, axs = plt.subplots(3, figsize=(8, 12))

    for j in range(3):
        upper = np.percentile(rndm_m[:, j*134 : (j + 1)*134], 97.5, axis = 0)
        lower = np.percentile(rndm_m[:, j*134 : (j + 1)*134], 2.5, axis = 0)
        median = np.percentile(rndm_m[:, j*134 : (j + 1)*134], 50, axis = 0)
        p1 = axs[j].plot(median, color = 'black')
        axs[j].fill_between(range(0, 134), lower, upper, color = 'grey')
        p3 = axs[j].plot(range(0, 134), real_data[j*134 : (j + 1)*134], 'ro' ,markersize = 5, color='red')
        if j == 0:
            axs[j].set_ylabel('COVID-19 Total Hospitalizations')
        elif j == 1:
            axs[j].set_ylabel('COVID-19 Hospital Admissions')
        elif j == 2:
            axs[j].set_ylabel('COVID-19 ICU Patients')
        axs[j].set_xlabel('Time (days)')  
    
        axs[j].legend([p1[0], p3[0]], ['prediction','observations'])
    fig.tight_layout()
    fig.subplots_adjust(top=0.9) 
    plt.show()

# Define a class for prior of 10 parameters
class prior_covid:
    """ This defines the class instance of priors provided to the method. """
    def lpdf(theta):
        return (sps.norm.logpdf(theta[:, 0], 2.5, 0.5) +
                sps.norm.logpdf(theta[:, 1], 4.0, 0.5) + 
                sps.norm.logpdf(theta[:, 2], 4.0, 0.5) + 
                sps.norm.logpdf(theta[:, 3], 1.875, 0.1) + 
                sps.norm.logpdf(theta[:, 4], 14, 1.5) + 
                sps.norm.logpdf(theta[:, 5], 18, 1.5) + 
                sps.norm.logpdf(theta[:, 6], 20, 1.5) + 
                sps.norm.logpdf(theta[:, 7], 14, 1.5) + 
                sps.norm.logpdf(theta[:, 8], 13, 1.5) + 
                sps.norm.logpdf(theta[:, 9], 12, 1.5)).reshape((len(theta), 1))


    def rnd(n):
        return np.vstack((sps.norm.rvs(2.5, 0.5, size=n),
                          sps.norm.rvs(4.0, 0.5, size=n),
                          sps.norm.rvs(4.0, 0.5, size=n),
                          sps.norm.rvs(1.875, 0.1, size=n),
                          sps.norm.rvs(14, 1.5, size=n),
                          sps.norm.rvs(18, 1.5, size=n),
                          sps.norm.rvs(20, 1.5, size=n),
                          sps.norm.rvs(14, 1.5, size=n),
                          sps.norm.rvs(13, 1.5, size=n),
                          sps.norm.rvs(12, 1.5, size=n))).T

obsvar = np.maximum(0.2*real_data, 5)

# Calibrator 1
cal_1 = calibrator(emu=emulator_1,
                   y=real_data,
                   x=x,
                   thetaprior=prior_covid,
                   method='directbayes',
                   yvar=obsvar, 
                   args={'theta0': np.array([[2, 4, 4, 1.875, 14, 18, 20, 14, 13, 12]]), 
                         'numsamp' : 1000,
                         'stepType' : 'normal', 
                         'stepParam' : np.array([0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])})

plot_pred_interval(cal_1)
