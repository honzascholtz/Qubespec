from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import truncnorm
from scipy.stats import lognorm
from scipy.stats import loguniform
import numpy as np
#import numba
#@numba.njit
def logprior_general_scipy_test(theta, priors):
    results = []
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results.append(norm.logpdf(t, p[1], p[2]))
        elif p[0]==1:
            results.append(uniform.logpdf(t, p[1], p[2]-p[1]))
        elif p[0]==2:
            results.append(  norm.logpdf(np.log10(t), p[1], p[2]))
        elif p[0]==3:
            results.append( uniform.logpdf(np.log10(t), p[1], p[2]-p[1]))
        elif p[0]==4:
            results.append(truncnorm.logpdf(t, a= (p[3]-p[1])/p[2], b= (p[4]-p[1])/p[2], loc=p[1], scale=p[2]))
        elif p[0]==5:
            results.append( truncnorm.logpdf(np.log10(t),a= (p[3]-p[1])/p[2], b= (p[4]-p[1])/p[2], loc=p[1], scale=p[2]))
    return results

def logprior_general_scipy(theta, priors):
    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results+= norm.logpdf(t, p[1], p[2])
        elif p[0]==1:
            results+= uniform.logpdf(t, p[1], p[2]-p[1])
        elif p[0]==2:
            results+=  norm.logpdf(np.log10(t), p[1], p[2])
        elif p[0]==3:
            results+= uniform.logpdf(np.log10(t), p[1], p[2]-p[1])
        elif p[0]==4:
            results += truncnorm.logpdf(t, a= (p[3]-p[1])/p[2], b= (p[4]-p[1])/p[2], loc=p[1], scale=p[2])
        elif p[0]==5:
            results += truncnorm.logpdf(np.log10(t),a= (p[3]-p[1])/p[2], b= (p[4]-p[1])/p[2], loc=p[1], scale=p[2])
    return results


import numba
@numba.njit
def logprior_general(theta, priors):
    results = 0.
    for t,p in zip( theta, priors):
        if p[0] ==0:
            results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            #results+= norm.pdf(t, p[1], p[2])
        elif p[0]==1:
            results+= np.log((p[1]<t<p[2])/(p[2]-p[1])) 
        elif p[0]==2:
            results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results+= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))
        elif p[0]==4:
            if p[3]<t<p[4]:
                results += -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            else:
                results += -np.inf
        elif p[0]==5:
            if p[3]<np.log10(t)<p[4]:
                results+= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
            else:
                results += -np.inf
    
    return results


@numba.njit
def logprior_general_test(theta, priors, labels):
    for t,p,lb in zip( theta, priors, labels):
        print(p)
        if p[0] ==0:
            results = -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
        elif p[0]==1:
            results= np.log((p[1]<t<p[2])/(p[2]-p[1])) 
        elif p[0]==2:
            results= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
        elif p[0]==3:
            results= np.log((p[1]<np.log10(t)<p[2])/(p[2]-p[1]))
        elif p[0]==4:
            if p[3]<t<p[4]:
                results = -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((t-p[1])/p[2])**2
            else:
                results = -np.inf
        elif p[0]==5:
            if p[3]<np.log10(t)<p[4]:
                results= -np.log(p[2]) - 0.5*np.log(2*np.pi) - 0.5 * ((np.log10(t)-p[1])/p[2])**2
            else:
                results = -np.inf
    
        print(lb, t, results)
