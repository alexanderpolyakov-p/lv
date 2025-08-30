from dataclasses import dataclass
from math import *
import numpy as np
import scipy.stats as st
from scipy.optimize import newton

def mc_iv(s0, t, k, s):
    v = np.mean(np.maximum(np.subtract.outer(s, k), 0), axis=0)
    return implied_vol(s0, t, k, v)

def bs_call_price(s, sigma, t, k):
    d1 = (np.log(s/k) + (0.5*sigma**2)*t) / (sigma*np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    return s*st.norm.cdf(d1) - k*st.norm.cdf(d2)

@np.vectorize
def implied_vol(s, t, k, v, theta=1):
    x = log(s/k)
    p = v/sqrt(s*k)
    if np.isclose(x, 0):  # Явная формула для опционов ATM
        return -2*st.norm.ppf((1-p)/2)/sqrt(t)
    def equation(sigma):
        return theta*(exp(x/2)*st.norm.cdf(theta*(x/sigma+sigma/2)) - exp(-x/2)*st.norm.cdf(theta*(x/sigma-sigma/2))) - p
    try:
        return newton(equation, x0=sqrt(2*abs(x)))/sqrt(t)
    except:
        return nan

class StochasticVolatilityModel: #can be Brownian motion.
    
    def __init__(self, s0: float):
        self.s0 = s0
    
    def kernel(self, delta_t: float, x: np.array):
        return st.norm.pdf(x, loc = 0, scale=np.sqrt(delta_t))

    def cdf(self, t: float, x: np.array): 
       return st.norm.cdf(x, loc = 0, scale=np.sqrt(t))

    def qf(self, t: float, x: np.array): 
        return st.norm.ppf(x, loc = 0, scale=np.sqrt(t))
    
    def simulate(self, n_paths: int, T: float, n_steps: int):
        dt = T / n_steps
        increments = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = 0
        paths[:, 1:] = np.cumsum(increments, axis=1)
        return paths
    
@dataclass
class Density:
	cdf: callable 
	qf: callable