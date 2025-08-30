
from dataclasses import dataclass
import numpy as np
import scipy.stats as st
from utils import bs_call_price, implied_vol, Density
from scipy.interpolate import interp1d, make_smoothing_spline

class ReferenceModel: #can be Brownian motion.
    
    def __init__(self, s0: float):
        self.s0 = s0
    
    def kernel(self, delta_t: float, x: np.array):
       pass

    def cdf(self, t: float, x: np.array): 
       pass

    def qf(self, t: float, x: np.array): 
        pass
    
    def simulate(self, n_paths: int, T: float, n_steps: int):
        pass

class BrownianMotion_R(ReferenceModel): 
    
    def __init__(self, s0: float, sigma: float):
        super().__init__(s0)
        self.sigma = sigma
    
    def kernel(self, delta_t: float, x: np.array):
        return st.norm.pdf(x, loc = 0, scale = self.sigma * np.sqrt(delta_t))

    def cdf(self, t: float, x: np.array): 
       return st.norm.cdf(x, loc = self.s0, scale= self.sigma * np.sqrt(t))

    def qf(self, t: float, x: np.array): 
        return st.norm.ppf(x, loc = self.s0, scale= self.sigma * np.sqrt(t))
    
    def simulate(self, n_paths: int, T: float, n_steps: int):
        dt = T / n_steps
        increments = np.random.normal(0, self.sigma * np.sqrt(dt), (n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = 0
        paths[:, 1:] = np.cumsum(increments, axis=1) 
        return paths + self.s0
    
