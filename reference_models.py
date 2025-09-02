
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
    
    
class ArithmeticSABR_R(ReferenceModel): 

    def __init__(self, s0: float, alpha: float, rho: float, nu: float):
        super().__init__(s0)
        self.alpha = alpha 
        self.rho = rho
        self.nu = nu 

        self.N = 30_000
        self.nsigma = 6

    #static method
    def kernel(self, delta_t: float, x: np.array):
        grid = np.linspace(- self.nsigma * np.sqrt(delta_t), self.nsigma * np.sqrt(delta_t), 2 * self.N + 3)
        strikes = grid + self.s0 + 0.000001 #костыль!
        dx = grid[1] - grid[0]

        pdf = np.diff(bs_call_price(self.s0, 
                                    self.iv(delta_t, strikes), 
                                    delta_t,
                                    strikes), n = 2)
        
        pdf[np.isclose(pdf, 0, atol = 1e-20)] = 0
        spl = make_smoothing_spline(grid[1:-1], pdf/(dx**2), lam=0)

        index = np.logical_and(x > grid[0], x < grid[-1])
        result = np.zeros(len(x))
        result[index] = spl(x)[index]
        return result

    def cdf(self, t: float, x: np.array): 
        strikes = np.linspace(- self.nsigma * np.sqrt(t), self.nsigma * np.sqrt(t), 2 * self.N + 3) + self.s0 + 0.000001
            
        dx = strikes[1] - strikes[0]

        cdf_values = np.diff(bs_call_price(self.s0,
                                           self.iv(t, strikes),
                                           t,
                                           strikes), n = 1)/dx + 1

        return interp1d(strikes[1:], cdf_values, kind = 'linear', bounds_error = False, fill_value = (0,1))(x)

    def qf(self, t: float, x: np.array): 
        assert np.all(np.logical_and(x <= 1, x >= 0))
        grid = np.linspace(- self.nsigma * np.sqrt(t), self.nsigma * np.sqrt(t), 2 * self.N + 3)
        strikes = grid + self.s0 + 0.000001
        dx = grid[1] - grid[0]

        cdf_values = np.diff(bs_call_price(self.s0,
                                           self.iv(t, strikes),
                                           t,
                                           strikes), n = 1)/dx + 1

        bounds = (cdf_values[0], cdf_values[-1])
        return interp1d(cdf_values, strikes[1:], kind = 'linear', bounds_error = False, fill_value = bounds)(x)
    
    def simulate(self, n_paths: int, T: float, n_steps: int, add_antithetic: bool = False):
        dt = T/n_steps
        sqrtdt = np.sqrt(dt)
        Z = st.norm.rvs(size=(2, n_steps, n_paths))

        if add_antithetic:
            Z = np.concatenate((Z, -Z), axis=2)
            n_paths *= 2

        Alpha = self.alpha*np.exp(np.concatenate([
            np.zeros(shape=(1, n_paths)),
            np.cumsum(Z[0]*self.nu*sqrtdt - 0.5*self.nu**2*dt, axis=0)]))

        S = np.cumsum(np.concatenate([
            np.ones(shape=(1, n_paths))*self.s0,
            Alpha[:-1] *
            (self.rho*Z[0] + np.sqrt(1-self.rho**2)*Z[1])*sqrtdt]), axis=0)
        
        # Transpose to get (n_paths, n_steps+1) format: [path, path, path, ...]
        return S.T

    #model methods 
    def call_price(self, t, k):
        return (bs_call_price(s=self.s0, sigma=self.iv(t, k), t=t, k=k))

    def iv(self, t, k):
        z = self.nu/self.alpha * np.sqrt(self.s0*k) * np.log(self.s0/k)
        x = np.log((np.sqrt(1-2*self.rho*z + z*z) + z - self.rho) /
                    (1-self.rho))
        return (
            self.alpha *

            # when f = k, we must have log(f/k)/(f-k) = k, z/x = 1
            # and we get the following formula
            np.divide(np.log(self.s0 / k)*z, (self.s0 - k)*x, where = np.abs(self.s0 - k) > 1e-12, out=np.array(k, dtype = float)) 
                        * (1 + t * (self.alpha**2 / (24 * self.s0 * k) + (2- 3 * self.rho**2) / 24 * self.nu**2)))