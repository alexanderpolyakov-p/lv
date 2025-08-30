from dataclasses import dataclass
from math import *
import cmath
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import newton

from utils import implied_vol
        
# МОЛЕЛЬ ХЕСТОНА
@dataclass
class Heston:
    s0: float       
    v0: float       
    kappa: float     
    theta: float
    xi: float
    rho: float

    def __post_init__(self):
        # Векторизуем функцию вычисления цены опциона, чтобы применять ее к массивам страйков
        self.call_price = np.vectorize(self._call_price, excluded='self')

    # Характеристическая функция логарифма цены акции в момент времени t
    def _cf(self, u: float | complex, t: float) -> complex:
        d = cmath.sqrt((1j*self.rho*self.xi*u - self.kappa)**2 + self.xi**2*(1j*u + u**2))
        g = ((1j*self.rho*self.xi*u - self.kappa + d) / (1j*self.rho*self.xi*u - self.kappa - d))
        C = self.kappa*self.theta/self.xi**2 * ((self.kappa - 1j*self.rho*self.xi*u - d)*t -
                                                2*cmath.log((1 - g*cmath.exp(-d*t))/(1-g)))
        D = (self.kappa - 1j*self.rho*self.xi*u - d)/self.xi**2 * ((1-cmath.exp(-d*t)) /
                                                                   (1-g*cmath.exp(-d*t)))
        return cmath.exp(C + D*self.v0)

    # Цена опциона колл с исполнением в момент t и страйком k
    def _call_price(self, t: float, k: float) -> float:
        def integrand(u):
            return (cmath.exp(1j*u*log(self.s0/k)) / (1j*u) *
                    (self._cf(u-1j, t) - k/self.s0 * self._cf(u, t))).real
        return self.s0 * ((1 - k/self.s0)/2 +
                          1/pi * quad(integrand, 0, inf, epsrel=1e-12, epsabs=1e-20)[0])

    # Подразумеваемая волатильность
    def implied_vol(self, t: float | np.ndarray, k: float | np.ndarray) -> float | np.ndarray:
        return implied_vol(self.s0, t, k, self.call_price(t, k))
    
    # Функция распределения цены
    def cdf(self, t: float, s: float) -> float:
        def integrand(u):
            return (cmath.exp(1j*u*log(self.s0/s)) / (1j*u) * self._cf(u, t)).real
        return 0.5 - 1/pi * quad(integrand, 0, inf)[0]
    
    # Квантиль распределения цены уровня p
    def quantile(self, t: float, p: float) -> float:
        return newton(lambda s: self.cdf(t, s) - p, self.s0)
    
    # Интерполяция функции распределения цены в момент t по n точкам с квантилями
    # 1/n, 2/n, ..., (n-1)/n. Функция распределения полагается равной нулю левее квантили 1/4n,
    # и равной 1 правее квантили 1 - 1/4n.
    # Возвращается объект, который можно вызывать как функцию.
    def cdf_interpolate(self, t: float, n: int = 1000, kind: str = 'linear') -> callable:
        P = np.linspace(0, 1, n+1)
        s_p0 = self.quantile(t, 1/(4*n))
        s_p1 = self.quantile(t, 1 - 1/(4*n))
        S = [s_p0] + [self.quantile(t, p) for p in P[1:-1]] + [s_p1]
        return interp1d(S, P, fill_value = (0, 1), bounds_error=False, kind=kind)

    # Интерполяция квантильной функции в момент t по n точкам
    def quantile_interpolate(self, t: float, n: int = 1000, kind: str = 'linear') -> callable:
        P = np.linspace(0, 1, n+1)
        s_p0 = self.quantile(t, 1/(4*n))
        s_p1 = self.quantile(t, 1 - 1/(4*n))
        S = [s_p0] + [self.quantile(t, p) for p in P[1:-1]] + [s_p1]
        return interp1d(P, S, bounds_error=True, kind=kind)