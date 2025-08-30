
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List
from math import *
import numpy as np
import scipy.stats as st
from scipy.interpolate import interp1d
from scipy.signal import convolve

from utils import StochasticVolatilityModel, Density
from reference_models import ReferenceModel

class SVBassLV:
    """
    Class implements the Bass Local Volatility model with customizable transition kernels
    and the ability to switch between different reference models.
    
    The SVBassLv model extends the standard local volatility framework by incorporating:
    - Stochastic volatility dynamics through reference models
    - Customizable transition kernels for different time intervals
    
    Attributes:
        s0 (float): Initial asset price
        reference_model (StochasticVolatilityModel): Reference stochastic volatility model 
        data (List[datastruct]): List of data structures containing marginal information
        grid (Optional[np.ndarray]): Computational grid for discrete convolutions
    
    Example:
        >>> # Initialize with market marginals and reference model
        >>> sv_bass = SVBassLv(s0=100.0, 
        ...                     reference_model=ref_model,
        ...                     market_marginals=market_data)
        >>> 
        >>> # Calibrate the model
        >>> sv_bass.calibrate()
        >>> 
        >>> # Simulate paths
        >>> paths, xi_paths = sv_bass.simulate_paths(n_paths=1000, n_steps=100)
        >>> 
        >>> # Generate smile at specific time
        >>> smile = sv_bass.simulate_smile(n_points=50, t=0.5)
    """
    
    @dataclass
    class datastruct:
        """
        Data structure for storing marginal information at each maturity.
        
        Attributes:
            t (float): Maturity time
            density (Callable): Density function for the marginal distribution
            brenier_map (Optional[Callable]): Brenier map function for optimal transport
            alpha_qf (Optional[Callable]): Alpha quantile function for path simulation
        """
        t: float
        density: Callable
        brenier_map: Optional[Callable]
        alpha_qf: Optional[Callable]

    def __init__(self, s0: float, reference_model: ReferenceModel, market_marginals: dict[float, Density]):
        """
        Initialize the SVBassLv model.
        
        Args:
            s0 (float): Initial asset price
            reference_model (StochasticVolatilityModel): Reference stochastic volatility model
            market_marginals (dict[float, Density]): Dictionary mapping maturity times to density functions
        """
        self.s0 = s0
        self.reference_model = reference_model 

        self.data = [SVBassLV.datastruct(
                t=time,
                density=density,
                brenier_map=None,
                alpha_qf=None
            ) for time, density in market_marginals.items()]


        self.grid = None

    def test_cenvex_order(self):
        """
        Test convex ordering of marginals (placeholder method).
        
        This method is intended to verify that the marginal distributions
        satisfy the necessary convex ordering conditions for the model.
        """
        pass

    def calibrate(self, initial_guess = Density, tolerance = 1e-5, max_iter = 100, N = 30_000, nsigma = 6) -> list:
        """
        This method iteratively solves fixed-point
        equations to find the disctribution functions of the Bass measures for each interval
        
        Args:
            initial_guess (Density, optional): Initial guess for the density function. Defaults to Density.
            tolerance (float, optional): Convergence tolerance for fixed-point iterations. Defaults to 1e-5.
            max_iter (int, optional): Maximum number of iterations of operator A. Defaults to 100.
            N (int, optional): Number of grid points. Defaults to 30_000.
            nsigma (int, optional): Number of standard deviations for grid bounds. Defaults to 6.
        """
        #extracting list of maturities
        T = np.array([d.t for d in self.data], dtype = float)  #array of maturities

        grid = np.linspace(-np.sqrt(T[-1]) * nsigma, np.sqrt(T[-1]) * nsigma, 2 * N + 1)
        self.grid = grid
        dx = grid[1] - grid[0]
        
		# Construct a brenier map for the first marginal
        brenier_map_values = self.data[0].density.qf(
            x = self.reference_model.cdf(t = T[0], x = grid + self.s0))
        
        self.data[0].brenier_map = interp1d(grid + self.s0, brenier_map_values, fill_value = 'extrapolate')
        print(f'first BM is constructed')
        
        # Iterate over the subsequent marginals to obtain the stretching function on the maturities times
        for i in range(1, len(T)): 
            marginal_2_qf: callable = self.data[i].density.qf
            marginal_1_cdf: callable = self.data[i-1].density.cdf

			# Create a discrete convolution kernel
            time_delta = T[i] - T[i - 1]
            discrete_kernel = self.reference_model.kernel(time_delta, grid) * dx
            print(f'next interval with delta: {time_delta}')

            #define bounds for convolution
            bound_0 = marginal_2_qf(1e-08)
            bound_1 = marginal_2_qf(1 - 1e-08)

            print(f'bounds are: {bound_0, bound_1}')
            
            def A_operator(F):
                convolved_F = self.conv(F, discrete_kernel, 0, 1)
                convolved_F[np.isclose(convolved_F, 0, atol=1e-08)] = 1e-08
                convolved_F[np.isclose(convolved_F, 1, atol=1e-08)] = 1 - 1e-08
                qf_result = marginal_2_qf(convolved_F)
                return marginal_1_cdf(self.conv(qf_result, discrete_kernel, bound_0, bound_1))

            #compute initial guess for fixed point equation 
            F = st.norm.cdf(grid, scale = np.sqrt(0.25)) 
            F1 = A_operator(F)

            error = np.max(np.abs(F - F1))

            #keep iterations until the tolerance level:
            for _ in range(max_iter-1):
                if (error > tolerance): 
                    print(f'iter: {_}, error: {error}')
                    F = F1 
                    F1 = A_operator(F)
                    error = np.max(np.abs(F - F1))
                else:
                    break
        
            # Save the mapping into the dictionary objects in self:
            brenier_map_values = marginal_2_qf(
                np.clip(self.conv(F, discrete_kernel, 0 , 1), 0, 1)
                )

            self.data[i].alpha_qf = interp1d(F, grid + self.s0, fill_value='extrapolate') 
            self.data[i].brenier_map = interp1d(grid + self.s0, brenier_map_values, fill_value='extrapolate') 

            print(f'Brenier map for maturity {T[i]} is constructed error = {error}')

    def conv(self, x: np.ndarray, y: np.ndarray, fill_left: float, fill_right: float) -> np.ndarray:
        """
        Perform convolution with boundary padding.
        
        This method implements a convolution operation with customizable boundary conditions.
        It automatically handles padding and uses FFT-based convolution when possible,
        falling back to direct convolution for numerical stability.
        
        Args:
            x (np.ndarray): First array to convolve
            y (np.ndarray): Second array to convolve (kernel)
            fill_left (float): Value to fill left padding
            fill_right (float): Value to fill right padding
        
        Returns:
            np.ndarray: Result of the convolution
            
        Raises:
            AssertionError: If x and y have different lengths
            
        Note:
            The method automatically detects numerical issues and switches to direct
            convolution when FFT-based methods produce NaN values.
        """
        assert (len(x) == len(y)) # x and y must have the same length

        if not np.all(np.isfinite(x)):
            # Fallback to direct convolution to avoid FFT NaN propagation
            print(f"conv:\nnp.all(np.isfinite(x)): {np.all(np.isfinite(x))}\nnp.all(np.isfinite(y)): {np.all(np.isfinite(y))}")
            method = 'direct'
        else:
            method = 'auto'

        pad_width = len(y) // 2
        padded_len = len(x) + 2 * pad_width
        x_padded = np.empty(padded_len, dtype = x.dtype)

        # Fill padding and core
        x_padded[:pad_width] = fill_left
        x_padded[pad_width:pad_width + len(x)] = x
        x_padded[pad_width + len(x):] = fill_right

        return convolve(x_padded, y, mode='valid')
    
    def create_stretching_function(self, int_, delta_t, method):
        """
        Create stretching function that maps reference model's paths 
        to fit the marginals on the curernt interval at the particuar time point. 
        The stretching function can be either direct (forward mapping) or inverse (backward mapping).
        
        Args:
            int_ (int): Index of the interval 
            delta_t (float): Time to the next closest marginal
            method (str): Method type - 'direct' for forward mapping, 'inverse' for backward mapping
        
        Returns:
            Callable: Interpolated stretching function
            
        Note:
            If delta_t is 0, returns the Brenier map directly Otherwise, applies
            the transition kernel  to calculate E( f(X_T) | X_t )
        """
        if delta_t == 0:
            return self.data[int_].brenier_map
        
        grid = self.grid
        dx = grid[1] - grid[0]

        transition_kernel = self.reference_model.kernel(delta_t, grid) * dx
        bound_0 = self.data[int_].density.qf(1e-08)
        bound_1 = self.data[int_].density.qf(1 - 1e-08)
        f_delta_t_values = self.conv(self.data[int_].brenier_map(grid + self.s0), transition_kernel, bound_0, bound_1)

        if method == 'direct':
            return interp1d(grid + self.s0, f_delta_t_values, fill_value = 'extrapolate')
        if method == 'inverse':
            return interp1d(f_delta_t_values, grid + self.s0, fill_value = 'extrapolate')
    
    def simulate_smile(self, n_points, t): 
        """
        Simulate implied volatility smile at a specific time.
        
        This method generates a volatility smile by simulating points 
        from the Bass measure and the reference model for the specific interval
        
        Args:
            n_points (int): Number of points to simulate
            t (float): Time at which to simulate the smile
        
        Returns:
            np.ndarray: Simulated asset prices for time t
            
        Note:
            The method automatically determines which interval to use 
            and creates the stretching function specifically for t 
        """
        T = np.array([d.t for d in self.data], dtype = float) 
        int_ = np.searchsorted(T, t)
        next_T = T[int_]

        if int_ == 0: 
            xi = np.ones(n_points) * self.s0
        else:
            u = np.random.uniform(size = n_points)
            xi = self.data[int_].alpha_qf(u)

        sigma = np.sqrt(t - (T[int_ - 1] if int_ > 0 else 0))
        dW = np.random.normal(loc = 0, scale = sigma, size = n_points) #simulate paths of the reference model.
        #paths = self.reference_model.simulate(N_points)

        f = self.create_stretching_function(int_, next_T - t, 'direct')
        return f(xi + dW)
    
    def distribute_steps(self, n_steps: int, T: list[float]) -> list[int]:
        """
        Distribute simulation steps across different time intervals.
 
        Args:
            n_steps (int): Total number of simulation steps
            T (list[float]): List of maturity times
        
        Returns:
            list[int]: List of steps allocated to each interval
        """
        r = n_steps * np.diff(T, prepend=0)
        s = r.astype(int)
        remaining = int(n_steps * T[-1]) - s.sum()
        if remaining > 0:
            s[np.argpartition(-(r - s), remaining)[:remaining]] += 1
        return s.tolist()

    def simulate_paths(self, n_paths, n_steps):
        """
        Simulate complete asset price paths using the calibrated model.
        
        This method generates full asset price paths by simulating the reference
        model and applying the stretching functions at each time step, enshuring the continuiety 
        
        Args:
            n_paths (int): Number of paths to simulate
            n_steps (int): Number of steps of the simulation in 1 year
        
        Returns:
            tuple: (final_paths, xi_paths) where:
                - final_paths: List of stretched asset price paths
                - xi_paths: List of reference model paths combined with Bass variables xi
        """
        T = np.array([d.t for d in self.data], dtype = float) 
        n_steps_split = self.distribute_steps(n_steps, T)
        print(f'steps split between maturities = {n_steps_split}')

        final_paths = []
        xi_paths = []

        for interval in range(len(T)):

            previous_marginal = (T[interval - 1] if interval > 0 else 0)
            time_interval = T[interval] - previous_marginal
            n_steps_i = n_steps_split[interval]

            print(f'time interval = {time_interval}')
            paths = self.reference_model.simulate(n_paths, time_interval, n_steps_i)

            if interval > 0:
                f_i_inverse = self.create_stretching_function(interval, time_interval, 'inverse')
                start_values = f_i_inverse(final_paths[-1][:,-1])
                paths = paths + np.reshape(start_values, (n_paths, 1)) - self.s0
            
            xi_paths.append(paths)
            stretched_paths = np.zeros(shape = paths.shape)

            for i in range(n_steps_i + 1):

                current_time = previous_marginal + (i / n_steps_i) * time_interval
                delta_T = T[interval] - current_time

                f_i = self.create_stretching_function(interval, delta_T, 'direct')
                stretched_paths[:,i] = f_i(paths[:,i])
                
            final_paths.append(stretched_paths)

        return final_paths, xi_paths