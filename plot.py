import numpy as np
import matplotlib.pyplot as plt
from utils import mc_iv

def plot_bass_result(bass, market, n_paths = 1_000_000):
    """
    Plot Bass model results comparing market implied volatilities with Bass Monte Carlo simulations.
    
    Args:
        bass: Calibrated SVBassLV model
        market: Market model (e.g., Heston) with implied_vol method
    """
    T_vals = np.array([d.t for d in bass.data], dtype=float) 
    mc_data = {T: bass.simulate_smile(n_paths, T) for T in T_vals} 
    marginals = [d.density for d in bass.data]

    # Create figure with 3D subplot
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(121)  # 2D plot
    ax2 = fig.add_subplot(122, projection='3d')  # 3D surface plot

    # 2D Plot with improved readability
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Distinct color palette

    for i, T in enumerate(T_vals):
        # Strike range
        marg = marginals[i]
        K = np.linspace(marg.qf(0.01), marg.qf(0.99), 100)

        # Plot market implied volatility
        market_vol = market.implied_vol(T, K)
        ax1.plot(K, market_vol, color='red', linestyle='-', linewidth=3, alpha=0.8, 
                label='Market Model' if i == 0 else "")
        
        # Plot Bass MC implied volatility
        bass_vol = mc_iv(market.s0, T, K, mc_data[T])
        ax1.plot(K, bass_vol, color=colors[i], linestyle='--', linewidth=2.5, alpha=1, 
                label=f'Bass MC T={T}')

    # Improve 2D plot appearance
    ax1.set_xlabel('Strike Price (K)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Implied Volatility (σ)', fontsize=12, fontweight='bold') 
    ax1.set_title('Volatility Smiles: Market vs Bass MC', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Set better axis limits and formatting
    ax1.set_xlim(left=0)  # Start from 0 for strikes
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 3D Surface Plot with fixed strike grid, cut off by maturity-specific bounds
    # Create fixed strike grid covering all maturities
    all_strikes = []
    for T, marginal in zip(T_vals, marginals):
        all_strikes.extend([marginal.qf(0.01), marginal.qf(0.99)])

    # Fixed strike range for all maturities
    K_min, K_max = min(all_strikes), max(all_strikes)
    strikes = np.linspace(K_min, K_max, 30)  # Fixed strike grid
    maturities = np.linspace(0.1, T_vals[-1], 25)   # Maturity grid up to max maturity

    # Generate surface points with fixed strikes, cut off by maturity bounds
    all_points = []
    for T in maturities:
        # Find the closest maturity with defined bounds
        marginal_idx = np.searchsorted(T_vals, T)
        if marginal_idx == 0:
            marginal = marginals[0]
        elif marginal_idx >= len(marginals):
            marginal = marginals[-1]
        else:
            # Use the marginal for the closest maturity
            marginal = marginals[marginal_idx - 1]
        
        # Get bounds for this maturity
        K_min_T = marginal.qf(0.01)
        K_max_T = marginal.qf(0.99)
        
        # Filter strikes that fall within this maturity's bounds
        valid_strikes = strikes[(strikes >= K_min_T) & (strikes <= K_max_T)]
        
        for K in valid_strikes:
            try:
                vol = market.implied_vol(T, K)
                if not np.isnan(vol) and vol > 0:
                    all_points.append([K, T, vol])
            except:
                continue

    # Convert to numpy arrays for triangulation
    all_points = np.array(all_points)
    if len(all_points) > 0:
        K_valid = all_points[:, 0]
        T_valid = all_points[:, 1]
        vol_valid = all_points[:, 2]
        
        # Create triangulation
        from matplotlib.tri import Triangulation
        triang = Triangulation(K_valid, T_valid)
        
        # Plot triangulated surface using plot_trisurf
        surf = ax2.plot_trisurf(K_valid, T_valid, vol_valid, 
                           triangles=triang.triangles,
                           cmap='viridis', alpha=0.8, 
                           linewidth=0.5, antialiased=True)

    # Plot market marginal lines on 3D surface
    for T, marginal in zip(T_vals, marginals):
        K_range = np.linspace(marginal.qf(0.01), marginal.qf(0.99), 100)  # More points for smooth lines
        vol_line = market.implied_vol(T, K_range)
        
        ax2.plot(K_range, [T] * len(K_range), vol_line, 
                color='red', linewidth=5, alpha=1.0, 
                label=f'Market Marginal T={T}', zorder=10)

    # Customize 3D plot
    ax2.set_xlabel('Strike', fontweight='bold')
    ax2.set_ylabel('Maturity', fontweight='bold')
    ax2.set_zlabel('Implied Volatility', fontweight='bold')
    ax2.set_title('3D Volatility Surface with Market Marginals', fontweight='bold')
    ax2.view_init(elev=25, azim=45)  # Set viewing angle

    # Add colorbar for the surface
    if len(all_points) > 0:
        fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()
    return None