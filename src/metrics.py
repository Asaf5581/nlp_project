import numpy as np
from scipy.optimize import curve_fit

def calculate_rho(steps_to_corrupt, steps_to_recover):
    """
    Calculates the recovery ratio rho(d) = steps_to_recover(d) / steps_to_corrupt(d)
    """
    return steps_to_recover / steps_to_corrupt

def exponential_decay(t, y0, y_inf, k):
    """
    Exponential decay function: y(t) = (y0 - y_inf) * exp(-k * t) + y_inf
    """
    return (y0 - y_inf) * np.exp(-k * t) + y_inf

def fit_exponential_decay(steps, toxicity):
    """
    Fits an exponential decay curve to the recovery trajectory.
    Returns optimal parameters: (y0, y_inf, k) and the covariance matrix.
    """
    # Initial guess: 
    # y0 = first toxicity value
    # y_inf = last toxicity value (plateau)
    # k = 0.001 (small positive rate)
    p0 = [toxicity.iloc[0], toxicity.iloc[-1], 0.001]
    
    # Bounds: y0 and y_inf >= 0, k >= 0
    bounds = (0, np.inf)
    
    try:
        popt, pcov = curve_fit(exponential_decay, steps, toxicity, p0=p0, bounds=bounds, maxfev=10000)
        return popt, pcov
    except RuntimeError:
        print("Curve fit failed to converge.")
        return [np.nan, np.nan, np.nan], None

def calculate_half_life(k):
    """
    Calculates the half-life of the exponential decay: ln(2) / k
    """
    if k <= 0 or np.isnan(k):
        return np.nan
    return np.log(2) / k

def bootstrap_ci(data, num_samples=1000, ci=95):
    """
    Computes the bootstrap confidence interval for the mean of the data.
    """
    bootstrapped_means = []
    n = len(data)
    
    if n == 0:
        return float('nan'), float('nan'), float('nan')
        
    for _ in range(num_samples):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_means.append(np.mean(sample))
        
    lower_bound = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower_bound, upper_bound
