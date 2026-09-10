import numpy as np


# Individual metric functions
# Each takes (observed, simulated) as 1D numpy arrays and returns a scalar.
# Metrics optimal at 1 (NSE, KGE, correlation) should be used with
# to_minimize() when passed to the Calibrator.

def nse(observed, simulated):
    """
    Nash-Sutcliffe Efficiency (NSE).

    Measures the relative magnitude of the residual variance compared
    to the observed variance. Sensitive to high flows due to the
    squared residuals.

    Range  : (-inf, 1], optimal at 1.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed timeseries of length t.
    simulated : numpy.ndarray
        Simulated timeseries of length t.

    Returns
    -------
    float
    """
    return 1 - (np.sum((observed - simulated)**2) /
                np.sum((observed - np.mean(observed))**2))


def log_nse(observed, simulated):
    """
    Log-transformed Nash-Sutcliffe Efficiency.

    Applies a log transformation before computing NSE, reducing
    sensitivity to high flows and increasing sensitivity to low
    flows. Useful for catchments where low flow behaviour is
    important.

    Range  : (-inf, 1], optimal at 1.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed timeseries of length t. Must be strictly positive.
    simulated : numpy.ndarray
        Simulated timeseries of length t. Must be strictly positive.

    Returns
    -------
    float
    """
    log_obs = np.log(observed + 1e-10)
    log_sim = np.log(simulated + 1e-10)
    return 1 - (np.sum((log_obs - log_sim)**2) /
                np.sum((log_obs - np.mean(log_obs))**2))


def kge(observed, simulated):
    """
    Kling-Gupta Efficiency (KGE).

    Decomposes model performance into correlation, bias, and
    variability components, giving a more balanced assessment
    than NSE. Less sensitive to high flow peaks.

    Range  : (-inf, 1], optimal at 1.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed timeseries of length t.
    simulated : numpy.ndarray
        Simulated timeseries of length t.

    Returns
    -------
    float
    """
    r = np.corrcoef(observed, simulated)[0, 1]
    alpha = np.std(simulated) / np.std(observed)     # variability ratio
    beta = np.mean(simulated) / np.mean(observed)    # bias ratio
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def rmse(observed, simulated):
    """
    Root Mean Square Error (RMSE).

    Measures the average magnitude of the residuals, with higher
    weight on large errors due to the squaring. Expressed in the
    same units as the observed variable.

    Range  : [0, inf), optimal at 0.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed timeseries of length t.
    simulated : numpy.ndarray
        Simulated timeseries of length t.

    Returns
    -------
    float
    """
    return np.sqrt(np.mean((observed - simulated)**2))


def nrmse(observed, simulated):
    """
    Normalized Root Mean Square Error (NRMSE).

    RMSE normalized by the mean of the observed timeseries, making
    it dimensionless and comparable across variables with different
    units or magnitudes. Useful when combining discharge and
    concentration objectives.

    Range  : [0, inf), optimal at 0.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed timeseries of length t. Must have non-zero mean.
    simulated : numpy.ndarray
        Simulated timeseries of length t.

    Returns
    -------
    float
    """
    return rmse(observed, simulated) / np.mean(observed)


def mae(observed, simulated):
    """
    Mean Absolute Error (MAE).

    Measures the average magnitude of the residuals without
    squaring, giving equal weight to all errors regardless of
    magnitude. More robust to outliers than RMSE.

    Range  : [0, inf), optimal at 0.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed timeseries of length t.
    simulated : numpy.ndarray
        Simulated timeseries of length t.

    Returns
    -------
    float
    """
    return np.mean(np.abs(observed - simulated))


def pbias(observed, simulated):
    """
    Percent Bias (PBIAS).

    Measures the average tendency of the simulated values to be
    larger or smaller than the observed. Positive values indicate
    model underestimation, negative values indicate overestimation.

    Range  : (-inf, inf), optimal at 0.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed timeseries of length t.
    simulated : numpy.ndarray
        Simulated timeseries of length t.

    Returns
    -------
    float
        Percent bias as a percentage.
    """
    return 100 * np.sum(observed - simulated) / np.sum(observed)


# ===========================================================================
# Transformation utilities
# ===========================================================================

def to_minimize(metric_function):
    """
    Transforms a metric that is optimal at 1 (e.g. NSE, KGE) into
    one that is optimal at 0, suitable for minimization by the
    Calibrator.

    Parameters
    ----------
    metric_function : callable
        A metric function of the form f(observed, simulated) -> float
        that is optimal at 1.

    Returns
    -------
    callable
        Transformed function optimal at 0.

    Example
    -------
    objective = to_minimize(nse)
    calibrator = Calibrator(..., objective_function=objective)
    """
    def wrapper(observed, simulated):
        return 1 - metric_function(observed, simulated)
    wrapper.__name__ = f'to_minimize({metric_function.__name__})'
    return wrapper


def to_minimize_negative(metric_function):
    """
    Transforms a metric that is optimal at 0 but can be negative
    (e.g. PBIAS) into one suitable for minimization by taking the
    absolute value.

    Parameters
    ----------
    metric_function : callable
        A metric function of the form f(observed, simulated) -> float
        that is optimal at 0.

    Returns
    -------
    callable
        Transformed function returning the absolute value.
    """
    def wrapper(observed, simulated):
        return np.abs(metric_function(observed, simulated))
    wrapper.__name__ = f'to_minimize_negative({metric_function.__name__})'
    return wrapper


# ===========================================================================
# Composite objective function builder
# ===========================================================================

def composite_objective(components):
    """
    Builds a single composite objective function from multiple metrics,
    each applied to a specific output variable and weighted by a scalar.
    The result is a weighted sum suitable for minimization.

    All component metrics must already be in minimization form (optimal
    at 0). Use to_minimize() or to_minimize_negative() to transform
    metrics beforehand if needed.

    Parameters
    ----------
    components : list of tuples
        Each tuple defines one component as:
        (metric_function, weight, output_index)
        where:
        - metric_function : callable
            A metric function f(observed, simulated) -> float,
            optimal at 0.
        - weight : float
            Relative weight of this component in the composite score.
            Weights do not need to sum to 1 but it is recommended
            for interpretability.
        - output_index : int
            Index into the observed and simulated tuples identifying
            which output variable this metric applies to (e.g. 0 for
            discharge, 1 for concentration).

    Returns
    -------
    callable
        Composite objective function of the form
        f(observed, simulated) -> float, suitable for passing
        directly to the Calibrator.

    Example
    -------
    objective = composite_objective([
        (to_minimize(nse),  0.6, 0),   # NSE on discharge, weight 0.6
        (nrmse,             0.4, 1),   # NRMSE on concentration, weight 0.4
    ])
    calibrator = Calibrator(..., objective_function=objective)
    """
    def objective(observed, simulated):
        score = 0.0
        for metric_function, weight, output_index in components:
            score += weight * metric_function(
                observed[output_index],
                simulated[output_index]
            )
        return score

    return objective

# ===========================================================================
# MCMC log likelihood functions
# ===========================================================================

def gaussian_log_likelihood(observed, simulated, sigma):
    """
    Log-likelihood assuming independent and identically distributed
    Gaussian errors with standard deviation sigma.

    Parameters
    ----------
    observed : tuple(numpy.ndarray)
        Tuple of observed timeseries.
    simulated : tuple(numpy.ndarray)
        Tuple of simulated timeseries.
    sigma : float
        Standard deviation of the residuals. Must be positive.

    Returns
    -------
    float
        Log-likelihood value. Returns -inf for invalid sigma.
    """
    if sigma <= 0:
        return -np.inf

    residuals = observed - simulated
    return -0.5 * np.sum(residuals**2 / sigma**2
                         + np.log(2 * np.pi * sigma**2))


def ar1_log_likelihood(observed, simulated, phi, sigma):
    """
    Log-likelihood for an AR(1) autocorrelated error model.

    Assumes residuals follow:
        epsilon_t = phi * epsilon_{t-1} + eta_t
        eta_t ~ N(0, sigma^2)
    
    /!\/!\/!\ 
    Assumes that all observations are recorded at the same frequency,
    i.e. does not account for varying observation intervals in the real-world data

    If residuals do not correspond to a constant timestep, the likelihood is misspecified in a systematic way.
    For pairs of observations close together in time, the true autocorrelation is high — residuals genuinely carry
    information about their neighbor. For pairs far apart, the residuals are nearly independent. The fixed-ϕ\phi
    ϕ AR(1) treats all pairs identically, which means:

    - It underestimates the dependence between close observations → overstates their independent information content → posterior too narrow for those periods
    - It overestimates the dependence between distant observations → understates their information content → posterior too wide for those periods

    Should switch to a Continuous-Time AR(1) process instead! -> TO DO
    /!\/!\/!\ 

    Parameters
    ----------
    observed : tuple(numpy.ndarray)
        Tuple of observed timeseries.
    simulated : tuple(numpy.ndarray)
        Tuple of simulated timeseries.
    phi : float
        AR(1) autocorrelation coefficient. Must be in (-1, 1)
        for stationarity.
    sigma : float
        Standard deviation of the AR(1) innovations. Must be positive.

    Returns
    -------
    float
        Log-likelihood value. Returns -inf for invalid parameters.
    """
    if not (-1 < phi < 1):
        return -np.inf
    if sigma <= 0:
        return -np.inf

    residuals = observed - simulated

    # Stationary variance of the AR(1) process
    sigma2_stationary = sigma**2 / (1 - phi**2)

    # Log-likelihood of the first residual (stationary distribution)
    ll = -0.5 * (np.log(2 * np.pi * sigma2_stationary)
                 + residuals[0]**2 / sigma2_stationary)

    # Log-likelihood of remaining residuals (innovations)
    innovations = residuals[1:] - phi * residuals[:-1]
    ll += -0.5 * np.sum(
        np.log(2 * np.pi * sigma**2) + innovations**2 / sigma**2
    )

    return ll


def ar1_heteroscedastic_log_likelihood(observed, simulated, phi,
                                        sigma_0, sigma_1):
    """
    Log-likelihood for an AR(1) autocorrelated, heteroscedastic error
    model. The innovation standard deviation scales linearly with the
    model prediction at each timestep:

        epsilon_t = phi * epsilon_{t-1} + eta_t
        eta_t ~ N(0, sigma_t^2)
        sigma_t = sigma_0 + sigma_1 * Y_sim(t)
    
    /!\/!\/!\ 
    Assumes that all observations are recorded at the same frequency,
    i.e. does not account for varying observation intervals in the real-world data

    If residuals do not correspond to a constant timestep, the likelihood is misspecified in a systematic way.
    For pairs of observations close together in time, the true autocorrelation is high — residuals genuinely carry
    information about their neighbor. For pairs far apart, the residuals are nearly independent. The fixed-ϕ\phi
    ϕ AR(1) treats all pairs identically, which means:

    - It underestimates the dependence between close observations → overstates their independent information content → posterior too narrow for those periods
    - It overestimates the dependence between distant observations → understates their information content → posterior too wide for those periods

    Should switch to a Continuous-Time AR(1) process instead! -> TO DO
    /!\/!\/!\ 

    Parameters
    ----------
    observed : tuple(numpy.ndarray)
        Tuple of observed timeseries.
    simulated : tuple(numpy.ndarray)
        Tuple of simulated timeseries.
    phi : float
        AR(1) autocorrelation coefficient. Must be in (-1, 1).
    sigma_0 : float
        Baseline standard deviation. Must be strictly positive.
    sigma_1 : float
        Proportional coefficient (dimensionless). Must be non-negative.
        When sigma_1=0, reduces to a homoscedastic AR(1) likelihood.

    Returns
    -------
    float
        Log-likelihood value. Returns -inf for invalid parameters.
    """
    if not (-1 < phi < 1):
        return -np.inf
    if sigma_0 <= 0:
        return -np.inf
    if sigma_1 < 0:
        return -np.inf

    residuals = observed - simulated

    # Time-varying innovation standard deviation
    sigma_t = sigma_0 + sigma_1 * simulated

    if np.any(sigma_t <= 0):
        return -np.inf

    # First residual: approximate stationary distribution
    sigma2_mean = np.mean(sigma_t**2)
    sigma2_stationary = sigma2_mean / (1 - phi**2)

    ll = -0.5 * (np.log(2 * np.pi * sigma2_stationary)
                 + residuals[0]**2 / sigma2_stationary)

    # Remaining residuals: conditional on previous residual
    innovations = residuals[1:] - phi * residuals[:-1]
    sigma_t_remaining = sigma_t[1:]

    ll += -0.5 * np.sum(
        np.log(2 * np.pi * sigma_t_remaining**2)
        + innovations**2 / sigma_t_remaining**2
    )

    return ll

def uniform_log_prior(parameter_values, bounds):
    """
    Uniform log-prior (non-informative prior). Returns 0 if all parameters are within
    bounds, -inf otherwise.
    """
    for value, (lo, hi) in zip(parameter_values, bounds):
        if not (lo <= value <= hi):
            return -np.inf
    return 0.0

class ErrorModel:
    """
    Container for the error model nuisance parameters.
    """
    def __init__(self, phi=0.5, sigma_0=1.0, sigma_1=0.01):
        self._parameters = {
            'phi':     phi,
            'sigma_0': sigma_0,
            'sigma_1': sigma_1
        }

    def get_parameter(self, key):
        if key not in self._parameters:
            raise KeyError(f'Unknown error model parameter "{key}".')
        return self._parameters[key]

    def set_parameter(self, key, value):
        if key not in self._parameters:
            raise KeyError(f'Unknown error model parameter "{key}".')
        self._parameters[key] = value
