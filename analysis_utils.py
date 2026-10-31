"""Data validation, time interpolation and consistent regime mixture estimates."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

SENSOR_COLUMNS = [f'{sensor}{axis}' for sensor in ('acc', 'gyro', 'mag') for axis in 'XYZ']


def load_imu(path):
    """Load the supplied IMU format, rejecting invalid or unordered measurements."""
    df = pd.read_csv(Path(path))
    missing = sorted(set(['name', 'time', *SENSOR_COLUMNS]) - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if len(df) < 3:
        raise ValueError('At least three rows are required')
    if df['name'].nunique(dropna=False) != 1:
        raise ValueError('Expected one IMU stream')
    if not pd.api.types.is_integer_dtype(df['time']):
        raise ValueError('time must contain integer nanosecond timestamps')
    if not (df['time'].diff().dropna() > 0).all():
        raise ValueError('time must be strictly increasing')
    values = df[SENSOR_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Input sensor values must be finite')
    return df


def interpolate_by_time(data, timestamps):
    """Offline interpolation on relative time; preserve index and edge NaNs."""
    ts = np.asarray(timestamps)
    if len(ts) != len(data) or len(ts) < 2 or not np.issubdtype(ts.dtype, np.integer):
        raise ValueError('Expected matching integer timestamps')
    if np.any(ts[1:] <= ts[:-1]):
        raise ValueError('time must be strictly increasing')
    relative_seconds = (ts - ts[0]).astype(float) / 1e9
    result = data.copy()
    result.index = relative_seconds
    result = result.interpolate(method='index', limit_area='inside')
    result.index = data.index
    return result


def regime_probability(x, moving, grid):
    """Bayes estimate using two class KDEs with a common absolute bandwidth."""
    x = np.asarray(x, dtype=float)
    moving = np.asarray(moving, dtype=bool)
    grid = np.asarray(grid, dtype=float)
    if x.shape != moving.shape or not np.isfinite(x).all():
        raise ValueError('Expected finite values and a matching class mask')
    if min(moving.sum(), (~moving).sum()) < 2:
        raise ValueError('Each regime needs at least two observations')
    h = gaussian_kde(x, bw_method='silverman').factor * x.std(ddof=1)
    density = []
    for mask in (moving, ~moving):
        subset = x[mask]
        if subset.std(ddof=1) == 0:
            raise ValueError('Each regime needs nonconstant observations')
        density.append(mask.mean() * gaussian_kde(subset, bw_method=h/subset.std(ddof=1))(grid))
    denominator = density[0] + density[1]
    return np.divide(density[0], denominator, out=np.full(grid.shape, np.nan), where=denominator > 0)
