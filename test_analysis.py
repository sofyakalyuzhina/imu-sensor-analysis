"""Small numerical regressions; no original dataset required."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm, t, cramervonmises
from analysis_utils import SENSOR_COLUMNS, load_imu, interpolate_by_time, regime_probability


def notebook_functions():
    notebook = json.loads(Path(__file__).with_name('imu_analysis.ipynb').read_text())
    namespace = dict(np=np, pd=pd, norm=norm, t=t, DIXON_N=20, DIXON_Q_CRITICAL=.342)
    for index in (74, 83, 88, 152, 153):
        tree = ast.parse(''.join(notebook['cells'][index]['source']))
        functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
        exec(compile(functions, f'cell-{index}', 'exec'), namespace)
    return namespace


class AnalysisTests(unittest.TestCase):
    def test_input_validation(self):
        frame = pd.DataFrame({col: [1., 2., 3.] for col in SENSOR_COLUMNS})
        frame['name'] = 'IMU8'; frame['time'] = [100, 200, 300]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.csv'
            frame.to_csv(path, index=False)
            self.assertEqual(load_imu(path).shape, (3, 11))
            frame.loc[1, 'time'] = 100
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'strictly'):
                load_imu(path)
            frame.drop(columns='accX').to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'Missing'):
                load_imu(path)

    def test_interpolation_relative_time(self):
        # At epoch scale, nanosecond offsets would be lost if cast to float first.
        stamps = np.array([1686062828391000000 + i for i in (0, 1, 3)], dtype=np.int64)
        data = pd.DataFrame({'x': [0., np.nan, 6.]}, index=[5, 6, 7])
        result = interpolate_by_time(data, stamps)
        np.testing.assert_allclose(result['x'], [0, 2, 6])
        self.assertTrue(result.index.equals(data.index))
        self.assertTrue(np.isnan(data.loc[6, 'x']))

    def test_interpolation_no_extrapolation(self):
        data = pd.DataFrame({'x': [np.nan, 1., np.nan, 3., np.nan]})
        result = interpolate_by_time(data, np.arange(5, dtype=np.int64))
        self.assertTrue(result.iloc[[0, -1]]['x'].isna().all())
        self.assertEqual(result.loc[2, 'x'], 2.)

    def test_regime_probability_complement(self):
        rng = np.random.default_rng(42)
        x = np.r_[rng.normal(-1, 1, 200), rng.normal(1, 1, 200)]
        mask = np.arange(400) >= 200
        grid = np.linspace(-3, 3, 20)
        p = regime_probability(x, mask, grid)
        q = regime_probability(x, ~mask, grid)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        np.testing.assert_allclose(p+q, 1)
        self.assertLess(p[0], p[-1])

    def test_cvm_matches_scipy(self):
        f = notebook_functions()['cvm_statistic_standardized']
        x = np.random.default_rng(10).normal(size=100)
        expected = cramervonmises((x-x.mean())/x.std(ddof=1), 'norm').statistic
        self.assertAlmostEqual(f(x), expected, places=12)

    def test_outlier_functions(self):
        ns = notebook_functions()
        s = pd.Series([-.1, .1, -.2, .2, 0., .1, -.1, 100.])
        self.assertTrue(ns['grubbs_test'](s)['outlier_detected'])
        steps, count = ns['generalized_esd'](s, max_outliers=2)
        self.assertEqual(count, 1)
        self.assertEqual(steps.loc[0, 'index'], 7)
        with self.assertRaises(ValueError):ns['grubbs_test'](pd.Series([1., 1., 1.]))
        with self.assertRaises(ValueError):ns['dixon_q_test'](np.arange(10))
        self.assertFalse(ns['dixon_q_test'](np.arange(20))['outlier_detected'])

    def test_shapiro_francia_scale_invariance(self):
        f = notebook_functions()['shapiro_francia_test']
        x = np.random.default_rng(42).normal(size=200)
        np.testing.assert_allclose(f(x), f(3*x+7), atol=1e-12)
        with self.assertRaises(ValueError):f(np.ones(20))

if __name__ == '__main__':
    unittest.main()
