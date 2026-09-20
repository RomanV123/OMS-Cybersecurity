import unittest
from platform import python_version
import sklearn
import scipy
import pandas as pd
import yellowbrick
import numpy as np
import matplotlib

# Expected versions from the ../env.yml file
EXPECTED_VERSIONS = {
    'python': '3.10.21',
    'sklearn': '1.7.2',
    'scipy': '1.15.2',
    'pandas': '2.3.3',
    'yellowbrick': '1.5',
    'numpy': '2.2.6',
    'matplotlib': '3.10.9'
}
class TestPackageVersions(unittest.TestCase):
    def test_python_version(self):
        self.assertEqual(EXPECTED_VERSIONS['python'], python_version())

    def test_sklearn_version(self):
        self.assertEqual(EXPECTED_VERSIONS['sklearn'], sklearn.__version__)

    def test_scipy_version(self):
        self.assertEqual(EXPECTED_VERSIONS['scipy'], scipy.__version__)

    def test_pandas_version(self):
        self.assertEqual(EXPECTED_VERSIONS['pandas'], pd.__version__)

    def test_yellowbrick_version(self):
        self.assertEqual(EXPECTED_VERSIONS['yellowbrick'], yellowbrick.__version__)

    def test_numpy_version(self):
        self.assertEqual(EXPECTED_VERSIONS['numpy'], np.__version__)

    def test_matplotlib_version(self):
        self.assertEqual(EXPECTED_VERSIONS['matplotlib'], matplotlib.__version__)

