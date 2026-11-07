# SPDX-License-Identifier: Apache-2.0

"""Running mean and standard deviation of a scalar series."""

import numpy as np


class RunningMeanStd:
    """Running mean and standard deviation of a scalar series.

    Attributes:
        n: Number of samples.
        x: Sum of the samples.
        x2: Sum of the squared samples.
    """

    def __init__(self):
        """Initialize an empty series."""
        self.reset()

    def reset(self):
        """Forget all samples."""
        self.n = 0
        self.x = 0.0
        self.x2 = 0.0

    def append(self, x):
        """Add a sample to the series.

        Args:
            x: New sample.
        """
        self.n += 1
        self.x += x
        self.x2 += x**2

    def average(self):
        """Mean of the series.

        Returns:
            Mean of the samples so far.
        """
        assert self.n >= 1
        return self.x / self.n

    def std(self):
        """Sample standard deviation of the series.

        Returns:
            Standard deviation of the samples so far, with Bessel's
            correction.
        """
        assert self.n > 1
        mu = self.average()
        unbiased = np.sqrt(1.0 + 1.0 / (self.n - 1))
        return unbiased * np.sqrt(self.x2 / self.n - mu**2)
