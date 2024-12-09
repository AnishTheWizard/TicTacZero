import unittest
from torch import nn
class MonteCarloTests(unittest.TestCase):

  def test_monte_carlo_with_equal_priors(self):

    #define a quick model
    class FakeModel(nn.Module):
      def predict(self, x):
        return np.array([])