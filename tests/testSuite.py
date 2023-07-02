import unittest

from simulationClass_tests import TestSimulationClass
from marketEnvironment_tests import MarketEnvironmentTestCase
from helpers_tests import TestHelpers

class ABM_TestSuite(unittest.TestSuite):
    def __init__(self):
        super().__init__()
        # adding test classes to the suite
        self.addTest(unittest.makeSuite(TestSimulationClass))
        self.addTest(unittest.makeSuite(MarketEnvironmentTestCase))
        self.addTest(unittest.makeSuite(TestHelpers))

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    suite = ABM_TestSuite()
    runner.run(suite)