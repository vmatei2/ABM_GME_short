import unittest
from classes.RedditTrader import RedditTrader
from helpers.network_helpers import calculate_average_commitment


class TestHelpers(unittest.TestCase):
    def test_average_commitment(self):
        agents = {}
        commitments = [1, 2, 3, 4, 5]
        for i in range(5):
            agent = RedditTrader(i, [], 0, commitments[i])
            agents[i] = agent
        expected_commitment = sum(commitments) / len(commitments)
        actual_commitment = calculate_average_commitment(agents)
        self.assertEqual(expected_commitment, actual_commitment, "average commitment values are not equal")


if __name__ == '__main__':
    unittest.main()