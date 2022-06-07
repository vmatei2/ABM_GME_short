import unittest
from classes.RedditTrader import RedditTrader
from helpers.network_helpers import calculate_average_commitment
from helpers.calculations_helpers import split_commitment_into_groups


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

    def test_split_commitment_into_groups(self):
        commitment_this_round = [0.2, 0.3, 0.2, 0.45, 0.5, 0.6, 0.7, 0.8, 0.8]
        trading_day = 1
        actual_zero_to_40, actual_forty_to_65, actual_sixty_five_to_1 = split_commitment_into_groups(
            commitment_this_round, trading_day)
        expected_zero_to_40 = [3, 1, "0-0.4"]
        self.assertEqual(expected_zero_to_40, actual_zero_to_40, "not equal")


if __name__ == '__main__':
    unittest.main()
