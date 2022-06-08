import unittest

import pandas as pd

from classes.RedditTrader import RedditTrader
from helpers.network_helpers import calculate_average_commitment, create_network_from_agent_dictionary
from helpers.calculations_helpers import split_commitment_into_groups, extract_values_counts_as_lists


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


    def test_create_network_from_agent_dictionary(self):
        mock_agent_dict = {}
        threshold = 2
        commitments = [1, 2, 4, 0.5, 5]
        for i in range(5):
            agent = RedditTrader(i, [], 0, commitments[i])
            mock_agent_dict[i] = agent
        expected_graph_node_length = 3
        actual_graph = create_network_from_agent_dictionary(mock_agent_dict, threshold)
        actual_graph_node_length = len(actual_graph.nodes())
        self.assertEqual(expected_graph_node_length, actual_graph_node_length,
                         "Number of nodes is not as expected")

    def test_extract_values_counts_as_list(self):
        mock_dict = {
            "mock_points": [0, 1, 0, 0, 2, 4, 5],
            "another_mock_column" : ['vlad', 'tim', 'marcus', 'stefan', 'stefan', 'tim', 'tim']
        }
        mock_df = pd.DataFrame.from_dict(mock_dict)
        values, counts = extract_values_counts_as_lists(mock_df, 'another_mock_column')
        expected_first_value = 'tim'  # because it appears 3 times and by default the function sorts
        actual_first_value = values[0]
        self.assertEqual(expected_first_value, actual_first_value)


if __name__ == '__main__':
    unittest.main()
