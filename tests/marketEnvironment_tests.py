import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import random
import matplotlib.pyplot as plt
import numpy as np
from classes.MarketEnvironment import MarketEnvironment

class MarketEnvironmentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.initialPrice = 100
        self.price_history = []
        self.start_date = datetime(2023, 1, 1)
        self.market = MarketEnvironment(self.initialPrice, 'Market Test', self.price_history, self.start_date)

    def test_update_market(self):
        #  Mocking retail traders and institution investors

        retail_traders = {
            'joe': MagicMock(demand=10),
            'jim': MagicMock(demand=20),
        }

        inst_investors = {
            'hf1': MagicMock(demand=5),
            'hf2': MagicMock(demand=15)
        }

        #  set a fixed random number for consistent test results
        random.seed(0.1234)

        demand_retail, demand_hf = self.market.update_market(retail_traders, inst_investors)

        #  assert that price history has been updated
        self.assertEqual(len(self.market.price_history), 1)

        #  assert that the current price has been updated as expected
        expected_price = self.initialPrice + self.market.tau * self.market.excess_demand
        self.assertEqual(self.market.current_price, expected_price, 'Price not updated as expected')

        #  has excess demand been calculated correctly
        total_demand = sum([retail_traders[id].demand for id in retail_traders]) + sum(
            [inst_investors[id].demand for id in inst_investors])
        normalized_demand = total_demand / (len(retail_traders) + len(inst_investors))
        self.assertEqual(self.market.excess_demand, normalized_demand)

        # Assert that the demand from retail traders and institutional investors has been returned correctly
        self.assertEqual(demand_retail, 30)
        self.assertEqual(demand_hf, 20)

    def test_update_day(self):
        initial_date = self.market.date
        self.market.update_day()
        self.assertEqual(self.market.date, initial_date + timedelta(days=1), "days not updated as expected")

    def test_update_simulation_history(self):
        self.market.update_simulation_history()
        self.assertEqual(len(self.market.simulation_history), 1, "issue in updating simulation history")

    def test_select_participating_agents(self):
        """
        Function takes in the average commitment value, then based calculates a threshold and returns particpating agents according to that
        Therefore, we want to assert that we have at least some number of agents returned
        :return:
        """
        average_commitment_value = 10
        num_agents = 10
        retail_agents = {}
        for i in range(num_agents):
            retail_agents[i] = MagicMock()
        # seed for consistent results
        np.random.seed(2)
        selected_agents = self.market.select_participating_agents(average_commitment_value, retail_agents)
        self.assertGreaterEqual(len(selected_agents), 1)

if __name__ == '__main__':
    unittest.main()