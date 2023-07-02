import datetime
import unittest

from classes.RedditInvestorTypes import RedditInvestorTypes
from classes.SimulationClass import SimulationClass
from classes.MarketEnvironment import MarketEnvironment

class TestSimulationClass(unittest.TestCase):
    def setUp(self) -> None:
        test_market = MarketEnvironment(initial_price=10, name='Test Market', price_history=[10, 10, 10], start_date=datetime.datetime(2023, 1, 1))
        self.simulation = SimulationClass(time_steps=100, N_agents=100, N_institutional_investors=10, m=3,
                                          market_environment=test_market, miu=0.5, commitment_scaler=0.8,
                                          volume_threshold=0.8, fundamental_price_inst_inv=100, lambda_parameter=0.2,
                                          n_influencers=5, commitment=(0.3, 0.6), d_parameter=None)

    def test_create_initial_network(self):
        # Test if the initial network is created correctly
        n_influencers = 5
        commitment = (0.3, 0.6) # we pass in the two variables for sampling when creating the agent network with comimtment vals in-between
        d_parameter = None

        social_media_agents, average_degree = self.simulation.create_initial_network(n_influencers, commitment,
                                                                                     d_parameter)

        # Check the number of agents in the network
        self.assertEqual(len(social_media_agents), self.simulation.N_agents)

        # Check the number of influential agents in the network
        count_influential_agents = sum(
            1 for agent in social_media_agents.values() if agent.investor_type == RedditInvestorTypes.FANATICAL)
        self.assertEqual(count_influential_agents, n_influencers)

    def test_halt_trading(self):
        # Test if trading is correctly halted and commitment values are adjusted
        commitment_threshold = 0.65
        commitment_lower_upper = [0.2, 0.2]

        # Call the halt_trading method
        ids_to_be_deleted = self.simulation.halt_trading(commitment_threshold, commitment_lower_upper)

        # Check if trading is halted
        self.assertTrue(self.simulation.trading_halted)

        # Check if commitment values are adjusted for eligible agents
        for agent in self.simulation.social_media_agents.values():
            if agent.commitment <= commitment_threshold:
                self.assertGreaterEqual(agent.commitment, commitment_lower_upper[0])
                self.assertLessEqual(agent.commitment, commitment_lower_upper[1])

    def test_create_institutional_investors(self):
        # Test if institutional investors are created correctly
        N_institutional_investors = 10

        institutional_investors = self.simulation.create_institutional_investors()

        # Check the number of institutional investors
        self.assertEqual(len(institutional_investors), N_institutional_investors)

        # Check if each institutional investor has the correct attributes
        for investor in institutional_investors.values():
            self.assertEqual(investor.demand, -1)
            self.assertEqual(investor.fundamental_price, self.simulation.fundamental_price_inst_inv)
            self.assertEqual(investor.lambda_parameter, self.simulation.lambda_parameter)

