import random

from classes.RedditTrader import RedditTrader


class InfluentialRedditUser(RedditTrader):
    """
    Child class of "RedditTrader" - this agent's main role in the simulation is to influence the rest of the network
    Whenever involved in the market, the agent updates his demand if the average network commitment passes the
    threshold (relatively low threshold as agent is fully committed)
    """
    def __init__(self, id, neighbours_ids, market_first_price, investor_type):
        super().__init__(id, neighbours_ids, investor_type=investor_type)
        self.price_scaling_factor = random.uniform(500, 1000)  # very large scaling factor, reflective of the agent's
        # belief that the stock is going to the moon
        self.fundamental_price = market_first_price * self.price_scaling_factor  # the agent's fundamental_price view

    def make_decision(self, average_network_commitment, threshold):
        if average_network_commitment >= threshold:
            self.demand += 100
        return
