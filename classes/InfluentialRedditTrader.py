import random

from classes.RedditTrader import RedditTrader


class InfluentialRedditUser(RedditTrader):
    def __init__(self, id, neighbours_ids, market_first_price):
        super().__init__(id, neighbours_ids)
        self.price_scaling_factor = random.uniform(500, 1000)  # very large scaling factor, reflective of the agent's
        # belief that the stock is going to the moon
        self.fundamental_price = market_first_price * self.price_scaling_factor  # the agent's fundamental_price view

    def make_decision(self, average_network_commitment, threshold):
        if average_network_commitment >= threshold:
            self.demand += 100  # option buying
        return
