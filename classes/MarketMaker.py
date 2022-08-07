import random

from classes.Option import Option
from helpers.calculations_helpers import calculate_gamma


class MarketMaker:
    def __init__(self):
        self.name = "Citadel"
        self.demand = 0
        self.options_sold = 1
        self.options_sold_dict = {}

    def hedge_position(self, option_id, current_price, price_history):
        option_to_be_hedged = self.options_sold[option_id]
        volatility = 0
        gamma = calculate_gamma(current_price, option_to_be_hedged.K, volatility, option_to_be_hedged.T)
        self.demand += gamma * 100  # calculate gamma and update demand accordingly (gamma returned as 0.25 for example)

    def sell_option(self, current_price):
        id = self.options_sold
        K = current_price + random.uniform(0.3, 0.5) * current_price  # adding some randomness in strike price of option
        volatility = 0.2  # replace by volatility calculation
        maturities = [10/365, 20/365, 30/365]
        type = "call"
        option = Option(id=id, K=K, volatility=volatility, T=random.choice(maturities), option_type=type)
        self.options_sold_dict[id] = option
        self.options_sold += 1
