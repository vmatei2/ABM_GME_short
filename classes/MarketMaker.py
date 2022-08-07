import random

from classes.Option import Option
from helpers.calculations_helpers import calculate_gamma, calculate_volatility


class MarketMaker:
    def __init__(self):
        self.name = "Citadel"
        self.demand = 0
        self.options_sold = 1
        self.options_sold_dict = {}
        self.risk_tolerance = random.choice([0, 0.25, 0.5, 0.75, 1])  # 1 - high risk-tolerance, 0 - low risk-tolerance

    def hedge_position(self, option, current_price, price_history, trading_day):
        if option.expired:
            return
        volatility = calculate_volatility(price_history) / 100
        r = 0.05
        days_in_a_year = 365
        T = (option.expiry_date - trading_day) / days_in_a_year
        gamma = calculate_gamma(current_price, option.K, r, volatility, T)
        hedge = gamma * 100
        return hedge

    def sell_option(self, current_price, trading_day, id):
        K = current_price + random.uniform(0.05,
                                           0.1) * current_price  # adding some randomness in strike price of option
        maturities = [5, 10, 15, 20]  # in days
        type = "call"
        expiry_date = trading_day + random.choice(maturities)
        option = Option(id=id, K=K, expiry_date=expiry_date, option_type=type, date_sold=trading_day)
        self.options_sold_dict[id] = [option, 0]
        self.options_sold += 1
        return option

    def hedge_all_positions(self, current_price, price_history, trading_day):
        for option_id, option in self.options_sold_dict.items():
            if trading_day > option.expiry_date:
                option.expired = True
            else:
                option.T = trading_day - option.date_sold
            self.hedge_position(option, current_price, price_history)
