class MarketEnvironment:
    def __init__(self, initial_price, reddit_traders, institutional_investors, market_makers):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.reddit_traders = reddit_traders
        self.institutional_investors = institutional_investors
        self.market_makers = market_makers
        self.excess_demand = {}
        self.noise_term = 0
        self.price_history = []

    def update_market(self):
        self.price_history.append(self.current_price)
        self.update_excess_demand()
        updated_price = self.current_price + self.noise_term * self.excess_demand
        pass

    def plot_price_history(self):
        """
        Function to observe the price evolution
        :return:
        """
        pass

    def select_participating_agents(self):
        """
        Selecting particpating agents, based on volume calcuation
        :return:
        """
        pass

    def update_excess_demand(self):
        """
        Updating the excess demand coming from retail trader and instituional investors
        :return:
        """
        pass
