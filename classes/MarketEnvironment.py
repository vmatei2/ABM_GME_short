class MarketEnvironment:
    def __init__(self, initial_price, reddit_traders, hedge_funds, market_makers):
        self.initial_price = initial_price
        self.reddit_traders = reddit_traders
        self.hedge_funds = hedge_funds
        self.market_makers = market_makers
        self.constants = {}
        self.price_history = []

    def update_market(self):
        pass

    def plot_price_history(self):
        pass

    def select_participating_agents(self):
        pass
