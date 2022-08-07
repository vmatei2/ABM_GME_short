class Option:
    def __init__(self, id, K, volatility, T, option_type):
        """

        :param id:
        :param K: strike price
        :param volatility:
        :param T: expiration date
        """
        self.id = id
        self.K = K
        self.volatility = volatility
        self.T = T
        self.option_type = option_type
