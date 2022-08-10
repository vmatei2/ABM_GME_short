class Option:
    def __init__(self, id, K, expiry_date, option_type, date_sold):
        """

        :param id:
        :param K: strike price
        :param volatility:
        :param T: days to expiration
        :param date_sold: trading day when sold
        """
        self.id = id
        self.K = K
        self.expiry_date = expiry_date
        self.option_type = option_type
        self.date_sold = date_sold
        self.expired = False
        self.delta = 0
