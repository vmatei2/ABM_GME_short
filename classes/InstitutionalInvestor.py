class InstitutionalInvestor:
    """
    Class modelling the insitutional investors involved in the GameStop saga
    This agent expects the stock to reach a value of 0 and greatly shorts the stock
    However, based on its own risk aversion, this agent can and will adapt its behaviour
    Going against its perceived opinion of the stock, in an effort to cut its losses
    """
    def __init__(self, id, demand, fundamental_price):
        self.id = id
        self.demand = demand
        self.fundamental_price = fundamental_price


    def make_decision(self, current_price):
        """
        Agent updates its demand of the stock according to the current market price
        :param current_price:
        :return:
        """
        pass