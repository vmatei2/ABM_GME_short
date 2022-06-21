import math
import random


class InstitutionalInvestor:
    """
    Class modelling the insitutional investors involved in the GameStop saga
    This agent expects the stock to reach a value of 0 and greatly shorts the stock
    However, based on its own risk aversion, this agent can and will adapt its behaviour
    Going against its perceived opinion of the stock, in an effort to cut its losses

    The above will be completed through an implementation of Asset Pricing under Prospect Theory
    """

    def __init__(self, id, demand, fundamental_price):
        self.id = id
        self.demand = demand
        self.fundamental_price = fundamental_price
        self.risk_loving = self.assign_risk_type()

    def make_decision(self, current_price, price_history):
        """
        Agent updates its demand of the stock according to the current market price
        :param current_price:
        :return:
        """
        self.utility_function(current_price, price_history)
        return True

    def utility_function(self, current_price, price_history):
        """
        Function to be used in the implementation of prospect theory
        Prospect theory tells us about the overall utility, which is computed through taking the sum
        of the value of each outcome, weighted by its probability of happening (or, more precisely by how
        we perceive it will happen (subjectively)
        Here we will have two formulas to be implemented - in case of gain and loss to separate between the chartist
        and fundamentalist and the probabilities will be according to the fund's risk aversion
        )
        :param current_price: extracting current price of the stock
        :param price_history: extracting previous price from this
        :return:
        """
        p_gain = 0.8
        p_loss = 0.2
        expected_price_chartist = self.compute_expected_price_chartist(fundamentalist_weight=0, chartist_weight=0,
                                                                       noise_weight=0,
                                                                       current_price=current_price,
                                                                       price_history=price_history)
        expected_price_fundamentalist = self.compute_expected_price_fundamentalist_apporoach(current_price)
        x_gain = abs(current_price - expected_price_fundamentalist)
        x_loss = abs(current_price - expected_price_chartist)

        return True

    def compute_expected_price_chartist(self, fundamentalist_weight, chartist_weight, noise_weight, current_price,
                                        price_history):
        """
        Do not want to give access to average commitment values, as this might over-complicate the model
        Given that the price is indeed driven by the commitment, it can be deemed as sufficient to simply
        have the previous price history, and depending on risk-aversion to assign weight to the chartist expectation
        More risk-averse fund will assign greater weight to the chartist parameter to ensure it does not get caught in a bad place
        despite its strong fundamental beliefs against the company
        :param fundamentalist_weight:
        :param chartist_weight:
        :param noise_weight:
        :param current_price:
        :param price_history:
        :return:
        """
        expected_return_chartist = self.compute_expected_return_chartist_approach(price_history)
        expected_price = current_price * math.exp(expected_return_chartist)
        return expected_price

    def compute_expected_return_chartist_approach(self, price_history):
        """
        Formula implemented here is observed in the paper "Impact of heterogenous trading rules on the limit order book and order flows
        By Chiarella et al - 2009
        :param price_history:
        :return:
        """
        spot_price_observation = 0
        for i in range(len(price_history) - 1):
            previous_price = price_history[i + 1]
            previous_previous_price = price_history[i]
            spot_price_observation += math.log(previous_price / previous_previous_price)

        future_expected_trend_chartist_approach = spot_price_observation / len(price_history)
        return future_expected_trend_chartist_approach

    def compute_expected_return_fundamentalist_approach(self, current_price):
        expected_return = math.log(self.fundamental_price / current_price)
        return expected_return

    def compute_expected_price_fundamentalist_apporoach(self, current_price):
        expected_return = self.compute_expected_return_fundamentalist_approach(current_price)
        expected_price = current_price * math.exp(expected_return)
        return expected_price

    def assign_risk_type(self):
        is_risk_loving = [False, True]
        probabilities = [0.33, 0.67]
        risk_type = random.choices(is_risk_loving, probabilities)
        return risk_type
