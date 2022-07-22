import math
import random

import numpy as np
from helpers.calculations_helpers import check_and_convert_imaginary_number


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
        self.alpha, self.betta = self.assign_risk_variables()

    def make_decision(self, current_price, price_history):
        """
        Agent updates its demand of the stock according to the current market price
        :param current_price:
        :return:
        """
        short_gme = self.utility_function(current_price, price_history)
        if short_gme:
            if self.demand != 0:
                self.demand -= self.demand
            else:
                self.demand = -1  # start shorting again after closing position

        else:
            self.demand = 0
        return short_gme

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
        p_gain = 0.8  # use in dissertation "such that p_gain + p_loss = 1 "
        p_loss = 0.2
        fundamentalist_weight = 1
        lambda_parameter = 1.75
        chartist_weight = 2.55
        noise_weight = 1
        added_noise = random.uniform(0, 1)
        expected_price_chartist = self.compute_expected_price(fundamentalist_weight=fundamentalist_weight,
                                                              chartist_weight=chartist_weight,
                                                              noise_weight=noise_weight, current_price=current_price,
                                                              price_history=price_history, added_noise=added_noise)

        fundamentalist_weight = 2
        chartist_weight = 0.9
        noise_weight = 1
        expected_price_fundamentalist = self.compute_expected_price(fundamentalist_weight, chartist_weight,
                                                                    noise_weight,
                                                                    current_price, price_history, added_noise)

        x_gain = abs(current_price - expected_price_fundamentalist)
        x_loss = abs(current_price - expected_price_chartist)
        V_loss = lambda_parameter * (x_loss ** self.alpha)
        V_gain = p_gain * (x_gain ** self.alpha) - (p_loss * lambda_parameter * (x_loss ** self.betta))

        V_gain = check_and_convert_imaginary_number(V_gain)

        should_gamble = V_gain > V_loss

        return should_gamble

    def compute_expected_price(self, fundamentalist_weight, chartist_weight, noise_weight, current_price,
                               price_history, added_noise):
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
        try:
            expected_return_fundamentalist = math.log(self.fundamental_price / current_price)
        except ValueError as e:
            expected_return_fundamentalist = 0

        expected_return = (1 / (fundamentalist_weight + chartist_weight + noise_weight)) * (
                fundamentalist_weight * expected_return_fundamentalist + chartist_weight * expected_return_chartist +
                noise_weight * added_noise
        )

        expected_price = current_price * np.exp(expected_return)

        return expected_price

    @staticmethod
    def compute_expected_return_chartist_approach(price_history):
        """
        Formula implemented here is observed in the paper "Impact of heterogenous trading rules on the limit order book and order flows
        By Chiarella et al - 2009
        :param price_history:
        :return:
        """
        price_history = price_history[-10:]  # getting last 10 entries in array
        spot_price_observation = 0
        for i in range(len(price_history) - 1):
            previous_price = price_history[i + 1]
            previous_previous_price = price_history[i]
            try:
                spot_price_observation += math.log(previous_price / previous_previous_price)
            except ValueError as e:
                spot_price_observation += 0

        future_expected_trend_chartist_approach = spot_price_observation / len(price_history)
        return future_expected_trend_chartist_approach

    def assign_risk_type(self):
        is_risk_loving = [False, True]
        probabilities = [0.33, 0.67]
        risk_type = random.choices(is_risk_loving, probabilities)
        return risk_type

    def assign_risk_variables(self):
        if self.risk_loving:
            return [2, 1]  # alpha betta for risk loving
        return [0.5, 1]  # alpha betta for risk averse
