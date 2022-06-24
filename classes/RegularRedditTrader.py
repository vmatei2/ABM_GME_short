import random
from classes.RedditInvestorTypes import RedditInvestorTypes
from classes.RedditTrader import RedditTrader
import numpy as np


class RegularRedditTrader(RedditTrader):
    def __init__(self, id, neighbours_ids, commitment=None, investor_type=None):
        demand = 0 # an agent's initial demand
        if commitment is None:
            commitment = random.uniform(0.3, 0.5)  # normal random distribution with mean = 0 and std deviation = 1
        self.d = random.uniform(0.1, 0.3)  # threshold for difference in commitment to be too high - or confidence
        # interval value - random choice rather than set values as all agents will be slightly different,
        # hence we want thought processes to be heterogeneous
        self.b = random.uniform(-1, 1)  # a parameter which gives the strength of the force calculated as simply (
        # current price - moving_average)
        self.expected_price = 0
        self.risk_aversion = np.random.normal(0, 1, 1)  # mean, std deviation and size of the array to be returned
        self.has_closed_position = False  # variable to replicate how, after commitment going down, if the agent sells
        # then he is completely out, believing the market to be rigged
        self.demand_history = []
        self.bought_option = False
        super().__init__(id, neighbours_ids, demand, commitment, investor_type)


    def update_commitment(self, agents, miu):
        """
        Function to update the commitment of a regular reddit trader in the network

        In the Deffuant Model, the agent updated his opinion simply based on the opinion of another randomly picked agent

        This does not replicate well what was observed in real-life, and we will calculate the average commitment of
        an agent's neighbours and compare that to the d value - makes more sense as our agents will base their
        commitment updates on what is happening in the total surrounding environment, not just one randomly chosen
        neighbour

        :param agents: retail trading agents
        :param d: the threshold where the difference in
        commitment is too high for this agent to update its own commitment (confidence level)
        :param miu: scaling
        parameter :return:
        """
        neighbour_choice_id = random.choice(self.neighbours_ids)  # randomly pick one neighbour for the interaction
        neighbour = agents[neighbour_choice_id]
        neighbour_commitment_value = 0
        for id in self.neighbours_ids:
            neighbour_commitment_value += agents[id].commitment
        average_neighbour_commitment = neighbour_commitment_value / len(self.neighbours_ids)
        if abs(neighbour.commitment - self.commitment) >= self.d:
            # this happens in the case when the difference in commitment between the agent and its neighbours is too
            # big - therefore we do not update opinion at this time point
            pass
        else:

            # otherwise, let's update this agent's opinion (being influenced)
            updated_commitment = average_neighbour_commitment + miu * abs(self.commitment - average_neighbour_commitment)
            self.commitment = min(updated_commitment, 1)

    def make_decision(self, average_network_commitment, current_price, price_history, white_noise, trading_halted):
        #  if agent out of trade, then stay out
        if self.has_closed_position:
            return
        if len(self.demand_history) > 0 and self.demand == 0 and trading_halted:
            self.has_closed_position = True
            if self.investor_type == RedditInvestorTypes.RATIONAL_SHORT_TERM:
                self.demand = -0.5
                print("Rational investor has shorted the stock")

        self.compute_price_expectation_chartist(current_price, price_history, white_noise)
        if self.commitment > 0.65:
            if self.bought_option:
                return
            self.demand = 15  # buys options
            print("Bought option")
            self.bought_option = True
        elif self.commitment > 0.57 and average_network_commitment > 0.45:
            self.demand = 4  # buys more stock
        elif self.commitment > 0.35 and self.expected_price > current_price:
            self.demand = 2  # slightly committed, still considers technical analysis
        elif self.expected_price > current_price:
            self.demand = 1 # closes open position as commitment is low and not happy with GME
        elif self.commitment < 0.35 and not trading_halted:
            self.demand = 0
            print("agent demand is 0, expects stock to go down")
        elif self.commitment < 0.35 and trading_halted:
            self.demand = -4
            self.has_closed_position = True
            print("Trading halted, agent got scared")
        if self.demand != 0:
            self.demand_history.append(self.demand)

    def compute_price_expectation_chartist(self, current_price, price_history, white_noise):
        """
        Chartists agents detect a trend through looking at the distance between the price and its smoothed profile (
        given by moving average in this case) :param current_price: :param current_trading_day: :param price_history:
        :param white_noise: :return:
        """
        #  below if statement considers whether the agent is simply looking for a quick profit, get in-get out or believe in GME
        if self.commitment > 0.5:
            rolling_average_window_length = 15
        else:
            rolling_average_window_length = 5
        rolling_average = self.compute_rolling_average(price_history, rolling_average_window_length)
        added_noise = self.b * white_noise
        expected_price = current_price + self.b * (current_price - rolling_average) + white_noise
        self.expected_price = expected_price
        return expected_price

    def compute_rolling_average(self, price_history, rolling_average_window_length):
        total_prices = 0
        price_history = price_history[-rolling_average_window_length:]
        for i in range(rolling_average_window_length):
            total_prices += price_history[i]
        rolling_average = total_prices / rolling_average_window_length
        return rolling_average
