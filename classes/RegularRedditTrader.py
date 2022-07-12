import random
from classes.RedditInvestorTypes import RedditInvestorTypes
from classes.RedditTrader import RedditTrader
import numpy as np


class RegularRedditTrader(RedditTrader):
    def __init__(self, id, neighbours_ids, commitment=None, investor_type=None):
        demand = 0  # an agent's initial demand
        if commitment is None:
            commitment = random.uniform(0.3, 0.5)  # normal random distribution with mean = 0 and std deviation = 1
        self.d = random.uniform(0.1, 0.3)  # threshold for difference in commitment to be too high - or confidence
        # interval value - random choice rather than set values as all agents will be slightly different,
        # hence we want thought processes to be heterogeneous
        self.b = random.uniform(-1, 1)  # a parameter which gives the strength of the force calculated as simply (
        # current price - moving_average)
        self.expected_price = 0
        self.has_closed_position = False  # variable to replicate how, after commitment going down, if the agent sells
        # then he is completely out, believing the market to be rigged
        self.demand_history = []
        self.bought_option = False
        self.has_trading_been_halted = False
        self.post_halting_decisions = {}
        self.post_halting_decisions['over 0.5 commitment'] = 0
        self.post_halting_decisions['long-term'] = 0
        self.post_halting_decisions['short-term-price-go-up'] = 0
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
            updated_commitment = average_neighbour_commitment + miu * abs(
                self.commitment - average_neighbour_commitment)
            self.commitment = min(updated_commitment, 1)

    def act_if_trading_halted(self, current_price, price_history, white_noise):
        if self.has_trading_been_halted:
            if self.commitment > 0.5: # 0.3 / 0.4
                self.demand += self.commitment
            else:
                if self.investor_type == RedditInvestorTypes.LONGTERM:
                    fundamental_price = 10  # in this case, our reddit agent will act rationally and consider the
                    if fundamental_price < current_price:
                        self.demand = -self.commitment
                elif self.investor_type == RedditInvestorTypes.RATIONAL_SHORT_TERM:
                    expected_price = self.compute_price_expectation_chartist(current_price, price_history, white_noise)
                    if expected_price > current_price:
                        self.demand = self.commitment
                    else:
                        self.demand -= self.commitment

        if not self.has_trading_been_halted:
            current_demand = self.demand / (1 / self.commitment)  # demand becomes a function of the agent's current
            # commitmemnt
            self.demand = -current_demand
            self.has_trading_been_halted = True

    def make_decision(self, average_network_commitment, current_price, price_history, white_noise, trading_halted):
        #  if agent out of trade, then stay out
        commitment_scaler = 0.95
        if trading_halted:
            self.act_if_trading_halted(current_price, price_history, white_noise)
            return
        if self.bought_option:  # not doing anything if we have bought an option already
            return
        self.compute_price_expectation_chartist(current_price, price_history, white_noise)
        if self.commitment > 0.6 and average_network_commitment > 0.58:
            self.demand = 100 * self.commitment  # buys options
            print("Bought option")
            self.bought_option = True
            return 1
        elif self.commitment > 0.4:
            self.demand = commitment_scaler * self.commitment  # slightly committed, still considers technical analysis
#
        elif self.commitment < 0.3:
            self.demand -= self.commitment * commitment_scaler
            print("agent demand is 0, expects stock to go down")
        if self.demand != 0:
            self.demand_history.append(self.demand)

    def compute_price_expectation_chartist(self, current_price, price_history, white_noise):
        """
        Taken from - minimal agent based model for financial markets

        Chartists agents detect a trend through looking at the distance between the price and its smoothed profile (
        given by moving average in this case) :param current_price: :param current_trading_day: :param price_history:
        :param white_noise: :return:
        """
        #  below if statement considers whether the agent is simply looking for a quick profit, get in-get out or believe in GME
        rolling_average_window_length = 15
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
