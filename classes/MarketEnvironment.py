import random

import numpy as np


class MarketEnvironment:
    def __init__(self, initial_price, name, price_history):
        self.name = name
        self.initial_price = initial_price
        self.current_price = initial_price
        self.excess_demand = {}
        self.tau = 100  # noise term for updating market price
        self.price_history = price_history

    def update_market(self, retail_traders, institutional_investors):
        self.price_history.append(self.current_price)
        self.update_excess_demand(retail_traders, institutional_investors)
        updated_price = self.current_price + self.tau * self.excess_demand
        print("Previous Price: ", self.current_price)
        self.current_price = updated_price
        print("Updated Price: ", self.current_price)

    def plot_price_history(self):
        """
        Function to observe the price evolution
        :return:
        """
        pass

    def select_participating_agents(self, average_commitment_value, retail_agents):
        """
        Selecting participating agents, based on volume calculation taking into account the average commitment value across the network
        :return:
        """
        probabilities_dict = {}
        all_agent_probabilities = np.random.uniform(0, 1, len(retail_agents))  # array of probabilities for each agent
        i = 0
        for id, agent in retail_agents.items():
            probabilities_dict[id] = all_agent_probabilities[i]
            i += 1
        commitment_scaler = 0.002
        noise_term = 0.012  #random.uniform(0.01, 0.015)
        threshold = average_commitment_value * commitment_scaler + noise_term
        # the line below loops over all probabilities and selects the ids of agents simply based on the thresholdf
        participating_agent_ids = [id for id, value in probabilities_dict.items() if value < threshold]
        return participating_agent_ids

    def update_excess_demand(self, retail_traders, institutional_investors):
        """
        Updating the excess demand coming from retail trader and instituional investors
        :return:
        """
        excess_demand = 0
        for id, retail_trader in retail_traders.items():
            excess_demand += retail_trader.demand
        for id, hedge_fund in institutional_investors.items():
            excess_demand += hedge_fund.demand
        number_of_agents = len(retail_traders) + len(institutional_investors)
        normalized_excess_demand = excess_demand / number_of_agents
        self.excess_demand = normalized_excess_demand
