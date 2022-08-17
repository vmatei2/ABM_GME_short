import datetime
import random

import matplotlib.pyplot as plt
import numpy as np
import datetime

class MarketEnvironment:
    def __init__(self, initial_price, name, price_history, start_date):
        self.name = name
        self.initial_price = initial_price
        self.current_price = initial_price
        self.excess_demand = {}
        self.tau = 1.09  # noise term for updating market price
        self.price_history = price_history
        self.simulation_history = {}
        self.date = start_date

    def update_market(self, retail_traders, institutional_investors):
        self.price_history.append(self.current_price)
        demand_from_retail, demand_from_hf = self.update_excess_demand(retail_traders, institutional_investors)
        updated_price = self.current_price + self.tau * self.excess_demand
        if updated_price < 0:
            updated_price = random.uniform(0, 0.05)
        self.current_price = updated_price
        # if new_date.weekday() == 5:  # this means that it is a Friday
        #     new_date = new_date + datetime.timedelta(days=2)
        print("Updated Price: ", self.current_price)
        self.update_simulation_history()
        self.update_day()
        return demand_from_retail, demand_from_hf

    def update_day(self):
        self.date = self.date + datetime.timedelta(days=1)

    def update_simulation_history(self):
        self.simulation_history[self.date] = self.current_price

    def plot_price_history(self, title):
        """
        Function to observe the price evolution
        :return:
        """
        plt.figure(figsize=(10, 10))
        plt.plot(self.simulation_history.keys(), self.simulation_history.values(), 'r')
        plt.xlabel("Date", fontsize=15)
        plt.ylabel("Price", fontsize=15)
        plt.title(title, fontsize=19)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(title+ ".jpg")
        plt.show()


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
        commitment_scaler = 1.5
        noise_term = 0.012  #random.uniform(0.01, 0.015)
        threshold = average_commitment_value * commitment_scaler + noise_term
        # the line below loops over all probabilities and selects the ids of agents simply based on the thresholdf
        participating_agent_ids = [id for id, value in probabilities_dict.items() if value < threshold]
        particpating_agents = {}
        for id in participating_agent_ids:
            particpating_agents[id] = retail_agents[id]

        return particpating_agents

    def update_excess_demand(self, retail_traders, institutional_investors):
        """
        Updating the excess demand coming from retail trader and instituional investors
        :return:
        """
        demand_from_retail = 0
        demand_from_hf = 0
        for id, retail_trader in retail_traders.items():
            demand_from_retail += retail_trader.demand
        for id, hedge_fund in institutional_investors.items():
            demand_from_hf += hedge_fund.demand
        number_of_agents = len(retail_traders) + len(institutional_investors)
        normalized_excess_demand = (demand_from_retail + demand_from_hf) / number_of_agents
        self.excess_demand = normalized_excess_demand
        print("Market Excess Demand is: ", self.excess_demand)
        return demand_from_retail, demand_from_hf
