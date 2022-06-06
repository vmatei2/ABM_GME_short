import random

import networkx as nx
import numpy as np
from helpers.network_helpers import get_sorted_degree_values
from helpers.network_helpers import calculate_average_commitment
from classes.InfluentialRedditTrader import InfluentialRedditUser
from classes.RegularRedditTrader import RegularRedditTrader


class SimulationClass:
    def __init__(self, time_steps, N_agents, m, market_first_price):
        self.N_agents = N_agents  # number of participating agents in the simulation
        self.m = m  # number of edges to attach from a new node to existing nodes
        self.time_steps = time_steps
        self.market_first_price = market_first_price
        self.tau = int((N_agents/2) * time_steps)  # parameter for updating the opinion profile of the population
        self.social_media_agents = self.create_initial_network() # the initial network of social media agents, we already have a few central nodes
                                                        # network is set to increase in size and add new agents throughout the simulation

    def create_initial_network(self):
        barabasi_albert_network = nx.barabasi_albert_graph(n=self.N_agents, m=self.m, seed=2)
        sorted_node_degree_pairs = get_sorted_degree_values(barabasi_albert_network)
        social_media_agents = {}
        for i, node_id_degree_pair in enumerate(sorted_node_degree_pairs):
            node_id = node_id_degree_pair[0]
            node_neighbours = list(barabasi_albert_network.neighbors(node_id))
            if i < 5:  # defining 5 largest nodes as being the influential ones in the network
                agent = InfluentialRedditUser(id=node_id, neighbours_ids=node_neighbours,
                                              market_first_price=self.market_first_price)
            else:
                agent = RegularRedditTrader(id=node_id, neighbours_ids=node_neighbours)
            social_media_agents[node_id] = agent
        return social_media_agents


    def barabasi_albert_graph(self, N, m):
        # 1. Start with a clique of m+1 nodes
        G = nx.complete_graph(m + 1)
        for i in range(G.number_of_nodes(), N):
            # 2. Select m different nodes at random, weighted by their degree.
            new_neighbors = []
            possible_neigbors = list(G.nodes)
            for _ in range(m):
                degrees = [G.degree(n) for n in possible_neigbors]
                j = random.choices(possible_neigbors, degrees)[0]
                new_neighbors.append(j)
                possible_neigbors.remove(j)
            # 3. Add a new node i and link it with the m nodes from the previous step
            for j in new_neighbors:
                G.add_edge(i, j)

        return G

    def run_simulation(self):
        trading_day = 0
        commitment_history = []
        for i in range(self.tau):
            agent_on_social_media = random.choice(self.social_media_agents)  # randomly picking an agent to update commitment
            if isinstance(agent_on_social_media, RegularRedditTrader): # checking here if the agent is an instance of
                # a regular reddit trader instead of an influential one, which does not update his commitment at all
                agent_on_social_media.update_commitment(agents=self.social_media_agents, miu=2)
            # the above is the updating of the commitment throughout the network, done in a more granular way the
            # below check ensures that we are at a trading day step and that's when we update the market + add new
            # users in the network
            if i % np.int(self.N_agents / 2) == 0:
                trading_day += 1
                #  here we update the number of agents in the network, and the number of agents to be added will be a
                #  a function of the percentage change in commitment value
                average_network_commitment = calculate_average_commitment(self.social_media_agents)
                commitment_history.append(average_network_commitment)
                if len(commitment_history) > 1:
                    # in this case we have more than one previous average commitment, hence we can calculate the percentage change
                    previous_average_commitment = commitment_history[-2]
                    percentage_change_in_commitment = 100 * (average_network_commitment - previous_average_commitment) / previous_average_commitment


        print(trading_day)

if __name__ == '__main__':
    simulation = SimulationClass(time_steps=100, N_agents=10000, m=4, market_first_price=20)
    simulation.run_simulation()

    stop = 0
