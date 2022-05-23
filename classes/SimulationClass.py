import random

import networkx as nx

from helpers.network_helpers import get_sorted_degree_values
from classes.InfluentialRedditTrader import InfluentialRedditUser
from classes.RegularRedditTrader import RegularRedditTrader


class SimulationClass:
    def __init__(self, time_steps, N_agents, m, market_first_price):
        self.N_agents = N_agents  # number of participating agents in the simulation
        self.m = m  # number of edges to attach from a new node to existing nodes
        self.time_steps = time_steps
        self.market_first_price = market_first_price
        self.tau = int((N_agents/2) * time_steps)
        self.social_media_agents = self.create_network()

    def create_network(self):
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


    def run_simulation(self):
        for i in range(self.tau):
            agent_on_social_media = random.choice(self.social_media_agents)
            if isinstance(agent_on_social_media, RegularRedditTrader): # checking here if the agent is an instance of a regular reddit trader instead of an influential one, which does not update his commitment at all
                agent_on_social_media.update_commitment(agents=self.social_media_agents, miu=2)

if __name__ == '__main__':
    simulation = SimulationClass(time_steps=100, N_agents=10000, m=4, market_first_price=20)
    simulation.run_simulation()
    stop = 0
