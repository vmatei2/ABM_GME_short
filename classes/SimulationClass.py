import random

import networkx as nx
import numpy as np
import seaborn as sns
from helpers.calculations_helpers import split_commitment_into_groups
from helpers.network_helpers import get_sorted_degree_values, gather_commitment_values
from helpers.network_helpers import calculate_average_commitment
from classes.InfluentialRedditTrader import InfluentialRedditUser
from classes.RegularRedditTrader import RegularRedditTrader
from helpers.plotting_helpers import plot_all_commitments, plot_commitment_into_groups, \
    simple_line_plot


class SimulationClass:
    def __init__(self, time_steps, N_agents, m, market_first_price):
        self.N_agents = N_agents  # number of participating agents in the simulation
        self.m = m  # number of edges to attach from a new node to existing nodes
        self.time_steps = time_steps
        self.market_first_price = market_first_price
        self.tau = int((N_agents / 2) * time_steps)  # parameter for updating the opinion profile of the population
        self.social_media_agents, self.average_degree = self.create_initial_network()  # the initial network of social media agents,
        # we already have a few central nodes network is set to increase in size and add new agents throughout the
        # simulation

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
        degree_values = [v for k, v in sorted_node_degree_pairs]
        average_degree = sum(degree_values) / barabasi_albert_network.number_of_nodes()
        average_degree = round(average_degree)
        return social_media_agents, average_degree

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
        average_commitment_history = []
        all_commitments_each_round = []
        commitment_changes = []
        df_data = []
        for i in range(self.tau):
            agents_on_social_media_keys = list(self.social_media_agents.keys())
            random_agent_key = random.choice(agents_on_social_media_keys)
            agent_on_social_media = self.social_media_agents[random_agent_key] # randomly picking an agent to update commitment
            if isinstance(agent_on_social_media, RegularRedditTrader):  # checking here if the agent is an instance of
                # a regular reddit trader instead of an influential one, which does not update his commitment at all
                agent_on_social_media.update_commitment(agents=self.social_media_agents, miu=0.13, average_network_degree=self.average_degree)
            # the above is the updating of the commitment throughout the network, done in a more granular way the
            # below check ensures that we are at a trading day step and that's when we update the market + add new
            # users in the network
            if i % np.int(self.N_agents / 2) == 0:

                #  here we update the number of agents in the network, and the number of agents to be added will be a
                #  a function of the percentage change in commitment value
                average_network_commitment = calculate_average_commitment(self.social_media_agents)
                average_commitment_history.append(average_network_commitment)
                commitment_this_round = gather_commitment_values(self.social_media_agents)
                all_commitments_each_round.append(commitment_this_round)
                if len(average_commitment_history) > 1:
                    if trading_day % 7 == 0:
                    # in this case we have more than one previous average commitment, hence we can calculate the
                    # percentage change
                        previous_average_commitment = average_commitment_history[-7]
                        percentage_change_in_commitment = (
                                average_network_commitment - previous_average_commitment) / previous_average_commitment
                        commitment_changes.append(percentage_change_in_commitment)
                        number_of_agents_to_be_added = int(percentage_change_in_commitment * self.N_agents)
                        for i in range(number_of_agents_to_be_added):
                            #  adding new agents to the network
                            new_neighbours = []
                            possible_neighbors = list(self.social_media_agents.keys())
                            for _ in range(self.average_degree):
                                # select different nodes, at random weighted by their degree, up to the average degree of the initial network
                                degrees = [len(agent.neighbours_ids) for id, agent in self.social_media_agents.items()]
                                j = random.choices(possible_neighbors, degrees)[0]
                                if j not in new_neighbours:
                                    new_neighbours.append(j)
                                stop = 0
                            new_id = round(random.uniform(10001, 100000))
                            if new_id not in self.social_media_agents:
                                new_agent = RegularRedditTrader(id=new_id, neighbours_ids=new_neighbours, commitment=average_network_commitment)
                                self.social_media_agents[new_id] = new_agent
                                stop = 0

                if trading_day % 20 == 0:
                    zero_to_40_list, forty_to_65_list, sixtyfive_to_one_list = split_commitment_into_groups(commitment_this_round, trading_day)
                    df_data.append(zero_to_40_list)
                    df_data.append(forty_to_65_list)
                    df_data.append(sixtyfive_to_one_list)
                trading_day += 1
                print("Finished Trading Day ", trading_day)


        plot_all_commitments(all_commitments_each_round)
        simple_line_plot(average_commitment_history, "Trading Day", "Average Commitment", "Average Commitment Evolution")
        simple_line_plot(commitment_changes, "Trading Week", "Change in commitment", "Percentage Changes in Average "
                                                                                    "Commitment")
        plot_commitment_into_groups(df_data)
        print(trading_day)


if __name__ == '__main__':
    sns.set_style("darkgrid")
    simulation = SimulationClass(time_steps=100, N_agents=10000, m=4, market_first_price=20)
    simulation.run_simulation()

    stop = 0
