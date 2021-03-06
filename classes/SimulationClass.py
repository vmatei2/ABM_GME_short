import datetime
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from classes.RedditInvestorTypes import RedditInvestorTypes
from classes.SensitivityAnalysis import calculate_rmse, plot_sens_analysis_results, write_results_dict_to_file
from helpers.calculations_helpers import split_commitment_into_groups
from helpers.stylized_facts import *
from helpers.network_helpers import get_sorted_degree_values, gather_commitment_values, \
    create_network_from_agent_dictionary
from helpers.network_helpers import calculate_average_commitment
from classes.InfluentialRedditTrader import InfluentialRedditUser
from classes.RegularRedditTrader import RegularRedditTrader
from classes.InstitutionalInvestor import InstitutionalInvestor
from classes.MarketEnvironment import MarketEnvironment
from helpers.plotting_helpers import plot_all_commitments, plot_commitment_into_groups, \
    simple_line_plot, visualise_network, get_price_history, scale_and_plot, plot_institutional_investors_decisions, \
    plot_demand_dictionary, barplot_options_bought, select_closing_prices


def store_commitment_values_split_into_groups(commitment_this_round, trading_day, df_data):
    zero_to_40_list, forty_to_65_list, sixtyfive_to_one_list = split_commitment_into_groups(
        commitment_this_round, trading_day)
    df_data.append(zero_to_40_list)
    df_data.append(forty_to_65_list)
    df_data.append(sixtyfive_to_one_list)
    return df_data


class SimulationClass:
    def __init__(self, time_steps, N_agents, N_institutional_investors, m, market_environment, miu,
                 commitment_scaler, volume_threshold, fundamental_price_inst_inv, lambda_parameter):
        self.N_agents = int(N_agents)  # number of participating retail traders in the simulation
        self.N_institutional_investors = int(N_institutional_investors)
        self.m = m  # number of edges to attach from a new node to existing nodes
        self.time_steps = time_steps
        self.tau = int((N_agents / 2) * time_steps)  # parameter for updating the opinion profile of the population
        self.market_environment = market_environment
        self.miu = miu  # opinion diffusion scaler
        self.commitment_scaler = commitment_scaler
        self.volume_threshold = volume_threshold
        self.lambda_parameter = lambda_parameter
        self.fundamental_price_inst_inv = fundamental_price_inst_inv
        self.social_media_agents, self.average_degree = self.create_initial_network()  # the initial network of social media agents,
        # we already have a few central nodes network is set to increase in size and add new agents throughout the
        # simulation
        self.institutional_investors = self.create_institutional_investors()
        self.trading_halted = False

    def create_initial_network(self):
        barabasi_albert_network = nx.barabasi_albert_graph(n=self.N_agents, m=self.m, seed=2)
        sorted_node_degree_pairs = get_sorted_degree_values(barabasi_albert_network)
        social_media_agents = {}
        for i, node_id_degree_pair in enumerate(sorted_node_degree_pairs):
            node_id = node_id_degree_pair[0]
            node_neighbours = list(barabasi_albert_network.neighbors(node_id))
            if i < 5:  # defining 5 largest nodes as being the influential ones in the network
                agent = InfluentialRedditUser(id=node_id, neighbours_ids=node_neighbours,
                                              market_first_price=self.market_environment.initial_price,
                                              investor_type=RedditInvestorTypes.FANATICAL)
            else:
                investor_type = [RedditInvestorTypes.LONGTERM, RedditInvestorTypes.RATIONAL_SHORT_TERM]
                investor_type_probabilities = [0.5, 0.5]
                agent = RegularRedditTrader(id=node_id, neighbours_ids=node_neighbours,
                                            investor_type=random.choices(investor_type, investor_type_probabilities)[0],
                                            commitment_scaler=self.commitment_scaler)
            social_media_agents[node_id] = agent
        degree_values = [v for k, v in sorted_node_degree_pairs]
        average_degree = sum(degree_values) / barabasi_albert_network.number_of_nodes()
        average_degree = round(average_degree)
        return social_media_agents, average_degree

    def halt_trading(self, commitment_threshold, commitment_lower_upper):
        self.trading_halted = True
        ids_to_be_deleted = []
        for agent_id, agent in self.social_media_agents.items():
            if agent.commitment <= commitment_threshold:
                agent.commitment = random.uniform(commitment_lower_upper[0], commitment_lower_upper[1])
        return ids_to_be_deleted

    def create_institutional_investors(self):
        institutional_investors = {}
        for i in range(self.N_institutional_investors):
            institutional_investors[i] = InstitutionalInvestor(i, demand=-200, fundamental_price=self.fundamental_price_inst_inv,
                                                               lambda_parameter=self.lambda_parameter)
        return institutional_investors

    @staticmethod
    def plot_agent_network_evolution(agent_network_evolution_dict, threshold):
        rows = int(len(agent_network_evolution_dict) / 2)
        fig, axs = plt.subplots(rows + 1, 2, figsize=(17, 16))
        i = 0
        for week, network in agent_network_evolution_dict.items():
            if week % 2 == 0:
                column = 0
            else:
                column = 1
            visualise_network(network, threshold, week, axs[i, column])
            if week % 2 != 0:
                i += 1  # only increase row number after visualising the network
        fig.delaxes(axs[i, 1])
        plt.savefig("agent_network_evolution")
        plt.show()

    def update_agent_commitment(self):
        agents_on_social_media_keys = list(self.social_media_agents.keys())
        random_agent_key = random.choice(agents_on_social_media_keys)
        agent_on_social_media = self.social_media_agents[
            random_agent_key]  # randomly picking an agent to update commitment
        if isinstance(agent_on_social_media, RegularRedditTrader):  # checking here if the agent is an instance of
            # a regular reddit trader instead of an influential one, which does not update his commitment at all
            agent_on_social_media.update_commitment(agents=self.social_media_agents, miu=self.miu)
        # the above is the updating of the commitment throughout the network, done in a more granular way the
        # below check ensures that we are at a trading day step and that's when we update the market + add new
        # users in the network

    def add_new_agents_to_network(self, average_network_commitment):

        #  adding new agents to the network
        new_neighbours = []
        possible_neighbors = list(self.social_media_agents.keys())
        for _ in range(self.average_degree):
            # select different nodes, at random weighted by their degree, up to the average degree of the initial network
            degrees = [len(agent.neighbours_ids) for id, agent in self.social_media_agents.items()]
            j = random.choices(possible_neighbors, degrees)[0]
            if j not in new_neighbours:
                new_neighbours.append(j)
        new_id = round(random.uniform(10001, 1000000))
        if new_id not in self.social_media_agents:
            # ensuring the random id chosen has not already been added to the agent dictionary
            new_agent = RegularRedditTrader(id=new_id, neighbours_ids=new_neighbours,
                                            commitment=average_network_commitment)
            self.social_media_agents[new_id] = new_agent

    def market_interactions(self, average_network_commitment, threshold, institutional_inv_decision_dict, trading_day):
        if self.market_environment.date.weekday() in [5, 6]:  # Saturday or Sunday:
            self.market_environment.update_day()
            return
        white_noise = random.uniform(-1, 1)
        participating_agents = self.market_environment.select_participating_agents(average_network_commitment,
                                                                                   self.social_media_agents)
        volume = len(participating_agents)
        institutional_inv_decision_dict[trading_day] = []
        options_bought = 0
        print("Number of agents involved in this trading day: ", volume)
        for id, agent in participating_agents.items():
            if isinstance(agent, InfluentialRedditUser):
                agent.make_decision(average_network_commitment, threshold)
            else:
                decision = agent.make_decision(average_network_commitment, self.market_environment.current_price,
                                               self.market_environment.price_history, white_noise, self.trading_halted)
                if decision == 1:  # above function returns 1 when agent buys option
                    options_bought += 1
        for i in range(int(len(self.institutional_investors) / 2)):
            institutional_inv_agent = random.choice(self.institutional_investors)
            decision = institutional_inv_agent.make_decision(self.market_environment.current_price,
                                                             self.market_environment.price_history)
            institutional_inv_decision_dict[trading_day].append(decision)
        demand_from_retail, demand_from_hf = self.market_environment.update_market(participating_agents,
                                                                              self.institutional_investors)
        return volume, demand_from_retail, demand_from_hf, options_bought

    def run_simulation(self, halt_trading):
        trading_day = 0
        step = 0  # we are splitting the 100 days into 20 days steps
        threshold = 0.65
        agent_network_evolution_dict = {}
        average_commitment_history = []
        all_commitments_each_round = []
        commitment_changes = []
        volume_history = []
        options_bought_history = []
        demand_dict = {'retail': [], 'institutional': []}
        hedge_fund_decision_dict = {}
        df_data = []  # used in plotting the commitments on separate bar charts and different values
        for i in range(self.tau):
            self.update_agent_commitment()
            if i % np.int(self.N_agents / 2) == 0:

                #  here we update the number of agents in the network, and the number of agents to be added will be a
                #  a function of the percentage change in commitment value
                average_network_commitment = calculate_average_commitment(self.social_media_agents)
                average_commitment_history.append(average_network_commitment)
                commitment_this_round = gather_commitment_values(self.social_media_agents)
                all_commitments_each_round.append(commitment_this_round)
                if len(average_commitment_history) > 1:
                    # in this case we have more than one previous average commitment, hence we can calculate the
                    # percentage change
                    if trading_day % 7 == 0:
                        # add new agents at the end of each weekly step
                        previous_average_commitment = average_commitment_history[-7]
                        percentage_change_in_commitment = (
                                                                  average_network_commitment - previous_average_commitment) / previous_average_commitment
                        commitment_changes.append(percentage_change_in_commitment)
                        number_of_agents_to_be_added = int(percentage_change_in_commitment * self.N_agents)
                        # for i in range(number_of_agents_to_be_added):
                        #     self.add_new_agents_to_network(average_network_commitment)
                if trading_day % 20 == 0:
                    df_data = store_commitment_values_split_into_groups(commitment_this_round, trading_day, df_data)
                    agent_network = create_network_from_agent_dictionary(self.social_media_agents, threshold=threshold)
                    agent_network_evolution_dict[step] = agent_network
                    step += 1
                try:
                    volume, demand_retail, demand_hf, options_bought = self.market_interactions(
                        average_network_commitment, threshold,
                        hedge_fund_decision_dict, trading_day)
                    demand_dict['retail'].append(demand_retail)
                    demand_dict['institutional'].append(demand_hf)
                    volume_history.append(volume)
                    options_bought_history.append(options_bought)
                except TypeError as e:
                    # Exception being hit when it is a weekend and the market is closed in the simulation
                    # the market_interaction function returns nothing in this case
                    pass

                trading_day += 1
                print("Average Network Commitment: ", average_network_commitment)
                print("Finished Trading Day ", trading_day)

                if volume >= self.volume_threshold * self.N_agents and halt_trading:
                    agent_ids_to_be_deleted = self.halt_trading(commitment_threshold=0.65,
                                                                commitment_lower_upper=[0.12, 0.2])
                    print("Trading halted")
                print()

        self.run_all_plots(self.market_environment, all_commitments_each_round, average_commitment_history,
                            commitment_changes, hedge_fund_decision_dict, demand_dict, df_data,
                           options_bought_history, agent_network_evolution_dict)
        simulated_price = list(self.market_environment.simulation_history.values())

        return simulated_price, average_commitment_history, hedge_fund_decision_dict

    def run_all_plots(self, market_environment, all_commitments_each_round, average_commitment_history,
                      commitment_changes, hedge_fund_decision_dict,
                      demand_dict, df_data, options_bought_history,
                      agent_network_evolution_dict):
        ### PLOTTING FUNCTIONS
        plot_all_commitments(all_commitments_each_round, self.N_agents, average_commitment_history,
                             "Evolution of all agent commitments")

        network_evolution_threshold = 0.65
        self.plot_agent_network_evolution(agent_network_evolution_dict, network_evolution_threshold)

        simple_line_plot(average_commitment_history, "Trading Day", "Average Commitment",
                          "Average Commitment Evolution")
        simple_line_plot(commitment_changes, "Trading Week", "Change in commitment", "Percentage Changes in Average "
                                                                                     "Commitment")

        plot_institutional_investors_decisions(hedge_fund_decision_dict, market_environment.simulation_history.keys())

        plot_demand_dictionary(demand_dict, market_environment.simulation_history.keys())

        plot_commitment_into_groups(df_data, title="Evolution of agent commitments in the network through each 20 days")

        market_environment.plot_price_history("Price evolution during simulation")

        barplot_options_bought(list(market_environment.simulation_history.keys()), options_bought_history)

        observe_fat_tails_returns_distribution(list(market_environment.simulation_history.values()))

        observe_volatility_clustering(list(market_environment.simulation_history.values()))

        observe_autocorrelation_abs_returns(list(market_environment.simulation_history.values()))

        observe_antileverage_effect(list(market_environment.simulation_history.values()))

        extract_weekend_data_effect(market_environment.simulation_history)


def start_simulation(miu=0.17, commitment_scaler=1.5, n_agents=10000,
                     n_institutional_investors=200, fundamental_price_inst_inv=1,
                     volume_threshold=0.97, lambda_parameter=1.75):
    gme = yf.Ticker("GME")
    gme_price_history = get_price_history(gme, "2020-11-15", "2020-12-08")
    gme_price_history = select_closing_prices(gme_price_history)
    start_date = datetime.datetime(2020, 12, 7)
    market_environment = MarketEnvironment(initial_price=16.35, name="GME Market Environment",
                                           price_history=gme_price_history, start_date=start_date)
    simulation = SimulationClass(time_steps=100, N_agents=n_agents, N_institutional_investors=n_institutional_investors, m=4,
                                 market_environment=market_environment, miu=miu,
                                 commitment_scaler=commitment_scaler, volume_threshold=volume_threshold,
                                 fundamental_price_inst_inv=fundamental_price_inst_inv,
                                 lambda_parameter=lambda_parameter)
    prices, average_commitment_history, hf_decision_dict = simulation.run_simulation(halt_trading=True)
    return prices, market_environment, simulation, average_commitment_history, hf_decision_dict

def run_sensitivity_analysis_miu_commitment_scaler(miu_values, commitment_scaler_values, rmse_dict, price_dict, i):
    for miu in miu_values:
        for commitment_scaler in commitment_scaler_values:
            simulation_prices, market_environment, simulation_object = start_simulation(miu=miu, commitment_scaler=commitment_scaler)
            rmse = calculate_rmse(simulation_prices[:51], gme_history_copy[:51])
            rmse_dict[i] = [rmse, miu, commitment_scaler]
            print("Run " + str(i) + " finished")
            key_name = [rmse, miu, commitment_scaler]
            key_name = str(key_name)
            max_price = np.max(simulation_prices)
            min_price = np.min(simulation_prices)
            price_dict[key_name] = [max_price, min_price]
            i += 1
    return rmse_dict, price_dict

def one_factor_at_a_time_sensitivity_analysis(n_reddit_agents_list, n_inst_investors_list, fund_prices_list, commitment_scaler_list, volume_threshold_list, miu_list):
    all_lists = [n_reddit_agents_list, n_inst_investors_list, fund_prices_list, commitment_scaler_list, volume_threshold_list, miu_list]
    results_dict = {}
    iteration = 1
    for index, parameter_list in enumerate(all_lists):
        for parameter in parameter_list:
            if index == 0:  #  update n reddit agents list
                prices, market_object, simulation_object = start_simulation(n_agents=parameter)
            elif index == 1: # update n inst investors
                prices, market_object, simulation_object = start_simulation(n_institutional_investors=parameter)
            elif index == 2:
                prices, market_object, simulation_object = start_simulation(fundamental_price_inst_inv=parameter)
            elif index == 3:
                prices, market_object, simulation_object = start_simulation(commitment_scaler=parameter)
            elif index == 4:
                prices, market_object, simulation_object = start_simulation(volume_threshold=parameter)
            elif index == 5:
                prices, market_object, simulation_object = start_simulation(miu=parameter)
            results_dict[iteration] = {}
            results_dict[iteration]["N_agents"] = simulation_object.N_agents
            results_dict[iteration]["N_inst_investors"] = simulation_object.N_institutional_investors
            results_dict[iteration]["fundamental_price_inst_inv"] = simulation_object.fundamental_price_inst_inv
            results_dict[iteration]["commitment_scaler"] = simulation_object.commitment_scaler
            results_dict[iteration]["volume_threshold"] = simulation_object.volume_threshold
            results_dict[iteration]["miu"] = simulation_object.miu
            rmse = calculate_rmse(prices[:len(gme_history_copy)], gme_history_copy)
            results_dict[iteration]["rmse"] = rmse
            max_price = np.max(prices)
            min_price = np.min(prices)
            results_dict[iteration]["max_price"] = max_price
            results_dict[iteration]["min_price"] = min_price
            iteration += 1
    return results_dict





if __name__ == '__main__':
    sns.set_style("darkgrid")
    start_simulation()
    n_simulations = 1
    #
    # for i in range(n_simulations):
    #     prices, market_environment = start_simulation(simulation_prices)
    #     simulation_prices.append(prices)
    # gme_price_history = get_price_history(gme, "2020-11-15", "2021-02-28")
    # gme_price_history = select_closing_prices(gme_price_history)
    # gme_history_copy = gme_price_history.copy()
    #  Sensitivity Analysis Section
    compare_up_to = 52  # around point when GME reached peak-price

    # calculate_rmse(average_simulation_prices[:compare_up_to], gme_price_history[:compare_up_to])

    rmse_dict = {}  # key = Run, Value = [0-RMSE, 1-Miu, 2-commitment_scaler]
    price_dict = {}
    i = 0
    miu_values = np.linspace(0.1, 0.4, 20)  # list from 0.1-0.3 step-size = 0.1
    commitment_scaler_values = np.linspace(0.75, 2, 20)

    run_sensitivity_analysis_miu_commitment_scaler(miu_values, commitment_scaler_values, rmse_dict, price_dict, i)
    plot_sens_analysis_results(rmse_dict)
    write_results_dict_to_file(rmse_dict, file_name="sensitivty_analysis_rmse_miu_comm_sclaer")
    write_results_dict_to_file(price_dict, file_name="sensitivity_analysis_price_dict")

    n_reddit_agents_list = np.linspace(1000, 15000, 50)
    n_inst_investors_list = np.linspace(100, 2000, 50)
    fund_prices_list = np.linspace(1, 50, 11)
    commitment_scaler_list = np.linspace(0.1, 5, 50)
    volume_threshold_list = np.linspace(0.6, 1, 11)
    miu_parameter_list = np.linspace(0.1, 2, 20)
    sa_results_dict = one_factor_at_a_time_sensitivity_analysis(n_reddit_agents_list, n_inst_investors_list, fund_prices_list, commitment_scaler_list, volume_threshold_list,
                                                                miu_parameter_list)
    write_results_dict_to_file(sa_results_dict, "ofat_sa_results")




