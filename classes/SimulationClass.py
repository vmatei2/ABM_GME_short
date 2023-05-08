import datetime
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import yfinance as yf

from classes.RedditInvestorTypes import RedditInvestorTypes
from classes.SensitivityAnalysis import calculate_rmse, plot_sens_analysis_results, write_results_dict_to_file
from helpers.calculations_helpers import split_commitment_into_groups, print_current_time, rescale_array
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
    plot_demand_dictionary, barplot_options_bought, select_closing_prices, plot_hedge_funds_involvment, stacked_plots, \
    two_y_axis_plots


def store_commitment_values_split_into_groups(commitment_this_round, trading_day, df_data):
    zero_to_40_list, forty_to_65_list, sixtyfive_to_one_list = split_commitment_into_groups(
        commitment_this_round, trading_day)
    df_data.append(zero_to_40_list)
    df_data.append(forty_to_65_list)
    df_data.append(sixtyfive_to_one_list)
    return df_data


class SimulationClass:
    def __init__(self, time_steps, N_agents, N_institutional_investors, m, market_environment, miu,
                 commitment_scaler, volume_threshold, fundamental_price_inst_inv, lambda_parameter, n_influencers, d_parameter=None):
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
        self.social_media_agents, self.average_degree = self.create_initial_network(n_influencers, d_parameter)  # the initial network of social media agents,
        # we already have a few central nodes network is set to increase in size and add new agents throughout the
        # simulation
        self.institutional_investors = self.create_institutional_investors()
        self.trading_halted = False

    def create_initial_network(self, n_influencers, d=None):
        barabasi_albert_network = nx.barabasi_albert_graph(n=self.N_agents, m=self.m, seed=2)
        sorted_node_degree_pairs = get_sorted_degree_values(barabasi_albert_network)
        social_media_agents = {}
        for i, node_id_degree_pair in enumerate(sorted_node_degree_pairs):
            node_id = node_id_degree_pair[0]
            node_neighbours = list(barabasi_albert_network.neighbors(node_id))
            if i < n_influencers:  # defining 5 largest nodes as being the influential ones in the network
                agent = InfluentialRedditUser(id=node_id, neighbours_ids=node_neighbours,
                                              market_first_price=self.market_environment.initial_price,
                                              investor_type=RedditInvestorTypes.FANATICAL)
            else:
                investor_type = [RedditInvestorTypes.LONGTERM, RedditInvestorTypes.RATIONAL_SHORT_TERM]
                investor_type_probabilities = [0.5, 0.5]
                agent = RegularRedditTrader(id=node_id, neighbours_ids=node_neighbours,
                                            investor_type=random.choices(investor_type, investor_type_probabilities)[0],
                                            commitment_scaler=self.commitment_scaler, d=d)
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
            institutional_investors[i] = InstitutionalInvestor(i, demand=-1,
                                                               fundamental_price=self.fundamental_price_inst_inv,
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
        fig.delaxes(axs[i, 0])
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

    def run_simulation(self, halt_trading, gme_copy=None):
        trading_day = 0
        step = 0  # we are splitting the 100 days into 20 days steps
        threshold = 0.65
        agent_network_evolution_dict = {}
        average_commitment_history = []
        all_commitments_each_round = []
        commitment_changes = []
        volume_history = []
        hf_involved_dict = {'involved': [], 'closed': []}  # dictionary used for plotting the evolution of hedge funds participaitng in the market, and how they close their positions
        options_bought_history = []
        demand_dict = {'retail': [], 'institutional': []}
        hedge_fund_decision_dict = {}
        df_data = []  # used in plotting the commitments on separate bar charts and different values
        for i in range(self.tau):
            self.update_agent_commitment()
            if i % np.int(self.N_agents / 2) == 0:
                investors_still_involved = 0
                for key, investor in self.institutional_investors.items():
                    if investor.still_involed:
                        investors_still_involved += 1
                average_network_commitment = calculate_average_commitment(self.social_media_agents)
                average_commitment_history.append(average_network_commitment)
                hf_involved_dict['involved'].append(investors_still_involved)
                hf_involved_dict['closed'].append(len(self.institutional_investors) - investors_still_involved)
                commitment_this_round = gather_commitment_values(self.social_media_agents)
                all_commitments_each_round.append(commitment_this_round)
                if len(average_commitment_history) > 1:
                    # in this case we have more than one previous average commitment, hence we can calculate the
                    # percentage change
                    if trading_day % 7 == 0:
                        previous_average_commitment = average_commitment_history[-7]
                        percentage_change_in_commitment = (
                                                                  average_network_commitment - previous_average_commitment) / previous_average_commitment
                        commitment_changes.append(percentage_change_in_commitment)
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
                print_current_time()
                if volume >= (self.volume_threshold * self.N_agents) and halt_trading:
                    agent_ids_to_be_deleted = self.halt_trading(commitment_threshold=0.65,
                                                                commitment_lower_upper=[0.2, 0.2])
                    print("Trading halted")
                print()
        print("Final pcgt volume is: ")
        print(volume_history[-1])
        self.run_all_plots(self.market_environment, all_commitments_each_round, average_commitment_history,
                           commitment_changes, hedge_fund_decision_dict, demand_dict, df_data,
                           options_bought_history, agent_network_evolution_dict, hf_involved_dict, gme_copy)
        simulated_price = list(self.market_environment.simulation_history.values())

        return simulated_price, average_commitment_history, hedge_fund_decision_dict

    def run_all_plots(self, market_environment, all_commitments_each_round, average_commitment_history,
                      commitment_changes, hedge_fund_decision_dict,
                      demand_dict, df_data, options_bought_history,
                      agent_network_evolution_dict, hf_involved_dict, gme_copy):
        ### PLOTTING FUNCTIONS
        plot_all_commitments(all_commitments_each_round, self.N_agents, average_commitment_history,
                             "Evolution of all agent commitments")

        network_evolution_threshold = 0.6
        # self.plot_agent_network_evolution(agent_network_evolution_dict, network_evolution_threshold)

        two_y_axis_plots(y1=average_commitment_history, y2=hf_involved_dict['involved'], xlabel='Simulation Day',
                         ylabel1='Average Commitment', ylabel2='Hedge Funds Involved',title='Average Commitment '
                                                                                            'Evolution and Hedge Fund'
                                                                                            ' Participation')

        simple_line_plot(average_commitment_history, "Trading Day", "Average Commitment",
                       "Average Commitment Evolution")
        # simple_line_plot(commitment_changes, "Trading Week", "Change in commitment", "Percentage Changes in Average "
        #                                                                              "Commitment")

        plot_demand_dictionary(demand_dict, market_environment.simulation_history.keys())

        grouped_commitment = plot_commitment_into_groups(df_data, title="Evolution of agent commitments in the network through each 20 days")

        stacked_plots(df_data, market_environment)

        # plot average commitment along with price evolution
        plt.figure(figsize=(8, 8))
        # before plotting, we need to rescale the arrays
        average_commitment_history = rescale_array(average_commitment_history)
        price_history = market_environment.simulation_history.values()
        # convert price history to float values
        price_history = [float(x) for x in price_history]
        price_history = rescale_array(price_history)
        plt.plot(average_commitment_history, 'bo', label='Average commitment across the network')
        plt.plot(price_history,'rx', label='Price evolution')
        plt.title('Normalized commitment and price evolution through simulation', fontsize=18)
        plt.legend()
        plt.xlabel('Trading day', fontsize=16)
        plt.ylabel('Rescaled values', fontsize=16)
        plt.show()

        market_environment.plot_price_history("Price evolution during simulation")

        plot_hedge_funds_involvment(hf_involved_dict)

        if gme_copy is not None:
            plot_simulation_against_real_values(market_environment.simulation_history.values(), gme_copy)

        # observe_fat_tails_returns_distribution(list(market_environment.simulation_history.values()))
        #
        # observe_volatility_clustering(list(market_environment.simulation_history.values()))
        #
        # observe_autocorrelation_abs_returns(list(market_environment.simulation_history.values()))
        #
        # observe_antileverage_effect(list(market_environment.simulation_history.values()))
        #
        # extract_weekend_data_effect(market_environment.simulation_history)



def start_simulation(miu=0.5, commitment_scaler=1.5, n_agents=10000,
                     n_institutional_investors=2000, fundamental_price_inst_inv=0.1,
                     volume_threshold=0.93, lambda_parameter=1.75, time_steps=160, n_influenecrs=15, d_parameter=0.6):
    gme = yf.Ticker("GME")

    gme_price_history_path = '../data/gme_price_history.csv'
    gme_empirical_data_simulation_path = '../data/gme_empirical_data_simulation'
    if os.path.exists(gme_price_history_path) and os.path.exists(gme_empirical_data_simulation_path):
        # if the path exists, then load in the CSVs
        gme_price_history = pd.read_csv("../data/gme_price_history.csv")
        gme_empirical_data_simulation = pd.read_csv('../data/gme_empirical_data_simulation')
    else:
        gme_price_history = get_price_history(gme, "2020-11-15", "2020-12-08")
        # gme_price_history.to_csv("../data/gme_price_history.csv", index=False)
        gme_empirical_data_simulation = get_price_history(gme, "2020-11-15", "2021-02-07")
        # gme_empirical_data_simulation.to_csv("../data/gme_empirical_data_simulation", index=False)

    gme_price_history = select_closing_prices(gme_price_history)
    gme_empirical_data_simulation = select_closing_prices(gme_empirical_data_simulation)

    start_date = datetime.datetime(2020, 12, 7)
    market_environment = MarketEnvironment(initial_price=16.35, name="GME Market Environment",
                                           price_history=gme_price_history, start_date=start_date)
    simulation = SimulationClass(time_steps=time_steps, N_agents=n_agents,
                                 N_institutional_investors=n_institutional_investors,
                                 m=4,
                                 market_environment=market_environment, miu=miu,
                                 commitment_scaler=commitment_scaler, volume_threshold=volume_threshold,
                                 fundamental_price_inst_inv=fundamental_price_inst_inv,
                                 lambda_parameter=lambda_parameter,n_influencers=n_influencers, d_parameter=d_parameter)
    halt_trading = True
    prices, average_commitment_history, hf_decision_dict = simulation.run_simulation(halt_trading=halt_trading)
    return prices, market_environment, simulation, average_commitment_history, hf_decision_dict


def run_sensitivity_analysis_miu_commitment_scaler(miu_values, commitment_scaler_values, rmse_dict, price_dict, i):
    for miu in miu_values:
        for commitment_scaler in commitment_scaler_values:
            simulation_prices, market_environment, simulation_object \
                , average_commitment_history, hedge_fund_decisions_dict = start_simulation(miu=miu,
                                                                                           commitment_scaler=commitment_scaler)
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


def sensitivty_analyis_mu_theta():
    """
    Calling the run function for this specific sensitivity analysis work and setting up all the prerequisites
    :return:
    """
    rmse_dict = {}  # key = Run, Value = [0-RMSE,  1-Miu, 2-commitment_scaler]
    price_dict = {}
    i = 0
    print("Start")
    print_current_time()
    miu_values = np.linspace(0.1, 0.9, 30)  # list from 0.1-0.3 step-size = 0.1
    commitment_scaler_values = np.linspace(0.5, 2, 30)

    run_sensitivity_analysis_miu_commitment_scaler(miu_values, commitment_scaler_values, rmse_dict, price_dict, i)
    plot_sens_analysis_results(rmse_dict)
    write_results_dict_to_file(rmse_dict, file_name="sensitivty_analysis_rmse_miu_comm_sclaer")
    write_results_dict_to_file(price_dict, file_name="sensitivity_analysis_price_dict")
    print("End")
    print_current_time()


def sensitivty_analysis_ofat():
    """
    Calling the run function, and setting up all the pre-requisites for running OFAT
    :return:
    """
    gme = yf.Ticker("GME")
    gme_price_history = get_price_history(gme, "2020-11-15", "2021-02-28")
    gme_price_history = select_closing_prices(gme_price_history)
    n_reddit_agents_list = np.linspace(1000, 15000, 50)
    n_inst_investors_list = np.linspace(100, 2000, 50)
    fund_prices_list = np.linspace(1, 50, 11)
    commitment_scaler_list = np.linspace(0.1, 5, 50)
    volume_threshold_list = np.linspace(0.6, 1, 11)
    miu_parameter_list = np.linspace(0.1, 2, 20)
    sa_results_dict = one_factor_at_a_time_sensitivity_analysis(n_reddit_agents_list, n_inst_investors_list,
                                                                fund_prices_list, commitment_scaler_list,
                                                                volume_threshold_list,
                                                                miu_parameter_list, gme_price_history)
    write_results_dict_to_file(sa_results_dict, "ofat_sa_results")


def one_factor_at_a_time_sensitivity_analysis(n_reddit_agents_list, n_inst_investors_list, fund_prices_list,
                                              commitment_scaler_list, volume_threshold_list, miu_list,
                                              gme_history_copy):
    all_lists = [n_reddit_agents_list, n_inst_investors_list, fund_prices_list, commitment_scaler_list,
                 volume_threshold_list, miu_list]
    results_dict = {}
    iteration = 1
    for index, parameter_list in enumerate(all_lists):
        for parameter in parameter_list:
            if index == 0:  # update n reddit agents list
                prices, market_object, simulation_object, avg_commitment_history, hf_decision_dict = start_simulation(
                    n_agents=parameter)
            elif index == 1:  # update n inst investors
                prices, market_object, simulation_object, avg_commitment_history, hf_decision_dict = start_simulation(
                    n_institutional_investors=parameter)
            elif index == 2:
                prices, market_object, simulation_object, avg_commitment_history, hf_decision_dict = start_simulation(
                    fundamental_price_inst_inv=parameter)
            elif index == 3:
                prices, market_object, simulation_object, avg_commitment_history, hf_decision_dict = start_simulation(
                    commitment_scaler=parameter)
            elif index == 4:
                prices, market_object, simulation_object, avg_commitment_history, hf_decision_dict = start_simulation(
                    volume_threshold=parameter)
            elif index == 5:
                prices, market_object, simulation_object, avg_commitment_history, hf_decision_dict = start_simulation(
                    miu=parameter)
            results_dict[iteration] = {}
            results_dict[iteration]["N_agents"] = simulation_object.N_agents
            results_dict[iteration]["N_inst_investors"] = simulation_object.N_institutional_investors
            object.N_institutional_investors
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


def run_x_simulations(n_simulations, d_parameters, n_influencers):
    """
    Main function to run x number of simulations, and store simulation prices time series array of each simulation
    start_simulation can take in model parameter values, otherwise these are set to base values
    :param n_simulations:
    :return:
    """
    simulation_prices = []
    for i in range(n_simulations):
        prices, market_env, sim_obj, avg_commitment_history, hf_decision_dict = start_simulation(d_parameter=d_parameters[i], n_influenecrs=n_influencers)
        simulation_prices.append(prices)
    return simulation_prices


def extract_statistics(simulation_values):
    """
    :param simulation_values: a list of lists - each list contains float values representing prices at different points in the simulation
    :return: dataframe containing calculated metrics
    mean / median - mean is the average value of the data set and is sensitive to outliers, while the median is the middle value and more resistant to outliers
    standard deviation - this gives us a measure of the volatility of the stock price
    min, max
    """
    min_vals = [np.min(sim) for sim in simulation_values]
    max_vals = [np.max(sim) for sim in simulation_values]
    mean_vals = [np.mean(sim) for sim in simulation_values]
    median_vals = [np.median(sim) for sim in simulation_values]
    std_vals = [np.std(sim) for sim in simulation_values]
    statistics = [min_vals, max_vals, mean_vals, median_vals, std_vals]

    statistics_dataframe = pd.DataFrame(data=statistics, columns=['Min', 'Max', 'Mean', 'Median', 'Standard Deviation']).T # transpose the dataframe so the lists in statistics are presented column-wise rather than row-wise
    return statistics_dataframe




if __name__ == '__main__':
    sns.set_style("darkgrid")
    n_simulations = 5
    d_parameters = np.linspace(0.3, 0.8, n_simulations)
    n_influencers = 5
    all_simulations = run_x_simulations(n_simulations, d_parameters=d_parameters, n_influencers=n_influencers)
    statistics_title = str(n_influencers) + "_influencers_statistics.csv"
    statistics_dataframe = extract_statistics(all_simulations)
    print(statistics_dataframe)
    statistics_dataframe.to_csv(statistics_title)
    # plot aimulations results
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, n_simulations)]
    plt.figure(figsize=(12,12))
    for i, prices in enumerate(all_simulations):
        plt.plot(prices, color=colors[i], label='opinion threshold={0}'.format(d_parameters[i]))
        plt.legend(loc='upper left')
        plt.title("Varying trigger threshold for trading halt", fontsize=20)
        plt.xlabel("Simulation step", fontsize=15)
        plt.ylabel("Simulation price", fontsize=15)
    plt.show()
