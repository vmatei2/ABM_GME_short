import collections

import networkx as nx
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import graph_tool.all as gt
from helpers.calculations_helpers import extract_values_counts_as_lists, rescale_array
from helpers.network_helpers import nx2gt


def get_price_history(ticker, start_date, end_date):
    """
    Function to connect to YFinance api and return df of price data for desired time period
    :param ticker:
    :return:
    """
    # getting the historical data
    price_history = ticker.history(start=start_date, end=end_date)
    return price_history


def plot_two_df_columns_together(data_frame, first_column, second_column, third_column=None, kind=None, rescale=False, title=""):
    plt.figure(figsize=(12, 10))
    if rescale:
        data_frame[first_column] = rescale_array(data_frame[first_column])
        data_frame[second_column] = rescale_array(data_frame[second_column])
        if third_column is not None:
            data_frame[third_column] = rescale_array(data_frame[third_column])
    first_plot = data_frame[first_column].plot(legend=True)
    if kind == None:
        second_plot = data_frame[second_column].plot(secondary_y=True, alpha=0.6, legend=True)
    else:
        second_plot = data_frame[second_column].plot(kind='area', secondary_y=True,
                                                 alpha=0.6, legend=True)

    if third_column is not None:
        third_plot = data_frame[third_column].plot(legend=True)
    first_plot.margins(0, 0)
    first_plot.yaxis.set_tick_params(labelsize=18)
    first_plot.xaxis.set_tick_params(labelsize=18)
    second_plot.yaxis.set_tick_params(labelsize=18)
    first_plot.set_ylabel("Closing price", fontsize=20)
    first_plot.set_xlabel("Date", fontsize=20)
    second_plot.set_ylabel("Volume (Shares traded)", fontsize=20)
    second_plot.margins(0, 0)
    plt.grid(True)
    plt.title(title, fontsize=20)
    # For some reason the file still thinks it's in ABM_GME_Short.wiki so need to move out to save in images
    plt.savefig("../images/" + title)
    plt.show()




def barplot_percentages_on_top(df, title, column, xlabel):
    fig, ax = plt.subplots(figsize=(10, 10))
    total_rows = int(len(df))
    sns.countplot(x=column, data=df)
    plt.title(title, fontsize=18)
    plt.ylabel("Count", fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    for p in ax.patches:
        percentage = '{:.2f}'.format(100 * p.get_height() / total_rows)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='center', fontsize=11, xytext=(0, 5),
                    textcoords='offset points')
    plt.show()


def line_plot(xvalues, yvalues, title, xlabel, ylabel, every_nth_showed, ylim=None, ):
    plt.figure(figsize=(14, 14))
    plt.plot(xvalues, yvalues)
    ax = plt.gca()
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    if ylim is not None:
        minimum = ylim[0]
        maximum = ylim[1]
        plt.ylim(minimum, maximum)
    plt.xticks(rotation=40, fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid(True)

    temp = ax.xaxis.get_ticklabels()
    # deciding the frequency with which we show the labels on the plot
    every_nth_showed = every_nth_showed
    temp = list(set(temp) - set(temp[::every_nth_showed]))
    for label in temp:
        label.set_visible(False)
    i = 0

    for x, y in zip(xvalues, yvalues):
        # annotating the plot on every 3rd entry - spaces out a bit more
        if i % every_nth_showed == 0 or i in [1, 4, 7, 11, 17]:
            if i == 0:
                # the user with the most posts skews the plot too much and power-law distribution can not be observed,
                # hence we limit 0 - 1000 and show the first value outside the plot
                y_position = 999
            else:
                y_position = y
            label = y
            plt.annotate(label, (x, y_position), textcoords="offset points",
                         xytext=(0, 10),
                         ha='center', fontsize=12)
        i += 1
    plt.show()


def log_log_plot(author_values, xlabel, ylabel, title):
    print("In the log log function")
    hist = dict(collections.Counter(author_values))
    print("Calculated author_hist vector")
    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.loglog(hist.keys(), hist.values(), 'ro-')

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.savefig("../images/" + title)
    plt.show()


def plot_all_commitments(all_commitments_each_round, number_of_agents, average_commitment_history, title):
    plt.figure(figsize=(10, 10))
    x = []
    for i in range(len(all_commitments_each_round)):
        x.append(i)
        all_commitments_each_round[i] = all_commitments_each_round[i][:number_of_agents]
    plt.plot(x, all_commitments_each_round)
    plt.plot(average_commitment_history, 'bo', label='Average commitment across the network')
    plt.xlabel("Trading Day")
    plt.ylabel("Commitment Values")
    plt.title(title, fontsize=20)
    plt.savefig("../images/" + title)
    plt.legend()
    plt.show()


def simple_line_plot(values_to_be_plotted, xlabel, ylabel, title):
    plt.figure(figsize=(10, 10))
    plt.plot(values_to_be_plotted)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_commitment_into_groups(commitment_this_round, title):
    columns = ["Number of agents in group", "Trading day", "Commitment group"]
    commitment_df = pd.DataFrame(commitment_this_round, columns=columns)
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(x="Trading day", y="Number of agents in group", hue="Commitment group", data=commitment_df)
    for container in ax.containers:
        ax.bar_label(container,fontsize=15)
    plt.title(title, fontsize=20)
    plt.savefig("../images/" + title)
    plt.show()


####  NETWORK PLOTTING HELPERS


def visualise_network(G, threshold, title, axs, use_graph_tool=False):

    fig = plt.figure(figsize=(12, 10))
    if use_graph_tool:
        graph = nx2gt(G)
        deg = graph.degree_property_map("total") # undirected graph so we are only working with total degrees
        deg.a = 4 * (np.sqrt(deg.a) * 0.5 + 0.4)
        pos = gt.sfdp_layout(graph)
        control = graph.new_edge_property("vector<double>")
        for e in graph.edges():
            d = np.sqrt(sum((pos[e.source()].a - pos[e.target()].a) **2)) / 5  # for curvy edges
            control[e] = [0.3, d, 0.7, d]
        gt.graph_draw(graph, mplfig=fig, pos=pos, vertex_size=deg, vertex_fill_color=deg, vorder=deg, edge_control_points=control)
        plt.savefig("../images/network week " + str(title))
        return
    d = nx.degree(G)
    degree_values = [v for k, v in d]
    nx.draw_networkx(G, pos=nx.spring_layout(G, k=0.99), nodelist=G.nodes(), node_size=[v*10 for v in degree_values], with_labels=False,
                     node_color='lightgreen', alpha=0.6, ax=axs)
    axs.set_title("Network of users with average commitment > " + str(threshold) + " week " + str(title))
    axs.title.set_size(20)



if __name__ == '__main__':
    sns.set_style("darkgrid")
    gme_ticker = "GME"
    gme = yf.Ticker(gme_ticker)
    gme_price_history = get_price_history(gme, "2020-12-08", "2021-02-04")
    plot_two_df_columns_together(gme_price_history, "Close", "Volume", "Open", "area")
    gme_institutional_holders = gme.institutional_holders
    gme_dates = gme_price_history.index
    gme_dates_as_string = []
    for date in gme_dates:
        date = date.strftime("%Y-%m-%d")
        gme_dates_as_string.append(date)

    cleaned_posts_path = "../kaggleData/cleaned_posts_data_dt_index"
    wsb_posts_data = pd.read_csv(cleaned_posts_path, infer_datetime_format=True)

    date_value, date_counts = extract_values_counts_as_lists(wsb_posts_data, 'datetime', False)

    date_post_counts_matching_gme = []
    for i, date in enumerate(date_value):
        if date in gme_dates_as_string:
            date_post_counts_matching_gme.append(date_counts[i])


    gme_price_history['number_of_posts'] = date_post_counts_matching_gme

    print(gme_institutional_holders)

    plot_two_df_columns_together(gme_price_history, "Close", "Volume", "number_of_posts", kind="area", rescale=True,
                                 title="Rescaled Closing Price, Volume and Number of Posts")
