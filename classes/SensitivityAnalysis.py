import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display import display

from helpers.plotting_helpers import extract_3d_plot_values, extract_prices_fixed_commitment, \
    extract_commitment_fixed_infl, plot_results_analysis, create_3d_plot


def calculate_rmse(simulation_price_history, gme_price_history):
    mse_array = np.square(np.subtract(simulation_price_history, gme_price_history))
    rmse_array = np.sqrt(mse_array)
    rmse = np.mean(rmse_array)
    print("RMSE between values is: ", rmse)
    return rmse


def plot_sens_analysis_results(rmse_dict):
    rmse_vals = []
    miu_vals = []
    commitment_scaler_vals = []
    for key, value in rmse_dict.items():
        rmse = value[0]
        miu = value[1]
        commitment_scaler = value[2]
        rmse_vals.append(rmse)
        miu_vals.append(miu)
        commitment_scaler_vals.append(commitment_scaler)
    fig = plt.figure(figsize=(9, 9))
    rmse_vals = np.array(rmse_vals)
    miu_vals = np.array(miu_vals)
    commitment_scaler_vals = np.array(commitment_scaler_vals)
    ax = plt.axes(projection='3d')

    my_cmap = plt.get_cmap('hot')
    trisurf = ax.plot_trisurf(miu_vals, commitment_scaler_vals, rmse_vals,
                              cmap=my_cmap, linewidth=0.2, antialiased=True, edgecolor="none")

    fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title(r"Sensitivity Analysis of $\mu$ and $\theta$", fontsize=18)
    ax.set_xlabel(r"$\mu$ values", fontsize=15)
    ax.set_ylabel("Commitment scaler values", fontsize=15)
    ax.set_zlabel("RMSE values", fontsize=15)
    plt.tight_layout()
    plt.savefig("../images/sensitivity_analysis_mu_commitment_scaler")
    plt.show()


def write_results_dict_to_file(results_dict, file_name):
    with open(file_name, "w") as file:
        file.write(json.dumps(results_dict))


def load_results(file_name, is_results_dict):
    """
    Loading results - in the case of the results dictionary, we do our analysis on the dictionary
    rather than the dataframe, hence boolean parameter for knowing if return before convert to df
    :param file_name:
    :param is_results_dict: see above
    :return:
    """
    f = open(file_name)
    data = json.load(f)
    if results_dict:
        return data
    df = pd.DataFrame.from_dict(data, orient='index')
    return df


def analyse_results(sa_df):
    rmse_values = sa_df['rmse']
    max_prices = sa_df["max_price"]
    min_prices = sa_df["min_price"]
    extract_statistics(rmse_values, "rmse")
    extract_statistics(max_prices, "max prices")
    extract_statistics(min_prices, "min prices")
    max_rmse_idx = sa_df["rmse"].idxmax()
    min_rmse_idx = sa_df["rmse"].idxmin()

    min_price_idx = sa_df["min_price"].idxmin()
    max_price_idx = sa_df["max_price"].idxmax()

    display(sa_df.loc[max_rmse_idx])
    print()
    display(sa_df.loc[min_rmse_idx])
    print()
    display(sa_df.loc[max_price_idx])
    print()
    display(sa_df.loc[min_price_idx])


def extract_statistics(list_, title):
    mean = np.mean(list_)
    std_dev = np.std(list_)
    max = np.max(list_)
    min = np.min(list_)
    q3, q1 = np.percentile(list_, [75, 25])
    iqr = q3 - q1
    print(f"Mean of {title} values is:  {mean}")
    print(f"Max of {title} values is:  {max}")
    print(f"Min of {title} values is:  {min}")
    print(f"Std dev of {title} values is:  {std_dev}")
    print(f"Interquantile range of {title} is: {iqr}")


def create_commitment_infl_price_analysis(results_dict):
    all_prices, all_commitments, influencer_vals = extract_3d_plot_values(results_dict)
    prices_fixed_commitment = extract_prices_fixed_commitment(results_dict)
    commitments_fixed_influencer, prices_fixed_influencer = extract_commitment_fixed_infl(results_dict)
    plot_results_analysis(influencer_vals, prices_fixed_commitment, '# of influencers', 'Max Price',
                          'Number of influencers against max price (starting commitment=0.45)')
    plot_results_analysis(commitments_fixed_influencer, prices_fixed_influencer, 'Starting commitment', 'Max Price',
                          'Starting commitment against max price (n_influencers=16)', True)

    create_3d_plot(all_prices, all_commitments, influencer_vals)


if __name__ == '__main__':
    sa_df = load_results("ofat_sa_results", is_results_dict=False)
    results_dict = load_results("results_dict.josn", is_results_dict=True)
    analyse_results(sa_df)
    create_commitment_infl_price_analysis(results_dict)
