import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display import display
from mpl_toolkits import mplot3d

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


def load_results(file_name):
    f = open(file_name)
    data = json.load(f)
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

    print()

if __name__ == '__main__':
    sa_df = load_results("ofat_sa_results")
    analyse_results(sa_df)