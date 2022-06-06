def get_sorted_degree_values(G):
    """
    Returns the descending node-degree pairs from a network G
    :param G:
    :return:
    """
    sorted_node_degree_pairs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return sorted_node_degree_pairs


def calculate_average_commitment(agents):
    number_of_agents = len(agents)
    total_commitment = 0
    for id, agent in agents.items():
        total_commitment += agent.commitment
    average_commitment = total_commitment / number_of_agents
    return average_commitment


