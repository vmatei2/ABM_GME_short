def get_sorted_degree_values(G):
    """
    Returns the descending node-degree pairs from a network G
    :param G:
    :return:
    """
    sorted_node_degree_pairs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return sorted_node_degree_pairs