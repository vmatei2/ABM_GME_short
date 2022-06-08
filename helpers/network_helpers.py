import networkx as nx


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


def gather_commitment_values(agents):
    commitments = []
    for id, agent in agents.items():
        commitments.append(agent.commitment)
    return commitments


def create_network_from_agent_dictionary(social_media_agents):
    G = nx.Graph()
    G.add_nodes_from(social_media_agents.keys())

    for k, v in social_media_agents.items():
        G.add_edges_from([(k, t) for t in v.neighbours_ids])
    return G