from classes.RedditInvestorTypes import RedditInvestorTypes

class RedditTrader:
    def __init__(self, id, neighbours_ids, demand=None, commitment=None, investor_type=None):
        self.id = id  # id of the agent in the network
        self.neighbours_ids = neighbours_ids  # IDs of the neigbours in the network of the Reddit Trader
        if commitment is None:
            self.commitment = 1  # when commitment is not passed, we have the case where the agent is a fully
            # committed user and hence their commitment is set to 1
        else:
            self.commitment = commitment
        if demand is None:
            self.demand = 2  # as above, the influential nodes have a non-changing high demand value
        else:
            self.demand = demand
        self.investor_type = investor_type

    def print_agent_demand(self):
        print("The demand of agent %d is %d" % (self.id, self.demand))



