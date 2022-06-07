import random

from classes.RedditTrader import RedditTrader
import numpy as np


class RegularRedditTrader(RedditTrader):
    def __init__(self, id, neighbours_ids, commitment=None):
        if commitment is None:
            commitment = random.uniform(0.2, 0.5)  # normal random distribution with mean = 0 and std deviation = 1
        demand = 0  # an agent's demand in the stock
        self.d = random.uniform(0.3, 0.5)  # threshold for difference in commitment to be too high - or confidence
        # interval value - random choice rather than set values as all agents will be slightly different,
        # hence we want thought processes to be heterogeneous
        super().__init__(id, neighbours_ids, demand, commitment)

    def update_commitment(self, agents, miu, average_network_degree):
        """
        Function to update the commitment of a regular reddit trader in the network

        In the Deffuant Model, the agent updated his opinion simply based on the opinion of another randomly picked agent

        This does not replicate well what was observed in real-life, and we will calculate the average commitment of an agent's
        neighbours and compare that to the d value - makes more sense as our agents will base their commitment updates on what is
        happening in the total surrounding environment, not just one randomly chosen neighbour

        :param neighbours: the neighbours whose opinion matters
        :param d: the threshold where the difference in
        commitment is too high for this agent to update its own commitment (confidence level)
        :param miu: scaling
        parameter :return:
        """
        neighbour_commitment_value = 0
        for id in self.neighbours_ids:
            neighbour_commitment_value += agents[id].commitment
        average_neighbour_commitment = neighbour_commitment_value / len(self.neighbours_ids)
        if abs(average_neighbour_commitment - self.commitment) >= self.d:
            # this happens in the case when the difference in commitment between the agent and its neighbours is too
            # big - therefore we do not update opinion at this time point
            pass
        else:
            neighbour_choice_id = random.choice(self.neighbours_ids)  # randomly pick one neighbour for the interaction
            neighbour = agents[neighbour_choice_id]
            # otherwise, let's update this agent's opinion (being influenced)
            updated_commitment = average_neighbour_commitment + miu * abs(self.commitment - neighbour.commitment)
            self.commitment = min(updated_commitment, 1)
