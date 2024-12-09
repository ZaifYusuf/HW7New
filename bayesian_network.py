""" Bayesian networks """

from probability import BayesNet, enumeration_ask, elimination_ask, rejection_sampling, likelihood_weighting, gibbs_ask
from timeit import timeit, repeat
import pickle
import numpy as np

T, F = True, False

class DataPoint:
    """
    Represents a single datapoint gathered from one lap.
    Attributes are exactly the same as described in the project spec.
    """
    def __init__(self, muchfaster, early, overtake, crash, win):
        self.muchfaster = muchfaster
        self.early = early
        self.overtake = overtake
        self.crash = crash
        self.win = win

def generate_bayesnet():
    """
    Generates a BayesNet object representing the Bayesian network in Part 2
    returns the BayesNet object
    """
    bayes_net = BayesNet()
    # load the dataset, a list of DataPoint objects
    data = pickle.load(open("data/bn_data.p","rb"))
    total_data = len(data)
    freq = {
        'MuchFaster': {T: 0, F: 0},
        'Early': {T: 0, F: 0},
        'Overtake': {(T, T): 0, (T, F): 0, (F, T): 0, (F, F): 0},
        'Crash': {(T, T): 0, (T, F): 0, (F, T): 0, (F, F): 0},
        'Win': {(T, T): 0, (T, F): 0, (F, T): 0, (F, F): 0}
    }

    # Calculate frequencies from the dataset
    for d in data:
        freq['MuchFaster'][d.muchfaster] += 1
        freq['Early'][d.early] += 1
        freq['Overtake'][(d.muchfaster, d.early)] += 1
        freq['Crash'][(d.muchfaster, d.early)] += 1
        freq['Win'][(d.overtake, d.crash)] += 1

    # Calculate probabilities
    P_MuchFaster = {k: v / total_data for k, v in freq['MuchFaster'].items()}
    P_Early = {k: v / total_data for k, v in freq['Early'].items()}
    P_Overtake = {(k0, k1): v / (freq['MuchFaster'][k0] + freq['Early'][k1]) for (k0, k1), v in freq['Overtake'].items()}
    P_Crash = {(k0, k1): v / (freq['MuchFaster'][k0] + freq['Early'][k1]) for (k0, k1), v in freq['Crash'].items()}
    P_Win = {(k0, k1): v / freq['Overtake'][k0, k1] for (k0, k1), v in freq['Win'].items()}

    # Create the BayesNet
    bayes_net = BayesNet([
        ('MuchFaster', '', P_MuchFaster),
        ('Early', '', P_Early),
        ('Overtake', 'MuchFaster Early', P_Overtake),
        ('Crash', 'MuchFaster Early', P_Crash),
        ('Win', 'Overtake Crash', P_Win)
    ])

    return bayes_net


def find_best_overtake_condition(bayes_net):
    """
    Finds the optimal condition for overtaking the car, as described in Part 3
    Returns the optimal values for (MuchFaster,Early)
    """
    # BEGIN_YOUR_CODE ######################################################
    best_condition = None
    best_probability = 0

    conditions = [(T, T), (T, F), (F, T), (F, F)]
    for condition in conditions:
        much_faster, early = condition
        result = elimination_ask('Win', dict(MuchFaster=much_faster, Early=early), bayes_net)
        win_prob = result[T]

        if win_prob > best_probability:
            best_probability = win_prob
            best_condition = condition

    return best_condition
    
    # END_YOUR_CODE ########################################################

def main():
    bayes_net = generate_bayesnet()
    cond = find_best_overtake_condition(bayes_net)
    print("Best overtaking condition: MuchFaster={}, Early={}".format(cond[0],cond[1]))

if __name__ == "__main__":
    main()

