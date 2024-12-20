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
    # BEGIN_YOUR_CODE ######################################################
    total = len(data)
    much_faster_prob = sum(dp.muchfaster for dp in data) / total
    early_prob = sum(dp.early for dp in data) / total

    overtake_counts = {}
    crash_counts = {}
    win_counts = {}

    for dp in data:
        mf_early = (dp.muchfaster, dp.early)
        if mf_early not in overtake_counts:
            overtake_counts[mf_early] = [0, 0]
        overtake_counts[mf_early][dp.overtake] += 1

        if mf_early not in crash_counts:
            crash_counts[mf_early] = [0, 0]
        crash_counts[mf_early][dp.crash] += 1

        overtake_crash = (dp.overtake, dp.crash)
        if overtake_crash not in win_counts:
            win_counts[overtake_crash] = [0, 0]
        win_counts[overtake_crash][dp.win] += 1

    # Normalize counts into probabilities
    overtake_probs = {k: v[1] / sum(v) for k, v in overtake_counts.items()}
    crash_probs = {k: v[1] / sum(v) for k, v in crash_counts.items()}
    win_probs = {k: v[1] / sum(v) for k, v in win_counts.items()}

    # Define the Bayesian Network structure and CPTs
    bayes_net.add(('MuchFaster', '', much_faster_prob))
    bayes_net.add(('Early', '', early_prob))
    bayes_net.add(('Overtake', 'MuchFaster Early', overtake_probs))
    bayes_net.add(('Crash', 'MuchFaster Early', crash_probs))
    bayes_net.add(('Win', 'Overtake Crash', win_probs))

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

