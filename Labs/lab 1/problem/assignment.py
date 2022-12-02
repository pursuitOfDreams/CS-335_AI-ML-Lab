import json
from typing import Tuple, List
import numpy as np


def generate_uniform(seed: int, num_samples: int) -> None:
    """
    Generate 'num_samples' number of samples from uniform
    distribution and store it in 'uniform.txt'
    """

    # TODO

    # END TODO

    assert len(np.loadtxt("uniform.txt", dtype=float)) == 100
    return None


def inv_transform(file_name: str, distribution: str, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO

    # END TODO
    assert len(samples) == 100
    return samples


def find_best_distribution(samples: list) -> Tuple[int, int, int]:
    """
    Given the three distributions of three different types, find the distribution
    which is most likely the data is sampled from for each type
    Return a tupple of three indices corresponding to the best distribution
    of each type as mentioned in the problem statement
    """
    indices = (0,0,0)

    # TODO

    # END TODO
    assert len(indices) == 3
    assert all([index >= 0 and index <= 2 for index in indices])
    return indices

def marks_confidence_intervals(samples: list, variance: float, epsilons: list) -> Tuple[float, List[float]]:

    sample_mean = 0
    deltas = [0 for e in epsilons] # List of zeros

    # TODO

    # END TODO

    assert len(deltas) == len(epsilons)
    return sample_mean, deltas

if __name__ == "__main__":
    seed = 21734

    # question 1
    generate_uniform(seed, 100)

    # question 2
    for distribution in ["categorical", "exponential", "cauchy"]:
        file_name = "q2_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        with open("q2_output_" + distribution + ".txt", "w") as f:
            for elem in samples:
                f.write(str(elem) + "\n")

    # question 3
    indices = find_best_distribution(np.loadtxt("q3_samples.csv", dtype=float))
    with open("q3_output.txt", "w") as f:
        f.write("\n".join([str(e) for e in indices]))

    # question 4
    q4_samples = np.loadtxt("q4_samples.csv", dtype=float)
    q4_epsilons = np.loadtxt("q4_epsilons.csv", dtype=float)
    variance = 5

    sample_mean, deltas = marks_confidence_intervals(q4_samples, variance, q4_epsilons)

    with open("q4_output.txt", "w") as f:
        f.write("\n".join([str(e) for e in [sample_mean, *deltas]]))
