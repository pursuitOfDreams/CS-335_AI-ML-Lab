import json
from typing import Tuple, List
import numpy as np
import sys

def generate_uniform(seed: int, num_samples: int) -> None:
    """
    Generate 'num_samples' number of samples from uniform
    distribution and store it in 'uniform.txt'
    """

    # TODO
    np.random.seed(seed = seed)
    nums = np.random.random(num_samples)
    return nums


def inv_transform(file_name: str, distribution: str, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []
    nums = np.loadtxt(file_name, dtype=float)
    # TODO
    if distribution == "categorical":
        values = kwargs["values"]
        probs = kwargs["probs"]
        cummulatives = []
        sum = 0
        for prob in probs:
            sum += prob
            cummulatives.append(sum)
        for num in nums:
            for i in range(len(cummulatives)):
                if num < cummulatives[i]:
                    samples.append(values[i])
                    break
    elif distribution == "exponential":
        def inv_exp(lam, x):
            return -np.log(1-x)/lam
        samples = inv_exp( kwargs["lambda"],nums)
    elif distribution == "cauchy":
        def inv_cauchy(mean, gamma, x):
            return mean + gamma*np.tan(np.pi*(x-0.5))
        samples = inv_cauchy(kwargs["peak_x"], kwargs["gamma"], nums)
        
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
    indices = list(indices)
    # TODO
    def log_gaussian_prob(x, mu, sigma):
        return -np.multiply(x-mu, x-mu)/(2*sigma*sigma) - np.log(sigma)
    def log_exponential(x, lam):
        return np.log(lam) - lam*x
    samples = np.array(samples)
    g1 = np.sum(log_gaussian_prob(samples, 0, 1))
    g2 = np.sum(log_gaussian_prob(samples, 0, 0.5))
    g3 = np.sum(log_gaussian_prob(samples, 1, 1))
    if g1 >= g2 and g1 >= g3:
        indices[0] = 0
    elif g2 >= g1 and g2 >= g3:
        indices[0] = 1
    else:
        indices[0] = 2
    e1 = np.sum(log_exponential(samples, 0.5))
    e2 = np.sum(log_exponential(samples, 1))
    e3 = np.sum(log_exponential(samples, 2))
    if e1 >= e2 and e1 >= e3:
        indices[2] = 0
    elif e2 >= e1 and e2 >= e3:
        indices[2] = 1
    else:
        indices[2] = 2
    min_num = np.min(samples)
    max_num = np.max(samples)
    if min_num >= 0 and max_num <= 1:
        indices[1] = 0
    elif min_num >= 0 and max_num <= 2:
        indices[1] = 1
    else:
        indices[1] = 2
    indices = tuple(indices)
    # END TODO
    assert len(indices) == 3
    assert all([index >= 0 and index <= 2 for index in indices])
    return indices

def marks_confidence_intervals(samples: list, variance: float, epsilons: list) -> Tuple[float, List[float]]:

    sample_mean = 0
    deltas = [0 for e in epsilons] # List of zeros

    # TODO
    sample_mean = np.mean(samples)
    deltas = [variance/(len(samples)*e*e) for e in epsilons]
    # END TODO

    assert len(deltas) == len(epsilons)
    return sample_mean, deltas

if __name__ == "__main__":

    roll_number = sys.argv[1]
    name = sys.argv[2]

    marks = [0]*4
    q_comments = ['']*4
    isImported = True
    
    try:
        import assignment
    except:
        isImported = False
        q_comments = ['Error importing functions']*4
    # question 1
    if isImported:
        try:
            q_comments[0] += "\""
            for seed in [56, 78, 1923]:
                assignment.generate_uniform(seed,100)
                student_answer = np.loadtxt('uniform.txt')
                correct_answer = generate_uniform(seed, 100)
                if np.all(np.abs(student_answer - correct_answer) <= 5e-4):
                    marks[0] += 0.33
                    if seed == 1923:
                        marks[0] += 0.01
                else:
                    q_comments[0] += f'Testcase with seed {seed} failed\n\n'
            q_comments[0] += "\""
        except Exception as e:
            marks[0] = 0
            q_comments[0] = f'\"Following exception occurred:\n{e}\"'
        

        np.random.seed(23)
        with open('uniform.txt','w') as f:
            f.write("\n".join([str(x) for x in np.random.random(100)]))
        # question 2
        try:
            q_comments[1] += "\""
            for distribution in ["categorical", "exponential", "cauchy"]:
                for index in [1,2,3]:
                    file_name = f"q2_{distribution}_{index}.json"
                    args = json.load(open(file_name, "r"))
                    correct_answer = inv_transform(**args)
                    student_answer = assignment.inv_transform(**args)
                    student_answer = [float(x) for x in student_answer]
                    if np.all(np.abs(np.array(student_answer) - np.array(correct_answer)) <= 5e-4):
                        marks[1] += 0.33
                        if index == 3:
                            marks[1] += 0.01
                    else:
                        q_comments[1] += f"Testcase with args {args} failed\n\n"
            q_comments[1] += "\""
        except Exception as e:
            marks[1] = 0
            q_comments[1] = f'\"Following exception occurred:\n{e}\"'
        # question 3
        try:
            q_comments[2] += "\""
            for i in [1,2,3]:
                correct_answer = find_best_distribution(np.loadtxt(f"q3_samples_{i}.csv", dtype=float))
                student_answer = assignment.find_best_distribution(np.loadtxt(f"q3_samples_{i}.csv", dtype=float))
                for j in range(3):
                    if correct_answer[j] == student_answer[j]:
                        marks[2] += 1/3
                    else:
                        failed_part = ["gaussian","uniform", "exponential"]
                        failed_part = failed_part[j]
                        q_comments[2] += f"Testcase {i} part {failed_part} failed\n\n"
            q_comments[2] += "\""
        except Exception as e:
            marks[2] = 0
            q_comments[2] = f'\"Following exception occurred:\n{e}\"'

        # question 4
        try:
            q_comments[3] += "\""
            q4_samples = np.loadtxt("q4_samples.csv", dtype=float)
            q4_epsilons = np.loadtxt("q4_epsilons.csv", dtype=float)
            variance = 5

            correct_sample_mean, correct_deltas = marks_confidence_intervals(q4_samples, variance, q4_epsilons)
            student_sample_mean, student_deltas = assignment.marks_confidence_intervals(q4_samples, variance, q4_epsilons)
            student_sample_mean = float(student_sample_mean)
            student_deltas = [float(x) for x in student_deltas]
            if np.abs(correct_sample_mean - student_sample_mean) <= 5e-4:
                    marks[3] += 0.5
            else:
                q_comments[3] += f"Incorrect sample mean\n\n"
            
            marks[3] += np.mean(np.bitwise_and(np.array(student_deltas) >= np.array(correct_deltas) - 5e-4, np.array(student_deltas) <= 3*np.array(correct_deltas)))*2.5
            if np.mean(np.bitwise_and(np.array(student_deltas) >= np.array(correct_deltas) - 5e-4, np.array(student_deltas) <= 3*np.array(correct_deltas))) != 1:
                q_comments[3] += f"Incorrect deltas for some epsilon values\n\n" 
            q_comments[3] += "\""
        except Exception as e:
            marks[3] = 0
            q_comments[3] = f'\"Following exception occurred:\n{e}\"'
        

    with open('grades.csv','a') as f:
        marks_string = ",".join([str(mark) for mark in marks])
        q_comments_string = ",".join(q_comments)
        f.write(f"{roll_number},{name},{marks_string},{sum(marks)},{q_comments_string}"+"\n")
