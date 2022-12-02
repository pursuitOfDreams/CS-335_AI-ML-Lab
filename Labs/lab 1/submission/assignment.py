import json
from typing import Tuple, List
import numpy as np


def generate_uniform(seed: int, num_samples: int) -> None:
    """
    Generate 'num_samples' number of samples from uniform
    distribution and store it in 'uniform.txt'
    """

    # TODO
    np.random.seed(seed)
    l = np.random.rand(100,1)
    f = open("uniform.txt",'w')
    np.savetxt("uniform.txt",l)

    # END TODO

    assert len(np.loadtxt("uniform.txt", dtype=float)) == 100
    return None


def inv_transform(file_name: str, distribution: str, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO
    unif_samples = np.loadtxt(file_name)
    if (distribution=="categorical"):
        values = kwargs['values']
        probs = kwargs['probs']
        for i in range(0,100):
            sum =0.0
            x = unif_samples[i]
            for j in range(0,len(probs)):
                if (x < probs[j]+sum) and (x> sum):
                    samples.append(values[j])
                    break
                sum += probs[j]
        
    elif (distribution=="exponential"):
        for i in range(0,100):
            x = (-1/kwargs['lambda'])*np.log(1-unif_samples[i])
            samples.append(x)
    
    elif (distribution=="cauchy"):
        for i in range(0,100):
            x = kwargs['gamma']*np.tan(np.pi*(unif_samples[i]-0.50)) +kwargs['peak_x']
            samples.append(x)
    
    # END TODO
    assert len(samples) == 100
    return samples

def gaussian(x, mu, sigma):
    x = np.exp(-np.power((x-mu),2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    pdt =1
    for i in range(0, len(x)):
        pdt = pdt*x[i]
    return pdt
    
def uniform(x):
    return [np.max(x), np.min(x)]
    
def exp(x, l):
    pdt =1
    x = l*np.exp(-1*l*x)
    for i in range(0, len(x)):
        pdt = pdt*x[i]
    return pdt

def find_best_distribution(samples: list) -> Tuple[int, int, int]:
    """
    Given the three distributions of three different types, find the distribution
    which is most likely the data is sampled from for each type
    Return a tupple of three indices corresponding to the best distribution
    of each type as mentioned in the problem statement
    """
    indices = [0,0,0]

    # TODO
    samples = np.array(samples)
    g1 = gaussian(samples, 0,1)
    g2 = gaussian(samples, 0, 0.5)
    g3 = gaussian(samples, 1,1)
    
    if g1 == max([g1, g2, g3]):
        indices[0]=0
    elif g2 == max([g1, g2, g3]):
        indices[0]=1
    elif g3 == max([g1, g2, g3]):  
        indices[0]=2
        
    u = uniform(samples)
    
    if u[1]>=0 and u[0]<=1:
        indices[1]=0
    elif u[1]>=0 and u[0]<=2:
        indices[1]=1
    elif u[1]>=-1 and u[0]<=1:
        indices[1]=1
            
    e1 = exp(samples, 0.5)
    e2 = exp(samples, 1)
    e3 = exp(samples,2)
    

    
    if e1 == max([e1, e2, e3]):
        indices[2]=0
    elif e2 == max([e1, e2, e3]):
        indices[2]=1
    elif e3 == max([e1, e2, e3]):  
        indices[2]=2
    
    indices =tuple(indices)
    print(indices)
    # END TODO
    assert len(indices) == 3
    assert all([index >= 0 and index <= 2 for index in indices])
    return indices

def marks_confidence_intervals(samples: list, variance: float, epsilons: list) -> Tuple[float, List[float]]:

    sample_mean = 0
    deltas = [0 for e in epsilons] # List of zeros

    # TODO
    for i in range(0, len(samples)):
        sample_mean+= samples[i]

    sample_mean/=len(samples)
    
    for i in range(0, len(deltas)):
        deltas[i] = (variance/len(samples))/(epsilons[i]**2)

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
