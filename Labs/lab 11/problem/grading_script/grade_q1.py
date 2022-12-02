import numpy as np
import json
from sklearn.metrics import adjusted_rand_score

def log(rollno, result):
    with open('grading.log', 'a') as file:
        file.write(f'Q1 : {rollno} : {result}\n')

def evaluate_q1(rollno):
    correct_k = 5
    correct_labels = np.array([0]*500 + [1]*500 + [2]*500 + [3]*500 + [4]*500)
    score = 0.0
    try:
        with open(f'submitted_folders/{rollno}_L11/Q1/labels.json') as file:
            result = json.load(file)
        if result['k'] == correct_k and len(result['labels']) == len(correct_labels):
            score = adjusted_rand_score(correct_labels, result['labels'])
        log(rollno, score)
    except Exception as e:
        log(rollno, e)
    return score