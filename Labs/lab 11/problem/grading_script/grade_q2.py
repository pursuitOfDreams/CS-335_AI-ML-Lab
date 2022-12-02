import os
import pickle
from sklearn.metrics import adjusted_rand_score

def log(rollno, result):
    with open('grading.log', 'a') as file:
        file.write(f'Q2 : {rollno} : {result}\n')

def evaluate_q2(rollno):
    score = 0.0
    try:
        assert os.system(f'cp submitted_folders/{rollno}_L11/Q2/utils.py .') == 0 
        assert os.system(f'cp submitted_folders/{rollno}_L11/Q2/main.py .') == 0
        for i in range(20):
            assert os.system(f'cp data_2/f{i}.csv ./mnist_samples.csv') == 0
            assert os.system(f'cp data_2/l{i}.pkl ./true_label.pkl') == 0
            assert os.system(f'python main_2.py') == 0
            with open('q2_labels.pkl', 'rb') as file:
                label = pickle.load(file)
            with open('true_label.pkl', 'rb') as file:
                true_label = pickle.load(file)
            assert len(label) == len(true_label)
            score = max(score, adjusted_rand_score(true_label, label))
        log(rollno, score)
    except Exception as e:
        log(rollno, e)
    try:
        os.system('rm utils.py main.py mnist_samples.csv true_label.pkl q2_labels.pkl')
    except:
        pass
    return score