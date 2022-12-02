import os
import sys
import csv
import time
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
from sklearn import svm
from tqdm import tqdm

def get_grade_q1(folder):
    os.chdir(folder)
    #print(os.getcwd())
    os.system(f"python assignment.py > text.txt")
    with open("text.txt", "rb") as f:
        data = f.readlines()
    #print(data)
    os.chdir("../../")
    ###### Grading scheme ######
    if "Error while solving Q1" in str(data[1]):
        return((0, "Error encountered while running the code"))

    else:
        i = 1
        accuracies = []
        while "Q2" not in str(data[i]):
            if "SVM" in str(data[i]):
                accuracies.append(float(str(data[i]).split(".")[1][:2]))
            i += 1
        
        if len(accuracies) >= 2:
            if accuracies[-1] > 95:
                return((3, "None"))
            elif accuracies[-1] >= accuracies[-2]:
                return((2.5, "Below margin of accuracy"))
        elif len(accuracies) == 1:
            if accuracies[-1] >= 90:
                return((2, 'One of the polynomial feature function not implemented'))
            else:
                return((1, "Below accuracy margin"))
        else:
            return((0, "No value returned"))

def get_grade_q2(folder):
    Q2_kernel_1 = getattr(__import__('temp.'+folder + '.Q2', fromlist=['Q2_kernel_1']), 'Q2_kernel_1')
    Q2_kernel_2 = getattr(__import__('temp.'+folder + '.Q2', fromlist=['Q2_kernel_2']), 'Q2_kernel_2')
    Q2_kernel_3 = getattr(__import__('temp.'+folder + '.Q2', fromlist=['Q2_kernel_3']), 'Q2_kernel_3')
    #print("Kernel 1")
    marks = 0.0
    comments = ""
    try:
        model = svm.SVC(kernel=Q2_kernel_1, C=10, gamma=1/2, shrinking=False)
        kernel_1_sample_1_x = np.loadtxt("input/q2_kernel_1_sample_1_x.txt")
        kernel_1_sample_1_y = np.loadtxt("input/q2_kernel_1_sample_1_y.txt")
        kernel_1_sample_2_x = np.loadtxt("input/q2_kernel_1_sample_2_x.txt")
        kernel_1_sample_2_y = np.loadtxt("input/q2_kernel_1_sample_2_y.txt")
        model.fit(kernel_1_sample_2_x,kernel_1_sample_2_y)
        kernel_1_sampel_1_ypred=  model.predict(kernel_1_sample_1_x)
        test1 = sum(kernel_1_sampel_1_ypred==kernel_1_sample_1_y)/len(kernel_1_sampel_1_ypred)
        if test1 > 0.9800:
            marks += 1
        else:
            comments += "Kernel 1 sample 1 failed"
    except Exception as e:
        comments += 'Error while solving Q2 kernel 1 : ' + str(e) + '\n'
    #print("Kernel 2")
    try:
        model = svm.SVC(kernel=Q2_kernel_2, C=10, gamma=1/2, shrinking=False)
        kernel_2_sample_1_x = np.loadtxt("input/q2_kernel_2_sample_1_x.txt")
        kernel_2_sample_1_y = np.loadtxt("input/q2_kernel_2_sample_1_y.txt")
        kernel_2_sample_2_x = np.loadtxt("input/q2_kernel_2_sample_2_x.txt")
        kernel_2_sample_2_y = np.loadtxt("input/q2_kernel_2_sample_2_y.txt")
        model.fit(kernel_2_sample_2_x,kernel_2_sample_2_y)
        kernel_2_sampel_1_ypred=  model.predict(kernel_2_sample_1_x)
        test2 =  sum(kernel_2_sampel_1_ypred==kernel_2_sample_1_y)/len(kernel_2_sampel_1_ypred)
        if test2 > 0.9750:
            marks += 0.5
        else:
            comments += "Kernel 2 sample 1 failed"
    except Exception as e:
        comments += 'Error while solving Q2 kernel 2 : ' + str(e) + '\n'
    #print("Kernel 3")
    try:
        model = svm.SVC(kernel=Q2_kernel_3, C=10, gamma=1/2, shrinking=False)
        kernel_3_sample_1_x = np.loadtxt("input/q2_kernel_3_sample_1_x.txt")
        kernel_3_sample_1_y = np.loadtxt("input/q2_kernel_3_sample_1_y.txt")
        kernel_3_sample_2_x = np.loadtxt("input/q2_kernel_3_sample_2_x.txt")
        kernel_3_sample_2_y = np.loadtxt("input/q2_kernel_3_sample_2_y.txt")
        model.fit(kernel_3_sample_2_x,kernel_3_sample_2_y)
        kernel_3_sampel_1_ypred=  model.predict(kernel_3_sample_1_x)
        test3 = sum(kernel_3_sampel_1_ypred==kernel_3_sample_1_y)/len(kernel_3_sampel_1_ypred)
        if test3 > 0.940:
            marks += 0.5
        else:
            comments += "Kernel 3 sample 1 failed"
    except Exception as e:
        comments += 'Error while solving Q2 kernel 3 : ' + str(e) + '\n'
    if comments == "":
        return((marks, "None"))
    return (marks,comments)

def get_grade_q3(folder):
    marks = 0.0
    comments = ""
    QUESTION_3_TOTALMARKS = 3 # can change accordingly
    QUESTION_3A_TOTALMARKS = 0.8*QUESTION_3_TOTALMARKS
    QUESTION_3B_TOTALMARKS = 0.2*QUESTION_3_TOTALMARKS
    q3 = getattr(__import__('temp.'+folder + '.Q3', fromlist=['q3']), 'q3')
    gram_func = getattr(__import__('temp.'+folder + '.Q3', fromlist=['gram']), 'gram')
    relative_error = getattr(__import__('temp.'+folder + '.Q3', fromlist=['relative_error']), 'relative_error')
    try:    
        x_shape = (2, 3, 4, 4)
        w_shape = (3, 3, 4, 4)
        x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
        b = np.linspace(-0.1, 0.2, num=3)

        conv_param = {'stride': 2, 'pad': 1}
        out, _ = q3(x, w, b, conv_param)
        correct_out = np.array([[[[-0.08759809, -0.10987781],
                                [-0.18387192, -0.2109216 ]],
                                [[ 0.21027089,  0.21661097],
                                [ 0.22847626,  0.23004637]],
                                [[ 0.50813986,  0.54309974],
                                [ 0.64082444,  0.67101435]]],
                                [[[-0.98053589, -1.03143541],
                                [-1.19128892, -1.24695841]],
                                [[ 0.69108355,  0.66880383],
                                [ 0.59480972,  0.56776003]],
                                [[ 2.36270298,  2.36904306],
                                [ 2.38090835,  2.38247847]]]])

        # Compare your output to ours; difference should be around e-8
        if(relative_error(out, correct_out) < 1e-8):
            marks+=QUESTION_3A_TOTALMARKS*1
        elif((relative_error(out, correct_out) < 1e-7) and (relative_error(out, correct_out) > 1e-8)):
            marks+=QUESTION_3A_TOTALMARKS*0.9
        elif((relative_error(out, correct_out) < 1e-6) and (relative_error(out, correct_out) > 1e-7)):
            marks+=QUESTION_3A_TOTALMARKS*0.8
        elif((relative_error(out, correct_out) < 1e-5) and (relative_error(out, correct_out) > 1e-6)):
            marks+=QUESTION_3A_TOTALMARKS*0.7
        elif((relative_error(out, correct_out) < 1e-4) and (relative_error(out, correct_out) > 1e-5)):
            marks+=QUESTION_3A_TOTALMARKS*0.5
        else:
            marks += 0    
   
    except(Exception) as e:
        comments += "Error  solving Q3 Convolution : " + str(e) + '\n'
    try:
        A = np.array([[1, 4, 5], 
                      [-5, 8, 9]])
        output = np.array([[ 26, -36, -40],
                           [-36,  80,  92],
                           [-40,  92, 106]])
        #print(gram(A))
        #output = np.matmul(A,A.T)
        #print(output)
        try:
            #print("GRAM",gram_func(A))
            if((gram_func(A) == output).all()):
                marks+=QUESTION_3B_TOTALMARKS*1 
            else:
                marks+=QUESTION_3B_TOTALMARKS*0
                comments += "Error solving Q3 Gram Matrix : " + str(e) + '\n'
        except Exception as e:
            A = np.array([[[1, 4, 5], 
                        [-5, 8, 9]],
                        [[1, 4, 5], 
                        [-5, 8, 9]],
                        [[1, 4, 5], 
                        [-5, 8, 9]]])
            A_new = np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]))
            output = np.matmul(np.transpose(A_new), A_new)
            if((gram_func(A) == output).all()):
                marks+=QUESTION_3B_TOTALMARKS*1 
            else:
                marks+=QUESTION_3B_TOTALMARKS*0
                comments += "Error  solving Q3 Gram Matrix : " + str(e) + '\n'

    except(Exception) as e:
        comments += "Error  solving Q3 Gram Matrix : " + str(e) + '\n'
    if comments == "":
        return((marks, "None"))
    return (marks,comments)


def get_grade_q4(folder):
    path_student = "temp/"+folder+"/output/imageWithEdges.png"
    path_correct = "correct/imageWithEdges.png"
    student = imread(path_student)
    correct = imread(path_correct)
    diff = abs(student - correct)
    if diff.max() < 0.05:
        return (2,"None")
    else:
        return (0,"Error in Q4 Edge Detection")

def grade():
    marks = []
    for folder in tqdm(os.listdir("all/")):
        if os.path.exists('temp/'):
            os.system('rm -rf temp/')
        os.mkdir('temp/')
        folder = folder.replace(" ", "\ ")
        os.system('cp -r all/' + folder + '/* temp/')
        #extract file name of tar file inside temp folder
        tar_file = os.listdir('temp/')[0]
        roll = tar_file.split('_')[0]
        #os.mkdir('temp/' + roll)
        os.system('tar -xf temp/' + tar_file + ' -C temp/')
        student_mark = {}
        student_mark['roll'] = roll
        try:
            student_mark['q1'],student_mark['q1_comments'] = get_grade_q1('temp/'+roll+'_L8/')
        except Exception as e:
            student_mark['q1'],student_mark['q1_comments'] = (0, "Error in Q1 : " + str(e))
        try:
            student_mark['q2'],student_mark['q2_comments'] = get_grade_q2(roll+'_L8')
        except Exception as e:
            student_mark['q2'],student_mark['q2_comments'] = (0, "Error in Q2 : " + str(e))
        try:
            student_mark['q3'],student_mark['q3_comments'] = get_grade_q3(roll+'_L8')
        except Exception as e:
            student_mark['q3'],student_mark['q3_comments'] = (0, "Error in Q3 : " + str(e))
        try:
            student_mark['q4'],student_mark['q4_comments'] = get_grade_q4(roll+'_L8')
        except Exception as e:
            student_mark['q4'],student_mark['q4_comments'] = (0, "Error in Q4 : " + str(e))
        os.system('rm -r temp/')
        student_mark['total'] = round(student_mark['q1'] + student_mark['q2'] + student_mark['q3'] + student_mark['q4'],2)
        #print(student_mark)
        marks.append(student_mark)
    return marks


if __name__ == "__main__":
    marks = grade()
    #Write to csv
    fields = ['roll', 'total','q1', 'q1_comments', 'q2', 'q2_comments', 'q3', 'q3_comments', 'q4', 'q4_comments']
    with open('marks.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fields)
        writer.writeheader()
        writer.writerows(marks)
