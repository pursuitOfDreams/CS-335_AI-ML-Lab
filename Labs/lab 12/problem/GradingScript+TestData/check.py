import numpy as np
import pickle as pkl
import sys
import math

if __name__ == "__main__":
    roll_number = sys.argv[1]

    isImported = True
    
    marks_tot = 0
    comments = ""       
     
    try:
        from assignment import * 
    except Exception as e:
        isImported = False
        comments += "Error importing functions from assignment.py: " + e
        
    if isImported:
        try:
            dataset = "mnist"
            with open(f"./data/{dataset}_test.pkl", "rb") as file:
                test_data = pkl.load(file)
            
            _, _, X_test, Y_test = split_data(test_data[0],test_data[1],train_ratio=0.0)
            Y_hat = predict(X_test,"mnist")
            score1 = (np.sum(Y_hat == Y_test)/Y_test.shape[0])
            
            X_test, Y_test = test_data[0],test_data[1]
            Y_hat = predict(X_test,"mnist")
            score2 = (np.sum(Y_hat == Y_test)/Y_test.shape[0])
            
            score_mnist=max([score1,score2])
        except Exception as e:
            score_mnist = 0
            comments += "Following exception occurred for mnist data: " +str(e)
            
        try:
            dataset = "flowers"
            with open(f"./data/{dataset}_test.pkl", "rb") as file:
                test_data = pkl.load(file)
            
            _, _, X_test, Y_test = split_data(test_data[0],test_data[1],train_ratio=0.0)
            Y_hat = predict(X_test,"flowers")
            score = (np.sum(Y_hat == Y_test)/Y_test.shape[0])
            
            X_test, Y_test = test_data[0],test_data[1]
            Y_hat = predict(X_test,"flowers")
            score2 = (np.sum(Y_hat == Y_test)/Y_test.shape[0])
            
            score_flowers=max([score1,score2])
        except Exception as e:
            score_flowers = 0
            comments += "Following exception occurred for flowers data: " +str(e)
                   
    marks_mnist = math.ceil(score_mnist*10)/2
    marks_flowers = math.ceil(score_flowers*10)/2
    marks_tot = marks_mnist + marks_flowers

# 	print(roll_number, score_mnist,score_flowers,marks_tot,comments)


    #if comments != "":        
    #    print(roll_number, score_mnist,score_flowers,marks_tot,comments)

    with open('grades.csv','a') as f:
        comments = comments.replace(',',' ')
        f.write(f"{roll_number},{str(score_mnist)},{str(score_flowers)},{str(marks_tot)},{comments}"+"\n")

