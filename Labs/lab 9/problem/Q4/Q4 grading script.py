
import time
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os
import sys
x_shape = (1, 6,6)
w_shape = (1,3,3)
x = np.array([[10,20,30,40,50,60],[2,4,6,8,10,12],[5,10,15,20,25,30],[4,8,12,16,20,24],[15,20,30,20,25,30],[8,16,24,32,40,48]]).reshape(x_shape)
w = np.array([[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]]).reshape(w_shape)
stride=1
corr_out=[[[[10, 20, 30, 40, 50, 60],
    [ 2, 10, 15, 19, 24, 12],
    [ 5,  8, 12, 16, 20, 30],
    [ 4, 12, 16, 19, 23, 24],
    [15, 17, 22, 24, 28, 30],
    [ 8, 16, 24, 32, 40, 48]]]]
if __name__ == '__main__':
    os.chdir(sys.argv[1])
    roll_number = sys.argv[2]
    name = sys.argv[3]
    marks = [0]*1
    comments=['']*1
      ######################## Q4 ####################################
    try:
      total_Marks=0
      os.chdir('./Q4')
      sys.path.insert(1, os.getcwd())
      os.system("ls")
      #target="./Q1.py"
      #original=os.path.join(os.getcwd(),"Q1.py")
      #shutil.copyfile(original,target)
      import Q4
      os.chdir('..')
      try:
        out=Q4.GaussianFilter(x, w, stride)
        print(out)
        p=1
        for i in range(6):
          for j in range(6):
            if int(out[0][0][i][j])!=corr_out[0][0][i][j]:
              print(out[0][0][i][j],corr_out[0][0][i][j])
              p=0
        if p==1:
          total_Marks+=3
        else:
          comments[0]+="return value of GaussianFilter function is not correct"
      except Exception as e: 
        comments[0] += f"Following exception occurred in GaussianFilter function:{e},"
    except Exception as e:
      os.chdir('..')
      comments[0] += f"Following exception occurred:{e}\t"
    marks[0]=total_Marks
    comments[0] = (str(comments[0])).replace('\n', '\t')
    with open('/home/yashwant/Music/grading/grades.csv','a') as f:
        marks_string = ";".join([str(mark) for mark in marks])
        q_comments_string = ";".join(comments)
        f.write(f"{roll_number};{name};{marks_string};{q_comments_string}"+"\n")
        # csvwriter = csv.writer(f,delimiter =",",quoting=csv.QUOTE_MINIMAL)
        # csvwriter.writerows(f"{roll_number},{name},{marks_string},{sum(marks)},{total_feedback}"+"\n",)
