from grade_q1 import evaluate_q1
from grade_q2 import evaluate_q2

"""
    Scale q1_grade by 50.0 / 1.0
    Scale q2_grade by 50.0 / 0.56
    Their sum will give you your final marks
"""

rollno = input("Enter rollnumber : ")

q1_grades = evaluate_q1(rollno)
q2_grades = evaluate_q2(rollno)

print(q1_grades, q2_grades)