from re import S
import numpy as np
import os 
import ast

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def scoring_fn(model, x_test, y_test):
    y_pred = model.predict(x_test)
    n = x_test.shape[0]
    return np.sum(abs(y_pred-y_test)<=1.5)/n

files = os.listdir('./Testcases/input/')
input_test_cases_path= './Testcases/input/'
output_test_cases_path= './Testcases/output/'


subject_list =[ "Physics","Chemistry", "English", "Biology", "PhysicalEducation", "Accountancy", "BusinessStudies", "ComputerScience", "Economics"]
# training 
x = None
# input vector will be of the size d = 10
subject_to_int = {subject: idx for idx, subject in enumerate(subject_list)}

with open('training.json','r') as f:
    lines = f.readlines()
    N = int(lines[0].strip())
    x = np.zeros((N, len(subject_list)))
    y = np.zeros(N)
    for i in range(N):
        record = ast.literal_eval(lines[i+1])
        for k,v in record.items():
            if k in subject_list:
                x[i][subject_to_int[k]] = int(v)
            elif k=="Mathematics":
                y[i] = int(v)-1

    # sc = StandardScaler()
    # x = sc.fit_transform(x)

with open('sample-test.in.json','r') as f:
    lines = f.readlines()
    N_test = int(lines[0].strip())
    x_test = np.zeros((N_test, len(subject_list)))

    for i in range(N_test):
        record = ast.literal_eval(lines[i+1])
        for k,v in record.items():
            if k in subject_list:
                x_test[i][subject_to_int[k]] = int(v)

    # sc = StandardScaler()
    # x_test = sc.fit_transform(x_test)

with open('sample-test.out.json','r') as f:
    lines = f.readlines()
    y_test= np.zeros(N_test)
    
    for i in range(N_test):
        y_test[i] = int(lines[i].strip())-1


# reg = LogisticRegression(random_state = 100)
# reg.fit(data,y)


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

categorical_pipeline = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

x = categorical_pipeline.fit_transform(x)
x_test = categorical_pipeline.fit_transform(x_test)

# model =  RandomForestClassifier(n_estimators=200, max_depth=400)
model = XGBClassifier()
# model = LGBMClassifier(max_depth=100)
model.fit(x,y)
print("training accuracy", model.score(x,y))
pred_y = model.predict(x_test)
pred_y_train = model.predict(x)
accuracy = accuracy_score(y_test, pred_y)
print("testing accuracy",accuracy)
s_test = (np.sum(abs(y_test-pred_y)<=1)/N_test)
s_train = (np.sum(abs(y-pred_y_train)<=1)/N)
print("testing score", s_test)
print("training score", s_train)


for idx,file in enumerate(files):
    dates =[]
    data = {}
    missing_item_idxs =  []
    with open(input_test_cases_path+file,'r') as f:
        lines = f.readlines()
        N_test = int(lines[0].strip())
        x_test = np.zeros((N_test, len(subject_list)))

        for i in range(N_test):
            record = ast.literal_eval(lines[i+1])
            for k,v in record.items():
                if k in subject_list:
                    x_test[i][subject_to_int[k]] = int(v)
    
    x_test= categorical_pipeline.fit_transform(x_test)
    predictions = model.predict(x_test)+1
    with open(output_test_cases_path+"output"+file[-6:],'w') as f:
        lines =[]
        for p in predictions:
            lines.append(str(p)+"\n")
        
        f.writelines(lines)
