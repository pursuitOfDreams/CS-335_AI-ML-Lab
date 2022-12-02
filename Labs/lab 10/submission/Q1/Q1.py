import os 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt


files = os.listdir('./Testcases/input/')
input_test_cases_path= './Testcases/input/'
output_test_cases_path= './Testcases/output/'

for idx,file in enumerate(files):
    dates =[]
    data = {}
    missing_item_idxs =  []
    with open(input_test_cases_path+file,'r') as f:
        lines  = f.readlines()
        # print(lines[0].strip())
        N = int(lines[0].strip())
        
        for i in range(N):
            line = lines[i+1].strip().split("\t")
            dates.append(line[0])
            if "Missing" in line[1]:
                missing_item_idxs.append(i)
            else:
                data[i] = (float(line[1]))
    
    
    x = np.array(list(data.keys())).reshape(-1,1)
    y = np.array(list(data.values()))
    polyreg = make_pipeline(PolynomialFeatures(7), LinearRegression())
    polyreg.fit(x,y)
    print(polyreg.score(x,y))
    predictions =[]
    for missing_idx in missing_item_idxs:
        predictions.append(polyreg.predict(np.array(missing_idx).reshape(-1,1))[0])
        
    with open(output_test_cases_path+"output"+file[-6:],'w') as f:
        lines =[]
        for p in predictions:
            lines.append(str(p)+"\n")

        f.writelines(lines)

    plt.plot(list(data.keys()), list(data.values()))
    plt.plot(list(data.keys()),polyreg.predict(x))
    plt.scatter(missing_item_idxs, predictions)
    plt.savefig(f"plt_{idx}.png")
    plt.clf()

    
