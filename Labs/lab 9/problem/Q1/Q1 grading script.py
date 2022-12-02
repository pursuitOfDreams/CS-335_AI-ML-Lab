import numpy as np

def test_gauss_kernel(kern2,x,y):
    comment, marks = "", 0
    kern2 = np.log(kern2)
    gamma = 1
    kernel = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            kernel[i,j] = np.linalg.norm(x[i]-y[j])**2
    kernel = kernel*(-1/(2*gamma**2))
    diff = np.abs(kernel - kern2)
    if(np.all(diff<0.01)):
        comment, marks = "Gaussian kernel correct ", 1
    elif (kern2.shape)==(kernel.shape):
        comment, marks = "Error in Gussian kernel computation ", 0.5
    else:
        comment, marks = "Gussian kernel incorrect ", 0
    comment += "Total "+str(marks) +" out of 1."
    return marks, comment  


def test_rbf_kernel(kern2,x,y):
    comment, marks = "", 0
    kern2 = np.log(kern2)
    gamma = 1
    kernel = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            kernel[i,j] = np.linalg.norm(x[i]-y[j])**2
    kernel = kernel*(-1*gamma)
    diff = np.abs(kernel - kern2)
    if(np.all(diff<0.01)):
        comment, marks = "RBF kernel correct ", 1
    elif (kern2.shape)==(kernel.shape):
        comment, marks = "Error in RBF kernel computation ", 0.5
    else:
        comment, marks = "RBF kernel incorrect ", 0
    comment += "Total "+str(marks) +" out of 1."
    return marks, comment  

def test_laplace_kernel(kern2,x,y):
    comment, marks = "", 0
    kern2 = np.log(kern2)
    gamma = 1
    kernel = np.zeros((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            kernel[i,j] = np.sum(np.abs(x[i]-y[j]))
    kernel = kernel*(-1/gamma)
    diff = np.abs(kernel - kern2)
    if(np.all(diff<0.01)):
        comment, marks = "Laplace kernel correct ", 1
    elif (kern2.shape)==(kernel.shape):
        comment, marks = "Error in Laplace kernel computation ", 0.5
    else:
        comment, marks = "Laplace kernel incorrect ", 0
    comment += "Total "+str(marks) +" out of 1."
    return marks, comment  

if __name__ == "__main__":
    try:
        from Q1 import gauss_kernel
        from Q1 import rbf_kernel
        from Q1 import laplace_kernel
    except Exception as e:
        isImported = False
        comments += "Error importing kernel functions: " + e
        
    # question 1
    if isImported:
        
        a = np.array([[0],[2]])
        b = np.array([[1],[4]])
        
        kernel_params = {'sigma_gauss':1,
                    'gamma_rbf':1,
                    'sigma_laplace':1}

        try:
            kernel = gauss_kernel(a,b)
            marks, comment = test_gauss_kernel(kernel,a,b)
            marks_tot+=marks
            comments+=comment
        except Exception as e:
            marks_tot += 0
            comments += "Following exception occurred in guass_kernel: " +str(e)
            
        try:
            kernel = rbf_kernel(a,b)
            marks, comment = test_rbf_kernel(kernel,a,b)
            marks_tot+=marks
            comments+=comment
        except Exception as e:
            marks_tot += 0
            comments += "Following exception occurred in guass_kernel: " +str(e)
            
        try:
            kernel = laplace_kernel(a,b)
            marks, comment = test_laplace_kernel(kernel,a,b)
            marks_tot+=marks
            comments+=comment
        except Exception as e:
            marks_tot += 0
            comments += "Following exception occurred in guass_kernel: " +str(e)
    
