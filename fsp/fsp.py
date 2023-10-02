import numpy as np
import d_cs
import time
import scipy.io as sio
from sklearn.cluster import KMeans
import random

def initializeStructures(opt=None):
    rmH = {
        'i': [],
        'k': [],
        'C': [],
        'rmCidx': [],
        'rmC': [],
        'dm_threshold': [],
        'dm': [],
        'rmDominantClassLabel': [],
        'rmDominantClassLabel_Proportion': [],
        'sum1': [],
        'rmXidx': [],
    }
    if opt['return_full_history']:
        H = {
            'i': [],
            'k': [],
            'C': [],
            'dm_threshold': [],
            'dm': [],
            'ClassLabelProportion': [],
            'DominantClassLabel': [],
            'DominantClassLabel_Proportion': [],
            'ClusterClassificationError': [],
            'ClassificationError': [],
            'sum1': [],
            'rmXidx': [],
            'ElapsedTime': [],
        }
    else:
        H = {}

    return rmH, H

def Feature_Space_Partition(X, y, opt=None):
    """
    Partitions the feature space based on the given input observations and their corresponding class labels.

    Args:
        X (ndarray of shape (N, d)): A numeric matrix where each row represents an observation 
            and each column corresponds to a feature.
        y (ndarray or list of shape (N, 1)): A numeric vector where the i-th row is the class 
            label of the i-th observation in X. Each value of y must be in the range [1, c], 
            where c is the number of unique class labels.
        opt (dict, optional): A dictionary containing optional settings for the partitioning process.
            Default values are provided if not specified.

    Returns:
        dict: A dictionary containing the results of the partitioning process.
    """
    tic = time.time()
    #define default values for optional parameters
    defaultOpt = {
        'dm_case': 2,
        'initial_k': 1,
        'p_parameter': 0.05,
        'dm_threshold': 0.5,
        'update_dm_threshold': True,
        'iteration_threshold': 2e6,
        'return_full_history': True,
    }
    #update default values with user specified values
    if opt is not None:
        defaultOpt.update(opt)
    opt = defaultOpt

    #Define initial values
    k = opt['initial_k']    #Current value of k for k-means
    i = 0                   #iteration counter
    rmi = 0                 #The iteration counterfor when cluster removal occured
    nCl = len(np.unique(y)) #Number of unique class labels
    N = X.shape[0]          #Number of observations
    p_parameter = opt['p_parameter'] #The p-almost homogeneous parameter
    dm_threshold = opt['dm_threshold'] #The divergence measure threshold
    sum2 = 0                #This variable is employed in the calculation of the classification 
    # Initialize the struct array rmH and H
    # In rmH, we keep a history of the segmentation process of iterations where cluster removal occurred.
    # In H, we have the complete history of the segmentation process.
    rmH, H = initializeStructures(opt)
    rmH_list = []
    H_list = []
    #While x is not empty
    #random.seed(1)
    while X.shape[0] > 0:
        print("X is not empty")
        print(f'X.shape[0]: {X.shape[0]}')
        i += 1
        print(f'i: {i}')

        if opt['return_full_history']:
            #Start measuring the loop execution time
            tStart_while = time.time()
            H['k'] = k #Save the current value of k
            H['i'] = i #Save the current iteration number
        Indices_Num = np.zeros(k)
        Indices = {}
        for j in range(k):
            Indices[j] = []
        ClassLabel_Num = np.zeros((k,nCl))
        ClassLabel_Indices = {}
        ClassLabel_Proportion = np.zeros((k,nCl))
        for j in range(k):
            ClassLabel_Indices[j] = []
        DominantClassLabel = np.zeros(k)
        DominantClassLabel_Proportion = np.zeros(k)
        nonDominantClassLabelNumber = np.zeros(k)
        dm = np.array([])
        rmXidx = np.array([])
        sum1 = np.array([])

        #Segment dataset X into K clusters using k-means
        #rand = random.randint(0,10000)

        kmeans = KMeans(n_clusters=k, max_iter= 1000, n_init='auto',random_state=1).fit(X)
        idx = kmeans.labels_
        C = kmeans.cluster_centers_
        print(f'C: {C}')
        # print(idx)
        #Evaluate each cluster
        print(f'k: {k}')
        for c in range(0,k):
            print("Evaluate cluster " + str(c))
            #Indices of the observations in X into cluster c
            Indices[c] = np.where(idx == c)[0]
            # print(Indices[c])
            # print("     ")
            #Number of observations in X into cluster c
            Indices_Num[c] = len(Indices[c])
            #Class labels of the observations in X into cluster c
            yc = y[Indices[c]]
            #For each class label, do>
            for l in range(0,nCl):
                print("Evaluate class label " + str(l))
                aux = l + 1
                ClassLabel_Indices[c,l] = np.where(yc == aux)[0]
                ClassLabel_Num[c,l] = len(ClassLabel_Indices[c,l])
                ClassLabel_Proportion[c,l] = ClassLabel_Num[c,l] / Indices_Num[c]
                print(f'ClassLabel_Num[c,l]: {ClassLabel_Num[c,l]}')
                print(f'ClassLabel_Indices[c,l]: {ClassLabel_Indices[c,l]}')
                print(f'ClassLabel_Proportion[c,l]: {ClassLabel_Proportion[c,l]}')
            # Cluster classification error
            # Let Xc and yc be the observations and class labels in cluster c
            # Let pc be the proportion of the dominant class label in cluste c
            # Let lc be the dominant class label in cluster c
            # N the oritional number of observations
            # Nc the number of observations in cluster c

            # The cluster classification error is defined as:
            # ClusterClassificationError(C) = sum(yc != lc) / Nc === mean(yc != lc)
            max_value = np.max(ClassLabel_Proportion[c])
            max_index = np.argmax(ClassLabel_Proportion[c])
            pc = max_value
            lc = max_index
            print(f'max_value: {max_value}')
            print(f'max_index: {max_index}')
            DominantClassLabel_Proportion[c] = pc
            DominantClassLabel[c] = lc
            nonDominantClassLabelNumber[c] = 0
            for x in range (len(yc)):
                if yc[x] != lc:
                    nonDominantClassLabelNumber[c] += 1
        ClusterClassificationError = np.divide(nonDominantClassLabelNumber, Indices_Num)
        #ClusterClassificationError = 1

        # Calculate the Classification error
        # Matlab to python:
        # ClassificationError = sum2 + sum(nonDominantClassLabelNumber)/N;
        ClassificationError = sum2 + sum(nonDominantClassLabelNumber) / N
        print(f'ClassificationError: {ClassificationError}')

        # Is there any homogenous region? 
        # By the definition, is said to be p-almost-homogenous if at 
        # least (100 - p)% of the obserations xa e Rc are labeled as belonging to the same class
        vec = []
        for j in range(k):
            aux = []
            for b in range(nCl):
                if ClassLabel_Proportion[j,b] >= 1 - p_parameter:
                    aux.append(True)
                    break
                else:
                    aux.append(False)
            vec.append(np.max(aux))

        print(f'vec: {vec}')

        # If there is homogenous region
        if sum(vec) > 0:
            #There is a homogenoeus region
            # Set them aside from X and y and set k=initial_k

            print("There is homogenous region remove it")
            rmCidx = np.where(vec)[0]
            print(f'rmCidx: {rmCidx}')
            for aux in rmCidx:
                rmXidx = np.concatenate((rmXidx, Indices[aux]))
                print(f'rmXidx: {rmXidx}')
            rmC = C[rmCidx, :]
            rmXidx = rmXidx.astype(int)
            print(f'rmXidx: {rmXidx}')
            
            rmNonDominantClassLabelNumberbyN = nonDominantClassLabelNumber[rmCidx] / N
            sum1 = sum(rmNonDominantClassLabelNumberbyN)
            sum2 = sum2 + sum1

            rmi = rmi + 1
            rmH['i'].append(i)
            rmH['k'].append(k)
            rmH['C'].append(C)
            rmH['rmCidx'].append(rmCidx)
            rmH['rmC'].append(rmC)
            rmH['dm_threshold'].append(dm_threshold)
            rmH['dm'].append(dm)
            rmH['rmDominantClassLabel'].append(DominantClassLabel[vec])
            rmH['rmDominantClassLabel_Proportion'].append(DominantClassLabel_Proportion[vec])
            rmH['sum1'].append(sum1)
            rmH['rmXidx'].append(rmXidx)
            rmH_list.append(rmH.copy())

            rmH, _ = initializeStructures(opt)

            X = np.delete(X, rmXidx, axis=0)
            y = np.delete(y, rmXidx, axis=0)
            k = opt['initial_k']


            
        # If there is no homogenous region
        else:
            print("There is no homogenous region")
            # Matlab to python:
            # if i>1 && opt.update_dm_threshold && ClassificationError~=0 && ClassificationError_previous~=0
            if i > 1 and opt['update_dm_threshold'] and ClassificationError != 0 and ClassificationError_previous != 0:
                print("Update dm_threshold")
                # Matlab to python:
                # dm_threshold = dm_threshold_previous/sqrt(ClassificationError/ClassificationError_previous);
                dm_threshold = dm_threshold_previous / np.sqrt(ClassificationError / ClassificationError_previous)
                print(dm_threshold)
            
            # Is there any separable region? 

            #Initialize matrix dm to store divergence measures
            dm = []
            for c in range(k):
                print(f'DM DO CLUSTER C: {c}')
                print(f'Indices de C: {Indices[c]}')
                for a in range(0,nCl-1):
                    Na = ClassLabel_Num[c,a]
                    print(f'Na: {Na}')
                    aIdx = Indices[c][ClassLabel_Indices[c,a]]
                    for b in range(a+1,nCl):
                        Nb = ClassLabel_Num[c,b]
                        bIdx = Indices[c][ClassLabel_Indices[c,b]]
                        print(f'Nb: {Nb}')
                        if Na > 2 and Nb > 2:
                            dm.append(d_cs.Divergence_Measure(
                                X[aIdx, :], 
                                X[bIdx, :], 
                                opt['dm_case']
                            ))  
                            print("DM " + str(dm[-1]))
            # Yes, there is separable region. Set k = k + 1
            for j in range(len(dm)):
                if dm[j] >= dm_threshold:
                    cst = 1
                    break
                else:
                    cst = 0
            if len(dm) == 0:
                cst = 0
            if cst > 0:
                print("There is separable region")
                k += 1
                if opt['update_dm_threshold']:
                    dm_threshold_previous = dm_threshold
            # No, there is no separable region. Set X to empty
            else: 
                print("There is no separable region")
                print("Set X to empty")
                rmNonDominantClassLabelNumberbyN = nonDominantClassLabelNumber / N
                sum1 = sum(rmNonDominantClassLabelNumberbyN)
                sum2 = sum2 + sum1
                rmCidx = np.arange(k)
                rmC = C
                rmXidx = np.arange(X.shape[0])
                rmH['i'] = i
                rmH['k'] = k
                rmH['C'] = C
                rmH['rmCidx'] = rmCidx
                rmH['rmC'] = rmC
                rmH['dm_threshold'] = dm_threshold
                rmH['dm'] = dm
                rmH['rmDominantClassLabel'] = DominantClassLabel
                rmH['rmDominantClassLabel_Proportion'] = DominantClassLabel_Proportion
                rmH['rmXidx'] = rmXidx
                rmH['sum1'] = sum1

                rmH_list.append(rmH)

                
                rmi += 1
                X = np.array([])
                y = np.array([])

        #Final settings
        dm_threshold_previous = dm_threshold
        ClassificationError_previous = ClassificationError

        #Fill the history structure fields
        if opt['return_full_history']:
            H['dm_threshold'] = dm_threshold
            H['dm'] = dm
            H['ClassLabel_Proportion'] = ClassLabel_Proportion
            H['DominantClassLabel'] = DominantClassLabel
            H['DominantClassLabel_Proportion'] = DominantClassLabel_Proportion
            H['ClusterClassificationError'] = ClusterClassificationError
            H['ClassificationError'] = ClassificationError
            H['ClassLabelProportion'] = ClassLabel_Proportion
            H['sum1'] = sum1
            H['C'] = C
            H['rmXidx'] = rmXidx
            H['ElapsedTime'] = time.time() - tStart_while
            H_list.append(H)

            _,H = initializeStructures(opt)
        # Check the stop criterion
        if i>=opt['iteration_threshold']:
            print('WARNING: The maximum number of iterations has been reached.')
            return -1 
    return rmH_list, H_list


data = sio.loadmat('mvnrnd01.mat')
rmH, H = Feature_Space_Partition(data['X'], data['y'])

for i in range(len(H)):
    print(f'Iteration: {i}, len(H[i][rmXidx]): {len(H[i]["rmXidx"])}')
    print("\n")