import numpy as np
import scipy as sp
from scipy.spatial import distance
from scipy.stats import multivariate_normal

def Divergence_Measure(A,B,Divergence_Measure_Case):
    """Decription of Divergence_Measure

    Args:
        A (matrix): A represents the observations of X that are in cluster and have class label A 
        B (matrix): B represents the observations of X that are in cluster and have class label B
        Divergence_Measure_Case (integer): Divergence_Measure_Case represents the case of divergence measure that is used {0,1,2,3}

    Returns:
        int: The Cauchy-Schwarz divergence measure between A and B
    """
    D_CS = 0
    #Define Na and Nb, d 
    Na, d = A.shape
    Nb, _ = B.shape
    #switch case for the divergence measure
    if Divergence_Measure_Case == 0: 
        # x = concatenation of A and B, y = matrix of ones with dimensions (Na+Nb,1)
        x = np.concatenate((A,B), axis=0)
        y = np.concatenate((np.ones((Na,1)), np.ones((Nb,1))*2), axis=0)
        
        std1 = np.std(A, axis=0)
        std1 = std1 * ((4/((d+2)*Na))**(1/(d+4)))
        # matlab reference:         std1(std1<10^(-6)) = 10^(-6);
        std1[std1<10**(-6)] = 10**(-6)

        std2 = np.std(B, axis=0)
        std2 = std2 * ((4/((d+2)*Nb))**(1/(d+4)))
        # matlab reference:         std2(std2<10^(-6)) = 10^(-6);
        std2[std2<10**(-6)] = 10**(-6)

        # matlab reference for sigma =         sigma = [std1' std2'];
        sigma = np.concatenate((std1[:, np.newaxis], std2[:, np.newaxis]), axis=1)
        
        #finding the index for each class {1 and 2}
        index1 = np.where(y==1)[0]
        index2 = np.where(y==2)[0]


        ## matlab reference: [G,G1,G2] = deal(zeros(Na+Nb,1));
        G = np.zeros((Na+Nb,1))
        G1 = np.zeros((Na+Nb,1))
        G2 = np.zeros((Na+Nb,1))

        Y = np.ones((Na+Nb, 1)) # vector of 1's of size = n (will be useful for kronecker product)
        Y1 = np.ones((Na, 1)) # vector of 1's of size = qty of class 1 obs
        Y2 = np.ones((Nb, 1)) # vector of 1's of size = qty of class 2 obs


        
        # M1 and M2 are defined in such a way that: if M1(i)*M2(j) = 1, then the indices
        # i and j correspond to observations of different classes. This definition is
        # similar to the Information Theoretic Clustering article
        M1 = np.concatenate((np.ones((Na,1)), np.zeros((Nb,1))), axis=0).T
        M2 = np.concatenate((np.zeros((Na,1)), np.ones((Nb,1))), axis=0).T
        for i in range(0, Na+Nb): 
            #  Information potential between the 2 clusters
            #  to avoid another 'for', I do the kron prod of x(i,:) by Y. So the
            #  q would be the second for and done all at once.
            #  the product of M1(i) by M2 makes q only class diferent observations
            #  are considered
            M = np.multiply(M1[0][i], M2)
            kron = np.kron(x[i,:],Y).reshape(-1, len(x[0,:])) - x
            zer0s = np.zeros(len(x[0,:]))
            dg = np.diag(sigma[:,0]**2 + sigma[:,1]**2)
            mvpdf = multivariate_normal.pdf(kron, zer0s, dg)
            G[i] = np.sum(M * multivariate_normal.pdf(kron, zer0s, dg))
            M = M1[0][i]
            kron = np.kron(x[i,:],Y1).reshape(-1, len(x[0,:])) - x[index1,:]
            G1[i] = np.sum(M * multivariate_normal.pdf(kron, zer0s, np.diag(2*(sigma[:,0]**2))))
            M = M2[0][i]
            kron = np.kron(x[i,:],Y2).reshape(-1, len(x[0,:])) - x[index2,:]
            G2[i] = np.sum(M * multivariate_normal.pdf(kron, zer0s, np.diag(2*(sigma[:,1]**2))))
            
        
        # Distance between the two clusters
        Cef = (1/(len(index1) * len(index2))) * np.sum(G)
        # Distance of the first cluster
        DCef = -2 * np.log(Cef)
        
        AuxG1 = (1/(len(index1) ** 2)) * np.sum(G1)
        DG1 = np.log(AuxG1)
        
        AuxG2 = (1/(len(index2) ** 2)) * np.sum(G2)
        DG2 = np.log(AuxG2)
        # Cauchy-Schwarz divergence measure
        D_CS = DCef + DG1 + DG2
    
    elif Divergence_Measure_Case == 1: 
        Va = np.var(A, axis=0)
        Vb = np.var(B, axis=0)
        ha2 = ( 4/((2*d+1)*Na) )**(2/(d+4))
        hb2 = ( 4/((2*d+1)*Nb) )**(2/(d+4))
        logha2 = (2/(d+4))*( np.log(4) - np.log(2*d+1) - np.log(Na) )
        loghb2 = (2/(d+4))*( np.log(4) - np.log(2*d+1) - np.log(Nb) )

        meanVa = np.mean(Va)
        meanVb = np.mean(Vb)
        sum_a = np.sum(np.exp( - distance.pdist(A, 'sqeuclidean')/(4*ha2*meanVa)))
        sum_b = np.sum(np.exp( - distance.pdist(B, 'sqeuclidean')/(4*hb2*meanVb)))
        sum_ab = np.sum(np.exp( - distance.cdist(A,B, 'sqeuclidean')/(2*ha2*meanVa+2*hb2*meanVb)))
        D_CS = -d*np.log(2) - (d/2)*( logha2 + loghb2 + np.log(meanVa) + np.log(meanVb) ) + d*np.log(ha2*meanVa+hb2*meanVb) -2*np.log(sum_ab) + np.log(Na + 2*sum_a) + np.log(Nb + 2*sum_b)
    
    elif Divergence_Measure_Case == 2: 
        Va = np.var(A, axis=0)
        Vb = np.var(B, axis=0)
        Va[Va<10**(-12)] = 10**(-12)
        Vb[Vb<10**(-12)] = 10**(-12)
        hb2 = ( 4/((d+2)*Nb) )**(2/(d+4))
        logha2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Na) )
        loghb2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Nb) )
        logprodVa = np.sum(np.log(Va))
        logprodVb = np.sum(np.log(Vb))
        logprodVab = np.sum(np.log(ha2*Va+hb2*Vb))
        sum_a = np.sum(np.exp( - distance.pdist(A, 'mahalanobis', VI=np.linalg.inv(np.diag(Va)))**2/(4*ha2)))
        sum_b = np.sum(np.exp( - distance.pdist(B, 'mahalanobis', VI=np.linalg.inv(np.diag(Vb)))**2/(4*hb2)))
        sum_ab = np.sum(np.exp( - distance.cdist(A,B, 'mahalanobis', VI=np.linalg.inv(np.diag(ha2*Va+hb2*Vb)))**2/2))
       
        D_CS = -d*np.log(2) - (d/2)*( logha2 + loghb2 ) - (1/2)*( logprodVa + logprodVb ) + logprodVab -2*np.log(sum_ab) + np.log(Na + 2*sum_a) + np.log(Nb + 2*sum_b)

    elif Divergence_Measure_Case == 3:
        covA = np.cov(A.T)
        covB = np.cov(B.T)

        ha2 = ( 4/((d+2)*Na) )**(2/(d+4))
        hb2 = ( 4/((d+2)*Nb) )**(2/(d+4))

        logha2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Na) )
        loghb2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Nb) )

        sum_a = np.sum(np.exp( - distance.pdist(A, 'mahalanobis', VI=np.linalg.inv(covA))**2/(4*ha2)))
        sum_b = np.sum(np.exp( - distance.pdist(B, 'mahalanobis', VI=np.linalg.inv(covB))**2/(4*hb2)))
        sum_ab = np.sum(np.exp( - distance.cdist(A,B, 'mahalanobis', VI=np.linalg.inv(ha2*covA+hb2*covB))**2/2))
        D_CS = -d*np.log(2) - (d/2)*( logha2 + loghb2 ) - (1/2)*( np.log(np.linalg.det(covA)) + np.log(np.linalg.det(covB)) ) + np.log(np.linalg.det(ha2*covA+hb2*covB)) -2*np.log(sum_ab) + np.log(Na + 2*sum_a) + np.log(Nb + 2*sum_b)

    elif Divergence_Measure_Case == 4: 
        Va = np.var(A, axis=0)
        Vb = np.var(B, axis=0)
        Va[Va<10**(-12)] = 10**(-12)
        Vb[Vb<10**(-12)] = 10**(-12)

        covA = np.cov(A.T)
        covB = np.cov(B.T)

        ha2 = ( 4/((d+2)*Na) )**(2/(d+4))
        hb2 = ( 4/((d+2)*Nb) )**(2/(d+4))

        logha2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Na) )
        loghb2 = (2/(d+4))*( np.log(4) - np.log(d+2) - np.log(Nb) )

        meanVa = np.mean(Va)
        meanVb = np.mean(Vb)

        sum_a = np.sum(np.exp( - distance.pdist(A, 'mahalanobis', VI=np.linalg.inv(covA))**2/(4*ha2*meanVa)))
        sum_b = np.sum(np.exp( - distance.pdist(B, 'mahalanobis', VI=np.linalg.inv(covB))**2/(4*hb2*meanVb)))
        sum_ab = np.sum(np.exp( - distance.cdist(A,B, 'mahalanobis', VI=np.linalg.inv(ha2*meanVa*covA+hb2*meanVb*covB))**2/2))

        D_CS = -d*np.log(2) - (d/2)*( logha2 + loghb2 + np.log(meanVa) + np.log(meanVb) ) - (1/2)*( np.log(np.linalg.det(covA)) + np.log(np.linalg.det(covB)) ) + np.log(np.linalg.det(ha2*meanVa*covA+hb2*meanVb*covB)) -2*np.log(sum_ab) + np.log(Na + 2*sum_a) + np.log(Nb + 2*sum_b)
    else :
        print('Divergence_Measure_Case not found')

    return D_CS