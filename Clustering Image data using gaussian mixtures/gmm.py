#utility script for building the Gaussian Mixtures model
#import libraries
import numpy as np
import random
import math
from scipy.stats import multivariate_normal as mvn

#computes the log of likelihood
def log_likelihood(data,weights,covs,means):
    K = len(weights)
    samples = data.shape[0]
    points_array = np.empty(shape = (samples,),dtype=int)
    for i in range(samples):

        sum_one_pt = 0
        for j in range(K):
            sum_one_pt = sum_one_pt + mvn.pdf(data[i,:],mean=means[j],cov=covs[j])
        points_array[i] = math.log(sum_one_pt)    

    return np.sum(points_array)

#Compute the responsibility matrix, each row vector in that matrix is responsibility taken for that data point by every cluster
def compute_resp(data,weights,means,covs):
    resp = np.empty(shape = (data.shape[0],len(weights)),dtype=float)
    for i in range(data.shape[0]):

        for j in range(len(weights)):
            resp[i,j] = weights[j] * mvn.pdf(data[i,:],mean=means[j],cov=covs[j])

        resp[i,:] = resp[i,:]/np.sum(resp[i,:])

    return resp        


#Update the cluster parameters given the responsibilities
def update_cluster_params(resp_vector,data):
    means = []
    dims = data.shape[1]
    covs = []
    weights = []
    for j in range(resp_vector.shape[1]):
        m = np.average(data,axis = 0,weights = resp_vector[:,j])
        w = (np.sum(resp_vector[:,j]))/(data.shape[0])
        c = 0
        for i in range(data.shape[0]):
            prdct = np.dot((data[i,:] - m).reshape(dims,1),(data[i,:] - m).reshape(dims,1).T)
            c = c + (resp_vector[i,j] * prdct)

        c = c/(np.sum(resp_vector[:,j]))
        means.append(m)
        covs.append(c)
        weights.append(w)
    return means,covs,weights 




def EM(data,init_means,init_covs,init_weights,iters,threshold):
    weights = init_weights
    means = init_means
    covs = init_covs
    ll_ = log_likelihood(data,weights,covs,means)
    log_likehoods = [ll_]

    for i in range(iters):
        #Expectation step
        resp = compute_resp(data,weights,means,covs)

        #Maximization step, updating all cluster parameters from responsibility vector
        means,covs,weights = update_cluster_params(resp,data)

        ll_latest = log_likelihood(data,weights,covs,means)
        log_likehoods.append(ll_latest)
        if(i%10==0):
            print("Iteration: " + str(i))
            print("Log likelihood:" + str(ll_latest))

        if(ll_latest-ll_<threshold):
            break
        ll_ = ll_latest    

    
    out_dict = {'Final weights':weights,'Final means':means,'Final covariances':covs,'Log likelihoods':log_likehoods}

    return out_dict
    


                






