from cmath import log
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import tqdm

######################################################################################################################################################
#Settings for plotting
######################################################################################################################################################
line_width = 3
fs = 40
fn = "Times New Roman"
markeredgewidth = line_width
capsize = 10
font = font_manager.FontProperties(family=fn,
                                style='normal', 
                                size=fs)

######################################################################################################################################################
#Log density of the gaussian distribution
######################################################################################################################################################
def log_p(x,sigma,Z,mean = 0):
    return  -np.log(Z) - (x - mean)**2 / (2 * sigma **2)

######################################################################################################################################################
#Get the Importance Sampling estimator for a range of different batcch sizes
######################################################################################################################################################
def importance_sampling(sigma_0,sigma_1,Z_1,n_max = 10000):

    data = np.zeros(n_max)

    for i in tqdm.tqdm(range(n_max)):

        x = np.random.normal(loc = 0,scale = sigma_1,size = i+1)

        r = np.exp(log_p(x,sigma = sigma_0,Z = 1) - log_p(x,sigma = sigma_1,Z = Z_1)).mean()

        data[i] = r
    
    return data

######################################################################################################################################################
#Iteratce calculation of the ratio of the partition funtions
######################################################################################################################################################
def iterative_approximation_r(n_iter,r_0,l_1,l_2):
    r_seq = np.zeros(n_iter)
    r_seq[0] = r_0

    n_1 = len(l_1)
    n_2 = len(l_2)

    s_1 = n_1 / (n_1+n_2)
    s_2 = n_2 / (n_1+n_2)

    for t in range(1,n_iter):
        r_seq[t] = n_1 / n_2 * (l_2 / (s_1*l_2 + r_seq[t-1] * s_2)).sum() / (1 / (s_1*l_1 + r_seq[t-1] * s_2)).sum()

    return r_seq

######################################################################################################################################################
#Get the Bridge Sampling estimator for a range of different batcch sizes
######################################################################################################################################################
def bridge_sampling(sigma_0,sigma_1,Z_1,n_max = 10000):

    data = np.zeros(n_max)

    for i in tqdm.tqdm(range(n_max)):
        
        x_0 = np.random.normal(loc = 0,scale = sigma_0,size = i+1)
        x_1 = np.random.normal(loc = 0,scale = sigma_1,size = i+1)

        l_1 = np.exp(log_p(x_0,sigma_0,1) - log_p(x_0,sigma_1,Z_1))
        l_2 = np.exp(log_p(x_1,sigma_0,1) - log_p(x_1,sigma_1,Z_1))

        r_seq = iterative_approximation_r(n_iter = 100,r_0 = 1,l_1 = l_1,l_2 = l_2)

        data[i] = r_seq[-1]
    
    return data
######################################################################################################################################################
#Set the standard deviations for the two experiments
######################################################################################################################################################
sigma_0_s = [1,5]
sigma_1_s = [5,1]



######################################################################################################################################################
#Initial settings for of the experiments
######################################################################################################################################################
n_samples_max = 10000
np.random.seed(47)

data_importance_sampling = []

plt.figure(figsize = (30,20))

######################################################################################################################################################
#Experiments
######################################################################################################################################################
for j in range(2):

    #Get the true partition functions
    Z_0 = np.sqrt(2 * np.pi * sigma_0_s[j] ** 2)
    Z_1 = np.sqrt(2 * np.pi * sigma_1_s[j] ** 2)

    #approximate the partition function of p_0 using the two differrent methods
    d_importance = importance_sampling(sigma_0 = sigma_0_s[j],sigma_1 = sigma_1_s[j],Z_1 = Z_1,n_max = n_samples_max)
    d_bridge = bridge_sampling(sigma_0 = sigma_0_s[j],sigma_1 = sigma_1_s[j],Z_1 = Z_1,n_max = n_samples_max)

    #Compatre the prediction as a function off the sample size with the true value Z_0
    plt.subplot(2,1,1+j)
    plt.yscale("log")
    plt.title(r"$\sigma_{proposal}$ = " + f"{sigma_1_s[j]}\t"+r"$\sigma_{unnorm.}$ = " + f"{sigma_0_s[j]}",fontsize = fs,fontname = fn)
    plt.hlines(y = Z_0,xmin=0,xmax=len(d_importance) + 100,color = "k",label = "True")
    plt.plot(np.arange(1,len(d_importance)+1),d_importance,color = "b",label = "Importance sampling, "+r"$\Delta$ = "+ f"{round(np.abs(d_importance[-1] - Z_0) / Z_0,5)}",linewidth = line_width)
    plt.plot(np.arange(1,len(d_bridge)+1),d_bridge,color = "g",label = "Bridge sampling, "+r"$\Delta$ = "+ f"{round(np.abs(d_bridge[-1] - Z_0) / Z_0,5)}",linewidth = line_width)
    plt.legend(prop = font)
    plt.xscale("log")
    plt.xticks(fontsize = fs,fontname = fn)
    plt.yticks(fontsize = fs,fontname = fn)
    plt.ylabel("Z",fontsize = fs,fontname = fn)
    plt.xlabel("n",fontsize = fs,fontname = fn)
    plt.tight_layout()

plt.savefig("TailBehaviour.jpg")
plt.close()