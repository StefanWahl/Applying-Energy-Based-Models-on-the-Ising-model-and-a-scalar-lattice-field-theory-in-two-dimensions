import torch
from toy_example import Data,Model
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from train_inn import INN
import matplotlib.font_manager as font_manager
import os

if not os.path.exists("./Bridge_Sampling_toy/Plots/"):
    os.makedirs("./Bridge_Sampling_toy/Plots/")

###############################################################################################################################################################
#Iterative approximation of the ratio between the two partition functions
###############################################################################################################################################################
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

ys = []
p_is = []
p_ts = []
p_targets = []
p_inns = []

modes = ["gmm","rings","cb"]        #Data modes forr which the evaluation is performed
x_conds = [0,0,0.5]                 #X values used in the conditionals
n_state_dicct_INN = 49999           #Training iteration of the used INN

###############################################################################################################################################################
#Settings for plotting
###############################################################################################################################################################
fs = 50
fn = "Times New Roman"
font = font_manager.FontProperties(family=fn,
                                style='normal', 
                                size=fs)

###############################################################################################################################################################
#Defined the EBM density
###############################################################################################################################################################
#density for the EBM
def p_EBM(x,E,Z = 1):
    return torch.exp(-E(x)-np.log(Z)).squeeze()

###############################################################################################################################################################
#Sample states following the EBM distribution via Langevin dynamic
###############################################################################################################################################################
def sample_EBM(N,E,epsilon,K = 250):
    #initial samples
    x = torch.randn([N,2]).requires_grad_(True)

    #update the samples with langevin dynamics
    for j in tqdm.tqdm(range(K)):
        g = torch.autograd.grad(E(x).sum(),[x])[0]
        x.data += - epsilon * g + np.sqrt(epsilon * 2) * torch.randn_like(x)

    return x.detach()

###############################################################################################################################################################
#Loop over the data modes and compute the partition functions for the corresponding EBM using Bridge Sampling
###############################################################################################################################################################
for i in range(len(modes)):
    n = 6000                #Number of samples per distribution
    mode = modes[i]         #Mode of the data set
    x_cond = x_conds[i]     #X value of used in the conditionals

    #Settings for the different data modes
    if mode == "gmm":
        n_modes = 6
        epsilon = 0.005

    elif mode == "cb":
        n_modes = 2
        epsilon = 0.0001

    elif mode == "rings":
        n_modes = 3
        epsilon = 0.005

    if not os.path.exists(f"./Bridge_Sampling_toy/Plots/{mode}/"):
        os.makedirs(f"./Bridge_Sampling_toy/Plots/{mode}/")


    plot_r = True           #Plot the approximation of the ratio between the partition functions as a function of the number of iterations in the iterative computation
    plot_samples = True     #Plot samples following the EBM distribution, the target distribution and the INN distribution
    path = "./Bridge_Sampling_toy/Plots/"+mode+f"/state_dict_inn_{n_state_dicct_INN}_"      #Path to store the results

    #initialize the the EBM model
    E = Model()
    E.load_state_dict(torch.load(f"./Learning_EBM_toy/{mode}/state_dict_i_49999.pt"))

    #initialize the the INN model
    I = INN(mode = mode,n_modes=n_modes,n_layers=10)
    I.load(path = f"./Bridge_Sampling_toy/INNs/{mode}/state_dicts/checkpoint_{n_state_dicct_INN}.pt")

    #Target distributions
    q = Data(mode = mode,n_modes = n_modes, res = 500)

    #Get samples following the involved distributions
    x_EBM = sample_EBM(N = n,E = E,epsilon=0.005)
    x_target= q.sample(n)
    x_INN = I.sample(n)

    #Get the factors needed for Bridge Sampling for the target distribution
    l_1_t = p_EBM(x_EBM,E = E).detach().numpy() / q.density(x_EBM.detach().numpy())
    l_2_t = p_EBM(x_target,E = E).detach().numpy() / q.density(x_target.detach().numpy())

    #Get the factors needed for Bridge Sampling for the INN distribution
    l_1_i = p_EBM(x_EBM,E = E).detach().numpy() / I.density(x_EBM.requires_grad_(True)).detach().numpy()
    l_2_i = p_EBM(x_INN,E = E).detach().numpy() / I.density(x_INN.requires_grad_(True)).detach().numpy()

    #Iterative computation of the ratio of the partition function
    r_seq_t = iterative_approximation_r(n_iter = 100,r_0 = 1,l_1 = l_1_t,l_2 = l_2_t)
    r_seq_i = iterative_approximation_r(n_iter = 100,r_0 = 1,l_1 = l_1_i,l_2 = l_2_i)

    #partition function of the EBM is given by the value obtained after the final iteration of the itertive computation
    Z_inn = r_seq_i[-1]
    Z_target = r_seq_t[-1]

    print(f"Z_inn = {Z_inn}")
    print(f"Z_target =  {Z_target}")

    #Plot the samples following the used distributions
    if plot_samples == True:
        plt.figure(figsize = (45,15))
        plt.subplot(1,3,1)
        plt.plot(x_INN.detach().numpy()[:,0],x_INN.detach().numpy()[:,1],ls = "",marker = ".",color = "k")
        plt.title("INN",fontsize = 40,fontname = fn)
        plt.xlabel("x",fontsize = fs,fontname = fn)
        plt.ylabel("y",fontsize = fs,fontname = fn)
        plt.xticks(fontsize = fs,fontname = fn)
        plt.yticks(fontsize = fs,fontname = fn)

        plt.subplot(1,3,2)
        plt.plot(x_EBM.detach().numpy()[:,0],x_EBM.detach().numpy()[:,1],ls = "",marker = ".",color = "k")
        plt.title("EBM",fontsize = 40,fontname = fn)
        plt.xlabel("x",fontsize = fs,fontname = fn)
        plt.ylabel("y",fontsize = fs,fontname = fn)
        plt.xticks(fontsize = fs,fontname = fn)
        plt.yticks(fontsize = fs,fontname = fn)

        plt.subplot(1,3,3)
        plt.plot(x_target.detach().numpy()[:,0],x_target.detach().numpy()[:,1],ls = "",marker = ".",color = "k")
        plt.title("Target",fontsize = 40,fontname = fn)
        plt.xlabel("x",fontsize = fs,fontname = fn)
        plt.ylabel("y",fontsize = fs,fontname = fn)
        plt.xticks(fontsize = fs,fontname = fn)
        plt.yticks(fontsize = fs,fontname = fn)
        plt.savefig(path + "samples.jpg")
        plt.close()

    #Plot the approximation of the ratio between the partition functions as a function of the number of iterations in the iterative computation
    if plot_r == True:
        plt.figure(figsize = (30,10))
        plt.xlabel("iteration",fontsize = fs,fontname = fn)
        plt.ylabel("Z",fontsize = fs,fontname = fn)
        plt.xticks(fontsize = fs,fontname = fn)
        plt.yticks(fontsize = fs,fontname = fn)
        plt.plot(np.arange(len(r_seq_i)),r_seq_i,label = "INN dist.",color = "b",linewidth = 4)
        plt.plot(np.arange(len(r_seq_t)),r_seq_t,label = "target dist.",color = "y",linewidth = 4)
        plt.legend(prop = font)
        plt.tight_layout()
        plt.savefig(path + "score_functions_secant.jpg")
        plt.close()

    #plot conditionals
    #Construct input for the EBM
    x = torch.ones(1000).view(-1,1) * x_cond
    y = torch.linspace(-q.x_lim,q.x_lim,1000).view(-1,1)
    points = torch.cat((x,y),dim = 1)

    #Get the density approximation from the different models
    learned_dist_inn = p_EBM(points,E,Z = Z_inn)
    learned_dist_target = p_EBM(points,E,Z = Z_target)
    target_dist = q.density(points.detach().numpy())

    #Plot the target distribution and the two normalized EBm distributions for the selected x valua as a function of y
    plt.figure(figsize=(30,30))
    plt.subplot(2,1,1)
    plt.title("normalized EBM density",fontsize = fs,fontname = fn)
    plt.plot(y.detach().squeeze().numpy(),learned_dist_inn.detach().numpy(),label = "norm. w. target",color = "b",linewidth = 4)
    plt.plot(y.detach().squeeze().numpy(),learned_dist_target.detach().numpy(),label = "norm. w. INN",color = "y",linewidth = 4)
    plt.plot(y.detach().squeeze().numpy(),target_dist,label = "Target",color = "k",ls = ":",linewidth = 4)
    plt.xticks(fontsize = fs,fontname = fn)
    plt.yticks(fontsize = fs,fontname = fn)
    plt.xlabel("y",fontsize = fs,fontname = fn)
    plt.ylabel(f"p(y|x = {x_cond})",fontsize = fs,fontname = fn)
    plt.legend(prop = font)

    #Plot the INN distribution for the selected x valua as a function of y
    plt.subplot(2,1,2)
    plt.plot(y.detach().squeeze().numpy(),target_dist,label = "Target",color = "k",ls = ":",linewidth = 4)
    plt.title("INN density",fontsize = fs,fontname = fn)
    p_INN = I.density(points.requires_grad_(True)).detach().numpy()
    plt.plot(y.detach().squeeze().numpy(),p_INN,label = "INN",color = "b",linewidth = 4)
    plt.xticks(fontsize = fs,fontname = fn)
    plt.yticks(fontsize = fs,fontname = fn)
    plt.legend(prop = font)
    plt.xlabel("y",fontsize = fs,fontname = fn)
    plt.ylabel(f"p(y|x = {x_cond})",fontsize = fs,fontname = fn)
    plt.savefig(path + "normalized densities.jpg")
    plt.close()

    ys.append(y.detach().squeeze().numpy())
    p_is.append(learned_dist_inn.detach().numpy())
    p_ts.append(learned_dist_target.detach().numpy())
    p_targets.append(target_dist)
    p_inns.append(p_INN)

###############################################################################################################################################################
#Summary plot containing the result of all evaluated data modes
###############################################################################################################################################################
if len(modes) > 0:
    plt.figure(figsize = (30,30))

    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.xticks(fontsize = fs,fontname = fn)
        plt.yticks(fontsize = fs,fontname = fn)
        plt.xlabel("y",fontsize = fs,fontname = fn)
        plt.ylabel(f"p(y|x = {x_conds[i]})",fontsize = fs,fontname = fn)
        plt.title(modes[i],fontsize = fs,fontname = fn)

        plt.plot(ys[i],p_inns[i],label = "INN density",linewidth = 6,color = "b",ls = ":")
        plt.plot(ys[i],p_targets[i],label = "target density",linewidth =6,color = "k",ls = ":")
        plt.plot(ys[i],p_is[i],label = "norm. w. INN",linewidth = 6,color = "r")
        plt.plot(ys[i],p_ts[i],label = "norm. w. target",linewidth = 6,color = "y")
        plt.tight_layout()

        plt.legend(prop = font)

    plt.savefig(f"./Bridge_Sampling_toy/Plots/summary_conditionals_bridge_inn_state_dict_{n_state_dicct_INN}.jpg")
    plt.close()

###############################################################################################################################################################
#evaluate the behaviou of the approximation for different n
###############################################################################################################################################################

#Get the sample sizes
n_min = 3       #Smalles tnumber os samples per distribution
n_max = 90000   #Bmalles tnumber os samples per distribution
dn = 2000       #Stepsize to increase the number os samples per distribution
ns = np.arange(n_min,n_max,dn)

#Traiing iteration of the INN
n_state_dicct_INN = 49999

#record the partition functios for different sample sizes
Z_recorded_i = np.zeros([len(ns)])
Z_recorded_t = np.zeros([len(ns)])

#initialize the the EBM 
E = Model()
E.load_state_dict(torch.load(f"./Learning_EBM_toy/gmm/state_dict_i_49999.pt"))

#initialize the the INN 
I = INN(mode = "gmm",n_modes=6,n_layers=10)
I.load(path = f"./Bridge_Sampling_toy/INNs/gmm/state_dicts/checkpoint_{n_state_dicct_INN}.pt")

#Initialize the target distributions
q = Data(mode = "gmm",n_modes = 6, res = 500)

#Compute the partition function for all sample sizes
for i in range(len(ns)):

    #Get the current number of samples
    n = ns[i]

    #Get samples following the different distributions
    x_EBM = sample_EBM(N = n,E = E,epsilon=0.005)
    x_target= q.sample(n)
    x_INN = I.sample(n)

    print(len(x_EBM),len(x_INN),len(x_target))

    #Get parameters for the score functions
    #Target distribution
    l_1_t = p_EBM(x_EBM,E = E).detach().numpy() / q.density(x_EBM.detach().numpy())
    l_2_t = p_EBM(x_target,E = E).detach().numpy() / q.density(x_target.detach().numpy())

    #INN distribution
    l_1_i = p_EBM(x_EBM,E = E).detach().numpy() / I.density(x_EBM.requires_grad_(True)).detach().numpy()
    l_2_i = p_EBM(x_INN,E = E).detach().numpy() / I.density(x_INN.requires_grad_(True)).detach().numpy()

    r_seq_t = iterative_approximation_r(n_iter = 100,r_0 = 1,l_1 = l_1_t,l_2 = l_2_t)
    r_seq_i = iterative_approximation_r(n_iter = 100,r_0 = 1,l_1 = l_1_i,l_2 = l_2_i)

    Z_inn = r_seq_i[-1]
    Z_target = r_seq_t[-1]

    Z_recorded_i[i] = Z_inn
    Z_recorded_t[i] = Z_target

    #Save the data
    with open(f"./Bridge_Sampling_toy/Plots/Normalization_const_as_function_of_samples.txt","a") as file:
        if i == 0:
            file.write(f"n_per_dist\tZ_norm_w_INN\tZ_norm_w_target\n")

        file.write(f"{ns[i]}\t{Z_recorded_i[i]}\t{Z_recorded_t[i]}\n")
    file.close()

#Plot the results
plt.figure(figsize=(30,12))
plt.plot(ns,Z_recorded_i,label = "norm. w. INN",linewidth =6)
plt.plot(ns,Z_recorded_t,label = "norm. w. target",linewidth =6)
plt.legend(prop = font)
plt.xlabel("samples per distribution",fontsize = fs, fontname = fn)
plt.ylabel("Z",fontsize = fs, fontname = fn)
plt.title("gmm",fontsize = fs, fontname = fn)
plt.xticks(fontsize = fs,fontname = fn)
plt.yticks(fontsize = fs,fontname = fn)

plt.savefig("./Bridge_Sampling_toy/Plots/Normalization_const_as_function_of_samples.jpg")
plt.close() 