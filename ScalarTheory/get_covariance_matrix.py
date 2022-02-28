'''
Fit a multivariate normal distribution to a data set and evaluate the fitted normal distributions
'''

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_states,gaussian_distribution,get_susceptibility,get_U_L,bootstrap
import tqdm
import os
import matplotlib.font_manager as font_manager

my_parser = argparse.ArgumentParser()

my_parser.add_argument("--N",                    type=int,      action = "store"                        ,required=True          ,help="Number of spins per row and column")
my_parser.add_argument("--kappa_min",            type=float,    action = "store",   default = "0.2"     ,required=False         ,help="Smallest hopping parameter")
my_parser.add_argument("--kappa_max",            type=float,    action = "store",   default = "0.39"    ,required=False         ,help="Biggest hopping parameter")
my_parser.add_argument("--dkappa",               type=float,    action = "store",   default = "0.02"    ,required=False         ,help="Step size of the hopping parameter")
my_parser.add_argument("--lambda_min",           type=float,    action = "store",   default = "0.02"    ,required=False         ,help="Smallest quartic coupling")
my_parser.add_argument("--lambda_max",           type=float,    action = "store",   default = "0.03"    ,required=False         ,help="Biggest quartic coupling")
my_parser.add_argument("--dlambda",              type=float,    action = "store",   default = "0.02"    ,required=False         ,help="Stepsize of the quartic coupling")
my_parser.add_argument("--path",                 type=str,      action = "store"                        ,required=True          ,help="Path containing the individual simulations")
my_parser.add_argument("--prefix",               type=str,      action = "store",   default = ""        ,required=False         ,help="Prefix added to path")
my_parser.add_argument("--n_samples_eval",       type=int,      action = "store",   default = 100000    ,required=False         ,help="Number of samples used to evaluate the fitted gaussian distributions")
args = my_parser.parse_args()

################################################################################################################################################
#get the quartic coupling and the hopping parameters
################################################################################################################################################
kappas = np.arange(args.kappa_min,args.kappa_max,args.dkappa)
lambdas = np.arange(args.lambda_min,args.lambda_max,args.dlambda)

N = args.N
prog_bar = tqdm.tqdm(total=len(kappas) * len(lambdas))

################################################################################################################################################
#Fit a multivariate normal distribution for each combination of quartic coupling and hopping parameter
################################################################################################################################################
for i in range(len(kappas)):
    for j in range(len(lambdas)):

        l = round(lambdas[j],4)
        k = round(kappas[i],4)

        #Initialize the normal distribution
        path = args.path + f"{args.prefix}kappa_{k}_lambda_{l}/"
        gaussian = gaussian_distribution(path = path)
    
        #Fit the normal distribution to the training set
        gaussian.fit(path = path,N = N)

        #plot some states following the normal distribution
        n_samples = 25
        s = gaussian.sample(n_samples=n_samples)
        plot_states(s,5,5,path + "gaussian_fit/visualization_covariance_fit.jpg",N = N)

        prog_bar.update(1)

################################################################################################################################################
#evaluate the gaussian distributions
################################################################################################################################################
prog_bar = tqdm.tqdm(total=len(kappas) * len(lambdas))

#Write the header of teh summary files:
for j in range(len(lambdas)):
    l = round(lambdas[j],4)

    if not os.path.exists(args.path + "summary_gaussian_fit/"):
        os.mkdir(args.path + "summary_gaussian_fit/")

    with open(args.path + f"summary_gaussian_fit/{args.prefix}summary_gausian_fit_lambda_{l}.txt","w") as file:
        file.write(f"n_samples_eval = {args.n_samples_eval}\n")
        file.write("kappa\t<m>\tsigma_m\t<U_L>\tsigma_U_L\t<chi>\tsigma_chi\n")
    file.close()

#evaluate
for i in range(len(kappas)):
    for j in range(len(lambdas)):

        l = round(lambdas[j],4)
        k = round(kappas[i],4)

        path = args.path + f"{args.prefix}kappa_{k}_lambda_{l}/"

        gaussian = gaussian_distribution(path = path)

        #Get samples 
        states = gaussian.sample(n_samples = args.n_samples_eval)

        #compute the observables
        ms = states.sum((1,2,3))

        m,sigma_m = bootstrap(ms.abs().detach() / N**2,torch.mean,args = {})
        U,sigma_U = bootstrap(ms.detach(),get_U_L,args = {"Omega":N**2})
        chi,sigma_chi = bootstrap(ms.detach(),get_susceptibility,args = {"Omega":N**2})

        #save the results
        with open(args.path + f"summary_gaussian_fit/{args.prefix}summary_gausian_fit_lambda_{l}.txt","a") as file:
            file.write(f"{k}\t{m}\t{sigma_m}\t{U}\t{sigma_U}\t{chi}\t{sigma_chi}\n")
        file.close()

        prog_bar.update(1)

################################################################################################################################################
#Plot the results
################################################################################################################################################
fs = 40
fontname = "Times New Roman"
marker_size = 10
line_width = 3
markeredgewidth = line_width
capsize = 10

font = font_manager.FontProperties(family=fontname,
                                style='normal', 
                                size=fs)

for j in range(len(lambdas)):
    l = round(lambdas[j],4)

    data = np.loadtxt(args.path + f"summary_gaussian_fit/{args.prefix}summary_gausian_fit_lambda_{l}.txt",skiprows = 2)
    ref = np.loadtxt(args.path + args.prefix + f"summary_lambda_{l}_0.txt",skiprows = 1)

    #Absolute mean magnetization per spin
    plt.figure(figsize = (30,15))
    plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
    plt.ylabel(r"m",fontsize = fs,fontname = fontname)
    plt.errorbar(kappas,data[:,1],yerr = data[:,2],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = "INN",ls = "",color = "k",marker = ".")
    plt.fill_between(x = ref[:,0],y1 = np.abs(ref[:,1]) - ref[:,2],y2 = np.abs(ref[:,1]) + ref[:,2],color = "orange",label = r"$1\sigma$"+" interval")
    plt.plot(ref[:,0],np.abs(ref[:,1]),ls = ":",linewidth = line_width,color = "k",label = "Simulation")
    plt.legend(prop = font)
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)
    plt.savefig(args.path + f"summary_gaussian_fit/{args.prefix}summary_gausian_fit_lambda_{l}_magnetization.jpg")
    plt.close()

    #Binder cumulant
    plt.figure(figsize = (30,15))
    plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
    plt.ylabel(r"U",fontsize = fs,fontname = fontname)
    plt.errorbar(kappas,data[:,3],yerr = data[:,4],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = "INN",ls = "",color = "k",marker = ".")
    plt.fill_between(x = ref[:,0],y1 = ref[:,5] - ref[:,6],y2 = ref[:,5] + ref[:,6],color = "orange",label = r"$1\sigma$"+" interval")
    plt.plot(ref[:,0],ref[:,5],ls = ":",linewidth = line_width,color = "k",label = "Simulation")
    plt.legend(prop = font)
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)
    plt.savefig(args.path + f"summary_gaussian_fit/{args.prefix}summary_gausian_fit_lambda_{l}_binder_cumulant.jpg")
    plt.close()

    #Susceptibility
    plt.figure(figsize = (30,15))
    plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
    plt.ylabel(r"\chi^2",fontsize = fs,fontname = fontname)
    plt.errorbar(kappas,data[:,5],yerr = data[:,6],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = "INN",ls = "",color = "k",marker = ".")
    plt.fill_between(x = ref[:,0],y1 = ref[:,7] - ref[:,8],y2 = ref[:,7] + ref[:,8],color = "orange",label = r"$1\sigma$"+" interval")
    plt.plot(ref[:,0],ref[:,7],ls = ":",linewidth = line_width,color = "k",label = "Simulation")
    plt.legend(prop = font)
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)
    plt.savefig(args.path + f"summary_gaussian_fit/{args.prefix}summary_gausian_fit_lambda_{l}_scusceptibility.jpg")
    plt.close()