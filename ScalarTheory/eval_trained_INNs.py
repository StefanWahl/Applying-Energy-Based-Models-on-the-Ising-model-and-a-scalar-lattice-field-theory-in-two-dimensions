import numpy as np
import matplotlib.pyplot as plt
from Training_INN import Trainer_INN
import os
import json
from models import fc_subnet
import matplotlib.font_manager as font_manager
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument("--Reference_path",              type=str,       action = "store",                                                           required=True       ,help = "File containing the reference data.")
my_parser.add_argument("--INN_path",                    type=str,       action = "store",                                                           required=True       ,help = "Folder containing the individdual INNs.")
my_parser.add_argument("--kappa_min",                   type=float,     action = "store",   default = 0.2,                                          required=False      ,help = "Smallest hopping parameter used in the training set.")
my_parser.add_argument("--kappa_max",                   type=float,     action = "store",   default = 0.39,                                         required=False      ,help = "Biggest hopping parameter used in the training set.")
my_parser.add_argument("--dkappa",                      type=float,     action = "store",   default = 0.02,                                         required=False      ,help = "Step size of the hopping parameter")
my_parser.add_argument("--lambd",                       type=float,     action = "store",   default = 0.02,                                         required=False      ,help = "Quartic coupling")
my_parser.add_argument("--n_epoch_state_dict",          type=int,       action = "store",   default = 45,                                           required=False      ,help = "Training epoch of the INNs")
my_parser.add_argument("--n_samples",                   type=int,       action = "store",   default = 100000,                                       required=False      ,help = "Number of samples used for the evaluation")
my_parser.add_argument("--name_training",               type=str,       action = "store",                                                           required=True       ,help = "Prefix of the folders containing the INNs")

args = my_parser.parse_args()

################################################################################################################################################
#Settings for plotting
################################################################################################################################################

marker_size = 10
line_width = 3
markeredgewidth = line_width
capsize = 10
fs = 40
fontname = "Times New Roman"

font = font_manager.FontProperties(
            family=fontname,
            style='normal', 
            size=fs)

################################################################################################################################################
#Constructor for the subnetworks if the INNs
################################################################################################################################################
subnet_dict = {
    "fc_subnet":fc_subnet
}

################################################################################################################################################
#Evaluate the unconditional INNs
################################################################################################################################################

#Get the hopping parameters and the quartic coupling that is evaluated
kappas = np.arange(args.kappa_min,args.kappa_max,args.dkappa)
lambdas = np.ones(len(kappas)) * args.lambd

#Storage
magnetization = np.zeros([len(kappas),2])
binder_cumulant = np.zeros([len(kappas),2])
chis = np.zeros([len(kappas),2])

#Load the reference file
reference_file = np.loadtxt(args.Reference_path,skiprows = 1)

#Create folder
if not os.path.exists(args.INN_path + "Images/"):
    os.mkdir(args.INN_path + "Images/")

#Write the header of the summary file
with open(args.INN_path + f"Images/summary_INN_fit_lambda_{args.lambd}.txt","w") as file:
    file.write(f"n_samples_eval = {args.n_samples}\n")
    file.write("kappa\t<m>\tsigma_m\t<U_L>\tsigma_U_L\t<chi>\tsigma_chi\n")
file.close()

#Loop over all hopping parameters
for i in range(len(kappas)):

    k = round(kappas[i],4)

    #Initialize the INN
    with open(args.INN_path + args.name_training + f"kappa_{k}_lambda_{args.lambd}/Code/config.json","r") as file:
        config = json.load(file)
    file.close()

    INN = Trainer_INN(
        mode = "SIMULATE",
        device = "cpu",
        SubNetFunctionINN = subnet_dict[config["SubNetFunctionINN"].split(" ")[1]],
        load_path = args.INN_path + args.name_training + f"kappa_{k}_lambda_{args.lambd}/",
        n_epoch_state_dict = args.n_epoch_state_dict
        )

    #Evaluate samples following the INN distribution
    d = INN.simulation(n_samples = args.n_samples)

    #store the results
    magnetization[i][0] = d["m"]
    magnetization[i][1] = d["sigma_m"]

    binder_cumulant[i][0] = d["U"]
    binder_cumulant[i][1] = d["sigma_U"]

    chis[i][0] = d["chi"]
    chis[i][1] = d["sigma_chi"]

    #Store the results
    with open(args.INN_path + f"Images/summary_INN_fit_lambda_{args.lambd }.txt","a") as file:
        file.write(f"{k}\t{d['m']}\t{d['sigma_m']}\t{d['U']}\t{d['sigma_U']}\t{d['chi']}\t{d['sigma_chi']}\n")
    file.close()

if not os.path.exists(args.INN_path + f"Images/"):
    os.mkdir(args.INN_path + f"Images/")

################################################################################################################################################
#Plot the results
################################################################################################################################################

#Absolute mean magnetization per spin
plt.figure(figsize = (30,15))
plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
plt.ylabel(r"m",fontsize = fs,fontname = fontname)
plt.errorbar(kappas,magnetization[:,0],yerr = magnetization[:,1],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = "INN",ls = "",color = "k",marker = ".")
plt.fill_between(x = reference_file[:,0],y1 = np.abs(reference_file[:,1]) - reference_file[:,2],y2 = np.abs(reference_file[:,1]) + reference_file[:,2],color = "orange",label = r"$1\sigma$"+" interval")
plt.plot(reference_file[:,0],np.abs(reference_file[:,1]),ls = ":",linewidth = line_width,color = "k",label = "Simulation")
plt.legend(prop = font)
plt.xticks(fontsize = fs,fontname = fontname)
plt.yticks(fontsize = fs,fontname = fontname)
plt.savefig(args.INN_path + f"Images/{args.name_training}_magnetization_INN_epoch_{args.n_epoch_state_dict}.jpg")
plt.close()

#Binder cumulant
plt.figure(figsize = (30,15))
plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
plt.ylabel(r"U",fontsize = fs,fontname = fontname)
plt.errorbar(kappas,binder_cumulant[:,0],yerr = binder_cumulant[:,1],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = "INN",ls = "",color = "k",marker = ".")
plt.fill_between(x = reference_file[:,0],y1 = reference_file[:,5] - reference_file[:,6],y2 = reference_file[:,5] + reference_file[:,6],color = "orange",label = r"$1\sigma$"+" interval")
plt.plot(reference_file[:,0],reference_file[:,5],ls = ":",linewidth = line_width,color = "k",label = "Simulation")
plt.legend(prop = font)
plt.xticks(fontsize = fs,fontname = fontname)
plt.yticks(fontsize = fs,fontname = fontname)
plt.savefig(args.INN_path + f"Images/{args.name_training}_binder_cumulant_INN_epoch_{args.n_epoch_state_dict}.jpg")
plt.close()

#Susceptibility
plt.figure(figsize = (30,15))
plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
plt.ylabel(r"\chi^2",fontsize = fs,fontname = fontname)
plt.errorbar(kappas,chis[:,0],yerr = chis[:,1],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = "INN",ls = "",color = "k",marker = ".")
plt.fill_between(x = reference_file[:,0],y1 = reference_file[:,7] - reference_file[:,8],y2 = reference_file[:,7] + reference_file[:,8],color = "orange",label = r"$1\sigma$"+" interval")
plt.plot(reference_file[:,0],reference_file[:,7],ls = ":",linewidth = line_width,color = "k",label = "Simulation")
plt.legend(prop = font)
plt.xticks(fontsize = fs,fontname = fontname)
plt.yticks(fontsize = fs,fontname = fontname)
plt.savefig(args.INN_path + f"Images/{args.name_training}_chis_INN_epoch_{args.n_epoch_state_dict}.jpg")
plt.close()