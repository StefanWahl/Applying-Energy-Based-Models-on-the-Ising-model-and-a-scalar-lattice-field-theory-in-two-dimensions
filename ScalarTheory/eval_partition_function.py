import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import matplotlib.font_manager as font_manager

my_parser = argparse.ArgumentParser()

my_parser.add_argument("--data_path",                   type=str,     action = "store",                       required= True                                     ,help = "Folder containing the recorded partition functions.")
my_parser.add_argument("--mode_unnormalized",           type=str,     action = "store",                       required= True                                     ,help = "EBM or TRUE")
my_parser.add_argument("--date_data_files",             type=str,     action = "store",                       required= True                                     ,help = "Time stamp of the recorded partition functions")
my_parser.add_argument("--epoch_EBM",                   type=int,     action = "store",                       required= False, default = 30                      ,help = "Epoch of the EBM that is evaluated\tdefault: 30")
my_parser.add_argument("--epoch_INN",                   type=int,     action = "store",                       required= False, default = 45                      ,help = "Epoch of the INN that is evaluated\tdefault: 45")
my_parser.add_argument("--kappa_min",                   type=float,   action = "store",                       required= False, default = 0.2                     ,help = "Smallest hopping parameter\tdefault: 0.2")
my_parser.add_argument("--kappa_max",                   type=float,   action = "store",                       required= False, default = 0.39                    ,help = "Biggest hopping parameter\tdefault: 0.39")
my_parser.add_argument("--dkappa",                      type=float,   action = "store",                       required= False, default = 0.02                    ,help = "Step size of the hopping parameter\tdefault: 0.02")
my_parser.add_argument("--lmbd",                        type=float,   action = "store",                       required= False, default = 0.02                    ,help = "Quartic coupling\tdefault: 0.02")

args = my_parser.parse_args()

########################################################################################################################################################################################################################################
#Set parameters for plotting
########################################################################################################################################################################################################################################
fs = 50
fontname = "Times New Roman"
marker_size = 10
line_width = 3
markeredgewidth = line_width
capsize = 10
font = font_manager.FontProperties(family=fontname,
                                style='normal', 
                                size=fs)

########################################################################################################################################################################################################################################
#Set the epoch of the EBM
########################################################################################################################################################################################################################################
if args.mode_unnormalized == "TRUE":
    epoch_EBM = 0

else:
    epoch_EBM = args.epoch_EBM

#Create folder for the plots
if not os.path.exists(args.data_path + "Images/"):
    os.makedirs(args.data_path + "Images/")

########################################################################################################################################################################################################################################
#Compare the convergence of the Montacarlo simulation and the Bridge sampling
########################################################################################################################################################################################################################################
if not os.path.exists(args.data_path + "Images/ComparisonConvergence/"):
    os.makedirs(args.data_path + "Images/ComparisonConvergence/")

kappas = np.arange(args.kappa_min,args.kappa_max,args.dkappa)

########################################################################################################################################################################################################################################
#Storage for the partition functions obtained from the biggest batch
########################################################################################################################################################################################################################################
final_Z_NORMAL_MC       = np.zeros([len(kappas),2])
final_Z_INN_MC          = np.zeros([len(kappas),2])
final_Z_NORMAL_BRIDGE   = np.zeros([len(kappas),2])
final_Z_INN_BRIDGE      = np.zeros([len(kappas),2])

########################################################################################################################################################################################################################################
#Storage for the relative deviation between the partition function and the partition function for the biggest batch
########################################################################################################################################################################################################################################
rel_dev_means_final_mean_INN_MC         = np.zeros([len(kappas),45])
rel_dev_means_final_mean_INN_BRIDGE     = np.zeros([len(kappas),45])
rel_dev_means_final_mean_NORMAL_MC      = np.zeros([len(kappas),45])
rel_dev_means_final_mean_NORMAL_BRIDGE  = np.zeros([len(kappas),45])

########################################################################################################################################################################################################################################
#Plot the results for each hopping parameter
########################################################################################################################################################################################################################################
for i in range(len(kappas)):

    kappa_action = round(kappas[i],4)

    plt.figure(figsize = (30,20))

    ########################################################################################################################################################################################################################################
    #Normal distribution
    ########################################################################################################################################################################################################################################

    plt.subplot(2,1,1)

    #Load the recorded partition functions
    data_MC     = np.loadtxt(args.data_path + f"{args.date_data_files}_MC_data_kappa-{kappa_action}_lambda-{args.lmbd}_NORMAL_{args.mode_unnormalized}_epochINN-0_epochEBM-{epoch_EBM}.txt",skiprows = 8)
    data_BRIDGE = np.loadtxt(args.data_path + f"{args.date_data_files}_BRIDGE_data_kappa-{kappa_action}_lambda-{args.lmbd}_NORMAL_{args.mode_unnormalized}_epochINN-0_epochEBM-{epoch_EBM}.txt",skiprows = 8)

    #Get the number of samples used to compute the partition functions
    n_samples_MC        = data_MC[:,0]
    n_samples_BRIDGE    = data_BRIDGE[:,0]

    #Get the average partition function over all repetitions
    mean_MC         = data_MC[:,1:].mean(-1)
    mean_BRIDGE     = data_BRIDGE[:,1:].mean(-1)

    #Get the standard deviation of the partition functions 
    std_MC      = data_MC[:,1:].std(-1)
    std_BRIDGE  = data_BRIDGE[:,1:].std(-1)

    #Plot the evolution of the partition function as a function of the batch size
    plt.title("Normal distribution",fontname = fontname,fontsize = fs)
    plt.errorbar(n_samples_MC,np.log(mean_MC),yerr = std_MC / mean_MC,label = "Importance Sampling",marker = ".",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
    plt.errorbar(n_samples_BRIDGE,np.log(mean_BRIDGE),yerr = std_BRIDGE / mean_BRIDGE,label = "Bridge sampling",marker = "*",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
    plt.xlabel("n",fontsize = fs,fontname = fontname)
    plt.ylabel("log Z",fontsize = fs,fontname = fontname)
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontname = fontname,fontsize = fs)
    plt.tight_layout()
    plt.xscale("log")
    plt.legend(prop = font)

    #Store the results from the biggest batch
    final_Z_NORMAL_BRIDGE[i] = np.array([mean_BRIDGE[-1],std_BRIDGE[-1]])
    final_Z_NORMAL_MC[i] = np.array([mean_MC[-1],std_MC[-1]])

    #Get the deviation of the means from the final mean
    rel_dev_means_final_mean_NORMAL_BRIDGE[i] = np.abs((mean_BRIDGE - mean_BRIDGE[-1]) / mean_BRIDGE[-1])
    rel_dev_means_final_mean_NORMAL_MC[i] = np.abs((mean_MC - mean_MC[-1]) / mean_MC[-1])

    ########################################################################################################################################################################################################################################
    #INN distribution
    ########################################################################################################################################################################################################################################

    plt.subplot(2,1,2)

    #Load the recorded partition functions
    data_MC     = np.loadtxt(args.data_path + f"{args.date_data_files}_MC_data_kappa-{kappa_action}_lambda-{args.lmbd}_INN_{args.mode_unnormalized}_epochINN-{args.epoch_INN}_epochEBM-{epoch_EBM}.txt",skiprows = 8)
    data_BRIDGE = np.loadtxt(args.data_path + f"{args.date_data_files}_BRIDGE_data_kappa-{kappa_action}_lambda-{args.lmbd}_INN_{args.mode_unnormalized}_epochINN-{args.epoch_INN}_epochEBM-{epoch_EBM}.txt",skiprows = 8)

    #Get the number of samples used to compute the partition functions
    n_samples_MC        = data_MC[:,0]
    n_samples_BRIDGE    = data_BRIDGE[:,0]

    #Get the average partition function over all repetitions
    mean_MC         = data_MC[:,1:].mean(-1)
    mean_BRIDGE     = data_BRIDGE[:,1:].mean(-1)

    #Get the standard deviation of the partition functions 
    std_MC      = data_MC[:,1:].std(-1)
    std_BRIDGE  = data_BRIDGE[:,1:].std(-1)

    plt.title("INN",fontname = fontname,fontsize = fs)
    plt.errorbar(n_samples_MC,np.log(mean_MC),yerr = std_MC / mean_MC,label = "Importance Sampling",marker = ".",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
    plt.errorbar(n_samples_BRIDGE,np.log(mean_BRIDGE),yerr = std_BRIDGE / mean_BRIDGE,label = "Bridge sampling",marker = "*",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
    plt.xlabel("n",fontsize = fs,fontname = fontname)
    plt.ylabel("log Z",fontsize = fs,fontname = fontname)
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontname = fontname,fontsize = fs)
    plt.tight_layout()
    plt.xscale("log")
    plt.legend(prop = font)

    #Store the results from the biggest batch
    final_Z_INN_BRIDGE[i] = np.array([mean_BRIDGE[-1],std_BRIDGE[-1]])
    final_Z_INN_MC[i] = np.array([mean_MC[-1],std_MC[-1]])

    #Get the deviation of the means from the final mean
    rel_dev_means_final_mean_INN_BRIDGE[i] = np.abs((mean_BRIDGE - mean_BRIDGE[-1]) / mean_BRIDGE[-1])
    rel_dev_means_final_mean_INN_MC[i] = np.abs((mean_MC - mean_MC[-1]) / mean_MC[-1])

    #Save the plot
    plt.savefig(args.data_path + f"Images/ComparisonConvergence/kappa_{kappa_action}_lambda_{args.lmbd}_convergence.jpg")
    plt.close()


########################################################################################################################################################################################################################################
#Plot the final partition function as a function of the hopping parameter
########################################################################################################################################################################################################################################
plt.figure(figsize = (30,24))

#Normal distributions
plt.subplot(2,1,1)
plt.title("Normal distribution",fontname = fontname,fontsize = fs)
plt.errorbar(kappas,np.log(final_Z_NORMAL_MC[:,0]),yerr = final_Z_NORMAL_MC[:,1] / final_Z_NORMAL_MC[:,0],label = "Importance sampling",marker = "s",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
plt.errorbar(kappas,np.log(final_Z_NORMAL_BRIDGE[:,0]),yerr = final_Z_NORMAL_BRIDGE[:,1] / final_Z_NORMAL_BRIDGE[:,0],label = "Bridge sampling",marker = "*",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
plt.ylabel("log Z",fontsize = fs,fontname = fontname)
plt.xticks(fontsize = fs,fontname = fontname)
plt.yticks(fontname = fontname,fontsize = fs)
plt.tight_layout()
plt.legend(prop = font)

print("deviation for Normal distribution: dZ = ",(np.abs((final_Z_NORMAL_MC[:,0] - final_Z_NORMAL_BRIDGE[:,0]) / np.sqrt(final_Z_NORMAL_MC[:,1]**2 + final_Z_NORMAL_BRIDGE[:,1]**2))).mean())

#INN distributions
plt.subplot(2,1,2)
plt.title("INN",fontname = fontname,fontsize = fs)
plt.errorbar(kappas,np.log(final_Z_INN_MC[:,0]),yerr = final_Z_INN_MC[:,1] / final_Z_INN_MC[:,0],label = "Importance sampling",marker = "s",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
plt.errorbar(kappas,np.log(final_Z_INN_BRIDGE[:,0]),yerr = final_Z_INN_BRIDGE[:,1] / final_Z_INN_BRIDGE[:,0],label = "Bridge sampling",marker = "*",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
plt.ylabel("log Z",fontsize = fs,fontname = fontname)
plt.xticks(fontsize = fs,fontname = fontname)
plt.yticks(fontname = fontname,fontsize = fs)
plt.tight_layout()
plt.legend(prop = font)

print("deviation for INN: dZ = ",(np.abs((final_Z_INN_MC[:,0] - final_Z_INN_BRIDGE[:,0]) / np.sqrt(final_Z_INN_MC[:,1]**2 + final_Z_INN_BRIDGE[:,1]**2))).mean())

plt.savefig(args.data_path + f"Images/partition_function_n_samples_{n_samples_BRIDGE[-1]}.jpg")
plt.close()

########################################################################################################################################################################################################################################
#Plot the deviation of the means from the final mean
########################################################################################################################################################################################################################################
plt.figure(figsize = (30,24))

#Normal distributions
plt.subplot(2,1,1)
plt.title("Normal distribution",fontname = fontname,fontsize = fs)
plt.errorbar(n_samples_MC,rel_dev_means_final_mean_NORMAL_MC.mean(axis = 0),yerr = rel_dev_means_final_mean_NORMAL_MC.std(axis = 0),label = "Importance Sampling",marker = "s",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
plt.errorbar(n_samples_BRIDGE,rel_dev_means_final_mean_NORMAL_BRIDGE.mean(axis = 0),yerr = rel_dev_means_final_mean_NORMAL_BRIDGE.std(axis = 0),label = "Bridge Sampling",marker = "*",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
plt.xlabel("n",fontsize = fs,fontname = fontname)
plt.ylabel("rel. deviation",fontsize = fs,fontname = fontname)
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize = fs,fontname = fontname)
plt.yticks(fontname = fontname,fontsize = fs)
plt.tight_layout()
plt.legend(prop = font)

#INN distributions
plt.subplot(2,1,2)
plt.title("INN distribution",fontname = fontname,fontsize = fs)
plt.errorbar(n_samples_MC,rel_dev_means_final_mean_INN_MC.mean(axis = 0),yerr = rel_dev_means_final_mean_INN_MC.std(axis = 0),label = "Importance Sampling",marker = "s",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
plt.errorbar(n_samples_BRIDGE,rel_dev_means_final_mean_INN_BRIDGE.mean(axis = 0),rel_dev_means_final_mean_INN_BRIDGE.std(axis = 0),label = "Bridge Sampling",marker = "*",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width)
plt.xlabel("n",fontsize = fs,fontname = fontname)
plt.ylabel("rel. deviation",fontsize = fs,fontname = fontname)
plt.xscale("log")
plt.yscale("log")
plt.xticks(fontsize = fs,fontname = fontname)
plt.yticks(fontname = fontname,fontsize = fs)
plt.tight_layout()
plt.legend(prop = font)

plt.savefig(args.data_path + f"Images/Deviations_means_from_final_mean_{n_samples_BRIDGE[-1]}.jpg")
plt.close()

########################################################################################################################################################################################################################################
#Write a file containing the average partition functions for the biggest number of samples and the error
########################################################################################################################################################################################################################################

with open(args.data_path +f"summary_partition_functions_{int(n_samples_BRIDGE[-1])}.txt","w") as file:
    file.write("kappa\tINN_MC_mean\tINN_MC_std\tINN_Bridge_mean\tINN_Bridge_std\tNORMAL_MC_mean\tNORMAL_MC_std\tNORMAL_Bridge_mean\tNORMAL_Bridge_std\n")
file.close()

data = np.zeros([len(kappas),9])

data[:,0] = kappas

data[:,1] = final_Z_INN_MC[:,0]
data[:,2] = final_Z_INN_MC[:,1]

data[:,3] = final_Z_INN_BRIDGE[:,0]
data[:,4] = final_Z_INN_BRIDGE[:,1]

data[:,5] = final_Z_NORMAL_MC[:,0]
data[:,6] = final_Z_NORMAL_MC[:,1]

data[:,7] = final_Z_NORMAL_BRIDGE[:,0]
data[:,8] = final_Z_NORMAL_BRIDGE[:,1]

with open(args.data_path +f"summary_partition_functions_{int(n_samples_BRIDGE[-1])}.txt","a") as file:
    np.savetxt(file,data)
file.close()