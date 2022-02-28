import torch
import numpy as np
import json
import tqdm
import os
import argparse
from utils import get_U_L,get_susceptibility,bootstrap,plotter_results

my_parser = argparse.ArgumentParser()

my_parser.add_argument("--kappa_min",                       type=float,   action = "store",   default = 0.2,                                         required=False      ,help = "Smallest hopping parameter that is evaluated.")
my_parser.add_argument("--kappa_max",                       type=float,   action = "store",   default = 0.39,                                        required=False      ,help = "Biggest hopping parameter that is evaluated.")
my_parser.add_argument("--dkappa",                          type=float,   action = "store",   default = 0.02,                                        required=False      ,help = "Step size of the hopping parameter")
my_parser.add_argument("--lmbd",                            type=float,   action = "store",   default = 0.02,                                        required=False      ,help = "Quartic coupling")
my_parser.add_argument("--path",                            type=str,     action = "store",                                                          required=True       ,help = "Folder containing the folder for the different hopping parameters")
my_parser.add_argument("--prefix_data_folders",             type=str,     action = "store",   default = "",                                          required=False      ,help = "Prefix of the subfolders")
my_parser.add_argument("--len_data_set",                    type=int,     action = "store",   default = 75000,                                       required=False      ,help = "Size of the data set")
my_parser.add_argument("--n_reps",                          type=int,     action = "store",   default = 10,                                          required=False      ,help = "Number of individual simulations")
my_parser.add_argument("--random_seed",                     type=int,     action = "store",   default = 47,                                          required=False      ,help = "Random Seed")
my_parser.add_argument("--n_taus",                          type=int,     action = "store",   default = 2,                                           required=False      ,help = "Number of correlation times between samples")

args = my_parser.parse_args()

################################################################################################################################################
#Set the random seed
################################################################################################################################################
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

################################################################################################################################################
#Get the hopping parameters
################################################################################################################################################
kappas =np.arange(args.kappa_min,args.kappa_max,args.dkappa)

################################################################################################################################################
#Storage for plotting
################################################################################################################################################
action_plot = np.zeros([len(kappas),2])
magnetization_plot = np.zeros([len(kappas),2])
U_L_plot = np.zeros([len(kappas),2])
chi_plot = np.zeros([len(kappas),2])

################################################################################################################################################
#Create a training set and a validation for each hopping parameter
################################################################################################################################################
for i in tqdm.tqdm(range(len(kappas))):

    k = round(kappas[i],4)

    storage = torch.zeros(0)

    #store the magnetization and the actions to compute the observables
    magnetizations_storage = torch.zeros(0)
    actions_storage = torch.zeros(0)

    #go through all recorded data files in the current folder
    for j in range(args.n_reps):

        #Load the data
        states = torch.load(args.path + f"{args.prefix_data_folders}kappa_{k}_lambda_{args.lmbd}/states_{j}.pt")

        #Get information about the simulation
        with open(args.path + f"{args.prefix_data_folders}kappa_{k}_lambda_{args.lmbd}/info_{j}.json","r") as file:
            info = json.load(file)
        file.close()

        N = info["N"]

        #Get the part of the data that is in equilibrium and use "independent" samples
        lower_lim = int(info["t_eq"] / info["freq_save_samples"])
        step_size = int(args.n_taus * info["tau_action"] / info["freq_save_samples"]) + 1
        states = states[lower_lim::step_size]

        #Get the magnetization and the action
        magn = torch.load(args.path + f"{args.prefix_data_folders}kappa_{k}_lambda_{args.lmbd}/Magnetization_{j}.pt")[int(lower_lim * info["freq_save_samples"])::int(step_size*info["freq_save_samples"])]
        act = torch.load(args.path + f"{args.prefix_data_folders}kappa_{k}_lambda_{args.lmbd}/Actions_{j}.pt")[int(lower_lim * info["freq_save_samples"])::int(step_size*info["freq_save_samples"])]

        #Add the states to the set
        storage = torch.cat((storage,states),dim = 0)
        actions_storage = torch.cat((actions_storage,act),dim = 0)
        magnetizations_storage = torch.cat((magnetizations_storage,magn),dim = 0)

    #Get shuffeled indices 
    indices_shuffle = np.random.permutation(len(storage))

    #shuffle the storeage
    storage = storage[indices_shuffle]
    magnetizations_storage = magnetizations_storage[indices_shuffle]
    actions_storage = actions_storage[indices_shuffle]

    #get a training set and a validation set
    training_set = storage[:args.len_data_set]
    validation_set = storage[args.len_data_set:]

    #Get the action and the magnetization of the validation set for the evaluation
    actions_validation = actions_storage[args.len_data_set:].detach().numpy()
    magnetizations_validation = magnetizations_storage[args.len_data_set:].detach().numpy()

    #Save the training set and the validation set
    torch.save(training_set,args.path + f"{args.prefix_data_folders}kappa_{k}_lambda_{args.lmbd}/training_set.pt")
    torch.save(validation_set,args.path + f"{args.prefix_data_folders}kappa_{k}_lambda_{args.lmbd}/validation_set.pt")

    #Output the lenght of the two sets
    print(f"training set: k = {k}\t {len(training_set)} states\t validation set: {len(validation_set)} states\n")

    #compute the observables based oin the validation set
    U_L_mean,sigma_U_L = bootstrap(x = magnetizations_validation,s = get_U_L,args={"Omega":N**2})
    m,sigma_m = bootstrap(x = np.abs(magnetizations_validation) / N**2,s = np.mean,args={"axis":0})
    a,sigma_a = bootstrap(x = actions_validation / N**2,s = np.mean,args={"axis":0})
    chi,sigma_chi = bootstrap(x = magnetizations_validation,s = get_susceptibility,args={"Omega":N**2})

    #Save the data for plotting 
    action_plot[i][0] = a
    action_plot[i][1] = sigma_a

    magnetization_plot[i][0] = m
    magnetization_plot[i][1] = sigma_m

    U_L_plot[i][0] = U_L_mean
    U_L_plot[i][1] = sigma_U_L

    chi_plot[i][0] = chi
    chi_plot[i][1] = sigma_chi

    #Store information about the data sets
    with open(args.path + f"{args.prefix_data_folders}kappa_{k}_lambda_{args.lmbd}/info_training_set.json","w") as file:
        info_training_set = {
            "len_training_set":len(training_set),
            "len_validation_set":len(validation_set),
            "n_taus":args.n_taus,
            "<U_L>":float(U_L_mean),
            "sigma_U_L":float(sigma_U_L),
            "<a>":float(a),
            "sigma_a":float(sigma_a),
            "<m>":float(m),
            "sigma_m":float(sigma_m),
            "<chi>":float(chi),
            "sigma_chi":float(sigma_chi),
        }

        json.dump(info_training_set,file)
    file.close()

############################################################################################################################################
#Plot the observables obtained from the validation set
############################################################################################################################################

#Get the reference file from the simulation with the true action function
reference_file = np.loadtxt(f"./Scalar_Theory/N_{N}_LANGEVIN_SPECIFIC/summary_lambda_{args.lmbd}_0.txt",skiprows = 1)

#create a folder for the results
if not os.path.exists(args.path + f"summary_training_set_lambda_{args.lmbd}/"):
    os.makedirs(args.path + f"summary_training_set_lambda_{args.lmbd}/")

labels = ["simulation","summary"]

#plot
plotter_results(
    n_reps = 0,
    target_path = args.path + f"summary_training_set_lambda_{args.lmbd}/",
    U_Ls = [reference_file[:,5:7].T,U_L_plot.T],
    labels_U_Ls = labels,
    ms = [reference_file[:,1:3].T,magnetization_plot.T],
    action = [reference_file[:,3:5].T,action_plot.T],
    chi_squares=[reference_file[:,7:9].T,chi_plot.T],
    kappas = [reference_file[:,0],kappas],
    labels_action=labels,
    labels_chi_squares= labels,
    labels_ms=  labels,
    labels_taus= [["",""],["",""]],
    taus = [[np.zeros(10),np.zeros(10)],[np.zeros(10),np.zeros(10)]],
    kappas_taus = [[np.zeros(10),np.zeros(10)],[np.zeros(10),np.zeros(10)]],
    title = "",
    reference = True,
    fontname = "Times New Roman"
)

############################################################################################################################################
#Save a summary file
############################################################################################################################################

with open(args.path + f"summary_training_set_lambda_{args.lmbd}/summary_lambda_{args.lmbd}_0.txt","w") as file:
    file.write("\tkappa	magnetization	std_magnetization	action	std_action	U_L	std_U_L	chi^2	std_chi^2\n")
file.close()


data = np.zeros([len(kappas),9])

data[:,0] = kappas

data[:,1] = magnetization_plot[:,0]
data[:,2] = magnetization_plot[:,1]

data[:,3] = action_plot[:,0]
data[:,4] = action_plot[:,1]

data[:,5] = U_L_plot[:,0]
data[:,6] = U_L_plot[:,1]

data[:,7] = chi_plot[:,0]
data[:,8] = chi_plot[:,1]

with open(args.path + f"summary_training_set_lambda_{args.lmbd}/summary_lambda_{args.lmbd}_0.txt","a") as file:
    np.savetxt(file,data)
file.close()