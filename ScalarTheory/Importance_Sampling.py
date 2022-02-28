from utils import S,gaussian_distribution
import torch
import numpy as np
import tqdm
import argparse
from utils import gaussian_distribution
from models import fc_subnet, LinearCondition5x5_one_condition
from Training_INN import Trainer_INN
from Training_EBM import Trainer_Maximum_Likelihood_Discrete
import os
from datetime import datetime

############################################################################################################################################################
#log of the unnormalized distribution defined by the true action function
############################################################################################################################################################
def log_p_True_action(X,lambdas,kappas):
    return - S(mus = X,lambdas=lambdas,kappas=kappas)

############################################################################################################################################################
#Sub net for the INNs
############################################################################################################################################################
subnet_dict = {
    "fc_subnet":fc_subnet
    }

############################################################################################################################################################
#Model for the EBM
############################################################################################################################################################
model_dict = {
    "LinearCondition5x5_one_condition":LinearCondition5x5_one_condition
    }

my_parser = argparse.ArgumentParser()

my_parser.add_argument("--mode_proposal",                   type=str,     action = "store",   default = "INN",                                       required=False      ,help = "Model used to normalize the unnormalized distribution defined by the action function. 'INN' or 'NORMAL")
my_parser.add_argument("--mode_unnormalized_model",         type=str,     action = "store",   default = "TRUE",                                      required=False      ,help = "Action function that is used to define the unnormalized distribution. 'TRUE' or 'EBM'")
my_parser.add_argument("--kappa_min",                       type=float,   action = "store",   default = 0.2,                                         required=False      ,help = "Smallest hopping parameter that is evaluated.")
my_parser.add_argument("--kappa_max",                       type=float,   action = "store",   default = 0.4,                                         required=False      ,help = "Biggest hopping parameter that is evaluated.")
my_parser.add_argument("--dkappa",                          type=float,   action = "store",   default = 0.02,                                        required=False      ,help = "Step size of the hopping parameter")
my_parser.add_argument("--lmbd",                            type=float,   action = "store",   default = 0.02,                                        required=False      ,help = "Quartic coupling")
my_parser.add_argument("--SubNetFunctionINN",               type=str,     action = "store",   default = "fc_subnet",                                 required=False      ,help = "Constructor of the subnetworks in the INNs")
my_parser.add_argument("--ModelClassEBM",                   type=str,     action = "store",   default = "LinearCondition5x5_one_condition",          required=False      ,help = "Class to define the action function.")
my_parser.add_argument("--load_path_proposal",              type=str,     action = "store",                                                          required=True       ,help = "Folder containing the subfolders with the normalized models.")
my_parser.add_argument("--load_path_EBM",                   type=str,     action = "store",   default = "",                                          required=False      ,help = "Folder containing the training outcomes of the EBM.")
my_parser.add_argument("--n_epoch_INN",                     type=int,     action = "store",   default = 0,                                           required=False      ,help = "Epoch of the INN")
my_parser.add_argument("--n_epoch_EBM",                     type=int,     action = "store",   default = 0,                                           required=False      ,help = "Epoch of the EBM")
my_parser.add_argument("--prefix_proposal",                 type=str,     action = "store",   default = "",                                          required=False      ,help = "Prefix of the folders containing the normalized models")
my_parser.add_argument("--n_reps",                          type=int,     action = "store",   default = 5,                                           required=False      ,help = "Number of repetitions")
my_parser.add_argument("--magnitude",                       type=int,     action = "store",   default = 5,                                           required=False      ,help = "Specifies the number of samples ued for Bridge sampling. Biggest batch size is given by 9 * 10 ^(magnitude - 1)")
my_parser.add_argument("--random_seed",                     type=int,     action = "store",   default = 47,                                          required=False      ,help = "Random seed")

args = my_parser.parse_args()

############################################################################################################################################################
#Get the hopping parameters
############################################################################################################################################################
kappas = np.arange(args.kappa_min,args.kappa_max,args.dkappa)

############################################################################################################################################################
#Set the random seed
############################################################################################################################################################
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

############################################################################################################################################################
#Output the mode of the calculation
############################################################################################################################################################
if args.mode_proposal == "INN":
    print("Normalized model:\tINN distribution")

elif args.mode_proposal == "NORMAL":
    print("Normalized model:\tNormal distribution")

if args.mode_unnormalized_model == "EBM":
    print("Unormalized model:\tEBM distribution")

elif args.mode_unnormalized_model == "TRUE":
        print("Unormalized model:\tTRUE distribution")

############################################################################################################################################################
#Get the number of samples that is evaluated
############################################################################################################################################################
mult = torch.arange(1,10)
numbers_of_samples = torch.zeros(len(mult )* args.magnitude)

for e in range(args.magnitude):
    n_samples = mult * 10**e
    numbers_of_samples[e * len(mult):(e+1) * len(mult)] = n_samples

############################################################################################################################################################
#Importance sampling for each hopping parameter
############################################################################################################################################################
for i in range(len(kappas)):
    k = round(kappas[i],4)

    print(f"\nkappa = {k}")

    ############################################################################################################################################################
    #load the normalized model
    ############################################################################################################################################################
    if args.mode_proposal == "INN":
        p_normalized = Trainer_INN(
                mode = "SIMULATE",
                device = "cpu",
                SubNetFunctionINN = subnet_dict[args.SubNetFunctionINN],
                load_path = args.load_path_proposal + f"INNs/{args.prefix_proposal}kappa_{k}_lambda_{args.lmbd}/",
                n_epoch_state_dict = args.n_epoch_INN
            )
    
    elif args.mode_proposal == "NORMAL":
        p_normalized = gaussian_distribution(path = args.load_path_proposal + f"{args.prefix_proposal}kappa_{k}_lambda_{args.lmbd}/")
    
    else:
        raise NotImplementedError()
    
    ############################################################################################################################################################
    #load the unnormalized model
    ############################################################################################################################################################
    if args.mode_unnormalized_model == "TRUE":
        log_p_action = log_p_True_action

    elif args.mode_unnormalized_model == "EBM":
        EBM = Trainer_Maximum_Likelihood_Discrete(
            mode =                  "SIMULATE",
            ModelClass =            model_dict[args.ModelClassEBM],
            load_path =             args.load_path_EBM,
            device =                "cpu",
            n_epoch_state_dict =    args.n_epoch_EBM,
            random_seed =           args.random_seed
        )

        log_p_action = EBM.log_p_EBM
    
    else:
        raise NotImplementedError()

    ############################################################################################################################################################
    #Importance sampling to estimate the partition function
    ############################################################################################################################################################
    storage = np.zeros([(len(numbers_of_samples)),args.n_reps])

    for j in tqdm.tqdm(range(args.n_reps)):

        #Get samples from the proposal distribution
        states_temp_proposal = p_normalized.sample(n_samples = int(numbers_of_samples[-1]))

        for h in range(len(numbers_of_samples)):

            #Get the hopping parameter and the quadric couplni for each of the samples
            lambda_action = torch.ones(int(numbers_of_samples[h])) * args.lmbd
            kappa_action = torch.ones(int(numbers_of_samples[h])) * k

            with torch.no_grad():

                #Get the batch form the proposal distribution
                mus_prop = states_temp_proposal[:int(numbers_of_samples[h])]

                #Get the ratio of the unnormalized model and th e normalized model
                r = torch.exp((log_p_action(X = mus_prop,lambdas = lambda_action,kappas = kappa_action) - p_normalized.log_p(X = mus_prop)).type(torch.double)).detach()
                print(min(r).item(),max(r).item(),torch.mean(r).item(),torch.median(r))

                #Store the average value
                storage[h][j] = r.mean() 

    ############################################################################################################################################################
    #save the recorded data
    ############################################################################################################################################################
    if not os.path.exists(args.load_path_proposal + "eval_partition_function/"):
        os.makedirs(args.load_path_proposal + f"eval_partition_function/")

    #Write the header of the data file:
    with open(args.load_path_proposal + f"eval_partition_function/{datetime.date(datetime.now())}_MC_data_kappa-{k}_lambda-{args.lmbd}_{args.mode_proposal}_{args.mode_unnormalized_model}_epochINN-{args.n_epoch_INN}_epochEBM-{args.n_epoch_EBM}.txt","w") as file:
        file.write("Monte Carlo Integration\n")
        file.write(f"Proposal distribution:  {args.mode_proposal} distribution\t location: {args.load_path_proposal}\n")
        file.write(f"Unnormalized distribution:  {args.mode_unnormalized_model}\t location: {args.load_path_EBM}\n")
        file.write(f"Epoch Proposal:  {args.n_epoch_INN}\n")
        file.write(f"Epoch unnormalized distribution:  {args.n_epoch_EBM}\n")
        file.write(f"Number of repetitions:  {args.n_reps}\n\n")

        q = "n_samples"
        for i in range(args.n_reps):
            q += f"\trun {i+1}"
        q += "\n"
        file.write(q)

    file.close()

    #Save the data
    with open(args.load_path_proposal + f"eval_partition_function/{datetime.date(datetime.now())}_MC_data_kappa-{k}_lambda-{args.lmbd}_{args.mode_proposal}_{args.mode_unnormalized_model}_epochINN-{args.n_epoch_INN}_epochEBM-{args.n_epoch_EBM}.txt","a") as file:
        data = np.zeros([len(numbers_of_samples),args.n_reps + 1])
        data[:,0] = numbers_of_samples.detach().cpu().numpy()
        data[:,1:] = storage

        np.savetxt(file,data)
    file.close()