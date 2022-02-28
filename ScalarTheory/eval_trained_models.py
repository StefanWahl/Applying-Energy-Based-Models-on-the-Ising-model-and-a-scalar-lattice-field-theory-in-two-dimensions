from models import LinearCondition5x5_one_condition
from Training_EBM import Trainer_Maximum_Likelihood_Discrete
import numpy as np
from utils import get_tau_eq,quick_eval_2D,plotter_results
import torch
import json
import argparse
import os

###################################################################################################################
#run simulations for a trained model at a certein temperature
###################################################################################################################
def run_simulation_Maximum_Likelihood_Discrete(random_seed,n_reps,record_freq,record_states,kappas_action,lambdas_action,n_iteration,load_path,date,n_epoch_state_dict,ModelClass,device,sampler_mode):
    '''
    Run a simulation using a trained energy model.

    parameters:
        random_seed:           Random seed
        n_reps:                Index of the simulation
        record_freq:           Frequency of storing the states during the MCMC
        record_states:         Record the states that occure during the MCMC
        kappas_action:         List containing the hopping parameters for which a simulation is performed.
        lambdas_action:        List containing the quartic couplings for which a simulation is performed.
        n_iteration:           Number of update steps performed in the simulations.
        load_path:             Parent folder of the training that is evaluated.
        date:                  Time stamp of the simulation.
        n_epoch_state_dict     Training epoch for which the action function is evaluated.
        ModelClass:            Class used to model the action function.
        device:                Device on which the simulation runs.
        sampler_mode:          Sampler used to generate new states.

    returns:
        None
    '''

    #Initialize the trained model
    Trainer = Trainer_Maximum_Likelihood_Discrete(
        mode =                  "SIMULATE",
        ModelClass =            ModelClass,
        load_path =             load_path,
        device =                device,
        n_epoch_state_dict =    n_epoch_state_dict,
        sampler_mode =          sampler_mode,
        record_freq =           record_freq,
        record_states =         record_states,
        random_seed=random_seed
        )

    #Perform the simulation.
    Trainer.simulation(
        n_iterations = n_iteration,
        date = date,
        kappas_action = torch.Tensor(kappas_action),
        lambdas_action=torch.Tensor(lambdas_action),
        n_reps = n_reps
        )

def eval(N,n_reps,random_seed,record_states,record_freq,sampler_mode,kappa_min,kappa_max,dkappa,lambda_min,lambda_max,dlambda,ModelClass,n_epochs,load_path,n_iter,date_simulation,device,l,dt_max,fs = 40,fontname = "Times New Roman"):
    '''
    Evaluate trained energy functions.

    parameters:
        N:                  Number of spins per tow and column.
        n_reps:             Index of the simulation
        random_seed:        Random seed
        record_states:      Record the states that occure during the MCMC
        record_freq:        Frequency of storing the states during the MCMC
        sampler_mode:       Sampler used to generate new states.
        kappa_min:          Samllest hopping parameter that is simulated.
        kappa_max:          Biggest hopping parameter that is simulated.
        dkappa:             Step size of the hopping parameter.
        lambda_min:         Samllest quartic coupling that is simulated.
        lambda_max:         Biggest quartic coupling that is simulated.
        dlambda:            Step size of the quartic coupling.
        ModelClass:         Class used to model the action function.
        n_epochs:           Epochs for which the training is evaluated.
        load_path:          Parent folder of the training that is evaluated.
        n_iter:             Number of update steps performed in the simulations.    
        date_simulation:    Time stamp of the simulation.
        device:             Device on which the simulation runs.
        l:                  Step size of the time lag of the computation of th eintegrated correlation time.
        dt_max:             Biggest time lag evaluated in teh coputation of the integrated correlation time.
        fs:                 Font size used for the plots.
        fontname:           Font style used for the plots.

    returns:
        None

    '''
    #Get the hopping parameters and the quadric couplings for which a simulation is performed
    kappas_action = np.arange(kappa_min,kappa_max,dkappa)
    lambdas_action = np.arange(lambda_min,lambda_max,dlambda)

    ###################################################################################################################
    ##Perform the simulations
    ###################################################################################################################

    for i in range(len(n_reps)):

        n_reps_temp = int(n_reps[i])
        seed_temp = int(random_seed[i])

        #Loop over all epochs that are evaluated
        for e in n_epochs:
            selected_kappas = []
            selected_lambdas = []

            for kappa_action in kappas_action:
                for lambda_action in lambdas_action:
                    kappa = round(kappa_action,7)
                    lambd = round(lambda_action,7)

                    #Check if the simulation has already be done
                    p = load_path + f"Simulations/"+date_simulation+ f"/epoch_{e}_kappa_{kappa}_lambda_{lambd}/"
                    if os.path.exists(p):
                        if os.path.exists(p+f"Actions_{n_reps_temp}.pt") and os.path.exists(p+f"Magnetization_{n_reps_temp}.pt") and os.path.exists(p+f"info_{n_reps_temp}.json"):
                            continue
                        else:
                            selected_kappas.append(kappa)
                            selected_lambdas.append(lambd)

                    else:
                        selected_kappas.append(kappa)
                        selected_lambdas.append(lambd)
            print(selected_lambdas,selected_kappas)
            #Perform the simulations
            if len(selected_kappas) > 0:
                run_simulation_Maximum_Likelihood_Discrete(n_reps=n_reps_temp,random_seed = seed_temp,kappas_action = selected_kappas,record_states = record_states,record_freq = record_freq,lambdas_action = selected_lambdas,n_iteration = n_iter,load_path = load_path,date = date_simulation,n_epoch_state_dict = e,ModelClass = ModelClass,device = device,sampler_mode = sampler_mode)

    ###################################################################################################################
    #Approximate the equillibrium time
    ###################################################################################################################

    for i in range(len(n_reps)):

        n_reps_temp = int(n_reps[i])

        for e in n_epochs:
            for kappa_action in kappas_action:
                for lambda_action in lambdas_action:
                    kappa = round(kappa_action,7)
                    lambd = round(lambda_action,7)

                    path = load_path + f"Simulations/"+date_simulation+ f"/epoch_{e}_kappa_{kappa}_lambda_{lambd}/"

                    #only determine the equillibrium time if it has not been determined yet.
                    with open(path + f"info_{n_reps_temp}.json","r") as file:
                        info = json.load(file)
                    file.close()

                    if "t_eq" in info.keys():
                        continue
                    else:
                        get_tau_eq(path = path,fs = fs,n_reps=n_reps_temp)
    ###################################################################################################################
    #Evaluate the simulations
    ###################################################################################################################

    for i in range(len(n_reps)):
        
        n_reps_temp = int(n_reps[i])


        for e in n_epochs:
            for kappa_action in kappas_action:
                for lambda_action in lambdas_action:
                    kappa = round(kappa_action,7)
                    lambd = round(lambda_action,7)

                    path = load_path + f"Simulations/"+date_simulation+ f"/epoch_{e}_kappa_{kappa}_lambda_{lambd}/"

                    with open(path + f"info_{n_reps_temp}.json","r") as file:
                        info = json.load(file)
                    file.close()

                    if ("tau_magnetization" in info.keys()) and ("tau_action" in info.keys()) and ("<S>" in info.keys()) and ("<M>" in info.keys()) and ("<U_L>" in info.keys()) and ("<chi^2>" in info.keys()):
                        continue

                    else:
                        magnetization = torch.load(path + f"Magnetization_{n_reps_temp}.pt")
                        energy = torch.load(path + f"Actions_{n_reps_temp}.pt")
                        quick_eval_2D(path = path,magnetizations=magnetization,actions=energy,l = l,dt_max = dt_max,fs = fs,n_reps = n_reps_temp)

        ################################################
        #Plot
        ################################################
        for lambda_action in lambdas_action:
            lamb = round(lambda_action,7)

            #Get a reference plot
            reference_path = f"./Scalar_Theory/N_{N}_LANGEVIN_SPECIFIC/summary_lambda_{lamb}_{0}.txt"
            data_reference = np.loadtxt(reference_path,skiprows = 1)

            ms = np.zeros([len(n_epochs),2,len(kappas_action)])
            actions = np.zeros([len(n_epochs),2,len(kappas_action)])
            ULs = np.zeros([len(n_epochs),2,len(kappas_action)])
            Chis = np.zeros([len(n_epochs),2,len(kappas_action)])
            tau_action = np.zeros([len(n_epochs),len(kappas_action)])
            tau_magnetization = np.zeros([len(n_epochs),len(kappas_action)])

            #plot it
            list_taus = [[],[]]
            list_kappas = []
            list__kappas_tau = [[],[]]
            list_actions = []
            list_ms = []
            list_ULs = []
            list_Chis = []

            list_labels_ms = []
            list_labels_taus = [[],[]] 

            list_taus[0].append(data_reference[:,9])
            list_taus[1].append(data_reference[:,10])
            list_kappas.append(data_reference[:,0])
            list__kappas_tau[0].append(data_reference[:,0])
            list__kappas_tau[1].append(data_reference[:,0])
            list_actions.append([data_reference[:,3],data_reference[:,4]])
            list_ms.append([np.abs(data_reference[:,1]),data_reference[:,2]])
            list_ULs.append([data_reference[:,5],data_reference[:,6]])
            list_Chis.append([data_reference[:,7],data_reference[:,8]])

            list_labels_ms = ["True action"]
            list_labels_taus = [[r"$\tau_{a}$"+f" true action"],[r"$\tau_{m}$"+f" true action"]] 



            #Get the data from the simulations
            for j in range(len(n_epochs)):

                #Summary file
                with open(load_path + f"Simulations/"+date_simulation+ f"/epoch_{n_epochs[j]}_summary_lambda_{lambd}_{n_reps_temp}.txt","w") as file:
                    file.write("kappa\tmagnetization\tstd_magnetization\taction\tstd_action\tU_L\tstd_U_L\tchi^2\tstd_chi^2\ttau_action\ttau_magnetization\n")
                file.close()

                for i in range(len(kappas_action)):
                    path = load_path + f"Simulations/"+date_simulation+ f"/epoch_{n_epochs[j]}_kappa_{round(kappas_action[i],7)}_lambda_{lambd}/"
                    with open(path + f"info_{n_reps_temp}.json","r") as file:
                        info = json.load(file)
                    file.close()

                    actions[j][0][i] = info["<S>"]
                    actions[j][1][i] = info["sigma_S"]

                    ms[j][0][i] = info["<M>"]
                    ms[j][1][i] = info["sigma_M"]

                    ULs[j][0][i] = info["<U_L>"]
                    ULs[j][1][i] = info["sigma_U_L"]

                    Chis[j][0][i] = info["<chi^2>"]
                    Chis[j][1][i] = info["sigma_chi^2"]


                    tau_action[j][i] = info["tau_action"]
                    tau_magnetization[j][i] = info["tau_magnetization"]

                    with open(load_path + f"Simulations/"+date_simulation+ f"/epoch_{n_epochs[j]}_summary_lambda_{lambd}_{n_reps_temp}.txt","a") as file:
                        file.write(f"{kappas_action[i]}\t{info['<M>']}\t{info['sigma_M']}\t{info['<S>']}\t{info['sigma_S']}\t{info['<U_L>']}\t{info['sigma_U_L']}\t{info['<chi^2>']}\t{info['sigma_chi^2']}\t{info['tau_action']}\t{info['tau_magnetization']}\n")
                    file.close()

                list_ms.append(ms[j])
                list_actions.append(actions[j])
                list_kappas.append(kappas_action)
                list_taus[0].append(tau_action[j])
                list_taus[1].append(tau_magnetization[j])
                list__kappas_tau[0].append(kappas_action)
                list__kappas_tau[1].append(kappas_action)
                list_ULs.append(ULs[j])
                list_Chis.append(Chis[j])

                list_labels_ms.append(f"Trained action function e = {n_epochs[j]}")
                list_labels_taus[0].append(r"$\tau_{a}$"+f" trained action function e = {n_epochs[j]}")
                list_labels_taus[1].append(r"$\tau_{m}$"+f" trained action function e = {n_epochs[j]}")

                plotter_results(
                    U_Ls = list_ULs,
                    labels_U_Ls = list_labels_ms,
                    target_path = load_path+f"Simulations/"+date_simulation+f"/lambda_{lambd}_",
                    chi_squares = list_Chis,
                    ms = list_ms,
                    action = list_actions,
                    taus = list_taus,
                    kappas = list_kappas,
                    kappas_taus = list__kappas_tau,
                    labels_chi_squares = list_labels_ms,
                    labels_ms = list_labels_ms,
                    labels_action = list_labels_ms,
                    labels_taus = list_labels_taus,
                    fontname = "Times New Roman",
                    n_correlation_times = 2,
                    reference = True,
                    fs = 40,
                    title = "",
                    n_reps= n_reps_temp
                    )

if __name__ == "__main__":

    #Get the parameters of the evaluation
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument("--kappa_min",            type = float,   action = "store",   default = 0.2,   	                            required=False      ,help = "Smallest evaluated hopping parameter \t default: 0.2")
    my_parser.add_argument("--kappa_max",            type = float,   action = "store",   default = 0.4,   	                            required=False      ,help = "Biggest evaluated hopping parameter \t default: 0.4")
    my_parser.add_argument("--dkappa",               type = float,   action = "store",   default = 0.01,   	                            required=False      ,help = "Step size of the hopping parameter \t default: 0.01")
    my_parser.add_argument("--lambda_min",           type = float,   action = "store",   default = 0.02,   	                            required=False      ,help = "Smallest evaluated quartic coupling \t default: 0.02")
    my_parser.add_argument("--lambda_max",           type = float,   action = "store",   default = 0.03,   	                            required=False      ,help = "Biggest evaluated quartic coupling \t default: 0.03")
    my_parser.add_argument("--dlambda",              type = float,   action = "store",   default = 0.02,   	                            required=False      ,help = "Step size of the quartic coupling \t default: 0.02")
    my_parser.add_argument("--load_path",            type = str,     action = "store", 	                                                required=True       ,help = "Parent folder of the training that is evaluated")
    my_parser.add_argument("--n_epochs",             type = str,     action = "store",   default = "30",                                required=False      ,help = "Epochs for which the training is evaluated \t default: '30'")
    my_parser.add_argument("--n_iter",               type = int,     action = "store",   default = 500000,                              required=False      ,help = "Number of update steps performed in the simulations \t default: 100000")
    my_parser.add_argument("--date_simulation",      type = str,     action = "store",                                                  required=True       ,help = "Time stamp of teh simulation")
    my_parser.add_argument("--ModelClass",           type = str,     action = "store",   default="LinearCondition5x5_one_condition",    required=False      ,help = "Class used to model the action function \t default: LinearCondition5x5_one_condition")
    my_parser.add_argument("--N",                    type = int,     action = "store",   default = 5,                                   required=False      ,help = "Number of spins per row and column \t default: 5")
    my_parser.add_argument("--l",                    type = int,     action = "store",   default = 10,                                  required=False      ,help = "Step size of the time lag of the computation of th eintegrated correlation time \t default: 10")
    my_parser.add_argument("--dt_max",               type = int,     action = "store",   default = 5000,                                required=False      ,help = "Biggest time lag evaluated in teh coputation of the integrated correlation time \t default:5000")
    my_parser.add_argument("--sampler_mode",         type = str,     action = "store",   default ="LANGEVIN",                           required=False      ,help = "Sampler used to generate new states \t default: LANGEVIN")
    my_parser.add_argument("--record_states",        type = int,     action = "store",   default = 0,                                   required=False      ,help = "Store the states that occure during the MCMC \t default: 0")
    my_parser.add_argument("--record_freq",          type = int,     action = "store",   default = 25,                                  required=False      ,help = "Frequency of storing the states during the MCMC \t default: 25")
    my_parser.add_argument("--seeds",                type = str,     action = "store",   default = "47",                                required=False      ,help = "Random seeds")
    my_parser.add_argument("--n_reps",               type = str,     action = "store",   default = "0",                                 required=False      ,help = "Indices of the simulation\tdefault 0")

    args = my_parser.parse_args()

    model_dict = {
        "LinearCondition5x5_one_condition":LinearCondition5x5_one_condition
        }

    #get the epochs of the trained model that are evaluated
    splits_n_epoch = args.n_epochs.split(',')

    #get the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #Get the indices of the repetitions
    splits_reps = args.n_reps.split(",")
    splits_seeds = args.seeds.split(",")

    #run the evaluation
    eval(
        N = args.N,
        kappa_min = args.kappa_min,
        kappa_max = args.kappa_max,
        dkappa = args.dkappa,
        lambda_min = args.lambda_min,
        lambda_max = args.lambda_max,
        dlambda = args.dlambda,
        ModelClass = model_dict[args.ModelClass],
        n_epochs = splits_n_epoch,
        load_path = args.load_path,
        n_iter = args.n_iter,
        date_simulation = args.date_simulation,
        device = device,
        l = args.l,
        dt_max = args.dt_max,
        sampler_mode = args.sampler_mode,
        record_states = bool(args.record_states),
        record_freq = args.record_freq,
        random_seed = splits_seeds,
        n_reps=splits_reps
    )