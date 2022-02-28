from models import ConvNet5x5,ConvNet_multi_Version_1,ConvNet20x20
from Training import Trainer_Maximum_Likelihood_Discrete
import numpy as np
from utils import get_tau_eq,quick_eval_2D,plotter_m_c_tau,get_T_c
from datetime import date
import torch
import json
import argparse
import os
import shutil

###################################################################################################################
#run simulations for a trained model at a certein temperature
###################################################################################################################
def run_simulation_Maximum_Likelihood_Discrete(Temperatures,save_state_freq,record_simulation,n_iteration,N,load_path,date,n_epoch_state_dict,ModelClass,device = "cpu"):
    '''
    Run a simulation using a trained energy model.

    parameters:
        Temperatures:               Temperatures that are simulated.
        n_iteration:                Number of update steps.
        N:                          Number of spins per row and column
        load_path:                  Location of the trained model.
        date:                       Time stamp for the simulation.
        n_epoch_state_dict:         Training epoch of the model which is evaluated.
        ModelClass:                 Class used to model the energy function.
        device:                     Device on which the simulation runs
        save_state_freq:            Frequency of saving the states during the simulation
        record_simulation:          Record states if True 

    returns:
        None
    '''

    #Load the model
    Trainer = Trainer_Maximum_Likelihood_Discrete(
        mode =                  "SIMULATE",
        ModelClass =            ModelClass,
        load_path =             load_path,
        device =                device,
        n_epoch_state_dict =    n_epoch_state_dict,
        record_simulation =     bool(record_simulation),
        save_state_freq = save_state_freq
    )

    Trainer.simulation(n_iterations = n_iteration,T = Temperatures,date = date,N = N)

def eval(N,save_state_freq,ModelClass,record_simulation,T_min,T_max,dT,n_epochs,load_path,iterations,date_simulation,upper_lim_c,full_eval = True,l = 10,dt_max = 3000,fs = 40,fontname = "Times New Roman"):
    '''
    Evaluate trained energy functions.

    parameters:
        N:                      Number of spins per row and column.
        ModelClass:             Class used as energy function
        T_min:                  Smallest temeprature that is evaluated.
        T_max:                  Biggest temeprature that is evaluated.
        dT:                     Step size of the Temperature.
        n_epochs:               Epochs that are evluated.
        load_path:              Location of the trained model
        iterations:             Number of update steps in the simulations.
        date_simulation:        Time stamp for the simulations.
        upper_lim_c:            Upper limit for the plot of the specific heat.
        l:                      Stepsize for the time lag of the calculation of the autocorrelation.
        dt_max:                 Biggest time lag evaluated in the calculation of the autocorrelation.
        fs:                     Fontsize of the plots.
        fontname:               Font type of the plots.
        full_eval:              Perfor full evaluation even if it has already be done
        save_state_freq:        Frequency of saving the states during the simulation
        record_simulation:      Record states if True

    returns:
        None

    '''
    #Get the Temperature values
    Temperatures_1 = np.arange(T_min,T_max,dT)

    #Select values close to the critical temperature
    T_c = round(get_T_c(J = 1,k = 1),4)
    Temperatures_2 = np.array([T_c-0.2,T_c-0.15,T_c-0.1,T_c-0.05,T_c,T_c+0.2,T_c+0.15,T_c+0.1,T_c+0.05])

    if bool(record_simulation):
        Temperatures = Temperatures_1

    else:
        Temperatures = np.concatenate((Temperatures_1,Temperatures_2))

    Temperatures = np.sort(Temperatures)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ###################################################################################################################
    ##Perform the simulations
    ###################################################################################################################
    for e in n_epochs:
        temps_selected = []
        for t in Temperatures:

            temp = round(t,7)

            #Check if the simulationhas already be done
            p = load_path + f"Simulations/" + date_simulation + f"_N_{N}/{date_simulation}_simulation_epoch_{e}_T_{temp}/"
            if os.path.exists(p):
                #Is it complete:
                files = os.listdir(p)

                files_expected = set(['Energies.pt', 'info.json', 'Magnetization.pt'])
                files_found = set(files)
                completed = files_expected.issubset(files_found)

                if completed == True: 
                    continue
                else: 
                    shutil.rmtree(p)
            temps_selected.append(temp)
            
        #Perform the simulation
        print("Temperatures to simulate:")
        print(temps_selected)
        if len(temps_selected) > 0:
            run_simulation_Maximum_Likelihood_Discrete(N = N,save_state_freq = save_state_freq,record_simulation = record_simulation,Temperatures = torch.tensor(temps_selected),ModelClass = ModelClass,n_iteration=iterations,load_path=load_path,n_epoch_state_dict=e,device = device,date = date_simulation)

    ###################################################################################################################
    #Approximate the equillibrium time
    ###################################################################################################################
    for e in n_epochs:
        for t in Temperatures:

            T = round(t,7)

            path = load_path + f"Simulations/"+date_simulation+f"_N_{N}/{date_simulation}_simulation_epoch_{e}_T_{T}/"

            #only determine the equillibrium time if it has not been determined yet.
            with open(path + "info.json","r") as file:
                info = json.load(file)
            file.close()

            if "t_eq" in info.keys() and full_eval == False:
                continue
            else:
                get_tau_eq(path = path,fs = fs)

    ###################################################################################################################
    #Evaluate the simulations
    ###################################################################################################################
    for e in n_epochs:
        for t in Temperatures:

            T = round(t,7)

            path = load_path + f"Simulations/"+date_simulation+f"_N_{N}/{date_simulation}_simulation_epoch_{e}_T_{T}/"

            #only determine the equillibrium time if it has not been determined yet.
            with open(path + "info.json","r") as file:
                info = json.load(file)
            file.close()

            '''if ("tau_energy" in info.keys()) and ("tau_magnetization" in info.keys()) and ("<c>" in info.keys()) and ("sigma_c" in info.keys()) and ("<m>" in info.keys()) and ("sigma_m" in info.keys()) and full_eval == False:
                continue

            else:'''
            magnetization = torch.load(path + "Magnetization.pt")
            energy = torch.load(path + "Energies.pt")
            quick_eval_2D(path = path,magnetizations=magnetization,energies=energy,l = l,dt_max = dt_max,fs = fs)

    ###################################################################################################################
    #plot the results and compare with the findings from the simulation using the correct hamiltonian
    ###################################################################################################################
    path_reference = f"./Discrete_Ising_Model/N_{N}_Metropolis/"

    data_cs_reference = np.loadtxt(path_reference + "specific_heat.txt",skiprows = 1)
    cs_reference = data_cs_reference[:,1:].T
    temps_reference = data_cs_reference[:,0]
    ms_reference = np.loadtxt(path_reference + "magnetization.txt",skiprows = 1,usecols = [1,2]).T
    es_reference = np.loadtxt(path_reference + "energies.txt",skiprows = 1,usecols = [1,2]).T
    taus_reference_energy = np.loadtxt(path_reference + "correlation_times.txt",skiprows = 1,usecols = 1)
    taus_reference_magnetization = np.loadtxt(path_reference + "correlation_times.txt",skiprows = 1,usecols = 2)

    cs = np.zeros([len(n_epochs),2,len(Temperatures)])
    ms = np.zeros([len(n_epochs),2,len(Temperatures)])
    es = np.zeros([len(n_epochs),2,len(Temperatures)])

    tau_energy = np.zeros([len(n_epochs),len(Temperatures)])
    tau_magnetization = np.zeros([len(n_epochs),len(Temperatures)])

    #plot it
    list_taus = [[taus_reference_energy],[taus_reference_magnetization]]
    list_temps = [temps_reference]
    list_temps_tau = [[temps_reference],[temps_reference]]
    list_cs = [cs_reference]
    list_ms = [ms_reference]
    list_es = [es_reference]

    list_labels_cs = ["True energy function"]
    list_labels_ms = ["True energy function"]
    list_labels_es = ["True energy function"]
    list_labels_taus = [[r"$\tau_{c}$"+" true energy function"],[r"$\tau_{m}$"+" true energy function"]] 

    #Get the data from the simulations
    for j in range(len(n_epochs)):
        #Wite header of the summary files:
        with open(load_path + f"Simulations/"+date_simulation+f"_N_{N}/correlation_times_epoch_{n_epochs[j]}.txt","w") as file:
            file.write("Temperature	tau_energy [iterations]	tau_magnetization [iterations]\n")
        file.close()

        with open(load_path + f"Simulations/"+date_simulation+f"_N_{N}/magnetization_epoch_{n_epochs[j]}.txt","w") as file:
            file.write("Temperature	magnetization per spin	std. dev. magnetization per spin\n")
        file.close()

        with open(load_path + f"Simulations/"+date_simulation+f"_N_{N}/specific_heat_epoch_{n_epochs[j]}.txt","w") as file:
            file.write("Temperature	specific heat per spin	std. dev. specific heat per spin\n")
        file.close()

        with open(load_path + f"Simulations/"+date_simulation+f"_N_{N}/energy_epoch_{n_epochs[j]}.txt","w") as file:
            file.write("Temperature	specific heat per spin	std. dev. specific heat per spin\n")
        file.close()
        

        for i in range(len(Temperatures)):

            T = round(Temperatures[i],7)

            path = load_path + f"Simulations/"+date_simulation+f"_N_{N}/{date_simulation}_simulation_epoch_{n_epochs[j]}_T_{T}/"
            with open(path + "info.json","r") as file:
                info = json.load(file)
            file.close()

            cs[j][0][i] = info["<c>"]
            cs[j][1][i] = info["sigma_c"]

            ms[j][0][i] = info["<m>"]
            ms[j][1][i] = info["sigma_m"]

            es[j][0][i] = info["<u>"]
            es[j][1][i] = info["sigma_u"]

            tau_energy[j][i] = info["tau_energy"]
            tau_magnetization[j][i] = info["tau_magnetization"]

            #Write summary file
            with open(load_path + f"Simulations/"+date_simulation+f"_N_{N}/correlation_times_epoch_{n_epochs[j]}.txt","a") as file:
                file.write(f"{T}\t{info['tau_energy']}\t{info['tau_magnetization']}\n")
            file.close()

            with open(load_path + f"Simulations/"+date_simulation+f"_N_{N}/magnetization_epoch_{n_epochs[j]}.txt","a") as file:
                file.write(f"{T}\t{info['<m>']}\t{info['sigma_m']}\n")
            file.close()

            with open(load_path + f"Simulations/"+date_simulation+f"_N_{N}/specific_heat_epoch_{n_epochs[j]}.txt","a") as file:
                file.write(f"{T}\t{info['<c>']}\t{info['sigma_c']}\n")
            file.close()

            with open(load_path + f"Simulations/"+date_simulation+f"_N_{N}/energy_epoch_{n_epochs[j]}.txt","a") as file:
                file.write(f"{T}\t{info['<u>']}\t{info['sigma_u']}\n")
            file.close()

        list_ms.append(ms[j])
        list_cs.append(cs[j])
        list_es.append(es[j])
        list_temps.append(Temperatures)
        list_taus[0].append(tau_energy[j])
        list_taus[1].append(tau_magnetization[j])
        list_temps_tau[0].append(Temperatures)
        list_temps_tau[1].append(Temperatures)

        list_labels_es.append(f"Trained energy function e = {n_epochs[j]}")
        list_labels_cs.append(f"Trained energy function e = {n_epochs[j]}")
        list_labels_ms.append(f"Trained energy function e = {n_epochs[j]}")
        list_labels_taus[0].append(r"$\tau_{c}$"+f" trained energy function e = {n_epochs[j]}")
        list_labels_taus[1].append(r"$\tau_{m}$"+f" trained energy function e = {n_epochs[j]}")

    plotter_m_c_tau(target_path=load_path + f"Simulations/"+date_simulation+f"_N_{N}/",title = f"N = {N}",cs = list_cs,es = list_es,ms = list_ms,taus = list_taus,temps=list_temps,temps_tau =list_temps_tau,labels_es = list_labels_es, labels_cs = list_labels_cs,labels_ms = list_labels_ms,labels_taus = list_labels_taus,reference=True,fs = fs,fontname = fontname,upper_lim_c = upper_lim_c)

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()

    my_parser.add_argument("--T_min",                type=float,   action = "store",    default=0.5,   	                            required=False      ,help = "Smallest temperature to evaluate\tdefault = 0.5")
    my_parser.add_argument("--T_max",                type=float,   action = "store",    default=5.0,   	                            required=False      ,help = "Biggest temperature to evaluate\tdefault = 5.0")
    my_parser.add_argument("--dT",                   type=float,   action = "store",    default=0.1,   	                            required=False      ,help = "Temperature stepsize\tdefault = 0.1")
    my_parser.add_argument("--load_path",            type=str,     action = "store", 	                                            required=True       ,help = "Location of the training results")
    my_parser.add_argument("--n_epochs",             type=str,     action = "store",    default="0,10,20,30",                       required=False      ,help = "Epochs for which the training is evaluated\tdefault = 0,10,20,30")
    my_parser.add_argument("--n_iter",               type=int,     action = "store",    default=500000,                             required=False      ,help = "Number of iterations in the suimulations\tdefault = 500000")
    my_parser.add_argument("--date_simulation",      type=str,     action = "store",                                                required=True       ,help = "Time stamp of the simulations")
    my_parser.add_argument("--upper_lim_c",          type=float,   action = "store",    default = 3.0,                              required=False      ,help = "Upper limit for the specific heat in the plots\tdefault = 3.0")
    my_parser.add_argument("--N",                    type=int,     action = "store",                                                required=True       ,help = "Number of spins per row and column of the grid")
    my_parser.add_argument("--ModelClass",           type=str,     action = "store",    default="ConvNet5x5",                       required=False      ,help = "Class used as energy function.\tdefault = ConvNet5x5")
    my_parser.add_argument("--full_eval",            type=int,     action = "store",    default=0,                                  required=False      ,help = "Perform full evaluation, even if it has already been done.\tdefault = 0")
    my_parser.add_argument("--dt_max",               type=int,     action = "store",    default=10000,                              required=False      ,help = "Biggest time lag in the calculation of the autocorrelation.\tdefault = 10000")
    my_parser.add_argument("--l",                    type=int,     action = "store",    default=10,                                 required=False      ,help = "Stepsize for the time lag in the computation of the correlation times\tdefault = 10")
    my_parser.add_argument("--record_simulation",    type=int,     action = "store",    default = 0,                                required=False      ,help = "Store the states that occure duing the simulation if set to 1\tdefault = 0")
    my_parser.add_argument("--save_state_freq",      type=int,     action = "store",    default = 25,                               required=False      ,help = "Frequency of saving the states\tdefault = 25")

    args = my_parser.parse_args()

    splits = args.n_epochs.split(',')

    model_dict = {
        "ConvNet5x5":ConvNet5x5,
        "ConvNet_multi_Version_1":ConvNet_multi_Version_1,
        "ConvNet20x20":ConvNet20x20
        }

    eval(
        save_state_freq = args.save_state_freq,
        N = args.N,
        record_simulation = args.record_simulation,
        ModelClass = model_dict[args.ModelClass],
        T_min = args.T_min,
        T_max = args.T_max,dT = args.dT,
        load_path = args.load_path,
        n_epochs = splits,
        iterations = args.n_iter,
        date_simulation = args.date_simulation,
        upper_lim_c = args.upper_lim_c,
        full_eval=bool(args.full_eval),
        dt_max=args.dt_max,
        l = args.l
        )