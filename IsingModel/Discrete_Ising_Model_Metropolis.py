import numpy as np
import os
import tqdm 
import torch
import json
import argparse
from utils import quick_eval_2D,get_tau_eq,plot_states,plotter_m_c_tau,get_T_c

class DiscreteIsingModel2D_Metropolis():
    def __init__(self,device,N = None,T  = None,J = 1,k = 1,fs = 40):
        '''
        parameters:
            N:          Number of spins per row and column. Total number of spins in the lattice is given by N*N
            T:          Tensor containing the temperatures that are simulated
            device:     Device, on which the sumulation runs
            J:          Interaction energy
            k:          Boltzmann constant
            fs:         Fontsize for the plots
        '''

        self.N = N
        self.T = T
        self.J = J
        self.k = k
        self.fs = fs
        self.device = device

        self.beta = 1 / (k * T)

        self.Energies = None
        self.Magnetization = None

    def H(self,v):
        '''
        Compute the energy of a state v. Periodic boundery conditions are applied.

        parameters:
            v:          Tensor of shape [B,1,N,N] containing the state.
        
        returns:
            E:          Energy of the state
        '''

        interactions = (v * torch.roll(input=v,shifts=1,dims = 2) + v * torch.roll(input=v,shifts=1,dims = 3)).to(self.device)
        s = torch.sum(interactions,dim = (1,2,3))

        E = -self.J * s

        return E
    
    def __call__(self,n_iter,init_mode = "inf",path = None,record = False,save_state_freq = 0):
        '''
        Run a simulation using the Metropolis algorithm.

        parameters:
            M:                  Number of simulation steps
            init_mode:          "inf" or "zero", specifies whether the initial sample is at temperature 
                                T = inf or T = 0
            path:               Location to store the resultd of the simulation
            record:             Save the states if True
            save_state_freq:    Frequency of saving data.

        returns:
            None
        '''

        #Get the number of different temperatures that are simulated in parallel
        B = self.T.shape[0]

        #Initialize the tensor used to store the energies and the magnetizations during the simulation
        self.Energies = torch.zeros(B,n_iter).to(self.device)
        self.Magnetization = torch.zeros(B,n_iter).to(self.device)

        #Get the initial state
        if init_mode == "inf":     
            mu = torch.ones([B,1,self.N,self.N]).to(self.device) * (-1)**torch.randint(0,2,[B,1,self.N,self.N]).to(self.device)
        
        elif init_mode == "zero":
            mu = torch.ones([B,1,self.N,self.N]).to(self.device) * (-1)**torch.randint(0,2,[1]).to(self.device)

        #Get the energy of the initial state
        E_mu = self.H(mu)
        self.Energies[:,0] = E_mu

        #Get the magnetization of the initial state
        M_mu = mu.sum(dim = (1,2,3))
        self.Magnetization[:,0] = M_mu

        #Save the initial state
        if record == True:
            self.states = torch.zeros([B,n_iter // save_state_freq,1,N,N]).to(self.device)
            counter = 0
            self.states[:,counter] = mu
            counter += 1

        #Store some states for plotting
        states_to_plot = torch.zeros([B,16,1,N,N]).to(self.device)
        plot_indices = np.random.permutation(np.arange(int(0.75 * n_iter),n_iter))[:16]
        plot_counter = 0

        #perform the simulations
        for k in tqdm.tqdm(range(1,n_iter)):
            #Select a spin to flip in each state of the batch
            pos = torch.randint(0,self.N,[2]).to(self.device)
            
            #Flip the spins to get new states
            v = mu.clone().to(device)
            v[:,0,pos[0],pos[1]] *= -1

            #Get the energy difference
            dE = 2 * self.J * mu[:,0,pos[0],pos[1]] * mu[:,0,[(pos[0]+1)%self.N,pos[0]-1,pos[0],pos[0]],[pos[1],pos[1],pos[1]-1,(pos[1]+1) % self.N]].sum(-1)

            #Accept the new state if the energy difference is negative
            mask = (dE <= 0)
            
            #Get the acceptance ratios for the other states with positive Energy difference
            indices = torch.arange(len(mu))[mask.cpu() == False]
            
            if len(indices) > 0:
                #Get the acceptance ratios:
                A = torch.exp(-dE[indices] * self.beta[indices]).to(self.device)
                
                #Get the value to compar
                r = torch.rand_like(A)

                mask_accept = (r < A)
                mask[indices] = mask_accept

            #Get the new state
            mu = torch.where(mask[:,None,None,None] == True,v,mu)
            E_v = E_mu + dE
            E_mu = torch.where(mask == True,E_v,E_mu)

            #Save the data
            self.Magnetization[:,k] = abs(mu.sum(dim = (1,2,3)))
            self.Energies[:,k] = E_mu

            if (record == True) and (k % freq == 0):
                self.states[:,counter] = mu
                counter += 1

            if k in plot_indices:
                states_to_plot[:,plot_counter] = mu
                plot_counter += 1

        if path is not None:
            #Save the individual simulations
            for j in range(len(self.T)):
                T = self.T[j]
                p = path+f"N_{self.N}_T_{T}/"
                os.makedirs(p)

                #information about the simulation
                d = {
                    "N":self.N,
                    "T":T.item(),
                    "n_iter":n_iter,
                    "k":self.k,
                    "J":self.J,
                }

                #Save the time between the different recorded samples
                if record == True:
                    d["freq_save_samples"] = save_state_freq

                with open(p+"info.json","w") as file:
                    json.dump(d,file)
                file.close()

                #Save the Recorded data
                energies = torch.tensor(self.Energies[j,:].detach().cpu().numpy())
                mags = torch.tensor(self.Magnetization[j,:].detach().cpu().numpy())
                torch.save(energies,p+"Energies.pt")
                torch.save(mags,p+"Magnetization.pt")

                #Save the recorded states
                if record == True:
                    s = torch.tensor(self.states[j,:].detach().cpu().numpy())
                    torch.save(s,p+"states.pt")
                    indices = np.random.permutation(np.arange(len(self.states[j]) //2,len(self.states[j])))[:25]
                    plot_states(self.states[j][indices],5,5,p+"states.jpg",N = self.N)

                #Plot the states
                plot_states(states_to_plot[j],4,4,p + f"visualization_N_{N}_T_{T}.jpg",N)

    def load(self,path):
        '''
        #Load recorded energies and magnetizations and teh arameters of the simulation.

        parameters:
            path:   Location of the simulation results
        
        return:
            None
        '''

        #Get the parrameters of the simulation
        with open(path+"info.json","r") as file:
            info = json.load(file)
        file.close()

        self.N = info["N"]
        self.T = info["T"]
        self.k = info["k"]
        self.n_iter = info["n_iter"]
        self.J = info["J"]
        self.t_eq = info["t_eq"]

        #Get the energy and the Magnetization
        self.Energies = torch.load(path+"Energies.pt")
        self.Magnetization = torch.load(path+"Magnetization.pt")

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()

    my_parser.add_argument("--T_min",           action='store',         type=float,     default=0.5         ,help = "Samllet Temperature that is simualted\t default: 0.5")      #Min Temperature considered in the simulation
    my_parser.add_argument("--T_max",           action='store',         type=float,     default=5.0         ,help = "Biggest Temperature that is simualted\t default: 5.0")      #Max Temperature considered in the simulation
    my_parser.add_argument("--dT",              action='store',         type=float,     default=0.25        ,help = "Steps size ffor the temperature\t default: 0.25")           #Temperature step size
    my_parser.add_argument("--N",               action='store',         type=int,       default=5           ,help = "Number of spins per row and column\t default: 5")           #Number of spins per row and column
    my_parser.add_argument("--n_iter",          action='store',         type=int,       default=int(5e5)    ,help = "Number of iterations\t default: 500000")                    #Number of iterations 
    my_parser.add_argument("--dt_max",          action='store',         type=int,       default=10000       ,help = "Biggest time lag evaluated in the calculation of the autocorrelation\t default: 10000")       #Biggest time lag in the calculation of the autocorrelation
    my_parser.add_argument("--l",               action='store',         type=int,       default=10          ,help = "Step size of the time lag for the calculation of the autocorrelation\t default: 10")          #Stepsize of the time lags during the calculation of the autocorrelation
    my_parser.add_argument("--record",          action='store',         type=int,       default=0           ,help = "Save sates as a training set\t default: 0")                                                   #Record states during the simulation
    my_parser.add_argument("--freq",            action = "store",       type = int,     default = 25        ,help = "Frequence of saving the states\t default: 25")                                                #Frequency of recording states during the training

    args = my_parser.parse_args()

    #Get the device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    #Get the Temperature values
    Temperatures_1 = np.arange(args.T_min,args.T_max,args.dT)

    #Select values close to the critical temperature
    T_c = round(get_T_c(J = 1,k = 1),4)
    Temperatures_2 = np.array([T_c-0.2,T_c-0.15,T_c-0.1,T_c-0.05,T_c,T_c+0.2,T_c+0.15,T_c+0.1,T_c+0.05])

    record = bool(args.record)

    #Simulation without savin the states
    if record == False:
        Temperatures = np.concatenate((Temperatures_1,Temperatures_2))
        Temperatures = np.sort(Temperatures)

    #Save the states
    else:
        Temperatures = np.sort(Temperatures_1)

    N = args.N
    l = args.l
    dt_max = args.dt_max
    n_iter = args.n_iter
    freq = args.freq

    if record == True: path = f"Discrete_Ising_Model/N_{N}_Metropolis_Data_Set/"
    else: path = f"Discrete_Ising_Model/N_{N}_Metropolis/"

    #Run the simulation and record the magnetization and the energy
    #Select simulations that are not done yet
    selected_Temperatures = []

    for T in Temperatures:
        p = path + f"N_{N}_T_{T}/"

        #Test if the simulation has already been done.
        if os.path.exists(p):
            #Is it complete:
            files = os.listdir(p)

            files_expected = set(['Energies.pt', 'info.json', 'Magnetization.pt'])
            files_found = set(files)
            completed = files_expected.issubset(files_found)

            if completed == True: 
                continue
            
        else:
            selected_Temperatures.append(T)

    if len(selected_Temperatures) > 0:
        I = DiscreteIsingModel2D_Metropolis(device = device,N = N,T = torch.tensor(selected_Temperatures).to(device))
        I(n_iter = n_iter,init_mode="inf",path = path,record = record,save_state_freq=freq)

    #Approximate the equillibrium time
    for T in Temperatures:
        p = path + f"N_{N}_T_{T}/"

        #only determine the equillibrium time if it has not been determined yet.
        with open(p + "info.json","r") as file:
            info = json.load(file)
        file.close()

        if "t_eq" in info.keys():
            continue
        else:
            I = DiscreteIsingModel2D_Metropolis(N = N,T = T,device = device)
            get_tau_eq(path = p,fs = I.fs)

    for i in range(len(Temperatures)):
        T = Temperatures[i]
        p = path + f"N_{N}_T_{T}/"

        I = DiscreteIsingModel2D_Metropolis(N = N,T = T,device = device)
        I.load(p)
        quick_eval_2D(path = p,magnetizations=I.Magnetization,energies=I.Energies,l = l,dt_max = dt_max,fs = I.fs)

    #plot 
    cs = np.zeros([2,len(Temperatures)])
    ms = np.zeros([2,len(Temperatures)])
    es = np.zeros([2,len(Temperatures)])

    tau_energy = np.zeros(len(Temperatures))
    tau_magnetization = np.zeros(len(Temperatures))

    for i in range(len(Temperatures)):
        T = Temperatures[i]
        p = path + f"N_{N}_T_{T}/"

        with open(p+"info.json","r") as file:
            info = json.load(file)
        file.close()

        cs[0][i] = info["<c>"]
        cs[1][i] = info["sigma_c"]

        ms[0][i] = info["<m>"]
        ms[1][i] = info["sigma_m"]

        es[0][i] = info["<u>"]
        es[1][i] = info["sigma_u"]

        tau_energy[i] = info["tau_energy"]
        tau_magnetization[i] = info["tau_magnetization"]

    labels_cs = ["Metropolis"]
    labels_ms = ["Metropolis"]
    labels_es = ["Metropolis"]
    labels_taus = [["Energy"],["Magnetization"]]

    #Plot the results
    plotter_m_c_tau(target_path=path,title = f"N = {N}",es = [es],cs = [cs],ms = [ms],taus = [[tau_energy],[tau_magnetization]],temps=[Temperatures],temps_tau = [[Temperatures],[Temperatures]],labels_es = labels_es,labels_cs = labels_cs,labels_ms = labels_ms,labels_taus = labels_taus,reference=False,upper_lim_c = 2,fontname = "Times New Roman")

    #Save the results
    #specific heat
    with open(path + "specific_heat.txt","w") as file:
        file.write("Temperature\tspecific heat per spin\tstd. dev. specific heat per spin\n")
        np.savetxt(file,np.concatenate((Temperatures.reshape([1,-1]),cs),0).T)
    file.close()

    #magnetization
    with open(path + "magnetization.txt","w") as file:
        file.write("Temperature\tmagnetization per spin\tstd. dev. magnetization per spin\n")
        np.savetxt(file,np.concatenate((Temperatures.reshape([1,-1]),ms),0).T)
    file.close()

    #Correlation times
    with open(path + "correlation_times.txt","w") as file:
        file.write("Temperature\ttau_energy [iterations]\ttau_magnetization [iterations]\n")
        np.savetxt(file,np.concatenate((Temperatures.reshape([1,-1]),tau_energy.reshape([1,-1]),tau_magnetization.reshape([1,-1])),0).T)
    file.close()

    #Energies
    with open(path + "energies.txt","w") as file:
        file.write("Temperature\tenergy per spin\tstd. dev. energy per spin\n")
        np.savetxt(file,np.concatenate((Temperatures.reshape([1,-1]),es),0).T)
    file.close()