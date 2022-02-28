import numpy as np
import os
import tqdm 
import torch
import json
import argparse
from utils import plot_states,get_tau_eq,quick_eval_2D,plotter_results

class Simulation_Scalar_Theory_2D():
    def __init__(self,N,device,epsilon,n_reps,random_seed,fs = 40):
        '''
        parameters:
            N:                  Number of spins per row and column. Total number of spins in the lattice is given by N*N
            device:             Device, on which the sumulation runs
            fs:                 Fontsize for the plots
            n_reps:             Number of this run
            random_seed:        Random_seed
            epsilon:            Langevin step size
            fs:                 Fontsize for plotting
        '''

        self.N = N
        self.fs = fs
        self.device = device
        self.epsilon = epsilon
        self.n_reps = n_reps
        self.random_seed = random_seed
        
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.Actions = None
        self.Magnetization = None

        #Volume
        self.Theta = N*N

    def S(self,mus,kappas,lambdas):
        '''
        True action function for 2D lattice.
        
        parameters:
            mus:        Initial states
            kappas:     Tensor containing the hopping-parameters
            lambdas:    Tensor containing the quadric-couplings

        returns:
            actions:    Containing the action of the different states
        '''

        #Get the quadric coupling
        actions = (1 - 2 * lambdas[:,None,None,None]) * mus.pow(2) +lambdas[:,None,None,None] * mus.pow(4)

        #Get the term depending on the hopping parameter
        actions += - 2 * kappas[:,None,None,None] * torch.roll(input=mus,shifts=1,dims=2) * mus
        actions += - 2 * kappas[:,None,None,None] * torch.roll(input=mus,shifts=1,dims=3) * mus

        actions = torch.sum(input=actions,dim = [1,2,3])

        return actions

    def K(self,mu,kappas,lambdas): 
        '''
        Compute the explicit drift drift term fro the tue action function

        parameters:
            mu:         Current state
            kappas:     Tensor containing the hopping-parameters
            lambdas:    Tensor containing the quartic-couplings

        returns:
            drift:      Drift term
        '''
        drift = 2 * mu* (2 * lambdas[:,None,None,None] * (1 - mu.pow(2)) - 1)

        drift += 2 * kappas[:,None,None,None] * (torch.roll(input=mu,shifts=1,dims=2) + torch.roll(input=mu,shifts= -1,dims=2))
        drift += 2 * kappas[:,None,None,None] * (torch.roll(input=mu,shifts=1,dims=3) + torch.roll(input=mu,shifts=-1,dims=3))

        return drift

    def langevin_sampler_general(self,n_iter,mu,save_state_freq,record,kappas,lambdas):
        '''
        Sample from the Boltzmann distribution using Langevin Dynamics 

        parameters:
            n_iter:             Number of update steps.
            mu:                 States.
            save_state_freq:    Frequnecy of saving the states.
            record:             Store the states
            kappas:             Tensor containing the hopping-parameters
            lambdas:            Tensor containing the quartic-couplings

        returns:
            None
        '''

        print("sample with general langevin")

        mu = mu.requires_grad_(True)

        for k in tqdm.tqdm(range(n_iter)):
            S_mu = self.S(mus = mu,kappas = kappas,lambdas = lambdas)
            g = torch.autograd.grad(outputs = S_mu.sum(),inputs = [mu])[0]

            mu.data += -self.epsilon * g + np.sqrt(2 * self.epsilon) * torch.randn_like(mu).to(self.device)

            mu = mu.detach().to(self.device)
            mu = mu.requires_grad_(True).to(self.device)

            self.Actions[:,k] = S_mu
            self.Magnetization[:,k] = torch.sum(input=mu,dim = [1,2,3])

            if (record == True) and (k % save_state_freq == 0):
                self.states[:,self.counter] = mu
                self.counter += 1

    def langevin_sampler_specific(self,n_iter,mu,save_state_freq,record,kappas,lambdas):
        '''
        Sample from the Boltzmann distribution using Langevin Dynamics with the explicit drift term for the action. 

        parameters:
            n_iter:             Number of update steps.
            mu:                 States.
            save_state_freq:    Frequnecy of saving the states.
            record:             Store the states
            kappas:             Tensor containing the hopping-parameters
            lambdas:            Tensor containing the quartic-couplings

        returns:
            None
        '''
        
        print("sample with explicit langevin")

        for k in tqdm.tqdm(range(n_iter)):
            mu.data += self.K(mu = mu,kappas = kappas,lambdas = lambdas) * self.epsilon + np.sqrt(2 * self.epsilon) * torch.randn_like(mu).to(self.device)

            self.Actions[:,k] = self.S(mus = mu,kappas=kappas,lambdas = lambdas)
            self.Magnetization[:,k] = torch.sum(input=mu,dim = [1,2,3])

            if (record == True) and (k % save_state_freq == 0):
                self.states[:,self.counter] = mu
                self.counter += 1

    def metropolis_sampler(self,n_iter,mu,save_state_freq,record,kappas,lambdas):
        '''
        Metropolis sampling with proposal states which follow a gaussian distribution centred at the current state.

        parameters:
            n_iter:             Number of update steps.
            mu:                 States.
            save_state_freq:    Frequnecy of saving the states.
            record:             Store the states
            kappas:             Tensor containing the hopping-parameters
            lambdas:            Tensor containing the quartic-couplings

        returns:
            None
        '''

        print("sample with metropolis")

        #Get the actions of the initial states
        S_mu = self.S(mus = mu, kappas = kappas, lambdas = lambdas)

        for k in tqdm.tqdm(range(n_iter)):
            #Get proposal states
            v = mu + np.sqrt(2 * self.epsilon) * torch.randn_like(mu)

            #Ge the action for the new proposal states
            S_v = self.S(mus = v, kappas = kappas, lambdas = lambdas)

            #Get the action differences
            dS = S_v - S_mu

            #accept the new state if the energy difference is negative
            mask = (dS <= 0)
            
            #Get the acceptance ratios for the other states with positive Energy difference
            indices = torch.arange(len(mu))[mask == False]
            
            if len(indices) > 0:
                #Get the acceptance ratios:
                A = torch.exp(-dS[indices])
                
                #Get the value to compar
                r = torch.rand_like(A)

                mask_accept = (r < A)
                mask[indices] = mask_accept

            #Get the new state
            mu = torch.where(mask[:,None,None,None] == True,v,mu)
            S_mu = torch.where(mask == True,S_v,S_mu)

            self.Actions[:,k] = S_mu
            self.Magnetization[:,k] = torch.sum(input=mu,dim = [1,2,3])

            if (record == True) and (k % save_state_freq == 0):
                self.states[:,self.counter] = mu
                self.counter += 1

    def __call__(self,path,n_iter,sampler_mode,record,save_state_freq,kappas,lambdas):
        '''
        Run a simulation.

        parameters:
            n_iter:             Number of simulation steps
            path:               Location to store the resultd of the simulation
            record:             Save the states if True
            save_state_freq:    Frequency of saving data.
            sampler_mode:       Method used to sample new states.
            kappas:             Hopping parameters simulated in parallel
            Lambdas:            Quartic couplings simulated in parallel

        returns:
            None
        '''

        #Get the number of simulations that are simulated in parallel
        B = len(kappas)

        self.Actions = torch.zeros(B,n_iter).to(self.device)
        self.Magnetization = torch.zeros(B,n_iter).to(self.device)

        #Get the initial state
        mu = torch.randn([B,1,self.N,self.N]).to(self.device)

        #Save the initial state
        if record == True:
            self.counter = 0
            self.states = torch.zeros([B,n_iter // save_state_freq,1,N,N]).to(self.device)


        if sampler_mode == "LANGEVIN_GENERAL":                  self.langevin_sampler_general(n_iter = n_iter,mu = mu,save_state_freq = save_state_freq,record = record,kappas = kappas,lambdas = lambdas)
        elif sampler_mode == "LANGEVIN_SPECIFIC":               self.langevin_sampler_specific(n_iter = n_iter,mu = mu,save_state_freq = save_state_freq,record = record,kappas = kappas,lambdas = lambdas)
        elif sampler_mode == "METROPOLIS":                      self.metropolis_sampler(n_iter = n_iter,mu = mu,save_state_freq = save_state_freq,record = record,kappas = kappas,lambdas = lambdas)
        else:                                                   raise NotImplementedError("Select valid sampler!")

        if path is not None:
            for i in range(B):
                p = path+f"kappa_{round(kappas[i].item(),6)}_lambda_{round(lambdas[i].item(),6)}/"

                if not os.path.exists(p):
                    os.makedirs(p)

                #information about the simulation
                d = {
                    "N":self.N,
                    "n_iter":n_iter,
                    "sampler_Mode":sampler_mode,
                    "lambda_action":round(lambdas[i].item(),6),
                    "kappa_action":round(kappas[i].item(),6),
                    "epsilon":self.epsilon,
                    "freq_save_samples":save_state_freq,
                    "random_seed":self.random_seed
                }

                with open(p+f"info_{self.n_reps}.json","w") as file:
                    json.dump(d,file)
                file.close()

                a = torch.tensor(self.Actions[i].detach().cpu().numpy())
                m = torch.tensor(self.Magnetization[i].detach().cpu().numpy())

                #Save the Recorded data
                torch.save(a,p+f"Actions_{self.n_reps}.pt")
                torch.save(m,p+f"Magnetization_{self.n_reps}.pt")

                #Create folder for Images
                if not os.path.exists(p+"Plt/"):
                    os.mkdir(p+"Plt/")

                #Save the date set
                if record == True:
                    current_states = torch.tensor(self.states[i,:].detach().cpu().numpy())
                    torch.save(current_states,p+f"states_{self.n_reps}.pt")
                    indices = np.random.permutation(np.arange(len(current_states) // 2,len(current_states)))[:25]
                    plot_states(current_states[indices],5,5,p+f"Plt/states_{self.n_reps}.jpg",N = self.N,titles=["" for i in range(len(indices))])

    def load(self,path):
        '''
        #Load recorded actions and magnetizations and the parameters of the simulation.

        parameters:
            path:   Location of the simulation results
        
        return:
            None
        '''

        #Get the parrameters of the simulation
        with open(path+f"info_{self.n_reps}.json","r") as file:
            info = json.load(file)
        file.close()

        self.N = info["N"]
        self.n_iter = info["n_iter"]
        self.t_eq = info["t_eq"]
        self.kappa_action = info["kappa_action"]
        self.lambda_action = info["lambda_action"]
        
        #Get the energy and the Magnetization
        self.Actions = torch.load(path+f"Actions_{self.n_reps}.pt")
        self.Magnetization = torch.load(path+f"Magnetization_{self.n_reps}.pt")

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument("--kappa_min",               action ='store',         type = float,     default = 0.20             ,help = "Smallet hopping parameter that is simualted\tdefault: 0.2")     
    my_parser.add_argument("--kappa_max",               action ='store',         type = float,     default = 0.40             ,help = "Biggest hopping parameter that is simualted\tdefault: 0.4")
    my_parser.add_argument("--dkappa",                  action ='store',         type = float,     default = 0.005            ,help = "Steps size for the hopping parameter\tdefault: 0.02")
    my_parser.add_argument("--lambda_min",              action ='store',         type = float,     default = 0.02             ,help = "Smallet quartic coupling that is simualted\tdefault: 0.02")
    my_parser.add_argument("--lambda_max",              action ='store',         type = float,     default = 0.03             ,help = "Biggest quartic coupling that is simualted\tdefault: 0.03")
    my_parser.add_argument("--dlambda",                 action ='store',         type = float,     default = 0.02             ,help = "Steps size for the quartic coupling\tdefault: 0.02")
    my_parser.add_argument("--N",               action = 'store',         type = int                                          ,help = "Number of spins per row and column")
    my_parser.add_argument("--n_iter",          action = 'store',         type = int,         default = int(5e4)              ,help = "Number of iterations\tdefault: 50000")
    my_parser.add_argument("--dt_max",          action = 'store',         type = int,         default = 5000                  ,help = "Biggest time lag evaluated in the calculation of the autocorrelation\tdefault: 5000")
    my_parser.add_argument("--l",               action = 'store',         type = int,         default = 10                    ,help = "Step size of the time lag for the calculation of the autocorrelation\tdefault: 250")
    my_parser.add_argument("--record",          action = 'store',         type = int,         default = 0                     ,help = "Save sates as a training set\tdefault: 0")
    my_parser.add_argument("--freq",            action = "store",         type = int,         default = 25                    ,help = "Frequence of saving the states\tdefault: 25")
    my_parser.add_argument("--sampler_mode",    action = "store",         type = str,         default = "LANGEVIN_SPECIFIC"   ,help = "Sampler used to generate new samples\tdefault: LANGEVIN_SPECIFIC")
    my_parser.add_argument("--epsilon",         action = "store",         type = float,       default = 0.01                  ,help = "Stepsize used in langevin dynamics\tdefault: 0.01")  
    my_parser.add_argument("--n_reps",          action = "store",         type = int,         default = 0                     ,help = "Number of the current simulations\tdefault: 0")  
    my_parser.add_argument("--seed",            action = "store",         type = int,         default = 47                    ,help = "Random seed\tdefault: 47")  

    args = my_parser.parse_args()

    ##################################################################################################################################
    #Get hopping parameters and the quartic couplings
    ##################################################################################################################################
    kappas_original = np.arange(args.kappa_min,args.kappa_max,args.dkappa)
    lambdas_original = np.arange(args.lambda_min,args.lambda_max,args.dlambda)

    N = args.N
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    ##################################################################################################################################
    #Get the parent folder containing the results of the simulations
    ##################################################################################################################################
    if args.record == 1: path = f"Scalar_Theory/N_{N}_{args.sampler_mode}_Data_Set/"
    else: path = f"Scalar_Theory/N_{N}_{args.sampler_mode}/"

    ##################################################################################################################################
    #Run the simulation and record the magnetization and the action
    ##################################################################################################################################

    #Get the cobinations of kappa and lambda
    len_kappas = len(kappas_original)
    len_lambdas = len(lambdas_original)

    kappas_combined = torch.zeros(len_kappas * len_lambdas)
    lambdas_combined = torch.zeros(len_kappas * len_lambdas)

    for i in range(len_lambdas):
        lambdas_combined[i * len_kappas:(i+1) * len_kappas] = torch.ones(len_kappas) * lambdas_original[i]
        kappas_combined[i * len_kappas:(i+1) * len_kappas] = torch.tensor(kappas_original)

    #sort out the combinations that already exist
    mask = torch.ones(len(lambdas_combined),dtype=bool)

    for i in range(len(kappas_combined)):
        p = path + f"kappa_{round(kappas_combined[i].item(),6)}_lambda_{round(lambdas_combined[i].item(),6)}/"

        if os.path.exists(p):
            if os.path.exists(p+f"Actions_{args.n_reps}.pt") and os.path.exists(p+f"Magnetization_{args.n_reps}.pt") and os.path.exists(p+f"info_{args.n_reps}.json"): mask[i] = False

    if mask.sum() > 0:

        lambdas_combined = lambdas_combined[mask]
        kappas_combined = kappas_combined[mask]

        print(kappas_combined)
        
        I = Simulation_Scalar_Theory_2D(N = N,epsilon = args.epsilon,device = device,n_reps = args.n_reps,random_seed=args.seed)
        I(n_iter = args.n_iter,sampler_mode=args.sampler_mode,path = path,record=bool(args.record),save_state_freq=args.freq,kappas=kappas_combined.to(device),lambdas=lambdas_combined.to(device))

    ##################################################################################################################################
    #Approximate the equillibrium time
    ##################################################################################################################################
    for kappe_action in kappas_original:
        for lambda_action in lambdas_original:
            
            #round the quadric coupling and the hopping parameter
            lambda_action = round(lambda_action,6)
            kappe_action = round(kappe_action,6)

            #Get the path for the current simulation
            p = path + f"kappa_{kappe_action}_lambda_{lambda_action}/"

            #Only determine the equillibrium time if it has not been determined yet.
            with open(p + f"info_{args.n_reps}.json","r") as file:
                info = json.load(file)
            file.close()

            if "t_eq" in info.keys():   continue
            else:
                #approximate the equillibrium time based on the recorded actions and magnetizations
                I = Simulation_Scalar_Theory_2D(N = N,epsilon = args.epsilon,device=device,n_reps = args.n_reps,random_seed=info["random_seed"])
                get_tau_eq(path = p,fs = I.fs,n_reps=args.n_reps)
    
    ##################################################################################################################################
    #evaluate the recorded data
    ##################################################################################################################################
    for lambda_action in lambdas_original:
        lambda_action = round(lambda_action,6)

        with open(path + f"summary_lambda_{lambda_action}_{args.n_reps}.txt","w") as file:
            #Write the header of the summary file
            file.write("kappa\tmagnetization\tstd_magnetization\taction\tstd_action\tU_L\tstd_U_L\tchi^2\tstd_chi^2\ttau_action\ttau_magnetization\n")

            for kappe_action in kappas_original:
                kappe_action = round(kappe_action,6)

                #Get the path for the current simulation
                p = path + f"kappa_{kappe_action}_lambda_{lambda_action}/"

                with open(p + f"info_{args.n_reps}.json","r") as info_file:
                  info = json.load(info_file)
                info_file.close()

                if ("tau_magnetization" in info.keys()) and ("tau_action" in info.keys()) and ("<S>" in info.keys()) and ("<M>" in info.keys()) and ("<U_L>" in info.keys()) and ("<chi^2>" in info.keys()): 
                    magnetization = info["<M>"]
                    std_magnetization = info["sigma_M"]
                    action = info["<S>"]
                    std_action = info["sigma_S"]
                    U_L = info["<U_L>"] 
                    std_U_L = info["sigma_U_L"]
                    chi_aquared = info["<chi^2>"] 
                    std_chi_aquared = info["sigma_chi^2"]
                    tau_action = info["tau_action"]
                    tau_magnetization = info["tau_magnetization"]

                else:  
                    #Load the simulation
                    I = Simulation_Scalar_Theory_2D(N = N,device = device,epsilon = args.epsilon,n_reps = args.n_reps,random_seed=info["random_seed"])
                    I.load(p)

                    #Evaluate the simulation
                    magnetization,std_magnetization,action,std_action,U_L,std_U_L,chi_aquared,std_chi_aquared,tau_action,tau_magnetization = quick_eval_2D(path = p,magnetizations=I.Magnetization,actions=I.Actions,l = args.l,dt_max =args.dt_max,fs = I.fs,n_reps = args.n_reps)

                #Add the results to the summary file
                np.savetxt(file,np.array([[kappe_action,magnetization,std_magnetization,action,std_action,U_L,std_U_L,chi_aquared,std_chi_aquared,tau_action,tau_magnetization]]))
        file.close()

    ##################################################################################################################################
    #plot the evaluated data
    ##################################################################################################################################

    #initialize lists to store the recorded data
    chi_squares = []
    labels_chi_squares = []
    labels_tau = [[],[]]
    
    ULs = []
    kappas = []
    ms = []
    actions = []
    taus = [[],[]]

    for lambda_action in lambdas_original:
        lambda_action = round(lambda_action,6)

        #Get the summary file for this quadric coupling
        data = np.loadtxt(path + f"summary_lambda_{lambda_action}_{args.n_reps}.txt",skiprows = 1)
        
        #Store the susceptibility
        chi_squares.append([data[:,7],data[:,8]])
        labels_chi_squares.append(r"$\lambda = $" + f"{lambda_action}")

        #Store the Binder cumulant
        ULs.append([data[:,5],data[:,6]])

        #Store the hopping parameters
        kappas.append(data[:,0])

        #store the absolute mean magnetization
        ms.append([np.abs(data[:,1]),data[:,2]])

        #store the action
        actions.append([data[:,3],data[:,4]])

        #Store the correlation times
        taus[0].append(data[:,9])
        taus[1].append(data[:,10])

        labels_tau[0].append(r"$\tau_a$"+"; "+r"$\lambda$"+f" = {lambda_action}")
        labels_tau[1].append(r"$\tau_m$"+"; "+r"$\lambda$"+f" = {lambda_action}")

    #Plot it
    plotter_results(
        n_reps = args.n_reps,
        U_Ls = ULs,
        labels_U_Ls = labels_chi_squares,
        target_path = path,
        chi_squares = chi_squares,
        ms = ms,
        action = actions,
        taus = taus,
        kappas = kappas,
        kappas_taus = [kappas,kappas],
        labels_chi_squares = labels_chi_squares,
        labels_ms = labels_chi_squares,
        labels_action = labels_chi_squares,
        labels_taus = labels_tau,
        fontname = "Times New Roman",
        n_correlation_times = 2,
        reference = False,
        fs = 40,
        title = f"N = {N}")