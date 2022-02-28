from DataSet import DataSetScalar2D
from models import LinearCondition5x5_one_condition
import torch
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from datetime import date
import shutil
import json
from utils import plot_states


class Buffer():
    def __init__(self,buffer_size,kappas_action,lambdas_action,N,device):
        '''
        parameters:
            buffer_size:        Number ssamples stored in the buffer
            kappas_action:      Hopping parametrs that are used.
            lambdas_action:     Quadric cuóuplings that are used.
            N:                  Numer of spins per row and column
            device:             Device on which the training runs.
        '''

        #Storage for the states
        self.states_storage = torch.rand([0,1,N,N])

        #Store the used parameters
        self.kappas_action = kappas_action
        self.lambdas_action = lambdas_action

        self.device = device
        self.buffer_size = buffer_size
        self.N = N

    def get(self,batch_size):
        '''
        Get a random batch form the buffer

        parameters:
            batch_size:     Size of the batch randomly drawn from the buffer

        returns:
            indices:                            Position of the samples of the batch in the buffer.
            self.kappas_storage[indices]:       Hopping parameters of the states in the batch.
            self.lambdas_storage[indices]:      Quadric couplings of the states in the batch.
            self.states_storage[indices]:       States of the bacth.
        '''

        #Fill the buffer with random states if teh buffer is called for the first time.
        if len(self.states_storage) == 0:
            #uniform initialization of the initail states mu \in [-1,1]^(N x N)
            self.states_storage = torch.rand([self.buffer_size,1,self.N,self.N]) * 2 - 1

            #Get a random combination of the parameters
            indices_lambda = np.random.randint(low = 0, high = len(self.lambdas_action),size = [self.buffer_size])
            indices_kappa = np.random.randint(low = 0, high = len(self.kappas_action),size = [self.buffer_size])

            #self.kappas_storage = torch.zeros(self.buffer_size)
            #self.lambdas_storage = torch.zeros(self.buffer_size)

            self.kappas_storage = self.kappas_action[indices_kappa]
            self.lambdas_storage = self.lambdas_action[indices_lambda]

        #get samples
        indices = np.random.randint(low = 0,high = self.buffer_size,size = batch_size)

        return indices,self.kappas_storage[indices].to(self.device),self.lambdas_storage[indices].to(self.device),self.states_storage[indices].to(self.device)

    def update(self,indices,states):
        '''
        Update certain statesin the buffer.

        parameters:
            indices:       Position of the upadated states.
            states:        States used to replace the buffer states.

        returns:
            None
        '''
        #Update the states at the given indices
        self.states_storage[indices] = states.detach().cpu()

class Trainer_Maximum_Likelihood_Discrete():
    def __init__(self,mode,ModelClass,device,random_seed,n_rep_data_set = None,record_freq = None,record_states = None,n_taus = None,epsilon = None,max_samples = None,N = None,kappas_action = None,lambdas_action = None,run_name = None,alpha = None,load_path = None,n_epoch_state_dict = None,n_epochs = None,n_sweeps = None,ModelParams = None,lr = None,OptimizerClass = None,buffer_size = None,batch_size = None,freq_lr_decay = None,lr_decay_factor = None,sampler_mode = None,fs = 40):
        '''
        parameters:
            mode:                       Mode of the instance ("TRAIN" or "SIMULATE")
            n_rep_data_set:             Index of the training set that is used.
            ModelClass:                 Class used to model the action function.
            device:                     Device on which the training runs.
            record_freq:                Frequency of storing the states during the MCMC
            record_states:              Record the states that occure during the MCMC
            n_taus:                     Number of correlation times used to seperate samples in the training set
            epsilon:                    Steps size for Langevin sampling
            max_samples:                Number of samples in the trainig set per hopping parameter / quadric coupling combination.
            N:                          Number of spins per row and column
            kappas_action:              Hopping parameters used in the training set.
            lambdas_action:             Quadric couplings used in the training set.
            run_name:                   Name of the training.
            alpha:                      Regularization factor.
            load_path:                  Parent folder of the training used for the simulations.
            n_epoch_state_dict:         Epoch of the training used in the simulation.
            n_epochs:                   Number of epochs in the training
            n_sweeps:                   Number of sweeps during the generation of negative samples during the training.
            ModelParams:                Parameters used to initialize the model during the training.
            lr:                         Learning rate.
            OptimizerClass:             Optimizer used to update the model parameters during the training.
            buffer_size:                Size of the buffer used to initialize the negative samples during the training.
            batch_size:                 Batch size.
            freq_lr_decay:              Number of epochs between learning rate decay.
            lr_decay_factor:            Factor used to decrease th elearning rate.
            sampler_mode:               Sampler used to generate new states.
            fs:                         Fontsize for the plots.
        '''

        #Set the random seed
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        #Sampler dict
        self.sampler_dict = {
            #"METROPOLIS":self.metropolis_sampler,
            "LANGEVIN":self.langevin_sampler
            }

        self.mode = mode

        #Case 1: Train a new model
        if mode == "TRAIN":
            self.n_rep_data_set = n_rep_data_set
            self.N = N 
            self.alpha = alpha
            self.device = device
            self.n_epochs = n_epochs
            self.ModelClass = ModelClass
            self.lr = lr
            self.OptimizerClass = OptimizerClass
            self.batch_size = batch_size
            self.buffer_size = buffer_size
            self.fs = fs
            self.sampler_mode = sampler_mode
            self.lr_decay_factor = lr_decay_factor
            self.n_iter = n_sweeps * N**2
            self.freq_lr_decay = freq_lr_decay
            self.kappas_action = kappas_action
            self.lambdas_action = lambdas_action
            self.epsilon = epsilon
            self.max_samples = max_samples
            self.ModelParams = ModelParams
            self.n_taus = n_taus

            #Origine of the data set
            self.path_data = f"./Scalar_Theory/N_{self.N}_LANGEVIN_SPECIFIC_Data_Set/"

            #Targte folder to store the results of the run
            self.path_target = f"./Scalar_Theory/Results/N_{N}_{sampler_mode}_{date.today()}_{run_name}/"

            #Create subfolders
            subfolders = ["Models","Data","Images","Code"]

            for folder in subfolders:
                os.makedirs(self.path_target+folder)

            #Copy the code used during the training and the generation of the training data
            files_to_save = ["DataSet.py","models.py","Training_EBM.py","utils.py","Simulation_Scalar_Theory.py","eval_trained_models.py"]

            for file in files_to_save:
                shutil.copy(file,self.path_target+"Code/copy_"+file)

            #Save the configuration of the training 
            d = {
                "N":self.N,
                "n_epochs":self.n_epochs,
                "n_iter":self.n_iter,
                "alpha":self.alpha,
                "run_name":run_name,
                "ModelClass":str(self.ModelClass),
                "ModelParams":self.ModelParams,
                "lr":self.lr,
                "OptimizerClass":str(self.OptimizerClass),
                "buffer_size":self.buffer_size,
                "batch_size":self.batch_size,
                "lr_decay_factor":self.lr_decay_factor,
                "device":self.device,
                "sampler_mode":self.sampler_mode,
                "fs":self.fs,
                "freq_lr_decay":freq_lr_decay,
                "kappas_action":list(kappas_action.detach().numpy().astype(float)),
                "lambdas_action":list(lambdas_action.detach().numpy().astype(float)),
                "epsilon":epsilon,
                "max_samples":max_samples,
                "n_taus":self.n_taus,
                "random_seed":self.random_seed,
                "n_rep_data_set":self.n_rep_data_set
            }

            with open(self.path_target + "Code/config.json","w") as file:
                json.dump(d,file)
            file.close()

        #Load a trained model to run a simulation
        elif mode == "SIMULATE":
            self.path_target = load_path

            #Load the properties necessary fo the simulation
            with open(self.path_target+"Code/config.json", "r") as file:
                config = json.load(file)
            file.close()

            self.N = config["N"]
            self.ModelParams = config["ModelParams"]
            self.device = device
            self.n_epoch_state_dict = n_epoch_state_dict
            self.fs = config["fs"]
            self.sampler_mode = sampler_mode
            self.epsilon = config["epsilon"]
            self.record_freq = record_freq
            self.record_states = record_states

            #initialize the energy function
            self.S = ModelClass(**self.ModelParams)
            self.S.to(self.device)

            #Load the state dict
            self.S.load_state_dict(torch.load(self.path_target + f"Models/state_dict_epoch_{n_epoch_state_dict}.pt",map_location=torch.device(self.device)))
            self.S.eval()

        else: raise NotImplementedError()

    def metropolis_sampler(self,n_iter,mu,kappas,lambdas):
        '''
        Metropolis sampling with gaussian proposal states.

        parameters:
            n_iter:     Number of update steps
            mu:         States used to initialze the MCMC
            kappas:     Tensor containing the hopping parameters of the states stored in mu
            lambdas:    Tensor containing the quadric couplings of the states stored in mu

        returns:
            mu:         States after the final MCMC step.
            S_mu:       Actions of the states stored in mu.

        
        '''

        #Get the actions of the initial states
        S_mu = self.S(X = mu, cond_1 = kappas,cond_2 = lambdas)

        for k in range(n_iter):
            #Get proposal states
            v = mu + np.sqrt(2 * self.epsilon) * torch.randn_like(mu)

            #Ge the action for the new proposal states
            S_v = self.S(X = v, cond_1 = kappas,cond_2 = lambdas)

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

            return mu,S_mu

    def langevin_sampler(self,n_iter,mu,kappas,lambdas):
        '''
        Sample from the Boltzmann distribution using Langevin Dynamics 

        parameters:
            n_iter:     Number of update steps
            mu:         States used to initialze the MCMC
            kappas:     Tensor containing the hopping parameters of the states stored in mu
            lambdas:    Tensor containing the quadric couplings of the states stored in mu

        returns:
            TRAINING:
                mu:         States after the final MCMC step.
                S_mu:       Actions of the states stored in mu.

            SIMULATE:
                actions:            Actions of all states that occured in the MCMC
                magnetizations:     Magnetizations of all states that occured in the MCMC
        '''

        mu.requires_grad_(True)

        if self.mode == "SIMULATE":
            actions = torch.zeros([len(mu),n_iter]).to(self.device)
            magnetizations = torch.zeros([len(mu),n_iter]).to(self.device)
            prog_bar = tqdm.tqdm(total = n_iter)

        for k in range(n_iter):

            mu.requires_grad_(True)

            S_mu = self.S(X = mu,cond_1 = kappas,cond_2 = lambdas)

            g = torch.autograd.grad(outputs = S_mu.sum(),inputs = [mu])[0]

            mu.data += -self.epsilon * g + np.sqrt(2 * self.epsilon) * torch.randn_like(mu).to(self.device)

            mu = mu.detach()

            if self.mode == "SIMULATE":

                actions[:,k] = S_mu.detach()
                magnetizations[:,k].add_(mu.sum(dim = (1,2,3)))# = mu.sum(dim = (1,2,3))

                prog_bar.update(1)

                if (self.record_states == True) and (k % self.record_freq == 0):
                    self.states[:,self.counter] = mu
                    self.counter += 1

        #Simulation
        if self.mode == "SIMULATE":
            return actions,magnetizations

        #Training
        else: return mu.detach(), S_mu

    def __call__(self):
        '''
        Training of the action function.
        '''

        #initialize the model
        self.S = self.ModelClass(**self.ModelParams)
        self.S.to(self.device)

        #Save the mode befor training
        torch.save(self.S.state_dict(),self.path_target + f"Models/state_dict_epoch_{0}.pt")

        #initialize the optimizer
        optimizer = self.OptimizerClass(params = self.S.parameters(),lr = self.lr)

        #Get the data set
        DS = DataSetScalar2D(path = self.path_data,kappas = self.kappas_action,lambdas = self.lambdas_action,max_samples = self.max_samples,N = self.N,n_taus = self.n_taus,n_reps_data_set = self.n_rep_data_set)
        DL = DataLoader(dataset=DS,batch_size=self.batch_size,shuffle=True)

        #Initialize the buffer used to initialze the negative samples
        buffer = Buffer(buffer_size = self.buffer_size,kappas_action = self.kappas_action,lambdas_action = self.lambdas_action,N = self.N,device = self.device)

        #Get the sampler used to draw samples from the learned energy function
        sampler = self.sampler_dict[self.sampler_mode]

        #Tensor to storre the losses
        counter = 0
        loss_tensor_len = 250
        losses = torch.zeros(loss_tensor_len).to(self.device)

        #Train the model
        for e in tqdm.tqdm(range(self.n_epochs)):

            #Progressbar for progress of the current epoch
            prog_bar = tqdm.tqdm(total = int(8 * self.max_samples * len(self.kappas_action) / self.batch_size+1))

            for i,(kappas_pos,lambdas_pos,states_pos) in enumerate(DL):
                
                #Get negative samples from the buffer
                indices_neg,kappas_neg,lambdas_neg,states_neg_initial = buffer.get(self.batch_size)

                #Get the updated states
                states_neg,S_neg = sampler(n_iter = self.n_iter,mu = states_neg_initial,kappas = kappas_neg,lambdas = lambdas_neg)

                #Get the loss
                F_neg = self.S(X = states_neg,cond_1 = kappas_neg,cond_2 = lambdas_neg)
                F_pos = self.S(X = states_pos,cond_1 = kappas_pos,cond_2 = lambdas_pos)

                loss =  F_pos.mean() - F_neg.mean() + self.alpha *(F_pos.pow(2).mean() + F_neg.pow(2).mean())

                #Update the model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Save the loss
                losses[counter] = loss
                counter += 1

                buffer.update(indices = indices_neg,states = states_neg)

                #dump the loss
                if counter == len(losses):
                    counter = 0
                    with open(self.path_target+"Data/Loss.txt","a") as file:
                        np.savetxt(file,losses.detach().cpu().numpy())
                    file.close()
                    losses = torch.zeros(loss_tensor_len).to(self.device)

                prog_bar.update(1)

            #Plot the states
            indices_neg_plot = np.random.permutation(len(states_neg))[:25]
            indices_pos_plot = np.random.permutation(len(states_pos))[:25]

            plot_states(states = states_neg[indices_neg_plot],rows = 5,cols = 5,path = self.path_target+f"Images/sampled_states_epoch_{e+1}.jpg",titles = [r"$\kappa$" + f" = {round(kappas_neg[j].item(),5)}; "+r"$\lambda$" + f" = {round(lambdas_neg[j].item(),5)}" for j in indices_neg_plot],N = self.N)
            plot_states(states = states_pos[indices_pos_plot],rows = 5,cols = 5,path = self.path_target+f"Images/training_states_epoch_{e+1}.jpg",titles = [r"$\kappa$" + f" = {round(kappas_pos[j].item(),5)}; "+r"$\lambda$" + f" = {round(lambdas_pos[j].item(),5)}" for j in indices_pos_plot],N = self.N)

            #plot the loss
            recorded_loss = np.loadtxt(self.path_target+"Data/Loss.txt")
            plt.figure(figsize = (30,15))
            plt.plot(recorded_loss)
            plt.xlabel("training iteration",fontsize = self.fs)
            plt.ylabel("loss",fontsize = self.fs)
            plt.xticks(fontsize = self.fs)
            plt.yticks(fontsize = self.fs)
            plt.savefig(self.path_target+"Images/loss.jpg")
            plt.close()

            #Save the model
            torch.save(self.S.state_dict(),self.path_target + f"Models/state_dict_epoch_{e+1}.pt")

            #Update the learning rate
            if (e % self.freq_lr_decay == 0) and (e != 0):
                optimizer.param_groups[0]['lr'] *= self.lr_decay_factor
                print(f"lr changed to {optimizer.param_groups[0]['lr']}")

    def simulation(self,n_iterations,date,kappas_action,lambdas_action,n_reps,n_parallel_sim = None): 
        '''
        parameters:
            n_iterations:        Number of iterations in the simulation.
            date:                Time stamp for the simulation
            kappas_action:       Hopping parameters used in the simulations
            lambdas_action:      Quadric couplings used in the simulations
            n_reps:              Index of this simulation / start index of the simulations that run in parallel
            n_parallel_sim:      Number of simulations running in parallel

        returns:
            None
        '''

        B = kappas_action.shape[0]

        #Get the initial states
        mu = (torch.rand([B,1,self.N,self.N]) * 2 - 1).to(self.device)

        #Get the sampler
        sampler = self.sampler_dict[self.sampler_mode]

        #Storage for the recorded states
        if self.record_states:
            self.counter = 0
            self.states = torch.zeros([B,n_iterations // self.record_freq + 1,1,self.N,self.N]).to(self.device)

        actions,magnetization = sampler(n_iter = n_iterations,mu = mu,kappas = kappas_action.to(self.device),lambdas = lambdas_action.to(self.device))

        if n_parallel_sim is None:
            print("Normal mode")

            for i in range(B):
                kappa = round(kappas_action[i].item(),7)
                lambd =  round(lambdas_action[i].item(),7)

                #Create the folder to store the results of the simulation
                if not os.path.exists(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"):
                    os.makedirs(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/")
                
                #Create a file to store the information about the longrun
                info = {
                    "N":self.N,
                    "kappa_action":kappa,
                    "lambda_action":lambd,
                    "n_iterations":n_iterations,
                    "n_epoch_state_dict":self.n_epoch_state_dict,
                    "record_states":self.record_states,
                    "freq_save_samples":self.record_freq,
                    "random_seed":self.random_seed
                }

                with open(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/info_{n_reps}.json","w") as file:
                    json.dump(info,file)
                file.close()

                a = torch.tensor(actions[i,:].detach().cpu().numpy())
                m = torch.tensor(magnetization[i,:].detach().cpu().numpy())

                #Save the recorded energies and magnetizations
                torch.save(a,self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"+f"Actions_{n_reps}.pt")
                torch.save(m,self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"+f"/Magnetization_{n_reps}.pt")

                #Create the folder for plots
                if not os.path.exists(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/Plt/"):
                    os.makedirs(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/Plt/")

                #Save the date set
                if self.record_states == True:
                    current_states = torch.tensor(self.states[i,:].detach().cpu().numpy())[:-1]
                    torch.save(current_states,self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"+f"states_{n_reps}.pt")
                    indices = np.random.permutation(np.arange(len(current_states) // 2,len(current_states)))[:25]
                    plot_states(current_states[indices],5,5,self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"+f"Plt/states_{n_reps}.jpg",N = self.N,titles=["" for j in range(len(indices))])

        else:

            print("Parallel mode")

            B = int(len(kappas_action) / n_parallel_sim)

            for j in range(n_parallel_sim):

                current_kappas = kappas_action[j * B : (j+1) * B]
                current_lambdas = lambdas_action[j * B : (j+1) * B]

                current_actions = actions[j * B : (j+1) * B]
                current_magnetizations = magnetization[j * B : (j+1) * B]
                current_states = self.states[j * B : (j+1) * B]

                for i in range(B):

                    kappa = round(current_kappas[i].item(),7)
                    lambd =  round(current_lambdas[i].item(),7)

                    #Create the folder to store the results of the simulation
                    if not os.path.exists(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"):
                        os.makedirs(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/")

                    info = {
                        "N":self.N,
                        "kappa_action":kappa,
                        "lambda_action":lambd,
                        "n_iterations":n_iterations,
                        "n_epoch_state_dict":self.n_epoch_state_dict,
                        "record_states":self.record_states,
                        "freq_save_samples":self.record_freq,
                        "random_seed":self.random_seed
                    }

                    with open(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/info_{n_reps + j}.json","w") as file:
                        json.dump(info,file)
                    file.close()

                    a = torch.tensor(current_actions[i,:].detach().cpu().numpy())
                    m = torch.tensor(current_magnetizations[i,:].detach().cpu().numpy())

                    #Save the recorded energies and magnetizations
                    torch.save(a,self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"+f"Actions_{n_reps + j}.pt")
                    torch.save(m,self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"+f"/Magnetization_{n_reps + j}.pt")

                    #Create the folder for plots
                    if not os.path.exists(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/Plt/"):
                        os.makedirs(self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/Plt/")

                    #Save the date set
                    if self.record_states == True:
                        save_states = torch.tensor(current_states[i,:].detach().cpu().numpy())[:-1]
                        torch.save(save_states,self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"+f"states_{n_reps + j}.pt")
                        indices = np.random.permutation(np.arange(len(save_states) // 2,len(save_states)))[:25]
                        plot_states(save_states[indices],5,5,self.path_target + f"Simulations/"+date+f"/epoch_{self.n_epoch_state_dict}_kappa_{kappa}_lambda_{lambd}/"+f"Plt/states_{n_reps + j}.jpg",N = self.N,titles=["" for j in range(len(indices))])
            
    def log_p_EBM(self,X,kappas,lambdas,Z = 1):
        log_p = - self.S(X = X,cond_1 = kappas,cond_2 = lambdas) - np.log(Z)
        return log_p.detach()

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()

    #Arguments needed ingeneral
    my_parser.add_argument("--mode",                        type=str,       action = "store",                                                           required=True       ,help = "Train a model if 'TRAIN' or run a simulation if 'SIMULATION'")
    my_parser.add_argument("--ModelClass",                  type=str,       action = "store",   default="LinearCondition5x5_one_condition",             required=False      ,help = "Class used to model the action function\tdefault: 'LinearCondition5x5_one_condition'")
    my_parser.add_argument("--kappa_min",                   type=float,     action = "store",   default = 0.2,                                          required=False      ,help = "Smallest hopping parameter used in the training set.\tdefault: 0.2")
    my_parser.add_argument("--kappa_max",                   type=float,     action = "store",   default = 0.4,                                          required=False      ,help = "Biggest hopping parameter used in the training set.\tdefault: 0.4")
    my_parser.add_argument("--dkappa",                      type=float,     action = "store",   default = 0.02,                                         required=False      ,help = "Step size of the hopping parameter\tdefault: 0.02")
    my_parser.add_argument("--lambda_min",                  type=float,     action = "store",   default = 0.02,                                         required=False      ,help = "Smallest quartic coupling used in the training set.\tdefault: 0.02")
    my_parser.add_argument("--lambda_max",                  type=float,     action = "store",   default = 0.03,                                         required=False      ,help = "Biggest quartic coupling  used in the training set.\tdefault: 0.03")
    my_parser.add_argument("--dlambda",                     type=float,     action = "store",   default = 0.02,                                         required=False      ,help = "Step size of the quartic coupling\tdefault: 0.02")
    my_parser.add_argument("--seed",                        type=int,       action = "store",   default = 47,                                           required=False      ,help = "Random seed\tdefault: 47")

    #Arguments needed for Training
    my_parser.add_argument("--N",                           type=int,       action = "store",   default=5,                                              required=False      ,help = "Number of spins per row / column of the lattice\tdefault: 5")
    my_parser.add_argument("--name",                        type=str,       action = "store",   default="Test_1",   	                                required=False      ,help = "Name of the training run.\tdefault: Test_1")
    my_parser.add_argument("--n_epochs",                    type=int,       action = "store",   default=30,                                             required=False      ,help = "Number of epochs during the training\tdefault: 30")
    my_parser.add_argument("--n_sweeps",                    type=int,       action = "store",   default=4,                                              required=False      ,help = "Number of iterations in the MCMC to update the samples measured in N^2\tdefault: 4")
    my_parser.add_argument("--alpha",                       type=float,     action = "store",   default=0.0,                                            required=False      ,help = "Strenght of the L2 regularization\tdefault: 0.0")
    my_parser.add_argument("--ModelParams",                 type=dict,      action = "store",   default={"in_channels" : 1,"out_channels" : 64},        required=False      ,help = "Parameters of the action function\tdefault: {'in_channels' : 1,'out_channels' : 64}")
    my_parser.add_argument("--lr",                          type=float,     action = "store",   default=1e-4,                                           required=False      ,help = "Initial learning rate\tdefault: 0.0001")
    my_parser.add_argument("--OptimizerClass",              type=str,       action = "store",   default="ADAM",                                         required=False      ,help = "Optimizer used to update the model parameters\tdefault: ADAM")
    my_parser.add_argument("--buffer_size",                 type=int,       action = "store",   default=10000,                                          required=False      ,help = "Size of the buffer used to initialze the negative samples\tdefault: 10000")
    my_parser.add_argument("--batch_size",                  type=int,       action = "store",   default=128,                                            required=False      ,help = "Batch size\tdefault: 128")
    my_parser.add_argument("--lr_decay_factor",             type=float,     action = "store",   default = 0.1,                                          required=False      ,help = "Factor used to decrease the learning rate\tdefault: 0.1")
    my_parser.add_argument("--sampler_mode",                type=str,       action = "store",   default = "LANGEVIN",                                   required=False      ,help = "Method used to sample from the learned Energy function\tdefault: LANGEVIN")
    my_parser.add_argument("--freq_lr_decay",               type=int,       action = "store",   default = 10,                                           required=False      ,help = "Frequence of reducing the learning rate\tdefault: 10")
    my_parser.add_argument("--epsilon",                     type=float,     action = "store",   default = 0.01,                                         required=False      ,help = "Step size used in Langevin dynamics\tdefault: 0.01")
    my_parser.add_argument("--max_samples",                 type=int,       action = "store",   default = 30000,                                        required=False      ,help = "Number of samples in the trainig set per hopping parameter / quadric coupling combination\tdefault: 30000")
    my_parser.add_argument("--n_taus",                      type=int,       action = "store",   default = 2,                                            required=False      ,help = "Number of correlation times used to seperate samples in the training set\tdefault: 2")
    my_parser.add_argument("--n_reps_data_set",             type=int,       action = "store",   default = 0,                                            required=False      ,help = "Index of the training set\tdefault: 0")

    #Arguments for simulation
    my_parser.add_argument("--n_iteration",                 type=int,       action = "store",   default = int(5e5),                                     required=False      ,help = "Number of iterations during the simulation\tdefault: 500000")
    my_parser.add_argument("--n_epoch_state_dict",          type=int,       action = "store",   default = 0,                                            required=False      ,help = "Epoch for which the state dict is loaded\tdefault: 0")
    my_parser.add_argument("--load_path",                   type=str,       action = "store",   default = "Test",                                       required=False      ,help = "Parent folder of the training used for the simulations\tdefault: 'Test'")
    my_parser.add_argument("--date",                        type=str,       action = "store",   default = "NoDateGiven",                                required=False      ,help = "Time stamp of the simulation\tdefault: 'NoDateGiven'")

    args = my_parser.parse_args()

    model_dict = {
        "LinearCondition5x5_one_condition":LinearCondition5x5_one_condition
        }

    optimizer_dict = {
        "ADAM":torch.optim.Adam,
        "SGD":torch.optim.SGD
        }
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    kappas_action = torch.arange(args.kappa_min,args.kappa_max,args.dkappa)
    lambdas_action = torch.arange(args.lambda_min,args.lambda_max,args.dlambda)

    if args.mode == "TRAIN":
        #Train a model
        T = Trainer_Maximum_Likelihood_Discrete(
            mode = args.mode,
            ModelClass = model_dict[args.ModelClass],
            device = device,
            epsilon = args.epsilon,
            max_samples = args.max_samples,
            N = args.N,
            kappas_action = kappas_action,
            lambdas_action = lambdas_action,
            run_name = args.name,
            alpha = args.alpha,
            n_epochs = args.n_epochs,
            n_sweeps = args.n_sweeps,
            ModelParams = args.ModelParams,
            lr = args.lr,
            OptimizerClass = optimizer_dict[args.OptimizerClass],
            buffer_size = args.buffer_size,
            batch_size = args.batch_size,
            freq_lr_decay = args.freq_lr_decay,
            lr_decay_factor = args.lr_decay_factor,
            sampler_mode = args.sampler_mode,
            n_taus = args.n_taus,
            random_seed=args.seed,
            n_rep_data_set = args.n_reps_data_set
            )

        T()

    elif args.mode == "SIMULATION":
        #Run a simulation
        T = Trainer_Maximum_Likelihood_Discrete(
            mode =                  "SIMULATE",
            ModelClass =            model_dict[args.ModelClass],
            load_path =             args.load_path,
            device =                device,
            n_epoch_state_dict =    args.n_epoch_state_dict,
            random_seed=            args.seed
        )

        T.simulation(n_iterations = args.n_iteration,date = args.date,kappa_action = kappas_action,lambdas_action=lambdas_action,n_reps = 0)