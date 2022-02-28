from DataSet import DataSetIsing2D
from models import ConvNet5x5,ConvNet_multi_Version_1,ConvNet20x20
import torch
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_states
import os
import argparse
from datetime import date
import shutil
import json

class Trainer_Maximum_Likelihood_Discrete():
    def __init__(self,mode,ModelClass,device,record_simulation = None,max_samples = None,save_state_freq = None,n_correlation_times = None,N = None,T = None,run_name = None,load_path = None,n_epoch_state_dict = None,n_epochs = None,n_sweeps_sample = None,lmbd = None,init_mode = None,ModelParams = None,lr = None,OptimizerClass = None,buffer_size = None,batch_size = None,freq_lr_decay = None,lr_decay_factor = None,sampler_mode = None,fs = 40,k = 1):
        '''
        parameters:
            general parameters:
                mode:                       Training or simulation
                ModelClass:                 Class used as energy model
                device:                     Device where the training / simulation runs
                init_mode:                  Initialization of the initial state in training and simulation
                sampler_mode:               sampler used to generate samples following the learned energy model
                fs:                         Fontsize for plotting
                k:                          Boltzmann constant

            parameters for training:
                N:                          Number of spins per row / column of the lattice
                T:                          Temperature
                run_name:                   Name of the training run
                n_epochs:                   Number of epochs during the training
                n_sweeps_sample:            Number of update steps in the MCM during the training measured in sweeps (1 sweep = N^2)
                lmbd:                       Energy regularization strenght
                ModelParams:                Parameters used to initializ the energy model
                lr:                         Learning rate
                OptimizerClass:             Optimizer used in the training       
                buffer_size:                Size of the buffer for persistent initialization
                batch_size:                 Batch size used during the training
                lr_decay_factor:            Factor to reduce the learning
                freq_lr_decay:              Frequence of reducing the learning rate
                max_samples:                Number of samples used to create the data set

            parameters for the simulation:
                load_path:                  Parent folder of the training used in the simulation
                n_epoch_state_dict:         Epoch number of the state dict used in the simulation
        '''

        self.mode = mode

        #Case 1: Train a new model
        if mode == "TRAIN":
            self.N = N 
            self.T = T
            self.lmbd = lmbd
            self.device = device
            self.n_epochs = n_epochs
            self.ModelClass = ModelClass
            self.ModelParams = ModelParams
            self.lr = lr
            self.OptimizerClass = OptimizerClass
            self.batch_size = batch_size
            self.buffer_size = buffer_size
            self.init_mode = init_mode
            self.fs = fs
            self.sampler_mode = sampler_mode
            self.lr_decay_factor = lr_decay_factor
            self.k = k
            self.n_sweeps_sample = n_sweeps_sample
            self.freq_lr_decay = freq_lr_decay
            self.n_correlation_times = n_correlation_times
            self.max_samples = max_samples

            #Origine of the data set
            self.path_data = f"./Discrete_Ising_Model/N_{N}_Metropolis_Data_Set/N_{N}_T_{T}/"

            #Targte folder to store the results of the run
            self.path_target = f"./Discrete_Ising_Model/Results/N_{N}_T_{T}_{sampler_mode}_{date.today()}_{run_name}/"

            #Create subfolders
            subfolders = ["Models","Data","Images","Code"]

            for folder in subfolders:
                os.makedirs(self.path_target+folder)

            #Copy the code used 
            files_to_save = ["DataSet.py","models.py","Training.py","utils.py","Discrete_Ising_Model_Metropolis.py"]

            for file in files_to_save:
                shutil.copy(file,self.path_target+"Code/copy_"+file)

            #Save the configuration of the training 
            d = {
                "N":self.N,
                "T":self.T,
                "n_epochs":self.n_epochs,
                "n_sweeps_sample":n_sweeps_sample,
                "lmbd":self.lmbd,
                "init_mode":self.init_mode,
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
                "k":k,
                "fs":self.fs,
                "freq_lr_decay":freq_lr_decay,
                "n_correlation_times":n_correlation_times,
                "max_samples":max_samples
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
            self.T = config["T"]
            self.k = config["k"]
            self.ModelParams = config["ModelParams"]
            self.device = device
            self.n_epoch_state_dict = n_epoch_state_dict
            self.init_mode = config["init_mode"]
            self.fs = config["fs"]
            self.sampler_mode = config["sampler_mode"]
            self.record_simulation = record_simulation
            self.save_state_freq = save_state_freq
        
            #initialize the energy function
            self.E = ModelClass(**self.ModelParams)
            
            #Load the state dict
            self.E.load_state_dict(torch.load(self.path_target + f"Models/state_dict_epoch_{n_epoch_state_dict}.pt",map_location=torch.device(self.device)))
            self.E.to(self.device)
            self.E.eval()

        else: raise NotImplementedError()
        #Sampler dict
        self.sampler_dict = {"Metropolis":self.sampler_metropolis}
        self.beta = 1 / (self.k * self.T)

    def sampler_metropolis(self,mu):
        '''
        Perform MCMC steps using the Metropolis algorithm to update a given initial state.

        parameters:
            mu:     Initial state.

        returns:
            mu:     Updated state.
        '''
        if self.mode == "SIMULATE":
            energies = torch.zeros([len(mu),self.n_iter]).to(self.device)
            magnetizations = torch.zeros([len(mu),self.n_iter])
            pbar = tqdm.tqdm(total=self.n_iter)

            if self.record_simulation:
                recorded_states = torch.zeros([len(mu),self.n_iter // self.save_state_freq + 1,1,self.N,self.N]).to(self.device)
                counter = 0
                recorded_states[:,counter] = mu
                counter += 1

        #Get the energies of the initial states in the batch
        E_mu = self.E(mu)

        #Training 
        if self.mode == "TRAIN":
            betas = self.beta * torch.ones(len(mu)).to(self.device)

        #Simulation:
        elif self.mode == "SIMULATE":
            betas = 1 / (self.T * self.k)

        else:
            raise NotImplementedError

        for k in range(self.n_iter):
            #Select a spin to flip in each state of the batch
            pos = torch.randint(0,self.N,[2]).to(self.device)
            
            #Flip the spins to get new states
            v = mu.clone()
            v[:,0,pos[0],pos[1]] *= -1

            #Get the energies of the new states
            E_v = self.E(v)

            #Get the energy difference
            dE = E_v - E_mu

            #accept the new state if the energy difference is negative
            mask = (dE <= 0)
            
            #Get the acceptance ratios for the other states with positive Energy difference
            indices = torch.arange(len(mu))[mask == False]
            
            if len(indices) > 0:
                
                #Get the acceptance ratios:
                A = torch.exp(-dE[indices] * betas[indices]).to(self.device)
                
                #Get the value to compar
                r = torch.rand_like(A)

                mask_accept = (r < A)
                mask[indices] = mask_accept

            #Get the new state
            mu = torch.where(mask[:,None,None,None] == True,v,mu)
            E_mu = torch.where(mask == True,E_v,E_mu)

            if self.mode == "SIMULATE":
                energies[:,k] = E_mu
                magnetizations[:,k] = torch.abs(torch.sum(mu,dim = (1,2,3)))
                pbar.update(1)

                #save the states
                if (self.record_simulation == True) and (k % self.save_state_freq == 0):
                    recorded_states[:,counter] = mu
                    counter += 1

        if self.mode == "SIMULATE":
            if self.record_simulation:
                return energies,magnetizations,recorded_states.detach()
            else:
                return energies,magnetizations
        
        else:
            return mu.detach(),E_mu.detach()

    def __call__(self):
        '''
        Perform the training of the energy function.
        '''

        #Get the number of iterations in the Data Generation
        if self.sampler_mode == "Metropolis":
            self.n_iter = self.N * self.N * self.n_sweeps_sample
        
        else:
            raise NotImplementedError()

        #initialize the model
        self.E = self.ModelClass(**self.ModelParams)
        self.E.to(self.device)

        #Save the mode befor training
        torch.save(self.E.state_dict(),self.path_target + f"Models/state_dict_epoch_{0}.pt")

        #initialize the optimizer
        optimizer = self.OptimizerClass(params = self.E.parameters(),lr = self.lr)

        #Get the data set
        DS = DataSetIsing2D(path = self.path_data,n_correlation_times=self.n_correlation_times,max_samples=self.max_samples)
        DL = DataLoader(dataset=DS,batch_size=self.batch_size,shuffle=True)

        #Initialize the buffer for persistent initialization
        if self.init_mode == "zero":
            buffer = (torch.ones([self.buffer_size,1,self.N,self.N]) * (-1)**torch.randint(0,2,[1])).to(self.device)

        elif self.init_mode == "inf":
            buffer = (torch.ones([self.buffer_size,1,self.N,self.N]) * (-1)**torch.randint(0,2,[self.buffer_size,1,self.N,self.N])).to(self.device)

        else:
            raise NotImplementedError()

        #Get the sampler used to draw samples from the learned energy function
        sampler = self.sampler_dict[self.sampler_mode]

        counter = 0
        losses = torch.zeros(250).to(self.device)

        self.beta = 1 / (self.k * self.T)

        #Train the model
        for e in tqdm.tqdm(range(self.n_epochs)):
            for i,batch in enumerate(DL):

                #Get samples from the buffer
                buffer_indices = torch.permute(torch.arange(len(buffer)),[0])[:len(batch)].to(self.device)
                mu = buffer[buffer_indices]

                with torch.no_grad():
                    #Perform MCMC to update the samples from the buffer
                    mu_new,E_mu = sampler(mu = mu)

                #Get the loss
                F_mu = self.E(mu_new) * self.beta
                F_batch = self.E(batch) * self.beta

                loss =  F_batch.mean() - F_mu.mean() + self.lmbd *(F_batch.pow(2).mean() + F_mu.pow(2).mean())

                #Update the model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Update the buffer
                buffer[buffer_indices] = mu_new

                #save the losse
                losses[counter] = loss.detach()
                counter += 1

                #dump the loss
                if counter == len(losses):
                    counter = 0
                    with open(self.path_target+"Data/Loss.txt","a") as file:
                        np.savetxt(file,losses.detach().cpu().numpy())
                    file.close()
                    losses = torch.zeros(250).to(self.device)

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

            #Plot samples from the latest batch of training samples

            n = int(np.sqrt(len(batch)))
            u = min(n,8)
            indices = np.random.permutation(len(batch))[:u*u]


            plot_states(states=batch[indices],rows = u,cols=u,path = self.path_target+f"Images/training_states_epoch_{e+1}.jpg",N = self.N)

            #Plot some of the latest generated samples
            plot_states(states=mu_new[indices],rows = u,cols=u,path = self.path_target+f"Images/sampled_states_epoch_{e+1}.jpg",N = self.N)

            #Save the model
            torch.save(self.E.state_dict(),self.path_target + f"Models/state_dict_epoch_{e+1}.pt")

            #Update the learning rate
            if (e % self.freq_lr_decay == 0) and (e != 0):
                optimizer.param_groups[0]['lr'] *= self.lr_decay_factor
                print(f"lr changed to {optimizer.param_groups[0]['lr']}")

    def simulation(self,n_iterations,T,date,N): 
        '''
        Use the trained energy to runa a simulation and record the energies and the magnetizations.

        parameters:
            n_iterations:        Number of iterations in the simulation.
            date:                Time stamp for the simulation
            N:                   Number of spins per row and column
            T:                   Temperatures

        returns:
            None
        '''

        B = T.shape[0]
        self.N = N

        energies = torch.zeros(B,n_iterations).to(self.device)
        magnetization = torch.zeros(B,n_iterations).to(self.device)

        self.T = T.to(self.device)

        #Get the initial sample
        if self.init_mode == "zero": mu = (torch.ones([B,1,self.N,self.N]) * (-1)**torch.randint(0,2,[1])).to(self.device)
        elif self.init_mode == "inf": mu = (torch.ones([B,1,self.N,self.N]) * (-1)**torch.randint(0,2,[B,1,self.N,self.N])).to(self.device)
        else: raise NotImplementedError()

        #Get the sampler
        sampler = self.sampler_dict[self.sampler_mode]

        #Perform only one Metropolis step
        self.n_iter = n_iterations

        if self.record_simulation:
            with torch.no_grad():
                energies,magnetization,states = sampler(mu)

        else:
            with torch.no_grad():
                energies,magnetization = sampler(mu)

        for i in range(B):
            temp = round(self.T[i].item(),7)
            #Create the folder to store the results of the simulation
            os.makedirs(self.path_target + f"Simulations/"+date+f"_N_{self.N}/{date}_simulation_epoch_{self.n_epoch_state_dict}_T_{temp}/")
            
            #Create a file to store the information about the longrun
            info = {
                "N":self.N,
                "T":temp,
                "n_iterations":n_iterations,
                "n_epoch_state_dict":self.n_epoch_state_dict,
                "k":self.k,
                "save_state_freq":self.save_state_freq
            }

            with open(self.path_target + f"Simulations/"+date+f"_N_{self.N}/{date}_simulation_epoch_{self.n_epoch_state_dict}_T_{temp}/info.json","w") as file:
                json.dump(info,file)
            file.close()

            e = torch.tensor(energies[i,:].detach().cpu().numpy())
            m = torch.tensor(magnetization[i,:].detach().cpu().numpy())

            #Save the recorded energies and magnetizations
            torch.save(e,self.path_target + f"Simulations/"+date+f"_N_{self.N}/{date}_simulation_epoch_{self.n_epoch_state_dict}_T_{temp}/"+"Energies.pt")
            torch.save(m,self.path_target + f"Simulations/"+date+f"_N_{self.N}/{date}_simulation_epoch_{self.n_epoch_state_dict}_T_{temp}/"+"/Magnetization.pt")

            #save the recorded states 
            if self.record_simulation == True:
                s = torch.tensor(states[i,:].detach().cpu().numpy())[:-1]
                torch.save(s,self.path_target + f"Simulations/"+date+f"_N_{self.N}/{date}_simulation_epoch_{self.n_epoch_state_dict}_T_{temp}/"+"states.pt")
                indices = np.random.permutation(np.arange(len(states[i]) //2,len(states[i])))[:25]
                plot_states(states[i][indices],5,5,self.path_target + f"Simulations/"+date+f"_N_{self.N}/{date}_simulation_epoch_{self.n_epoch_state_dict}_T_{temp}/"+"states.jpg",N = self.N)

    def log_p_EBM(self,X,T,Z = 1):
        '''
        Compute the logarithm of the density defined by the energy function

        parameters:
            X:      States for whcih the density is evaluated
            T:      Temperature at which the density is evaluated
            Z:      Partition function

        returns:
            p_x:    Density of X normalized using Z
        '''

        p_x = - self.E(X) / T - np.log(Z)

        return p_x.detach()

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()

    #Arguments needed ingeneral
    my_parser.add_argument("--mode",                    type=str,   action = "store",                                                       required=True       ,help="Train a model if 'TRAIN' or run a simulation if 'SIMULATION'")
    my_parser.add_argument("--ModelClass",              type=str,   action = "store",   default="ConvNet5x5",                               required=False      ,help = "Class used as energy function.\tdefault: ConvNet20x20")
    my_parser.add_argument("--N",                       type=int,   action = "store",   default=5,                                          required=False      ,help = "Number of spins per row / column of the lattice.\tdefault: 5")
    my_parser.add_argument("--T_min",                   type=float, action = "store",   default=2.5,                                        required=False      ,help = "Smallest temperature for which a model is trained.\tdefault: 2.5")
    my_parser.add_argument("--T_max",                   type=float, action = "store",   default=3.0,                                        required=False      ,help = "Biggest temperature for which a model is trained.\tdefault: 3.0")
    my_parser.add_argument("--dT",                      type=float, action = "store",   default=0.5,                                        required=False      ,help = "Stepsize of the temperature.\tdefault: 0.5")
    my_parser.add_argument("--name",                    type=str,   action = "store",   default="Test_1",   	                            required=False      ,help = "Name of the training run.\tdefault: Test_1")
    my_parser.add_argument("--n_epochs",                type=int,   action = "store",   default=30,                                         required=False      ,help = "Number of epochs during the training.\tdefault: 30")
    my_parser.add_argument("--n_sweeps_sample",         type=int,   action = "store",   default=5,                                          required=False      ,help = "Number of iterations in the MCMC to update the samples measured in N^2.\tdefault: 5")
    my_parser.add_argument("--lmbd",                    type=float, action = "store",   default=0.0,                                        required=False      ,help = "Strenght of the Energy regularization.\tdefault: 0.0")
    my_parser.add_argument("--init_mode",               type=str,   action = "store",   default="inf",                                      required=False      ,help = "Init mode of the samples in the buffer.\tdefault: inf")
    my_parser.add_argument("--ModelParams",             type=dict,  action = "store",   default={"in_channels" : 1,"out_channels" : 64},    required=False      ,help = "Parameters of the energy function.\tdefault: {'in_channels' : 1,'out_channels' : 32}")
    my_parser.add_argument("--lr",                      type=float, action = "store",   default=1e-2,                                       required=False      ,help = "Initial learning rate.\tdefault: 0.02")
    my_parser.add_argument("--OptimizerClass",          type=str,   action = "store",   default="ADAM",                                     required=False      ,help = "Optimizer used to update the model parameters.\tdefault: ADAM")
    my_parser.add_argument("--buffer_size",             type=int,   action = "store",   default=10000,                                      required=False      ,help = "Size of the buffer.\tdefault: 10000")
    my_parser.add_argument("--batch_size",              type=int,   action = "store",   default=128,                                        required=False      ,help = "Batch size.\tdefault: 128")
    my_parser.add_argument("--lr_decay_factor",         type=float, action = "store",   default = 0.1,                                      required=False      ,help = "Factor to multiply the learning rate after eachh epoch with.\tdefault: 0.1")
    my_parser.add_argument("--sampler_mode",            type=str,   action = "store",   default = "Metropolis",                             required=False      ,help = "Method used to sample from the learned Energy function.\tdefault: Metropolis")
    my_parser.add_argument("--freq_lr_decay",           type=int,   action = "store",   default = 10,                                       required=False      ,help = "Frequence of reducing the learning rate.\tdefault: 10")
    my_parser.add_argument("--n_correlation_times",     type=int,   action = "store",   default = 2,                                        required=False      ,help = "Number of correlation times between samples used in the data set.\tdefault: 2")
    my_parser.add_argument("--n_iteration",             type=int,   action = "store",   default = int(5e5),                                 required=False      ,help = "Number of iterations during the simulation.\tdefault: 500000")
    my_parser.add_argument("--n_epoch_state_dict",      type=int,   action = "store",   default = 0,                                        required=False      ,help = "Epoch for which the state dict is loaded.\tdefault: 0")
    my_parser.add_argument("--load_path",               type=str,   action = "store",   default = "Test",                                   required=False      ,help = "Path where the trained model is stored.\tdefault: Test")
    my_parser.add_argument("--record_simulation",       type=int,   action = "store",  default = 0,                                         required=False      ,help = "Save the states during the simulation.\tdefault: 0")
    my_parser.add_argument("--save_state_freq",         type=int,   action = "store",  default = 25,                                        required=False      ,help = "Frequency of saving states.\tdefault: 25")
    my_parser.add_argument("--max_samples",             type=int,   action = "store",  default = 10000,                                     required=False      ,help = "Number of samples used in the training set.\tdefault: 25")

    args = my_parser.parse_args()

    model_dict = {
        "ConvNet5x5":ConvNet5x5,
        "ConvNet_multi_Version_1":ConvNet_multi_Version_1,
        "ConvNet20x20":ConvNet20x20
        }

    optimizer_dict = {
        "ADAM":torch.optim.Adam,
        "SGD":torch.optim.SGD
        }
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.mode == "TRAIN":

        Temperatures = np.arange(args.T_min,args.T_max,args.dT)

        for temp in Temperatures:
            #Train a model
            T = Trainer_Maximum_Likelihood_Discrete(
                mode =                  "TRAIN",
                N =                     args.N,
                T =                     round(temp,6),
                ModelClass =            model_dict[args.ModelClass],
                device =                device,
                run_name =              args.name,
                n_epochs =              args.n_epochs,
                n_sweeps_sample =       args.n_sweeps_sample,
                lmbd =                  args.lmbd,
                init_mode =             args.init_mode,
                ModelParams =           args.ModelParams,
                lr =                    args.lr,
                OptimizerClass =        optimizer_dict[args.OptimizerClass],
                buffer_size =           args.buffer_size,
                batch_size =            args.batch_size,
                lr_decay_factor =       args.lr_decay_factor,
                sampler_mode =          args.sampler_mode,
                freq_lr_decay=          args.freq_lr_decay,
                n_correlation_times=    args.n_correlation_times,
                max_samples=            args.max_samples
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
            record_simulation = bool(args.record_simulation),
            save_state_freq = args.save_state_freq
        )

        T.simulation(n_iterations = args.n_iteration)