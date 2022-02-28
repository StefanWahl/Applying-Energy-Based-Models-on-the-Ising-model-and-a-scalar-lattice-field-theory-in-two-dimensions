from DataSet import DataSetScalar2D_INN_No_Condition
from models import fc_subnet
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
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from utils import bootstrap,get_susceptibility,get_U_L

class Trainer_INN():
    def __init__(self,mode,device,random_seed = None,data_path = None,load_path = None,n_epoch_state_dict = None,max_samples = None,SubNetFunctionINN = None,NumberCouplingBlocksINN = None,N = None,run_name = None,n_epochs = None,lr = None,OptimizerClass = None,batch_size = None,freq_lr_decay = None,lr_decay_factor = None,fs = 40):
        '''
        parameters:
            mode:                   TRAIN or SIMULATE
            device:                 Device on which the training runs
            random_seed:            Random sedd
            data_path:              Folder containing the states and the info file used for the data set
            load_path:              Parent folder of the training used for the the simulation
            n_epoch_state_dict:     Epoch of the INN used for the simulation
            max_samples:            Number of samples from the stored states used for the training set
            SubNetFunctionINN:      Constructor for the sub netzs used in the INN blocks
            N:                      Number of spins per row and column of the lattice
            run_name:               Name of the training run
            n_epochs:               Numer of epochs in the training
            lr:                     Initial learning rate 
            OptimizerClass:         Optimizer used to update the model parameters
            batch_size:             Batch size
            freq_lr_decay:          Number of epochs between the learning rate decay
            lr_decay_factor:        Decay factor for the learning rate
            fs:                     Fontsize for Plotting
        '''

        self.mode = mode

        #Case 1: Train a new model
        if mode == "TRAIN":
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

            self.N = N 
            self.device = device
            self.n_epochs = n_epochs
            self.lr = lr
            self.OptimizerClass = OptimizerClass
            self.batch_size = batch_size
            self.fs = fs
            self.lr_decay_factor = lr_decay_factor
            self.freq_lr_decay = freq_lr_decay
            self.NumberCouplingBlocksINN = NumberCouplingBlocksINN
            self.max_samples = max_samples
            self.SubNetFunctionINN = SubNetFunctionINN
            self.data_path = data_path

            splits = self.data_path.split("/")

            home_path = ""
            for i in range(len(splits)-2):
                home_path += splits[i]
                home_path += "/"

            home_path += "INNs/"

            #Get the quadric coupling and the hopping parameter used to generate the training set
            with open(self.data_path + f"info_0.json","r") as file:
                info_training_data = json.load(file)
            file.close()

            #Targte folder to store the results of the training
            self.path_target = home_path + f"INN_Training_N_{N}_{date.today()}_{run_name}_kappa_{info_training_data['kappa_action']}_lambda_{info_training_data['lambda_action']}/"

            #Create subfolders
            subfolders = ["Models","Data","Images","Code"]

            for folder in subfolders:
                os.makedirs(self.path_target+folder)

            #Copy the code used for the training
            files_to_save = ["DataSet.py","models.py","Training_INN.py","utils.py","Simulation_Scalar_Theory.py","eval_trained_models.py"]

            for file in files_to_save:
                shutil.copy(file,self.path_target+"Code/copy_"+file)

            #Save the configuration of the training 
            d = {
                "random_seed":random_seed,
                "N":self.N,
                "n_epochs":self.n_epochs,
                "run_name":run_name,
                "lr":self.lr,
                "OptimizerClass":str(self.OptimizerClass),
                "batch_size":self.batch_size,
                "lr_decay_factor":self.lr_decay_factor,
                "device":self.device,
                "fs":self.fs,
                "freq_lr_decay":freq_lr_decay,
                "lambda_action":info_training_data["lambda_action"],
                "kappa_action":info_training_data["kappa_action"],
                "NumberCouplingBlocksINN":NumberCouplingBlocksINN,
                "SubNetFunctionINN":str(SubNetFunctionINN),
                "max_samples":max_samples,
                "data_path":data_path
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
            self.device = device
            self.n_epoch_state_dict = n_epoch_state_dict
            self.fs = config["fs"]
            self.kappa_action = config["kappa_action"]
            self.lambda_action = config["lambda_action"]
            self.NumberCouplingBlocksINN = config["NumberCouplingBlocksINN"]
            
            #initialize the model
            self.inn = Ff.SequenceINN(self.N*self.N)

            for k in range(self.NumberCouplingBlocksINN):
                self.inn.append(Fm.AllInOneBlock,subnet_constructor = SubNetFunctionINN)

            #Load the state dict
            self.inn.load_state_dict(torch.load(self.path_target + f"Models/state_dict_epoch_{n_epoch_state_dict}.pt"))
            self.inn.eval()

    def __call__(self):
        '''
        Perform the training of the INN.
        '''

        #Initialize the model
        self.inn = Ff.SequenceINN(self.N*self.N)

        for k in range(self.NumberCouplingBlocksINN):
            self.inn.append(Fm.AllInOneBlock,subnet_constructor = self.SubNetFunctionINN)

        #Save the model
        torch.save(self.inn.state_dict(),self.path_target + f"Models/state_dict_epoch_{0}.pt")

        #Initialize the optimizer
        optimizer = self.OptimizerClass(params = self.inn.parameters(),lr = self.lr)

        #Get the data set
        DS = DataSetScalar2D_INN_No_Condition(path = self.data_path,max_samples = self.max_samples,N = self.N)
        DL = DataLoader(dataset=DS,batch_size=self.batch_size,shuffle=True)

        #Store the loss
        counter = 0
        losses = torch.zeros(10).to(self.device)

        #Train the model
        for e in tqdm.tqdm(range(self.n_epochs)):
            for i,states in enumerate(DL):
                
                #Flatten the states
                mus = states.reshape(len(states),self.N * self.N)

                #Get the log determinant of the jacobi matrix and the latent representaion of the states
                Z_latent,log_det_J = self.inn(mus)

                #Get the loss
                loss = (Z_latent.pow(2).sum(-1) / 2 - log_det_J).mean()

                #Update the model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #Save the losse
                losses[counter] = loss.detach()
                counter += 1

                #Dump the loss
                if counter == len(losses):
                    counter = 0
                    with open(self.path_target+"Data/Loss.txt","a") as file:
                        np.savetxt(file,losses.detach().cpu().numpy())
                    file.close()
                    losses = torch.zeros(250).to(self.device)

            #Plot the loss
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
            n = int(np.sqrt(len(mus)))
            u = min(n,5)
            indices = np.random.permutation(len(mus))[:u*u]
            plot_states(states=mus[indices].view(len(indices),self.N,self.N),rows = u,cols=u,path = self.path_target+f"Images/training_states_epoch_{e+1}.jpg",N = self.N)

            #Plot some samples generated using the INN
            Z_sampling = torch.randn([25,self.N * self.N])
            sampled_states,_ = self.inn(Z_sampling,rev = True)
            
            plot_states(states=sampled_states.view(25,self.N,self.N),rows = 5,cols=5,path = self.path_target+f"Images/sampled_states_epoch_{e+1}.jpg",N = self.N)

            #Save the model
            torch.save(self.inn.state_dict(),self.path_target + f"Models/state_dict_epoch_{e+1}.pt")

            #Update the learning rate
            if (e % self.freq_lr_decay == 0) and (e != 0):
                optimizer.param_groups[0]['lr'] *= self.lr_decay_factor
                print(f"lr changed to {optimizer.param_groups[0]['lr']}")

    def simulation(self,n_samples): 
        '''
        Use the trained energy to runa a simulation and record the energies and the magnetizations.

        parameters:
            n_samples:  Numbe of samples generated using the INN

        returns:
            None
        
        '''

        #Get latennt states
        Z = torch.randn([n_samples,self.N ** 2])

        #Get states 
        states,_ = self.inn(Z,rev = True)

        #Get the magnetization of the states
        ms = torch.sum(states,dim = -1)

        #Get the observables 
        m,sigma_m = bootstrap(ms.abs().detach() / self.N**2,torch.mean,args = {})
        U,sigma_U = bootstrap(ms.detach(),get_U_L,args = {"Omega":self.N**2})
        chi,sigma_chi = bootstrap(ms.detach(),get_susceptibility,args = {"Omega":self.N**2})

        d = {
            "m":m,
            "sigma_m":sigma_m,
            "U":U,
            "sigma_U":sigma_U,
            "chi":chi,
            "sigma_chi":sigma_chi,
        }

        return d

    def log_p(self,X):
        '''
        Compute the  log density defined by the INN

        parameters:
            X:      States for whcih the density is evaluated

        returns:
            p_x:    Density of
        '''

        N = X.shape[-1]

        mus = X.reshape(len(X),N * N)

        Z_latent,log_det_J = self.inn(mus,[torch.ones([len(X),1])],jac = True)
        log_p_x = - Z_latent.pow(2).sum(-1) / 2 + log_det_J - np.log((2* np.pi) ** (self.N * self.N / 2))

        return log_p_x.detach()

    def sample(self,n_samples):
        '''
        Samples states following the INN distribution.
            
        parameters:
            n_samples:      Numbr of samples

        returns:
            s:              Generated samples
        '''

        Z = torch.randn([n_samples,self.N ** 2])
        states,_ = self.inn(Z,rev = True)

        s = torch.Tensor(states.reshape([n_samples,1,5,5])).detach()

        return s

if __name__ == "__main__":

    my_parser = argparse.ArgumentParser()

    my_parser.add_argument("--mode",                    type=str,   action = "store",                                                       required=True       ,help = "Train a model if 'TRAIN' or run a simulation if 'SIMULATION'")
    my_parser.add_argument("--SubNetFunction",          type=str,   action = "store",   default = "fc_subnet",                              required=False      ,help = "Constructor used for the subnetworks in the coupling blocks of the INN\tdefault: 'fc_subnet'")
    my_parser.add_argument("--N",                       type=int,   action = "store",   default = 5,                                        required=False      ,help = "Number of spins per row / column of the lattice\tdefault: 5")
    my_parser.add_argument("--name",                    type=str,   action = "store",   default = "INN_Test_1",   	                        required=False      ,help = "Name of the training run\tdefault: INN_Test_1'")
    my_parser.add_argument("--n_epochs",                type=int,   action = "store",   default = 45,                                       required=False      ,help = "Number of epochs during the training\tdefault: 45")
    my_parser.add_argument("--lr",                      type=float, action = "store",   default = 1e-3,                                     required=False      ,help = "Initial learning rate\tdefault: 0.001")
    my_parser.add_argument("--OptimizerClass",          type=str,   action = "store",   default = "ADAM",                                   required=False      ,help = "Optimizer used to update the model parameters\tdefault: ADAM")
    my_parser.add_argument("--batch_size",              type=int,   action = "store",   default = 128,                                      required=False      ,help = "Batch size\tdefault: 128")
    my_parser.add_argument("--lr_decay_factor",         type=float, action = "store",   default = 0.1,                                      required=False      ,help = "Factor used for learning rate decay\tdefault: 0.1")
    my_parser.add_argument("--freq_lr_decay",           type=int,   action = "store",   default = 15,                                       required=False      ,help = "Frequence of reducing the learning rate\tdefault: 15")
    my_parser.add_argument("--NumberCouplingBlocksINN", type=int,   action = "store",   default = 12,                                       required=False      ,help = "Number of coupling Blocks in the INN\tdefault: 12")
    my_parser.add_argument("--max_samples",             type=int,   action = "store",   default = 75000,                                    required=False      ,help = "Number of samples taken from the stored data to initialize the training set\tdefault: 75000")
    my_parser.add_argument("--data_path",               type=str,   action = "store",   default = "",                                       required=False      ,help = "Folder containing the states and the info file used for the data set\tdefault: ''")
    my_parser.add_argument("--load_path",               type=str,   action = "store",   default = "",                                       required=False      ,help = "Parent folder of the training used for the the simulation\tdefault: ''")
    my_parser.add_argument("--n_epoch_state_dict",      type=int,   action = "store",   default = 45,                                       required=False      ,help = "Epoch of the INN used for the simulation\tdefault: 45")
    my_parser.add_argument("--n_samples_sim",           type=int,   action = "store",   default = 100000,                                   required=False      ,help = "Number of samples generated in teh simulation\tdefault: 100000")
    my_parser.add_argument("--date_simulation",         type=str,   action = "store",   default = "YYYY-MM-DD",                             required=False      ,help = "Time stamp for the simulation\tdefault: YYYY-MM-DD")
    my_parser.add_argument("--random_seed",             type=int,   action = "store",   default = 123,                                      required=False      ,help = "Random seed\tdefault: 123")
   
    args = my_parser.parse_args()

    #Subnet for the couling blocks of the INN
    subnet_dict = {
        "fc_subnet":fc_subnet
        }

    optimizer_dict = {
        "ADAM":torch.optim.Adam,
        "SGD":torch.optim.SGD
        }

    device = "cpu"

    if args.mode == "TRAIN":
        #Train a model
        T = Trainer_INN(
                mode = "TRAIN",
                device = device,
                SubNetFunctionINN = subnet_dict[args.SubNetFunction],
                NumberCouplingBlocksINN = args.NumberCouplingBlocksINN,
                N = args.N,
                run_name = args.name,
                n_epochs = args.n_epochs,
                lr = args.lr,
                OptimizerClass = optimizer_dict[args.OptimizerClass],
                batch_size = args.batch_size,
                freq_lr_decay = args.freq_lr_decay,
                lr_decay_factor = args.lr_decay_factor,
                max_samples = args.max_samples,
                data_path=args.data_path,
                random_seed = args.random_seed,
            )
        T()

    if args.mode == "SIMULATE":
        T = Trainer_INN(
                mode = "SIMULATE",
                device = device,
                SubNetFunctionINN = subnet_dict[args.SubNetFunction],
                load_path = args.load_path,
                n_epoch_state_dict = args.n_epoch_state_dict
            )

        T.simulation(n_samples = args.n_samples_sim,date = args.date_simulation)