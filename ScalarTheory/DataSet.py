import torch
from torch.utils.data import Dataset
import json
import numpy as np

class DataSetScalar2D(Dataset):
    def __init__(self,path,kappas,lambdas,max_samples,N,n_reps_data_set,n_taus = 2):
        '''
        parameters:
            path:               Location of the individual simulations used for the training set
            kappas:             Tensor containing the values of the hopping parameter included in the data set.
            lambdas:            Tensor containing the values of the quartic coupling included in the data set.
            max_samples:        Max. samples taken form one combination of kappa and lambda (from the stored data set, due to to the data augmentation , the actual number included in the trainig set is eight times max_samples)
            n_reps_data_set:    Index of the training set that is used
            n_taus:             Number of correlation times between samples included in the trainig set
            N:                  Number of spins per row and column
        '''

        super().__init__()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        #Internal storage for the training data
        self.data = torch.zeros(0)
        self.kappas_storage = torch.zeros(0)
        self.lambdas_storage = torch.zeros(0)

        print("###########################################################################################################################################################################################################################")
        print("Initialize the data set...\n")

        #loop over all hopping parameters
        for j in range(len(kappas)):

            #loop over all quadric couplings
            for i in range(len(lambdas)):

                #Get the current lambda and kappa
                k = round(kappas[j].item(),5)
                l = round(lambdas[i].item(),5)
                
                #Location of the stored training data
                p = path + f"kappa_{k}_lambda_{l}/"
                data = torch.load(p + f"states_{n_reps_data_set}.pt")

                #Get the inforamtion about the simulation
                with open(p + f"info_{n_reps_data_set}.json","r") as file:
                    info = json.load(file)
                file.close()

                #Get the part of the stored states that is in equillibrium
                lower_lim = int(info["t_eq"] / info["freq_save_samples"])+1
                data = data[lower_lim:]

                #Select states that are at least two correlation times away from each other
                step_size = int(n_taus * info["tau_action"] / info["freq_save_samples"])+1
                data = data[::step_size].view(-1,1,N,N)
                
                #Select desired number of samples by random selection
                indices = np.random.permutation(len(data))[:min(max_samples,len(data))]
                data = data[indices]

                #Use the symmetry of the problem to increase the number of training samples 
                #Horizontal flip
                data_horizontal_flip = torch.flip(data,[2])

                #Vertical flip
                data_vertical_flip = torch.flip(data,[3])

                #Horizontal and vertical flip
                data_horizontal_vertical_flip = torch.flip(data,[2,3])

                data = torch.cat((data,data_horizontal_flip,data_vertical_flip,data_horizontal_vertical_flip),dim = 0)

                #Invert the data set
                data_inverted = -1 * data
                
                #Concatenate the data set and th einverted data set
                data = torch.cat((data,data_inverted),dim = 0)

                #Concatenat with the over all data set 
                self.data = torch.cat((self.data,data),dim = 0)

                #Store the kappas and lambdas
                self.kappas_storage = torch.cat((self.kappas_storage,k * torch.ones(len(data))))
                self.lambdas_storage = torch.cat((self.lambdas_storage,l * torch.ones(len(data))))

                print(f"\t{len(data)} states for {N} x {N} lattice; \t lambda = {l}\tkappa = {k}")
        print("\n###########################################################################################################################################################################################################################\n\n")

    def __len__(self):
        '''
        return lenght of the training set
        '''
        return self.data.shape[0]

    def __getitem__(self, index):
        '''
        Get an entry from the training set

        parameters:
            index:   Index of an entry of the training set.

        return:
            self.kappas_storage[index]:         Hopping parameter of the state with index "index" in the training set
            self.lambdas_storage[index]:        Quadric coupling of the state with index "index" in the training set
            self.data[index]:                   State of the training set with index "index"
        '''
        return self.kappas_storage[index].to(self.device),self.lambdas_storage[index].to(self.device),self.data[index].to(self.device)

class DataSetScalar2D_INN_No_Condition(Dataset):
    def __init__(self,path,max_samples,N):
        '''
        parameters:
            path:               Location of the simulation and the corresponding info file
            max_samples:        Number of samples taken
            N:                  Number of spins per row and column
        '''

        super().__init__()

        print("###########################################################################################################################################################################################################################")
        print("Initialize the data set...\n")

        training_set = torch.load(path + "/training_set.pt")
        self.data = training_set[:min(len(training_set),max_samples)]

        print(f"\t{self.data.shape} states for {N} x {N} lattice")
        print("\n###########################################################################################################################################################################################################################\n\n")

    def __len__(self):
        '''
        return lenght of the training set
        '''
        return self.data.shape[0]

    def __getitem__(self, index):
        '''
        Get an entry from the training set

        parameters:
            index:   Index of an entry of the training set.

        return:
            self.data[index]:                   State of the training set with index "index"
        '''
        return self.data[index]