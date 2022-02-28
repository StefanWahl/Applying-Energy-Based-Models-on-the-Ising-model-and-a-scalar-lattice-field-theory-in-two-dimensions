import torch
from torch.utils.data import Dataset
import json
import numpy as np

class DataSetIsing2D(Dataset):
    def __init__(self,path,n_correlation_times = 2,max_samples = 6250):
        '''
        parameters:
            path:                       Location of the recorded states
            n_correlation_times:        Number of correlation times between the staes used as training date
            max_samples:                Number od samples taken from the recorded data set to build the data set
        '''
        super().__init__()

        #Load the stored states
        data = torch.load(path+f"states.pt")

        #Get the device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        #Get the inforamtion about the simulation
        with open(path+ "info.json","r") as file:
            info = json.load(file)
        file.close()

        print("########################################################################################################################################\nData Set:")
        print(f"\t{len(data)} \tstates loaded")
        
        #Get the part of the stored states that is in equillibrium
        lower_lim = int(info["t_eq"] / info["freq_save_samples"])+1
        data = data[lower_lim:]
        print(f"\t{len(data)} \tstates in equillibrium")

        #Select states that are at least two correlation times away from each other
        N = info["N"]
        step_size = int(n_correlation_times * info["tau_energy"] / info["freq_save_samples"])+1
        data = data[::step_size].view(-1,1,N,N)
        print(f"\t{len(data)} \tindependent states")

        #Select the desired number of states
        indices = np.random.permutation(len(data))[:min([len(data),max_samples])]
        data = data[indices]

        #Use the symmetry of the problem to increase the number of training samples 
        #Horizontal flip
        data_horizontal_flip = torch.flip(data,[2])

        #Vertical flip
        data_vertical_flip = torch.flip(data,[3])

        #Horizontal and vertical flip
        data_horizontal_vertical_flip = torch.flip(data,[2,3])

        self.data = torch.cat((data,data_horizontal_flip,data_vertical_flip,data_horizontal_vertical_flip),dim = 0)

        #Use the negative data set
        data_neg = -1 * self.data

        self.data = torch.cat((self.data,data_neg),dim = 0)

        print(f"\t{len(self.data)}","\tstates by mirroring")

        print("########################################################################################################################################\n")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].to(self.device)