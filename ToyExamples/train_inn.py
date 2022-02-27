from toy_example import Data
import os
import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import tqdm
import matplotlib.pyplot as plt
import numpy as np

###############################################################################################################################################################
#Constructor for the subnetworks of the INN blocks
###############################################################################################################################################################
def fc_subnet(c_in,c_out):
        return nn.Sequential(
            nn.Linear(c_in,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,c_out)
        )

###############################################################################################################################################################
#INN class
###############################################################################################################################################################
class INN():
    def __init__(self,mode,n_modes,n_layers):
        #initialize the INN 
        self.inn = Ff.SequenceINN(2)

        for i in range(n_layers):
            self.inn.append(Fm.AllInOneBlock,subnet_constructor = fc_subnet, permute_soft = True)

        self.q = Data(mode = mode,n_modes = n_modes,res = 200)
        self.mode = mode

    def train(self,n_iter,lr,batch_size,freq_save = 1000,fs = 40,start_decay = 2000,r = 0.01):
        #Set the random seed
        torch.manual_seed(123)

        #Get the decay factor
        C = (n_iter - start_decay) * r / (1 - r)

        #initialize the optimizer
        optim = torch.optim.Adam(params = self.inn.parameters(),lr = lr)

        #store the training loss
        losses = torch.zeros(n_iter)

        #training iterations
        for i in tqdm.tqdm(range(n_iter)):

            #sample from the true distribution
            x = self.q.sample(N = batch_size)

            #Get the loss
            z,log_det_J = self.inn(x.float())
            loss = 0.5*torch.sum(z**2, 1) - log_det_J
            loss = loss.mean()

            #update the model parameters
            optim.zero_grad()
            loss.backward()
            optim.step()

            #save the loss
            losses[int(i % freq_save)] = loss
           
            if i == (n_iter -1) or i % freq_save == 0:
                #Save the state dict of the inn
                torch.save(self.inn.state_dict(),"./Bridge_Sampling_toy/INNs/"+self.mode+"/state_dicts/"+f"checkpoint_{i}.pt")

                #save the density aproximation
                plt.figure(figsize = (30,15))
                plt.subplot(1,2,1)
                plt.title("target density",fontsize = fs,fontname = "Times New Roman")
                plt.imshow(self.q.target_density)
                plt.xticks([-0.5,self.q.res / 2-0.5,self.q.res-0.5],[-self.q.x_lim,0.0,self.q.x_lim],fontsize = fs,fontname = "Times New Roman")
                plt.yticks([-0.5,self.q.res / 2-0.5,self.q.res-0.5],[self.q.x_lim,0.0,-self.q.x_lim],fontsize = fs,fontname = "Times New Roman")

                plt.subplot(1,2,2)
                plt.title("INN density",fontsize = fs,fontname = "Times New Roman")
                p = self.density(torch.Tensor(self.q.points).requires_grad_(True)).reshape(self.q.res,self.q.res).detach().numpy()
                p = np.flip(p,axis = 1)

                plt.imshow(p)
                plt.xticks([-0.5,self.q.res / 2-0.5,self.q.res-0.5],[-self.q.x_lim,0.0,self.q.x_lim],fontsize = fs,fontname = "Times New Roman")
                plt.yticks([-0.5,self.q.res / 2-0.5,self.q.res-0.5],[self.q.x_lim,0.0,-self.q.x_lim],fontsize = fs,fontname = "Times New Roman")

                plt.savefig(f"./Bridge_Sampling_toy/INNs/{self.mode}/plots/landscape_{i}.jpg")
                plt.close()

                with open("./Bridge_Sampling_toy/INNs/"+self.mode+"/loss.txt", 'a') as f:
                    a = losses.detach().numpy()
                    a = a[(a != 0)]

                    np.savetxt(f,a)
                f.close()
                losses = torch.zeros(freq_save)

            if i >= start_decay:
                optim.param_groups[0]['lr'] = lr * C / (C + i - start_decay)
        
        #Plot the loss
        loss = np.loadtxt("./Bridge_Sampling_toy/INNs/"+self.mode+"/loss.txt")

        plt.figure(figsize = (30,15))
        plt.xlabel("iteration",fontsize = fs)
        plt.ylabel("loss",fontsize = fs)
        plt.plot(loss)
        plt.savefig(f"./Bridge_Sampling_toy/INNs/{self.mode}/plots/loss.jpg")
        plt.close()

    def p_0(self,z):
        '''
        Asssumed latent distribution of the INN (standard normal)

        parameters:
            z:      Points for which the density is evaluated
        returns:
            p_z:    Density approximation for the given points
        '''

        p_z = 1 / (2 * np.pi) * torch.exp(-0.5 * z.pow(2).sum(-1))

        return p_z

    def density(self,x,p_0 = None):
        '''
        Get an approximation for the density defined by INN

        parameters:
            x:      Points for which the density is evaluated
        returns:
            p_x:    Density approximation for the given points
        '''

        #Get the latent representation of the grid in the data space
        z,log_det_J = self.inn(x)

        #Get the volume correction term
        det_J = torch.exp(log_det_J)

        if p_0 is None: 
            p_z = self.p_0(z).squeeze()
        else:
            p_z = p_0(z).squeeze()

        p_x = (p_z * torch.abs(det_J))

        return p_x

    def load(self,path):
        self.inn.load_state_dict(torch.load(path))
       
    def sample(self,N):
        '''
        Sample from the distribution efined by the INN
        
        parameters:
            N:          Number of samples
            config:     Dictionary containing hyperparameters of the training
        
        returns:
            X:          Samples from the learned distribution
        '''
        #Get the latent samples
        Z = torch.randn([N,2])

        #Get the samples in the data space
        X,_ = self.inn(Z,rev = True)

        return X


if __name__ == "__main__":
    n_iter = 50000

    def make_dirs():
        if not os.path.exists("./Bridge_Sampling_toy/INNs/"+mode+"/"):
            os.makedirs("./Bridge_Sampling_toy/INNs/"+mode+"/")
            os.makedirs("./Bridge_Sampling_toy/INNs/"+mode+"/state_dicts/")
            os.makedirs("./Bridge_Sampling_toy/INNs/"+mode+"/plots/")

    modes = ["gmm","rings","cb"]

    ###############################################################################################################################################################
    #Train the INNs
    ###############################################################################################################################################################

    for mode in modes:
        if mode == "gmm":
            make_dirs()
            I = INN(mode = "gmm",n_modes = 6,n_layers=10)
            I.train(n_iter = n_iter,lr = 1e-3,batch_size=128,freq_save=1000)

    
        elif mode == "rings":
            make_dirs()
            I = INN(mode = "rings",n_modes =3,n_layers=10)
            I.train(n_iter = n_iter,lr = 1e-3,batch_size=128,freq_save=1000)

        elif mode == "cb":
            make_dirs()
            I = INN(mode = "cb",n_modes =6,n_layers=10)
            I.train(n_iter = n_iter,lr = 1e-3,batch_size=128,freq_save=1000)

    ###############################################################################################################################################################
    #Summary plot for the densities
    ###############################################################################################################################################################
    n_state_discts_INN = [1000,49999]
    modes = ["gmm","rings","cb"]
    n_modes = [6,3,2]
    fs = 50
    fn = "Times New Roman"

    n_n = len(n_state_discts_INN)

    if not os.path.exists("./Bridge_Sampling_toy/Plots/"):
        os.makedirs("./Bridge_Sampling_toy/Plots/")

    for i in range(n_n):
        INNs = []
        for j in range(len(modes)):
            I = INN(mode = modes[j],n_modes = n_modes[j],n_layers=10)
            I.load(path = f"./Bridge_Sampling_toy/INNs/{modes[j]}/state_dicts/"+f"checkpoint_{n_state_discts_INN[i]}.pt")
            INNs.append(I)
        
        #Plot the target densities
        if i == 0:
            plt.figure(figsize = (15 * (n_n + 1),37.5))
            for j in range(len(modes)):
                plt.subplot(n_n + 1,3,1+j)
                plt.xticks([-0.5,INNs[j].q.res / 2-0.5,INNs[j].q.res-0.5],[-INNs[j].q.x_lim,0.0,INNs[j].q.x_lim],fontsize = fs,fontname = fn)
                plt.yticks([-0.5,INNs[j].q.res / 2-0.5,INNs[j].q.res-0.5],[INNs[j].q.x_lim,0.0,-INNs[j].q.x_lim],fontsize = fs,fontname = fn)
                plt.imshow(INNs[j].q.target_density)
                plt.title(f"target dist. {modes[j]}",fontsize = fs,fontname = fn)
                plt.xlabel("x",fontsize = fs,fontname = fn)
                plt.ylabel("y",fontsize = fs,fontname = fn)
                plt.tight_layout()

        for j in range(len(modes)):
            plt.subplot(n_n + 1,3,(i+1)*3+j+1)
            plt.xticks([-0.5,INNs[j].q.res / 2-0.5,INNs[j].q.res-0.5],[-INNs[j].q.x_lim,0.0,INNs[j].q.x_lim],fontsize = fs,fontname = fn)
            plt.yticks([-0.5,INNs[j].q.res / 2-0.5,INNs[j].q.res-0.5],[INNs[j].q.x_lim,0.0,-INNs[j].q.x_lim],fontsize = fs,fontname = fn)
            plt.xlabel("x",fontsize = fs,fontname = fn)
            plt.ylabel("y",fontsize = fs,fontname = fn)
            plt.title(f"INN dist. {modes[j]} e = {n_state_discts_INN[i] + 1}",fontsize = fs,fontname = fn)
            plt.tight_layout()

            p_inn = INNs[j].density(torch.tensor(INNs[j].q.points).requires_grad_(True).float()).detach().numpy().reshape(INNs[j].q.res,INNs[j].q.res)
            p_inn = np.flip(p_inn,axis = 0)

            plt.imshow(p_inn)
        
    plt.savefig("./Bridge_Sampling_toy/Plots/summary_INN_density.jpg")
    plt.close()
