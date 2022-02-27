import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
import os

###############################################################################################################################################################
#define the energy model class
###############################################################################################################################################################

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Linear(2,128),
            nn.LeakyReLU(),

            nn.Linear(128,256),
            nn.LeakyReLU(),

            nn.Linear(256,128),
            nn.LeakyReLU(),

            nn.Linear(128,1)
        )

    def forward(self,X):
        return self.sequence(X.float())

###############################################################################################################################################################
#Data set
###############################################################################################################################################################
class Data():
    def __init__(self,mode,n_modes,res = 500):
        self.n_modes = n_modes
        self.mode = mode
        self.sigma = 0.15
        self.r = 1.0
        self.res = res

        if mode == "gmm": 
            self.x_lim = 1.5
            self.phis = np.linspace(0,np.pi * 2,n_modes+1)[:n_modes]

        elif mode == "rings":
            self.x_lim = self.r * (n_modes + 0.5)

        elif mode == "cb":
            self.x_lim = 1.5


        x,y = self.eval_points = np.meshgrid(np.linspace(-self.x_lim,self.x_lim,self.res),np.linspace(-self.x_lim,self.x_lim,self.res))

        x = x.flatten().reshape(-1,1)
        y = y.flatten().reshape(-1,1)

        self.points = np.concatenate((x,y),axis = 1)

        self.target_density = self.density(self.points).reshape(self.res,self.res)
        self.target_density = np.flip(self.target_density,axis = 0)
    
    def sample(self,N):
        samples = torch.zeros(0,2)
        sample_group = np.random.multinomial(N, np.ones(self.n_modes) / self.n_modes)

        if self.mode == "gmm":
            for i in range(self.n_modes):
                s = torch.randn(sample_group[i],2) * self.sigma + torch.tensor([self.r * np.sin(self.phis[i]),self.r * np.cos(self.phis[i])])
                samples = torch.cat((samples,s),dim = 0)

        elif self.mode == "rings":
            for i in range(self.n_modes):
                phis = torch.rand(sample_group[i]) * np.pi * 2
                rs = torch.randn(sample_group[i]) * self.sigma + self.r * (i + 1)

                xs =(torch.sin(phis) * rs).view(-1,1)
                ys =(torch.cos(phis) * rs).view(-1,1)

                s = torch.cat((xs,ys),dim = 1)

                samples = torch.cat((samples,s),dim = 0)

        elif self.mode == "cb":
            samples_1 = torch.rand(sample_group[0],2) 
            samples_2 = torch.rand(sample_group[0],2) + torch.tensor([-1,-1])

            samples = torch.cat((samples_1,samples_2),dim = 0)

        else:
            raise NotImplementedError()

        return samples

    def density(self,x):

        p = np.zeros(x.shape[0])

        if self.mode == "gmm":
            for i in range(self.n_modes):
                mu = np.array([self.r * np.sin(np.pi * 2 / self.n_modes * i),self.r * np.cos(np.pi * 2 / self.n_modes * i)])

                p += 1 / (2 * np.pi * self.sigma**2) * np.exp( - np.square(x - mu).sum(-1) / (2 * self.sigma**2))

        elif self.mode == "rings":
            for i in range(self.n_modes):
                rs = np.sqrt(x[:,0]**2 + x[:,1]**2)

                prop = 1 / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(- ((1+i) * self.r - rs) ** 2 / (2 * self.sigma ** 2)) / (2 * np.pi * (1+i) * self.r)
                p += prop

        elif self.mode == "cb":

            p = np.where((x[:,0] <= 1) * (x[:,0] >=0) * (x[:,1] <= 1) * (x[:,1] >=0),np.ones_like(p) * 1,p)
            p = np.where((x[:,0] <= 0) * (x[:,0] >=-1) * (x[:,1] <= 0) * (x[:,1] >= -1),np.ones_like(p) * 1,p)

        p = p / self.n_modes

        return p

    def plot_density(self,Model = None,path = None):

        fs = 40

        plt.figure(figsize = (30,15))

        plt.subplot(1,2,1)
        plt.title("target density",fontsize = fs,fontname = "Times New Roman")
        plt.imshow(self.target_density)
        plt.xticks([-0.5,self.res / 2-0.5,self.res-0.5],[-self.x_lim,0.0,self.x_lim],fontsize = fs,fontname = "Times New Roman")
        plt.yticks([-0.5,self.res / 2-0.5,self.res-0.5],[self.x_lim,0.0,-self.x_lim],fontsize = fs,fontname = "Times New Roman")

        if Model is not None:
            plt.subplot(1,2,2)
            plt.title("learned density",fontsize = fs,fontname = "Times New Roman")
            
            energy = Model(torch.tensor(self.points))
            energy = energy - torch.min(energy)
            p = torch.exp(- energy)
            p = p.detach().numpy().reshape(self.res,self.res)
            p = np.flip(p,axis = 1)

            plt.imshow(p)
            plt.xticks([-0.5,self.res / 2-0.5,self.res-0.5],[-self.x_lim,0.0,self.x_lim],fontsize = fs,fontname = "Times New Roman")
            plt.yticks([-0.5,self.res / 2-0.5,self.res-0.5],[self.x_lim,0.0,-self.x_lim],fontsize = fs,fontname = "Times New Roman")

        if path is not None:
            plt.savefig(path)
            return

        plt.show()

    def plot_points(self,x):

        x = x.detach().numpy()

        plt.plot(x[:,0],x[:,1],marker = ".",color = "k",ls = "")
        plt.show()

###############################################################################################################################################################
#Training routine
###############################################################################################################################################################

def training(n_iter = 10000,K = 500,mode = "gmm",n_modes = 4,epsilon = 0.005,path  = None,lr = 0.0050,r = 0.001,start_decay = 2000):

    #Create the folders
    if not os.path.exists(path):
        os.makedirs(path)

    #Get the decay factor
    C = (n_iter - start_decay) * r / (1 - r)

    #Initialize the Model
    E = Model()
    optimizer = torch.optim.Adam(params=E.parameters(),lr = lr)

    #Date class
    q = Data(mode = mode,n_modes = n_modes)

    #Buffer for persistetn initialization
    buffer = torch.randn(10000,2)

    #Initialize a file to store the loss
    with open(path +"loss.txt","w") as file:
        file.write("iteration\tloss\n")
    file.close()

    for i in tqdm.tqdm(range(n_iter)):

        #Get positiv samples
        x_pos = q.sample(128)

        #Get initial negativa samples from the buffer
        indices = np.random.permutation(len(buffer))[:128]
        x_neg = buffer[indices]
        x_neg.requires_grad_(True)

        #update the samples with langevin dynamics
        for j in range(K):
            g = torch.autograd.grad(E(x_neg).sum(),[x_neg])[0]
            x_neg.data += - epsilon * g + np.sqrt(epsilon * 2) * torch.randn_like(x_neg)

        x_neg = x_neg.detach()

        #Get the Loss
        loss = E(x_pos).mean() - E(x_neg).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update the buffer
        buffer[indices] = x_neg

        #save the loss
        with open(path + "loss.txt","a") as file:
            file.write(f"{i}\t{loss.item()}\n")
        file.close()

        #Plot the density
        if i % 1000 == 0 or i == n_iter - 1 :
            q.plot_density(Model = E,path=path + f"density_i_{i}.jpg")
            torch.save(E.state_dict(),path+f"state_dict_i_{i}.pt")

        if i >= start_decay:
            optimizer.param_groups[0]['lr'] = lr * C / (C + i - start_decay)

if __name__ == "__main__":

    n_iter = 50000
    ###############################################################################################################################################################
    #perform the training
    ###############################################################################################################################################################
    training(n_iter = n_iter,K = 250,mode = "cb",n_modes = 3,path = "./Learning_EBM_toy/cb/",epsilon = 0.0001,start_decay=100)
    training(n_iter = n_iter,K = 250,mode = "rings",n_modes = 3,path = "./Learning_EBM_toy/rings/")
    training(n_iter = n_iter,K = 250,mode = "gmm",n_modes = 6,path = "./Learning_EBM_toy/gmm/")

    ###############################################################################################################################################################
    #eval the trained models:
    ###############################################################################################################################################################

    E_cb = Model()
    E_rings = Model()
    E_gmm = Model()

    E_cb.load_state_dict(torch.load(f"./Learning_EBM_toy/cb/state_dict_i_{n_iter-1}.pt"))
    E_rings.load_state_dict(torch.load(f"./Learning_EBM_toy/rings/state_dict_i_{n_iter-1}.pt"))
    E_gmm.load_state_dict(torch.load(f"./Learning_EBM_toy/gmm/state_dict_i_{n_iter-1}.pt"))

    D_cb = Data(mode = "cb",n_modes=1)
    D_rings = Data(mode = "rings",n_modes=3)
    D_gmm = Data(mode = "gmm",n_modes=6)

    ###############################################################################################################################################################
    #Plot the density approximation of the trained EBMs
    ###############################################################################################################################################################
    fs = 50
    fn = "Times New Roman"
    plt.figure(figsize=(45,25))

    #Checkerboard
    plt.subplot(2,3,1)
    plt.title("learned density cb",fontsize = fs,fontname = "Times New Roman")
    energy = E_cb(torch.tensor(D_cb.points))
    energy = energy - torch.min(energy)
    p = torch.exp(- energy)
    p = p.detach().numpy().reshape(D_cb.res,D_cb.res)
    p = np.flip(p,axis = 1)
    plt.imshow(p)
    plt.xticks([-0.5,D_cb.res / 2-0.5,D_cb.res-0.5],[-D_cb.x_lim,0.0,D_cb.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.yticks([-0.5,D_cb.res / 2-0.5,D_cb.res-0.5],[D_cb.x_lim,0.0,-D_cb.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.xlabel("x",fontsize = fs,fontname = fn)
    plt.ylabel("y",fontsize = fs,fontname = fn)
    plt.tight_layout()

    plt.subplot(2,3,4)
    plt.title("target density cb",fontsize = fs,fontname = "Times New Roman")
    plt.imshow(D_cb.target_density)
    plt.xticks([-0.5,D_cb.res / 2-0.5,D_cb.res-0.5],[-D_cb.x_lim,0.0,D_cb.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.yticks([-0.5,D_cb.res / 2-0.5,D_cb.res-0.5],[D_cb.x_lim,0.0,-D_cb.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.xlabel("x",fontsize = fs,fontname = fn)
    plt.ylabel("y",fontsize = fs,fontname = fn)
    plt.tight_layout()

    #rings
    plt.subplot(2,3,2)
    plt.title("learned density rings",fontsize = fs,fontname = "Times New Roman")
    energy = E_rings(torch.tensor(D_rings.points))
    energy = energy - torch.min(energy)
    p = torch.exp(- energy)
    p = p.detach().numpy().reshape(D_rings.res,D_rings.res)
    p = np.flip(p,axis = 1)
    plt.imshow(p)
    plt.xticks([-0.5,D_rings.res / 2-0.5,D_rings.res-0.5],[-D_rings.x_lim,0.0,D_rings.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.yticks([-0.5,D_rings.res / 2-0.5,D_rings.res-0.5],[D_rings.x_lim,0.0,-D_rings.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.xlabel("x",fontsize = fs,fontname = fn)
    plt.ylabel("y",fontsize = fs,fontname = fn)

    plt.subplot(2,3,5)
    plt.title("target density rings",fontsize = fs,fontname = "Times New Roman")
    plt.imshow(D_rings.target_density)
    plt.xticks([-0.5,D_rings.res / 2-0.5,D_rings.res-0.5],[-D_rings.x_lim,0.0,D_rings.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.yticks([-0.5,D_rings.res / 2-0.5,D_rings.res-0.5],[D_rings.x_lim,0.0,-D_rings.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.xlabel("x",fontsize = fs,fontname = fn)
    plt.ylabel("y",fontsize = fs,fontname = fn)

    #gmm
    plt.subplot(2,3,3)
    plt.title("learned density gmm",fontsize = fs,fontname = "Times New Roman")
    energy = E_gmm(torch.tensor(D_gmm.points))
    energy = energy - torch.min(energy)
    p = torch.exp(- energy)
    p = p.detach().numpy().reshape(D_gmm.res,D_gmm.res)
    p = np.flip(p,axis = 1)
    plt.imshow(p)
    plt.xticks([-0.5,D_gmm.res / 2-0.5,D_gmm.res-0.5],[-D_gmm.x_lim,0.0,D_gmm.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.yticks([-0.5,D_gmm.res / 2-0.5,D_gmm.res-0.5],[D_gmm.x_lim,0.0,-D_gmm.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.xlabel("x",fontsize = fs,fontname = fn)
    plt.ylabel("y",fontsize = fs,fontname = fn)

    plt.subplot(2,3,6)
    plt.title("target density gmm",fontsize = fs,fontname = "Times New Roman")
    plt.imshow(D_gmm.target_density)
    plt.xticks([-0.5,D_gmm.res / 2-0.5,D_gmm.res-0.5],[-D_gmm.x_lim,0.0,D_gmm.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.yticks([-0.5,D_gmm.res / 2-0.5,D_gmm.res-0.5],[D_gmm.x_lim,0.0,-D_gmm.x_lim],fontsize = fs,fontname = "Times New Roman")
    plt.xlabel("x",fontsize = fs,fontname = fn)
    plt.ylabel("y",fontsize = fs,fontname = fn)

    plt.savefig("./Learning_EBM_toy/Summary_density.jpg")
    plt.close()

    ###############################################################################################################################################################
    #Plot the conditionals
    ###############################################################################################################################################################

    plt.figure(figsize=(45,30))

    #gmm
    plt.subplot(3,1,1)
    conditional_points_gmm = torch.cat((torch.zeros(1000).view(-1,1),torch.linspace(-D_gmm.x_lim,D_gmm.x_lim,1000).view(-1,1)),dim = 1)
    conditional_points_gmm_long = np.zeros((D_gmm.res * D_gmm.res,2))
    conditional_points_gmm_long[:len(conditional_points_gmm)]  = conditional_points_gmm

    energies_gmm = E_gmm(conditional_points_gmm)
    energies_gmm = energies_gmm - torch.min(energies_gmm)
    prop_gmm = torch.exp(-energies_gmm)
    prop_gmm_target = D_gmm.density(conditional_points_gmm_long).reshape(-1)[:len(conditional_points_gmm)]
    prop_gmm_target /= prop_gmm_target.max()

    plt.plot(conditional_points_gmm.detach().numpy()[:,1],prop_gmm.detach().numpy(),label = r"$p_{\theta}(y|x = 0.0)$",color = "k",linewidth = 4)
    plt.plot(conditional_points_gmm.detach().numpy()[:,1],prop_gmm_target,label = r"$p_{gmm}(y|x = 0.0)$",color = "b",linewidth = 4)
    plt.legend(fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.xlabel("y",fontsize = fs)
    plt.ylabel(r"$p(y|x = 0.0)$",fontsize = fs)

    #rings
    plt.subplot(3,1,2)
    conditional_points_rings = torch.cat((torch.zeros(1000).view(-1,1),torch.linspace(-D_rings.x_lim,D_rings.x_lim,1000).view(-1,1)),dim = 1)
    conditional_points_rings_long = np.zeros((D_rings.res * D_rings.res,2))
    conditional_points_rings_long[:len(conditional_points_rings)]  = conditional_points_rings

    energies_rings = E_rings(conditional_points_rings)
    energies_rings = energies_rings - torch.min(energies_rings)
    prop_rings = torch.exp(-energies_rings)
    prop_rings_target = D_rings.density(conditional_points_rings_long).reshape(-1)[:len(conditional_points_rings)]
    prop_rings_target /= prop_rings_target.max()

    plt.plot(conditional_points_rings.detach().numpy()[:,1],prop_rings.detach().numpy(),label = r"$p_{\theta}(y|x = 0.0)$",color = "k",linewidth = 4)
    plt.plot(conditional_points_rings.detach().numpy()[:,1],prop_rings_target,label = r"$p_{rings}(y|x = 0.0)$",color = "b",linewidth = 4)
    plt.legend(fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.xlabel("y",fontsize = fs)
    plt.ylabel(r"$p(y|x = 0.0)$",fontsize = fs)

    #checkerboard
    plt.subplot(3,1,3)
    conditional_points_cb = torch.cat((torch.ones(1000).view(-1,1)*0.5,torch.linspace(-D_cb.x_lim,D_cb.x_lim,1000).view(-1,1)),dim = 1)

    energies_cb = E_cb(conditional_points_cb)
    energies_cb = energies_cb - torch.min(energies_cb)
    prop_cb = torch.exp(-energies_cb)

    prob_cb_target = np.zeros(len(conditional_points_cb))
    mask = ((conditional_points_cb[:,1] >= 0) * (conditional_points_cb[:,1] <= 1))
    prob_cb_target[mask] = np.ones(mask.sum())

    plt.plot(conditional_points_cb.detach().numpy()[:,1],prop_cb.detach().numpy(),label = r"$p_{\theta}(y|x = 0.5)$",color = "k",linewidth = 4)
    plt.plot(conditional_points_cb.detach().numpy()[:,1],prob_cb_target,label = r"$p_{cb}(y|x = 0.5)$",color = "b",linewidth = 4)
    plt.legend(fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.xlabel("y",fontsize = fs)
    plt.ylabel(r"$p(y|x = 0.5)$",fontsize = fs)

    plt.savefig("./Learning_EBM_toy/summary_conditionals.jpg") 
    plt.close()