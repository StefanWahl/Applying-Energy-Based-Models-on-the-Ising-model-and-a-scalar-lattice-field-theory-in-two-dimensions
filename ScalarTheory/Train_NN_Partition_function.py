import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
import matplotlib.font_manager as font_manager
import argparse
from utils import bootstrap
from models import LinearCondition5x5_one_condition

#########################################################################################################################################################
#Model class
#########################################################################################################################################################
class model(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(1,128),
            nn.Softplus(),
            nn.Linear(128,128),
            nn.Softplus(),
            nn.Linear(128,128),
            nn.Softplus(),
            nn.Linear(128,2)
        )
    
    def forward(self,X):
        Y = self.linear(X.float())
        return Y.squeeze()

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()

    my_parser.add_argument("--mode",                            type=str,     action = "store",                       required= True                                     ,help = "Combination of proposal density and sampling method for the partition function ('INN_MC','INN_BRIDGE','NORMAL_MC','NORMAL_BRIDGE')")
    my_parser.add_argument("--data_path",                       type=str,     action = "store",                       required= True                                     ,help = "Location of the summary file containing the partition functions.")
    my_parser.add_argument("--n_samples_max",                   type=int,     action = "store",                       required= True                                     ,help = "Number of samples per repetition to computet the partition function")
    my_parser.add_argument("--n_iter",                          type=int,     action = "store",                       required= False,     default = 60000               ,help = "Number of training iterrations\tdefault: 60000")
    my_parser.add_argument("--save_freq",                       type=int,     action = "store",                       required= False,     default = 5000                ,help = "Frequency of saving the model\tdefault: 5000")
    my_parser.add_argument("--lr_decay_freq",                   type=int,     action = "store",                       required= False,     default = 15000               ,help = "Frequency of lr decay\tdefault: 15000")
    my_parser.add_argument("--batch_size",                      type=int,     action = "store",                       required= False,     default = 128                 ,help = "Batch size\tdefault: 128")
    my_parser.add_argument("--lr",                              type=int,     action = "store",                       required= False,     default = 0.01                ,help = "Learning rate\tdefault: 0.01")
    my_parser.add_argument("--train_NN_partitionfunction",      type=int,     action = "store",                       required= False,     default = 0                   ,help = "Train NN for the partition function as a funktion of the hopping parameter\tdefault: 0")
    my_parser.add_argument("--train_NN_add_term",               type=int,     action = "store",                       required= False,     default = 0                   ,help = "Train NN for the additional term as a funktion of the hopping parameter\tdefault: 0")
    my_parser.add_argument("--mode_action_function",            type=str,     action = "store",                       required= False,     default = "TRUE"              ,help = "'EBM' or 'TRUE'\tdefault: 'True'")
    my_parser.add_argument("--lmbd",                            type=float,   action = "store",                       required= False,     default = 0.02                ,help = "Quartic coupling\tdefault: 0.02")
    my_parser.add_argument("--kappa_min",                       type=float,   action = "store",                       required= False,     default = 0.2                 ,help = "Smallest hopping parameter\tdefault: 0.2")
    my_parser.add_argument("--kappa_max",                       type=float,   action = "store",                       required= False,     default = 0.39                ,help = "Biggest hopping parameter\tdefault: 0.39")
    my_parser.add_argument("--dkappa",                          type=float,   action = "store",                       required= False,     default = 0.02                ,help = "Stepsize of the hopping parameter\tdefault: 0.02")
    my_parser.add_argument("--path_EBM_model",                  type=str,     action = "store",                       required= False,     default = ""                  ,help = "location of teh state dict of the EBM\tdefault: ''")
    my_parser.add_argument("--n_samples_additional_term",       type=int,     action = "store",                       required= False,     default = 100000              ,help = "Number of samples used to approximate the additional terms used in the training set\tdefault: 100000")
    my_parser.add_argument("--n_iter_eval",                     type=int,     action = "store",                       required= False,     default = 60000               ,help = "Index of the state dicts used in the evaluation\tdefault: 60000")
    my_parser.add_argument("--N",                               type=int,     action = "store",                       required= False,     default = 5                   ,help = "Number spins per row and column\tdefault: 5")
    my_parser.add_argument("--reference_file",                  type=str,     action = "store",                       required= False,     default = ""                 ,help = "file containing the summary for the validation sets\tdefault: ''")

    args = my_parser.parse_args()

    #########################################################################################################################################################
    #Settings for plotting
    #########################################################################################################################################################
    fs = 50
    fontname = "Times New Roman"
    marker_size = 10
    line_width = 3
    markeredgewidth = line_width
    capsize = 10
    font = font_manager.FontProperties(family=fontname,
                                    style='normal', 
                                    size=fs)

    #########################################################################################################################################################
    #Visualization of the results
    #########################################################################################################################################################
    def plotter_part_fct(iteration,M,mode,kappas_TS,Z_TS,dZ_TS):
        '''
        Visualize the results during the training of the partition function

        parameters:
            iteration:      Current Training iteration
            M:              Model
            mode:           Training mode 
            kappas_TS:      Hopping parameters of the data set
            Z_TS:           Partition functions of the training set
            dZ_TS:          Error of the partition functions of the training set   

        returns:
            None
        '''

        #Create folder to store the results of the training
        if not os.path.exists(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/{mode}/"):
            os.makedirs(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/{mode}/")

        #Plot the result
        kappas = torch.linspace(kappas_TS.min()-0.02,kappas_TS.max()+0.02,100).view(-1,1)
        Y = M(kappas.view(-1,1))

        plt.figure(figsize = (30,15))
        plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
        plt.ylabel("log Z",fontsize = fs,fontname = fontname)
        plt.xticks(fontsize = fs,fontname = fontname)
        plt.yticks(fontsize = fs,fontname = fontname)
        plt.errorbar(kappas_TS.detach().numpy(),np.log(Z_TS.detach().numpy()),yerr = (dZ_TS / Z_TS).detach().numpy(),label = "measurement",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width,ls = "",marker = ".")
        plt.errorbar(kappas.detach().numpy(),Y[:,0].detach().numpy(),yerr = np.exp(Y[:,1].detach().numpy()),label = "fit",linewidth = line_width,capsize = capsize, markeredgewidth = markeredgewidth,ls = "",marker = ".")
        plt.legend(prop = font)
        plt.savefig(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/{mode}/{mode}_log_Z_{iteration}.jpg")
        plt.close()

    def plotter_add_term(iteration,M,kappas_TS,A_TS,dA_TS):
        '''
        Visualize the results during the training of the additional term

        parameters:
            iteration:      Current Training iteration
            M:              Model
            mode:           Training mode 
            kappas_TS:      Hopping parameters of the data set
            A_TS:           Additional term of the training set
            dA_TS:          Error of the additional terms of the training set   

        returns:
            None
        '''

        #Plot the result
        kappas = torch.linspace(kappas_TS.min()-0.02,kappas_TS.max()+0.02,100).view(-1,1)
        Y = M(kappas.view(-1,1))

        plt.figure(figsize = (30,15))
        plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
        plt.ylabel("A",fontsize = fs,fontname = fontname)
        plt.xticks(fontsize = fs,fontname = fontname)
        plt.yticks(fontsize = fs,fontname = fontname)
        plt.errorbar(kappas_TS,A_TS,yerr = dA_TS.detach().numpy(),label = "measurement",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width,ls = "",marker = ".")
        plt.errorbar(kappas.detach().numpy(),Y[:,0].detach().numpy(),yerr = np.exp(Y[:,1].detach().numpy()),label = "fit",linewidth = line_width,capsize = capsize, markeredgewidth = markeredgewidth,ls = "",marker = ".")
        plt.legend(prop = font)
        plt.savefig(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/AdditionalTerm_{iteration}.jpg")
        plt.close()

    #########################################################################################################################################################
    #Train a NN to model the logarithm of teh partition function
    #########################################################################################################################################################

    if bool(args.train_NN_partitionfunction) == True:

        #########################################################################################################################################################
        #Get the training training set
        #########################################################################################################################################################
        data = np.loadtxt(args.data_path +  f"eval_partition_function/summary_partition_functions_{args.n_samples_max}.txt",skiprows = 1)
        kappas_TS = data[:,0]

        if args.mode == "INN_MC":
            means_Z = data[:,1]
            std_Z = data[:,2]

        elif args.mode == "INN_BRIDGE":
            means_Z = data[:,3]
            std_Z = data[:,4]

        elif args.mode == "NORMAL_MC":
            means_Z = data[:,5]
            std_Z = data[:,6]

        elif args.mode == "NORMAL_BRIDGE":
            means_Z = data[:,7]
            std_Z = data[:,8]

        else:
            raise NotImplementedError

        kappas_TS = torch.tensor(kappas_TS,dtype = torch.double)
        means_Z = torch.tensor(means_Z,dtype = torch.double)
        std_Z = torch.tensor(std_Z,dtype = torch.double)

        #########################################################################################################################################################
        #Initialize the model and the optimizer
        #########################################################################################################################################################
        M = model()
        optimizer = torch.optim.Adam(params = M.parameters(),lr = args.lr)
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,step_size=args.lr_decay_freq,gamma = 0.1)

        #########################################################################################################################################################
        #Training
        #########################################################################################################################################################
        for i in tqdm.tqdm(range(args.n_iter)):

            #get samples form the training set and perturbe them with gausssian noise 
            indices = np.random.randint(len(kappas_TS),size = [args.batch_size])

            log_Z = torch.log(means_Z[indices])
            d_log_Z = std_Z[indices] / means_Z[indices]
            log_Z_perturbed = log_Z + torch.randn_like(log_Z) * d_log_Z
            
            #Construct the target vector
            Y_1 = log_Z.reshape(-1,1)
            Y_2 = torch.log(d_log_Z).reshape(-1,1)

            Y = torch.cat((Y_1,Y_2),dim = 1)

            kappas = kappas_TS[indices].view(-1,1)

            #Get the MSE loss
            loss = (Y - M(kappas)).pow(2).mean()

            #Optimize the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Update the learning rate
            lr_schedule.step()

            #Save the state dict and plot the current aproximation
            if i % args.save_freq == 0 or i == args.n_iter - 1:
                plotter_part_fct(iteration = i+1,M = M,mode = args.mode,kappas_TS = kappas_TS,Z_TS = means_Z,dZ_TS = std_Z)

                torch.save(M.state_dict(),args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/{args.mode}/{args.mode}_state_dict_partition_function_{i+1}.pt")

    #########################################################################################################################################################
    #Train a NN to model the additional term in the expectation value of the action
    #########################################################################################################################################################

    if bool(args.train_NN_add_term) == True:

        #########################################################################################################################################################
        #Approximat th additional term from the recorded states
        #########################################################################################################################################################

        if not os.path.exists(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/AdditionalTerm.txt"):

            kappas = np.arange(args.kappa_min,args.kappa_max,args.dkappa)

            #write the header of the summary file
            with open(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/AdditionalTerm.txt","w") as file:
                file.write("kappa\ta\tsigma_a\n")
            file.close()

            for i in range(len(kappas)):

                k = round(kappas[i],6)

                #########################################################################################################################################################
                #Samples generated using the true action function
                #########################################################################################################################################################
                if args.mode_action_function == "TRUE":
                    #get the validation set
                    data = torch.load(args.data_path + f"/kappa_{k}_lambda_{args.lmbd}/validation_set.pt")
                    data = data[np.random.permutation(len(data))[:args.n_samples_additional_term]]

                    def d(mus):
                        lambdas = torch.ones(len(mus)) * args.lmbd
                        u = (1 - 2 * lambdas[:,None,None,None]) * mus.pow(2) +lambdas[:,None,None,None] * mus.pow(4)
                        u = torch.sum(input=u,dim = [1,2,3])

                        return u.mean()

                    a,sigma_a = bootstrap(data,s = d,args = {})

                #########################################################################################################################################################
                #Samples generated using the learned action function
                #########################################################################################################################################################
                if args.mode_action_function == "EBM":
                    data = torch.load(args.data_path + f"/epoch_30_kappa_{k}_lambda_{args.lmbd}/validation_set.pt")
                    data = data[np.random.permutation(len(data))[:args.n_samples_additional_term]]

                    Action_model = LinearCondition5x5_one_condition(1,64)
                    Action_model.load_state_dict(torch.load(args.path_EBM_model,map_location="cpu"))
                    Action_model.eval()

                    def d(mus):
                        with torch.no_grad():
                            u = Action_model.convs_2(mus.float()).squeeze()
                            u = Action_model.lins_2(u).squeeze()

                        return u.mean()

                    a,sigma_a = bootstrap(data,s = d,args = {})

                #########################################################################################################################################################
                #Save the approximation
                #########################################################################################################################################################
                with open(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/AdditionalTerm.txt","a") as file:
                    file.write(f"{k}\t{a}\t{sigma_a}\n")
                file.close()

                print(k,a,sigma_a)
        
        #########################################################################################################################################################
        #Initialize the model and the optimzer
        #########################################################################################################################################################
        M = model()
        optimizer = torch.optim.Adam(params = M.parameters(),lr = args.lr)
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer = optimizer,step_size=args.lr_decay_freq,gamma = 0.1)

        #########################################################################################################################################################
        #get the data file
        #########################################################################################################################################################
        data = np.loadtxt(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/AdditionalTerm.txt",skiprows = 1)
        kappas_TS = data[:,0]
        means_A = torch.tensor(data[:,1])
        std_A = torch.tensor(data[:,2])

        for i in tqdm.tqdm(range(args.n_iter)):

            #get samples
            indices = np.random.randint(len(kappas_TS),size = [args.batch_size])

            A = (means_A[indices] + torch.randn(len(indices)) * std_A[indices]).view(-1,1)
            dA = torch.log(std_A[indices]).view(-1,1)
            
            Y = torch.cat((A,dA),dim = 1)

            kappas = torch.tensor(kappas_TS[indices]).view(-1,1)

            #Get the MSE loss
            loss = (Y - M(kappas)).pow(2).mean()

            #Optimize the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Update the learning rate
            lr_schedule.step()

            #Save the state dict and plot the current aproximation
            if i % args.save_freq == 0 or i == args.n_iter - 1:
                plotter_add_term(iteration = i+1,M = M,kappas_TS = kappas_TS,A_TS = means_A,dA_TS = std_A)

                torch.save(M.state_dict(),args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/AdditionalTerm_state_dict_partition_function_{i+1}.pt")


    #########################################################################################################################################################
    #Approximate the expectation value of the action
    #########################################################################################################################################################

    #Load the model for the additional term
    B = model()
    B.load_state_dict(torch.load(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/AdditionalTerm_state_dict_partition_function_{args.n_iter_eval}.pt"))
    B.eval()

    #Get the reference file 
    reference_file = np.loadtxt(args.reference_file,skiprows = 1)

    #comput the deviations between the absolute deviation between the training data and teh model prediction
    #Expectation value:
    data = np.loadtxt(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/AdditionalTerm.txt",skiprows = 1)
    kappas_TS = data[:,0]
    means_A = torch.tensor(data[:,1])
    std_A = torch.tensor(data[:,2])

    print("\nAdditional term:")
    Y = B(torch.tensor(kappas_TS).view(-1,1))
    print("\tB:",torch.abs((means_A - Y[:,0]) / means_A).mean().item())
    print("\tdB:",torch.abs((std_A - torch.exp(Y[:,1])) / std_A).mean().item(),"\n")

    #log Z models:
    d = np.loadtxt(args.data_path +  f"eval_partition_function/summary_partition_functions_{args.n_samples_max}.txt",skiprows = 1)
    kappas_TS = d[:,0]

    for m in ["INN_BRIDGE","INN_MC","NORMAL_BRIDGE","NORMAL_MC"]:

        kappas_TS = d[:,0]

        if m == "INN_MC":
            means_Z = d[:,1]
            std_Z = d[:,2]

        elif m == "INN_BRIDGE":
            means_Z = d[:,3]
            std_Z = d[:,4]

        elif m == "NORMAL_MC":
            means_Z = d[:,5]
            std_Z = d[:,6]

        elif m == "NORMAL_BRIDGE":
            means_Z = d[:,7]
            std_Z = d[:,8]

        kappas_TS = torch.tensor(kappas_TS,dtype = torch.double)
        means_Z = torch.tensor(means_Z,dtype = torch.double)
        std_Z = torch.tensor(std_Z,dtype = torch.double)

        log_Z = model()
        log_Z.load_state_dict(torch.load(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/{m}/{m}_state_dict_partition_function_{args.n_iter_eval}.pt"))
        log_Z.eval()

        print(f"\n{m}:")
        Y = log_Z(kappas_TS.view(-1,1))
        print("\tlog Z:",torch.abs((torch.log(means_Z) - Y[:,0]) / torch.log(means_Z)).mean().item())
        print("\tdlog Z:",torch.abs((std_Z / means_Z - torch.exp(Y[:,1])) / (std_Z / means_Z)).mean().item(),"\n")

    #Get the action 
    plt.figure(figsize = (40,25))
    j = 1

    #Plot the approximation of the action and comapre it to the reference file
    for mode in ["INN_BRIDGE","INN_MC","NORMAL_BRIDGE","NORMAL_MC"]:

        #Load the model for the logarithm of the partition function
        log_Z = model()
        log_Z.load_state_dict(torch.load(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/{mode}/{mode}_state_dict_partition_function_{args.n_iter_eval}.pt"))
        log_Z.eval()
        
        kappas = torch.linspace(0.2,0.38,100).view(-1,1).requires_grad_(True)

        #Get the derivative of the logarithm of the partition function with respect too the hopping parameter
        log_part = log_Z(kappas)[:,0]
        d_log_Z_d_kappa = torch.autograd.grad(log_part.sum(),kappas)[0].detach().squeeze().numpy()

        #Get the additional term
        u = B(kappas).detach().numpy()
        A = u[:,0]
        dA = np.exp(u[:,1])

        #Get the approximation for the expectation of the action
        a = (- kappas.detach().squeeze().numpy() * d_log_Z_d_kappa) + A
        a = a / args.N ** 2
        da = dA / args.N **2

        plt.subplot(2,2,j)
        plt.plot(kappas.detach().squeeze().numpy(),a,label = f"{mode.split('_')[0]} {mode.split('_')[1]}",markersize = marker_size,linewidth = line_width,ls = "",marker = ".")
        plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
        plt.ylabel("a",fontsize = fs,fontname = fontname)
        plt.xticks(fontsize = fs,fontname = fontname)
        plt.yticks(fontsize = fs,fontname = fontname)
        plt.errorbar(reference_file[:,0],reference_file[:,3],yerr = reference_file[:,4],label = "validation set",color = "k",markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,linewidth = line_width,ls = "",marker = "s")
        plt.legend(prop = font)

        j += 1
    plt.savefig(args.data_path+f"eval_partition_function/TrainingNNPartitionFunction/summary_plot_N_{args.N}_lambda_{args.lmbd}.jpg")
    plt.close()