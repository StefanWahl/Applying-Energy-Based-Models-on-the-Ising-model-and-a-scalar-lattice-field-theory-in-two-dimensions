import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import json
import matplotlib.font_manager as font_manager
import os

def S(mus,kappas,lambdas):
    '''
    True action function for 2D lattice.
    
    parameters:
        mus:        Initial states
        kappas:     Tensor containing the hopping parameters
        lambdas:    Tensor containing the quartic couplings

    returns:
        actions:    Containing the action of the different states
    '''

    #Get the quartic coupling
    actions = (1 - 2 * lambdas[:,None,None,None]) * mus.pow(2) +lambdas[:,None,None,None] * mus.pow(4)

    #Get the term depending on the hopping parameter
    actions += - 2 * kappas[:,None,None,None] * torch.roll(input=mus,shifts=1,dims=2) * mus
    actions += - 2 * kappas[:,None,None,None] * torch.roll(input=mus,shifts=1,dims=3) * mus

    actions = torch.sum(input=actions,dim = [1,2,3])

    return actions

def bootstrap(x,s,args,n_bootstrap = 250):
    '''
    Get an approximation foe the error and the mean by using the bootstrap method.

    parameters:
        x:                  Full time series
        s:                  Function returning the examined property. First argument must be the time series.
        args:               Dictionary containing additional arguments for s
        n_bootstrap:        Number of bootstrap samples

    returns:
        mean:   Approximation for the value of s based on x
        error:  Approximation for the error of s based on x
    '''

    #Estimate the error of the examined quantity
    samples = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        indices = np.random.randint(0,len(x),len(x))
        subset = x[indices]

        samples[i] = s(subset,**args)

    mean_samples = samples.mean()
    error = np.sqrt(np.square(samples - mean_samples).sum() / (n_bootstrap - 1))

    #Get the mean of the evaluated property
    mean = s(x,**args)

    return mean,error

def plot_states(states,rows,cols,path,N,titles = None):
    '''
    Visualize the states of the two-dimensional scalar theory

    parameters:
        states:     Tensor containing the states that are plotted
        rows:       Number of rows
        cols:       Number of columns
        path:       Location where the plot is stored
        N:          Number of spins per row and column
        titles:     List containing the titles of the individual subplots

    returns:
        None
    '''

    states = states.detach().cpu().numpy().reshape(-1,N,N)

    c = 1
    plt.figure(figsize = (cols * 5 ,cols * 5))
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows,cols,c)
            plt.imshow(states[c-1],cmap = "PiYG",vmax=4.0,vmin=-4.0)
            plt.axis("off")

            if titles is not None:
                plt.title(titles[c-1],fontsize = 20)

            #Plot a frame around the map
            plt.hlines(y = -0.5,xmin=-0.5,xmax= N-0.5,color = "k",linewidth = 2)
            plt.hlines(y = N-0.5,xmin=-0.5,xmax= N-0.5,color = "k",linewidth = 2)
            plt.vlines(x = -0.5,ymin=-0.5,ymax= N-0.5,color = "k",linewidth = 2)
            plt.vlines(x = N-0.5,ymin= -0.5,ymax= N-0.5,color = "k",linewidth = 2)
            c += 1

    plt.savefig(path)
    plt.close()

def get_tau_eq(path,fs,n_reps,step_plot = 2500):
    '''
    Plot the recorded magnetization and action. By visual examination, one can approximate the time needed to get to the equillibrium. This is done by clicking 
    on the plot at the desired position. Clicking outside the plot, leads to invalid t_eq.

    parameters:
        path:       Location where the results of the simulation are stored
        step_plot:  Frequency of time steps that are plotted.
        n_reps:     Number of the current simulation.
        fs:         Fontsize for the plots.

    returns:
        None
    '''

    #load the action and the magnetization
    actions = torch.load(path+f"Actions_{n_reps}.pt")
    magnetization = torch.load(path+f"Magnetization_{n_reps}.pt")

    #Get the infos about the simulation
    with open(path + f"info_{n_reps}.json","r") as file:
        info = json.load(file)
    file.close()

    #Load the recorded action and magnetization
    M = magnetization[::step_plot]
    S = actions[::step_plot]

    N = info["N"]

    #Plot the absolute magnetization per spin and select the equillibrium time by clicking
    iters = np.arange(len(M)) * step_plot

    fig = plt.figure(figsize=(30,15))
    plt.title(f"Magnetization per lattice site, " + r"$\kappa$" + f" = {info['kappa_action']}; " +r"$\lambda$" + f" = {info['lambda_action']}"+ f" for {info['N']**2} spins",fontsize = fs)
    plt.xlabel("iterations",fontsize = fs)
    plt.ylabel("m",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    
    plt.plot(iters,torch.abs(M.detach() / (N**2)),color = "k")

    plt.savefig(path+f"Plt/magnetization_{n_reps}.jpg")

    C = Clicker(n = 1)  
    fig.canvas.mpl_connect('button_press_event', C)
    plt.show()
    t_eq_energy = C.x[0]

    #Plot the action and select the equillibrium time by clicking
    fig = plt.figure(figsize=(30,15))
    plt.title(f"Action per lattice site, "+ r"$\kappa$" + f" = {info['kappa_action']}; " +r"$\lambda$" + f" = {info['lambda_action']}"+ f" for {info['N']**2} spins",fontsize = fs)
    plt.xlabel("iterations",fontsize = fs)
    plt.ylabel("u",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.plot(iters,S.detach() / (N**2),color = "k")
    plt.savefig(path+f"Plt/action_{n_reps}.jpg")
        
    C = Clicker(n = 1)  
    fig.canvas.mpl_connect('button_press_event', C)
    plt.show()
    t_eq_mag = C.x[0]

    #Select the bigger of the two selected equillibrium times ore raise an error if the values are both invalid.
    if t_eq_mag is None and t_eq_energy is not None:            t_eq = int(t_eq_energy)
    elif t_eq_mag is not None and t_eq_energy is None:          t_eq = int(t_eq_mag)
    elif t_eq_mag is None and t_eq_energy is None:              raise ValueError()
    elif t_eq_mag is not None and t_eq_energy is not None:      t_eq = int(max(t_eq_energy,t_eq_mag))

    #Save the equillibrium time
    with open(path+f"info_{n_reps}.json","w") as file:
        info["t_eq"] = t_eq
        json.dump(info,file)
    file.close()

def chi(dt,ts):
    '''
    Compute the autocorrelatoion of a time series for a given time lag.

    parameters:
        dt:     Time lag to calculate the autocorrelation for
        ts:     Time series

    return:
        x:      Autocorrelation 
    
    '''

    #Get th elenght of the time series
    t_max = len(ts)

    a = (ts[0:t_max-dt]*ts[dt:t_max]).sum() / (t_max - dt)
    b = ts[0:t_max-dt].sum() / (t_max - dt) 
    c = ts[dt:t_max].sum() / (t_max - dt)

    x = a - b*c

    return x 

def get_acorr(dt_max,l,ts):
    '''
    Get the autocorrelation of a time series.

    parameters:
        dt_max:     Biggest time lag that is considered. Measured in iteratiotions.
        l:          Step size for increasing the time lag. Measured in iteratiotions.
        ts:         Time series to calculate the autocorrelation for.

    returns:
        time_lags:  Time lags for which the autocorrelation has been considered
        acorr:      Autocorrelation.
    '''

    #Get the time lags
    time_lags = np.arange(0,dt_max,l)

    #Compute the autocorrelation for all the time lags
    acorr = np.zeros(len(time_lags))

    for i in tqdm.tqdm(range(len(time_lags))):
        acorr[i] = chi(dt = time_lags[i],ts = ts) 

    #Normalize the autocorrelation
    acorr /= acorr[0]

    return time_lags,acorr

def get_tau(acorr,dV):
    '''
    Get the integrated correlation time based on the autocorrelation.

    parameters:
        acorr:     Autocorrelation
        dV:        Volume element dor the integration (time lag difference between two entries in acorr). Measured in iterations.
    
    raturns:
        tau:       Approximation of teh integrated correlation time based on the autocorrelation. Measured in iterations.
    '''

    a = acorr[1:]
    b = acorr[0:-1]

    tau = (a+b).sum() / 2 * dV

    return tau

def get_U_L(magnetization,Omega):
    '''
    parameters:
        magnetization:      Magnetizations used to determine the susceptibility
        Omega:              Volume

    returns:
        U_L:                Appproximation for the Binder cumulant based in the given magnetizations.
    '''

    exp_mag_4 = np.power(magnetization / Omega,4).mean()
    exp_mag_2 = np.power(magnetization / Omega,2).mean()

    U_L = 1 - (1 / 3) * (exp_mag_4 / exp_mag_2 ** 2)

    return U_L

def get_susceptibility(magnetization,Omega):
    '''
    parameters:
        magnetization:      Magnetizations used to determine the susceptibility
        Omega:              Volume

    returns:
        susceptibility:     Appproximation for the susceptibility based in the given magnetizations.
    '''

    exp_mag_squared = np.power(magnetization / Omega,2).mean()
    exp_mag = (magnetization / Omega).mean()

    susceptibility = Omega * (exp_mag_squared - exp_mag**2)

    return susceptibility

def quick_eval_2D(path,magnetizations,actions,l,dt_max,fs,n_reps):
    '''
    parameters:
        magnetizations: Recorded magnetizations
        actions:        Recorded action
        path:           Path where the results of the simulation are stored.
        l:              Steps size of the time lags during the calculation of the autocorrelation. Measured in iterations.
        dt_max:         Biggest time lag for which the autocorrelation is determined. Measured in iterations.
        n_reps:         Number of the current simulation.
        fs:             Fontsize for the plots.

    returns:
        None
    '''

    ###################################################################################################
    #Load the info file of the simulation
    ###################################################################################################
    with open(path+f"info_{n_reps}.json","r") as file:
        info = json.load(file)
    file.close()

    t_eq = info["t_eq"]
    N = info["N"]

    ###################################################################################################
    #Select the part of the recorded data which is in the equillibrium
    ###################################################################################################

    M = np.array(magnetizations.detach().numpy())[t_eq:]
    S = np.array(actions.detach().numpy())[t_eq:]

    ####################################################################################################
    #Auto correlation
    ####################################################################################################
    #Autocorrelation for the energy time series
    time_lags,acorr_action = get_acorr(dt_max = dt_max,l = l,ts = S)
    tau_action = get_tau(acorr = acorr_action,dV = l)

    #auto correlation for the Magnetization time series
    time_lags,acorr_magnetization = get_acorr(dt_max = dt_max,l = l,ts = np.abs(M))
    tau_magnetization = get_tau(acorr = acorr_magnetization,dV = l)

    #Plot the magnetization and the decay modeled by the correlation time
    plt.figure(figsize = (30,15))

    plt.subplot(2,1,1)
    plt.title("Autocorrelation of the action",fontsize = fs)
    plt.xlabel("iterations",fontsize = fs)
    plt.ylabel(r"$\chi(t)$",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.plot(time_lags,acorr_action,color = "k",label = "acorr")
    plt.plot(time_lags,np.exp(-time_lags / tau_action),color = "y",label = "fit")
    plt.legend(fontsize = fs)

    plt.subplot(2,1,2)
    plt.title("Auto correlation of the magnetization.",fontsize = fs)
    plt.xlabel("iterations",fontsize = fs)
    plt.ylabel(r"$\chi(t)$",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.plot(time_lags,acorr_magnetization,color = "k",label = "acorr")
    plt.plot(time_lags,np.exp(-time_lags / tau_magnetization),color = "g",label = "fit")
    plt.legend(fontsize = fs)

    plt.savefig(path+f"Plt/auto_correlation_{n_reps}.jpg")
    plt.close()

    ####################################################################################################
    #Store the correlation time
    ####################################################################################################

    info["tau_action"] = tau_action
    if len(acorr_magnetization.shape) > 1:
        info["tau_magnetization"] = list(tau_magnetization)
    else:
        info["tau_magnetization"] = tau_magnetization

    ####################################################################################################
    #Get samples that are three correlation times apert to ensure independance
    ####################################################################################################

    step_size = int(info["tau_action"]) * 2

    magnetization_independent = M[::step_size]
    action_independent = S[::step_size]

    ####################################################################################################
    #Get the Binder Cumulant
    ####################################################################################################

    U_L_mean,sigma_U_L = bootstrap(x = magnetization_independent,s = get_U_L,args={"Omega":N**2})

    info["<U_L>"] = U_L_mean
    info["sigma_U_L"] = sigma_U_L

    ####################################################################################################
    #Get Susceptibility
    ####################################################################################################
    #susceptibility_mean,sigma_susceptibility = bootstrap(x = magnetization_independent,s = get_susceptibility,args={"Omega":N**2})
    susceptibility_mean,sigma_susceptibility = bootstrap(x = np.abs(magnetization_independent),s = get_susceptibility,args={"Omega":N**2})

    info["<chi^2>"] = susceptibility_mean
    info["sigma_chi^2"] = sigma_susceptibility

    ####################################################################################################
    #Action
    ####################################################################################################
    mean_action,std_action = bootstrap(x = action_independent / N**2 ,s = np.mean,args={"axis":0})

    info["<S>"] = float(mean_action)
    info["sigma_S"] = float(std_action)

    ####################################################################################################
    #Magnetization
    ####################################################################################################
    mean_magnetization,std_magnetization = bootstrap(x = np.abs(magnetization_independent) / N**2,s = np.mean,args={"axis":0})

    info["<M>"] = float(mean_magnetization)
    info["sigma_M"] = float(std_magnetization)

    ####################################################################################################
    #Save the findings to the info file
    ####################################################################################################

    with open(path+f"info_{n_reps}.json","w") as file:
        json.dump(info,file)
    file.close()

    return mean_magnetization,std_magnetization,mean_action,std_action,U_L_mean,sigma_U_L,susceptibility_mean,sigma_susceptibility,tau_action,tau_magnetization

class Clicker():
    def __init__(self,n):
        self.x = []
        self.y = []
        self.counter = 0
        self.n = n

    def __call__(self,event):
        self.x.append(event.xdata)
        self.y.append(event.ydata)

        self.counter += 1

        if self.counter == self.n:
            plt.close()

def plotter_results(n_reps,target_path,chi_squares,U_Ls,labels_U_Ls,ms,action,taus,kappas,kappas_taus,labels_chi_squares,labels_ms,labels_taus,labels_action,fontname,title,n_correlation_times = 2,reference = False,connect = False,fs = 40):
    '''
    Plot the magnetizations, the specific heat and the correlation times as a function of the temperature
    
    parameters:
        target_path                 Directory to store the results
        chi_squares                 List containing recorded susceptibility time series
        U_Ls                        List containing recorded binder cumulant time series
        labels_U_Ls                 List containing the labels for the binder cumulant time series
        ms                          List containing recorded magnetization time series
        action                      List containing recorded action time series
        taus                        List containing recorded correlation time time series
        kappas                      List containing the hopping parameters for the time series
        kappas_taus                 List containing the hopping parameters for the correlation time time series
        labels_chi_squares          List containing the labels for the susceptibility time series
        labels_ms                   List containing the labels for the magnetization time series
        labels_taus                 List containing the labels for the correlation time time series
        labels_action               List containing the labels for the action time series
        fontname                    Font used for the labels
        n_correlation_times         Number of differetn correlation times
        reference                   Plot the first entry in the lists as a reference plot with colored one sigma intervall
        fs                          Fontsize for the plots
        title                       Title for the plots.
        n_reps:                     Number of the current simulation.

    reurns:
        None
    '''

    markers = [".","s","*","p","X","D","1","v","^"]
    colors = ["b","r","c","g","k","m","orange","y","maroon"]
    marker_size = 10
    line_width = 3
    markeredgewidth = line_width
    capsize = 10
    font = font_manager.FontProperties(family=fontname,
                                   style='normal', 
                                   size=fs)

    if connect == True: ls = "-"
    else: ls = ""

    #####################################################################################################################################################################################################
    #Plot susceptibility
    #####################################################################################################################################################################################################
    plt.figure(figsize = (30,12))
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)

    plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
    plt.ylabel(r"$\chi_2$",fontsize = fs,fontname = fontname)
    plt.title(title,fontsize = fs,fontname = fontname)
    
    start = 0

    if reference == True:
        plt.fill_between(x = kappas[0],y1 = chi_squares[0][0] - chi_squares[0][1],y2 = chi_squares[0][0] + chi_squares[0][1],color = "orange",label = r"$1\sigma$"+" interval")
        plt.plot(kappas[0],chi_squares[0][0],ls = ":",linewidth = line_width,color = "k",label = labels_chi_squares[0])
        start = 1

    for i in range(start,len(chi_squares)):
        plt.errorbar(x = kappas[i],y = chi_squares[i][0],yerr = chi_squares[i][1],ls = ls,marker = markers[i-start],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = labels_chi_squares[i],color = colors[i-start])

    plt.legend(prop = font)
    plt.savefig(target_path+f"susceptibility_{n_reps}.jpg")
    plt.close()

    #####################################################################################################################################################################################################
    #Plot Binder cumulant
    #####################################################################################################################################################################################################
    plt.figure(figsize = (30,12))
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)

    plt.ylabel(r"$U_l$",fontsize = fs,fontname = fontname)
    plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
    plt.title(title,fontsize = fs,fontname = fontname)
    
    start = 0

    if reference == True:
        plt.fill_between(x = kappas[0],y1 = U_Ls[0][0] - U_Ls[0][1],y2 = U_Ls[0][0] + U_Ls[0][1],color = "orange",label = r"$1\sigma$"+" interval")
        plt.plot(kappas[0],U_Ls[0][0],ls = ":",linewidth = line_width,color = "k",label = labels_U_Ls[0])
        start = 1

    for i in range(start,len(chi_squares)):
        plt.errorbar(x = kappas[i],y = U_Ls[i][0],yerr = U_Ls[i][1],ls = ls,marker = markers[i-start],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = labels_U_Ls[i],color = colors[i-start])

    plt.legend(prop = font)
    plt.savefig(target_path+f"binder_cumulant_{n_reps}.jpg")
    plt.close()


    #####################################################################################################################################################################################################
    #Plot the magnetization
    #####################################################################################################################################################################################################

    plt.figure(figsize = (30,12))
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)

    plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
    plt.ylabel("abs. mean magnetization per spin",fontsize = fs,fontname = fontname)
    plt.title(title,fontsize = fs,fontname = fontname)

    start = 0

    if reference == True:
        plt.fill_between(x = kappas[0],y1 = ms[0][0] - ms[0][1],y2 = ms[0][0] + ms[0][1],color = "orange",label = r"$1\sigma$"+" interval")
        plt.plot(kappas[0],ms[0][0],ls = ":",linewidth = line_width,color = "k",label = labels_ms[0])
        start = 1

    for i in range(start,len(ms)):
        plt.errorbar(x = kappas[i],y = np.abs(ms[i][0]),yerr = ms[i][1],ls = ls,marker = markers[i-start],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = labels_ms[i],color = colors[i-start])

    plt.legend(prop = font)
    plt.savefig(target_path+f"magnetization_{n_reps}.jpg")
    plt.close()

    #####################################################################################################################################################################################################
    #Plot the action
    #####################################################################################################################################################################################################

    plt.figure(figsize = (30,12))
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)

    plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
    plt.ylabel("mean action per spin",fontsize = fs,fontname = fontname)
    plt.title(title,fontsize = fs,fontname = fontname)
    start = 0

    if reference == True:
        plt.fill_between(x = kappas[0],y1 = action[0][0] - action[0][1],y2 = action[0][0] + action[0][1],color = "orange",label = r"$1\sigma$"+" interval")
        plt.plot(kappas[0],action[0][0],ls = ":",linewidth = line_width,color = "k",label = labels_ms[0])
        start = 1

    for i in range(start,len(ms)):
        plt.errorbar(x = kappas[i],y = action[i][0],yerr = action[i][1],ls = ls,marker = markers[i-start],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = labels_action[i],color = colors[i-start])

    plt.legend(prop = font)
    plt.savefig(target_path+f"action_{n_reps}.jpg")
    plt.close()

    #####################################################################################################################################################################################################
    #Plot the correlation times
    #####################################################################################################################################################################################################

    plt.figure(figsize = (30,10 * n_correlation_times))

    for j in range(n_correlation_times):
        plt.subplot(n_correlation_times,1,j+1)

        plt.xticks(fontsize = fs,fontname = fontname)
        plt.yticks(fontsize = fs,fontname = fontname)

        plt.xlabel(r"$\kappa$",fontsize = fs,fontname = fontname)
        plt.ylabel("Correlation time [iter.]",fontsize = fs,fontname = fontname)
        plt.title(title,fontsize = fs,fontname = fontname)

        if reference == True:
            for i in range(1):
                plt.plot(kappas_taus[j][i],taus[j][i],ls = ":",linewidth = line_width,label = labels_taus[j][i])
            start = 1

        for i in range(start,len(taus[j])):
            plt.plot(kappas_taus[j][i],taus[j][i],ls = ls,marker = markers[i-start],markersize = marker_size,label = labels_taus[j][i],color = colors[i-start])

        plt.legend(prop = font)
    plt.savefig(target_path+f"correlation_times_{n_reps}.jpg")
    plt.close()
    
class gaussian_distribution():
    def __init__(self,path):
        '''
        parameters:
            path: Location of the files containing the parameters of the multivariate normal distribution
        '''

        #Load the files if they exist
        #Mean
        if os.path.exists(path + "gaussian_fit/mean_fit.pt"):
            self.mean = torch.load(path + "gaussian_fit/mean_fit.pt")
            self.N = int(np.sqrt(len(self.mean)))

        #Covariance
        if os.path.exists(path + "gaussian_fit/covariance_fit.pt"):
            self.cov = torch.load(path + "gaussian_fit/covariance_fit.pt")

            #get the inverse of the covariance matrix
            self.cov_inv = torch.linalg.inv(self.cov)

            #get the determinant of the covariance matrix
            self.det = torch.linalg.det(self.cov)

    def fit(self,path,N):
        '''
        Fit a multivariate gaussian distribution to the data set.

        parameters:
            path:       Location of the data set
            N:          Number of samples per row and column
        
        returns:
            None
        '''

        #Load the training data
        training_data = torch.load(path + f"training_set.pt")
        X = training_data.view(-1,N**2).T.numpy()

        #Get the covariance and the meanof the data set
        self.cov = torch.Tensor(np.cov(X))
        self.mean = X.mean(1)

        #save the fit
        if not os.path.exists(path + "gaussian_fit/"):
            os.makedirs(path + "gaussian_fit/")
            
        torch.save(self.cov,path + "gaussian_fit/covariance_fit.pt")
        torch.save(self.mean,path + "gaussian_fit/mean_fit.pt")
         
    def log_p(self,X):
        '''
        Compute the log density of the data set X

        parameters:
            X:          Data set

        returns:
            log_p_X:    Logarithm of the probability of X
        '''
        
        X = X.view(-1,self.N * self.N).detach()    
        log_p_X = - np.log(np.sqrt((2 * np.pi)**(self.N * self.N) * self.det)) - (((X-self.mean) @ self.cov_inv) * (X-self.mean)).sum(axis = 1) / 2

        return log_p_X

    def sample(self,n_samples):
        '''
        Generate samples that follow the normal distribution.

        parameters:
            n_samples:      Number of samples that are generated

        returns:
            s:              Generated samples
        '''

        s = np.random.multivariate_normal(mean = self.mean,cov = self.cov,size=[n_samples])
        s = torch.Tensor(s.reshape([n_samples,1,5,5]))

        return s

def iterative_approximation_r(n_iter,r_0,l_1,l_2):

    r_seq = np.zeros(n_iter)
    r_seq[0] = r_0

    n_1 = len(l_1)
    n_2 = len(l_2)

    s_1 = n_1 / (n_1+n_2)
    s_2 = n_2 / (n_1+n_2)

    for t in range(1,n_iter):
        r_seq[t] = n_1 / n_2 * ((l_2 / (s_1*l_2 + r_seq[t-1] * s_2)).sum()) / ((1 / (s_1*l_1 + r_seq[t-1] * s_2)).sum())
        
    return r_seq