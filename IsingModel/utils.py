import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import json
import matplotlib.font_manager as font_manager

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

def get_T_c(J,k):
    '''
    Compute the critical temperature for the two dimensional ising model with no external field.

    parameters:
        J:   Interaction between the spins
        k:   Boltzmann constant

    returns:
        critical temperature for the given parameters
    '''
    return 2 * J / (k * np.log(1 + np.sqrt(2)))

def get_theoretical_prediction(T_min,T_max,dT,stepsize = 1 / 100000,calculate = True):
    '''
    Approximate the theoretical prediction for the energy nad the specific heat for the 2D Ising model.

    parameters:
        T_min:      Smallest temperature which is evaluated
        T_max:      Biggest temperature which is evaluated
        dT:         Stepsize of the Temperature.
        stepsize:   Stepsize numerical integration
        calculate:  Calculate the properties if True. If False only load and plot the results
    
    returns:
        None
    '''
    J = 1
    k = 1
    T_c = get_T_c(J,k)

    if calculate:
        Ts = np.arange(T_min,T_max,dT)
        results = np.zeros([len(Ts),4]) 

        for i in tqdm.tqdm(range(len(Ts))):
            T = Ts[i]
            beta = 1 / ( k * T)


            k_1 = 2 * np.sinh(2 * beta) / np.square(np.cosh(2 * beta))
            k_1_pp = 2 * np.square(np.tanh(2 * beta)) - 1

            phi = np.arange(0,np.pi / 2,stepsize)
            arg = np.sqrt(1 - np.square(k_1 * np.sin(phi)))

            E_1 = 0.5 * (arg[:-1] + arg[1:]).sum() * stepsize
            K_1 = 0.5 * (1 / arg[:-1] + 1 / arg[1:]).sum() * stepsize

            E = - J * 1 / np.tanh(2 * beta) * ( 1 + 2 / np.pi * k_1_pp * K_1)
            C = k * np.square(beta / np.tanh(2 * beta)) * 2 / np.pi * (2 * K_1 - 2 * E_1 - (1 - k_1_pp) * (0.5 * np.pi + k_1_pp * K_1))

            results[i][0] = T
            results[i][1] = C
            results[i][2] = E

            if T < T_c: 
                results[i,3] = (1 - np.sinh(2 * beta.item() * J)**(-4))**(1 / 8)

        with open("./Discrete_Ising_Model/approximation_theory.txt","w") as file:
            file.write("Temperature\tSpecific heat per spin\tEnergy per spin\tMagnetization per spin\n")
            np.savetxt(file,results)
        file.close()
    
    #Plot the results
    fs = 60

    results = np.loadtxt("./Discrete_Ising_Model/approximation_theory.txt",skiprows = 1)

    plt.figure(figsize = (30,12))
    plt.xticks(fontsize = fs,fontname = "Times New Roman")
    plt.yticks(fontsize = fs,fontname = "Times New Roman")
    plt.xlabel("T",fontsize = fs,fontname = "Times New Roman")
    plt.ylabel("c",fontsize = fs,fontname = "Times New Roman")
    plt.plot(results[:,0],results[:,1],color = "k",linewidth = 4)
    ind = np.arange(len(results[:,0]))[(results[:,0] > T_c)][0]
    #plt.title("Specific heat per spin",fontsize = 40,fontname = "Times New Roman")
    plt.tight_layout()
    plt.savefig("./Discrete_Ising_Model/specific_heat_theory.jpg")
    plt.close()

    plt.figure(figsize = (30,12.5))
    plt.xticks(fontsize = fs,fontname = "Times New Roman")
    plt.yticks(fontsize = fs,fontname = "Times New Roman")
    plt.xlabel("T",fontsize = fs,fontname = "Times New Roman")
    plt.ylabel("u",fontsize = fs,fontname = "Times New Roman")
    plt.plot(results[:,0],results[:,2],color = "k",linewidth = 4)
    #plt.title("Energy per spin",fontsize = 40,fontname = "Times New Roman")
    plt.tight_layout()
    plt.savefig("./Discrete_Ising_Model/energy_theory.jpg")
    plt.close()

    plt.figure(figsize = (30,12))
    plt.xticks(fontsize = fs,fontname = "Times New Roman")
    plt.yticks(fontsize = fs,fontname = "Times New Roman")
    plt.xlabel("T",fontsize = fs,fontname = "Times New Roman")
    plt.ylabel("m",fontsize = fs,fontname = "Times New Roman")
    plt.plot(results[:,0],results[:,3],color = "k",linewidth = 4)
    #plt.title("Magnetization per spin",fontsize = 40,fontname = "Times New Roman")
    plt.tight_layout()
    plt.savefig("./Discrete_Ising_Model/magnetization_theory.jpg")
    plt.close()

def plot_states(states,rows,cols,path,N,titles = None,fs = None):
    '''
    Visualize the states of the two-dimensional Ising Model

    parameters:
        states:     Tensor containing the states that are plotted
        rows:       Number of rows
        cols:       Number of columns
        path:       Location where the plot is stored
        N:          Number of spins per row and column
        titles:     List containing the titles of the individual subplots
        fs:         Fontsize

    returns:
        None
    '''

    states = states.detach().cpu().numpy().reshape(-1,N,N)

    c = 1
    plt.figure(figsize = (cols * 5 ,cols * 5))
    for i in range(rows):
        for j in range(cols):
            plt.subplot(rows,cols,c)
            plt.imshow(states[c-1],cmap = "gray",vmax=1,vmin=-1)
            plt.axis("off")

            #Plot a frame around the map
            plt.hlines(y = -0.5,xmin=-0.5,xmax= N-0.5,color = "k",linewidth = 2)
            plt.hlines(y = N-0.5,xmin=-0.5,xmax= N-0.5,color = "k",linewidth = 2)
            plt.vlines(x = -0.5,ymin=-0.5,ymax= N-0.5,color = "k",linewidth = 2)
            plt.vlines(x = N-0.5,ymin= -0.5,ymax= N-0.5,color = "k",linewidth = 2)
            
            if titles is not None:
                plt.title(titles[c-1],fontsize = fs)

            c += 1

    plt.savefig(path)
    plt.close()

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
        time_lags:  Time lags for which the autocorrelation has been computed
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

def get_c(E,N,T,k):
    '''
    Compute the specific heat per lattice site.

    parameters:
        E:  Array containing the recorded energies
        N:  Number of spins per row and column
        T:  Temperature
        k:  Boltzmann constant

    returns:
        c:  Specific heat per spin, calculated using the given energies.
    '''
    beta = 1 / (T *k)

    return (beta / N)**2 * np.std(E)**2

def quick_eval_2D(path,magnetizations,energies,l,dt_max,fs):
    '''
    Determine the correlation times based on the recorded magnetizations and energies and visualize the correlation time, 
    the magnetization and the specific heat. Determine the mean magnetization and the mean specific heat.

    parameters:
        path:           Path where the results of the simulation are stored.
        magnetizations: Recorded magnetizations
        energies:       Recorded energies
        l:              Steps size of the time lags during the calculation of the autocorrelation. Measured in iterations.
        dt_max:         Biggest time lag for which the autocorrelation is determined. Measured in iterations.
        fs:             Fontsize for plotting.

    returns:
        None
    '''

    ###################################################################################################
    #Load the info file of the simulation
    ###################################################################################################
    with open(path+"info.json","r") as file:
        info = json.load(file)
    file.close()

    t_eq = info["t_eq"]
    N = info["N"]
    k = info["k"]
    T = info["T"]

    ###################################################################################################
    #Select the part of the recorded data which is in the equillibrium
    ###################################################################################################

    M = np.array(magnetizations.detach().numpy())[t_eq:]
    E = np.array(energies.detach().numpy())[t_eq:]

    ####################################################################################################
    #Auto correlation
    ####################################################################################################
    #Autocorrelation for the energy time series
    time_lags,acorr_energy = get_acorr(dt_max = dt_max,l = l,ts = E)
    tau_energy = get_tau(acorr = acorr_energy,dV = l)

    #auto correlation for the Magnetization time series
    if len(M.shape) == 1:
        time_lags,acorr_magnetization = get_acorr(dt_max = dt_max,l = l,ts = M)
        tau_magnetization = get_tau(acorr = acorr_magnetization,dV = l)

    else:
        acorr_magnetization = np.zeros([M.shape[1],len(acorr_energy)])
        tau_magnetization = np.zeros(M.shape[1])

        for i in range(M.shape[1]):
            time_lags,a = get_acorr(dt_max = dt_max,l = l,ts = M[:,i])
            acorr_magnetization[i] = a
            tau_magnetization[i] = get_tau(acorr = a,dV = l)

    #Plot the magnetization and the decay modeled by the correlation time
    plt.figure(figsize = (30,30))

    plt.subplot((2),1,1)
    plt.title("Autocorrelation of the energy",fontsize = fs)
    plt.xlabel("iterations",fontsize = fs)
    plt.ylabel(r"$\chi(t)$",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.plot(time_lags,acorr_energy,color = "k",label = "acorr")
    plt.plot(time_lags,np.exp(-time_lags / tau_energy),color = "y",label = "fit")
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

    plt.savefig(path+"correlation_times.jpg")


    ####################################################################################################
    #Store the correlation time
    ####################################################################################################

    info["tau_energy"] = tau_energy
    if len(acorr_magnetization.shape) > 1:
        info["tau_magnetization"] = list(tau_magnetization)
    else:
        info["tau_magnetization"] = tau_magnetization

    ####################################################################################################
    #Get samples that are three correlation times apert to ensure independance
    ####################################################################################################
    
    step_size = int(info["tau_energy"]) * 2
    energy_independent = E[::step_size]
    magnetization_independent = M[::step_size]

    ####################################################################################################
    #Specific heat based on the independent samples using the bootstrap method
    ####################################################################################################

    c,sigma_c = bootstrap(x = energy_independent,s = get_c,args = {"N":N,"k":k,"T":T})

    info["<c>"] = c
    info["sigma_c"] = sigma_c

    ####################################################################################################
    #Magnetization based on all recorded energie
    ####################################################################################################
    
    m,sigma_m = bootstrap(x = magnetization_independent / (N ** 2),s = np.mean,args = {})

    info["<m>"] = float(m)
    info["sigma_m"] = float(sigma_m)

    ####################################################################################################
    #Energy based on all recorded energie
    ####################################################################################################

    u,sigma_u = bootstrap(x = energy_independent / (N ** 2),s = np.mean,args = {})

    info["<u>"] = float(u)
    info["sigma_u"] = float(sigma_u)

    ####################################################################################################
    #Save the findings to the info file
    ####################################################################################################

    with open(path+"info.json","w") as file:
        json.dump(info,file)
    file.close()

def get_tau_eq(path,fs,step_plot = 100):
    '''
    Plot the recorded magnetization and energy. By visual examination, one can approximate the time needed to get to the equillibrium. This is done by clicking 
    on the plot at the desired position. Clicking outside the plot, leads to invalid t_eq.

    parameters:
        path:       Location where the results of the simulation are stored
        fs:         Fontsize for plotting
        step_plot:  Frequency of time steps that are plotted.

    returns:
        None
    '''

    #load the energy and the magnetization
    energies = torch.load(path+"Energies.pt")
    magnetization = torch.load(path+"Magnetization.pt")

    #Get the infos about the simulation
    with open(path + "info.json","r") as file:
        info = json.load(file)
    file.close()

    #Load the recorded energy and magnetization
    M = magnetization[::step_plot]
    E = energies[::step_plot]

    N = info["N"]

    #Plot the Magnetization and select the equillibrium time by clicking
    iters = np.arange(len(M)) * step_plot

    fig = plt.figure(figsize=(30,15))
    plt.title(f"Magnetization per lattice site, T = {info['T']} and {info['N']**2} spins",fontsize = fs)
    plt.xlabel("iterations",fontsize = fs)
    plt.ylabel("m",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    if len(M.shape) == 1:
        plt.plot(iters,M.detach() / (N**2),color = "k")

    else:
        labels = [r"$m_x$",r"$m_y$"]
        colors = ["k","y"]
        for i in range(M.shape[1]):
            plt.plot(iters,M[:,i].detach() / (N**2),label = labels[i],color = colors[i])
        plt.legend(fontsize = fs)

    plt.savefig(path+"magnetization.jpg")

    C = Clicker(n = 1)  
    fig.canvas.mpl_connect('button_press_event', C)
    plt.show()
    t_eq_energy = C.x[0]

    #Plot the inner energy and select the equillibrium time by clicking
    fig = plt.figure(figsize=(30,15))
    plt.title(f"Inner Energy per lattice site, T = {info['T']} and {info['N']**2} spins",fontsize = fs)
    plt.xlabel("iterations",fontsize = fs)
    plt.ylabel("u",fontsize = fs)
    plt.xticks(fontsize = fs)
    plt.yticks(fontsize = fs)
    plt.plot(iters,E.detach() / (N**2),color = "k")
    plt.savefig(path+"inner_energy.jpg")
        
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
    with open(path+"info.json","w") as file:
        info["t_eq"] = t_eq
        json.dump(info,file)
    file.close()

def plotter_m_c_tau(target_path,title,es,cs,ms,taus,temps,temps_tau,labels_cs,labels_ms,labels_es,labels_taus,fontname,upper_lim_c,n_correlation_times = 2,reference = False,fs = 40):
    '''
    Plot the magnetizations, the specific heat and the correlation times as a function of the temperature
    
    parameters:
        target_path:            Location where the plots are stored
        title:                  Title for each plot
        cs:                     Specific heat. List containing the different time series. Structure:[:,[[specific heat],[std. dev. of the specific heat]]]
        ms:                     Magnetization. List containing the different time series. Structure:[:,[[magnetization],[std. dev. magnetization]]]
        es:                     Energy. List containing the different time series. Structure:[:,[[energy],[std. dev. energy]]]
        taus:                   Correlation times. List containing the different time series. Structure:[[tau_1_1,...,tau_1y_n],...,[tau_n_correlation_times_1,...,tau_n_correlation_times_n]]
        temps:                  Temperatures. Structure:[:,[Temperatures]]
        temps_tau               Temperatures to plot the correlation times. Structure as for taus.
        labels_cs               Labels for the specific heats. Structure: [label_1, ..., label_n]
        labels_ms               Labels for the magnetization. Structure: [label_1, ..., label_n]
        labels_taus             Labels for the correlation times. Structure: [[label_1_1, ..., label_1_n], ... , [label_n_correlation_times_1, ..., label_n_correlation_times_n]]
        reference:              If True, the initial tiem series in the above lists is plotted as a dotted line as a reference line
        n_correlation_times:    Number of different correlation time types that are examined. p. ex. for energy and magnetization n_correlation_times = 2
        fontname:               Fontname for plotting
        fs:                     Fontsize of the plots
        labels_es:              Labels of the energy plots
        upper_lim_c:            Upper limit for the plots of the specific heat
    
    reurns:
        None
    '''

    markers = [".","s","*","p","X","D"]
    colors = ["g","r","b","c","k","m"]
    marker_size = 10
    line_width = 3
    markeredgewidth = line_width
    capsize = 10
    font = font_manager.FontProperties(family=fontname,
                                   style='normal', 
                                   size=fs)

    #####################################################################################################################################################################################################
    #Plot the specific heat
    #####################################################################################################################################################################################################

    plt.figure(figsize = (30,15))
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)
    plt.ylim([-0.1,upper_lim_c])
    plt.title(title,fontsize = fs,fontname = fontname)

    plt.xlabel("T",fontsize = fs,fontname = fontname)
    plt.ylabel("Specific heat per spin",fontsize = fs,fontname = fontname)
    
    #plot the theoretical prediction
    theory = np.loadtxt("./Discrete_Ising_Model/approximation_theory.txt",skiprows = 1)
    plt.plot(theory[:,0],theory[:,1],color = "k",label = "theory")

    start = 0

    if reference == True:
        plt.fill_between(x = temps[0],y1 = cs[0][0] - cs[0][1],y2 = cs[0][0] + cs[0][1],color = "orange",label = r"$1\sigma$"+" interval")
        plt.plot(temps[0],cs[0][0],ls = ":",linewidth = line_width,color = "k",label = labels_cs[0])
        start = 1

    for i in range(start,len(cs)):
        plt.errorbar(x = temps[i],y = cs[i][0],yerr = cs[i][1],ls = "",marker = markers[i-start],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = labels_cs[i],color = colors[i-start])

    plt.legend(prop = font)
    plt.savefig(target_path+"specific_heat.jpg")
    plt.close()

    #####################################################################################################################################################################################################
    #Plot the magnetization
    #####################################################################################################################################################################################################

    plt.figure(figsize = (30,15))
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)

    plt.xlabel("T",fontsize = fs,fontname = fontname)
    plt.ylabel("Magnetization per spin",fontsize = fs,fontname = fontname)
    plt.title(title,fontsize = fs,fontname = fontname)
    start = 0

    #plot the theoretical prediction
    plt.plot(theory[:,0],theory[:,3],color = "k",label = "theory")

    if reference == True:
        plt.fill_between(x = temps[0],y1 = ms[0][0] - ms[0][1],y2 = ms[0][0] + ms[0][1],color = "orange",label = r"$1\sigma$"+" interval")
        plt.plot(temps[0],ms[0][0],ls = ":",linewidth = line_width,color = "k",label = labels_ms[0])
        start = 1

    for i in range(start,len(ms)):
        plt.errorbar(x = temps[i],y = ms[i][0],yerr = ms[i][1],ls = "",marker = markers[i-start],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = labels_ms[i],color = colors[i-start])

    plt.legend(prop = font)
    plt.savefig(target_path+"magnetization.jpg")
    plt.close()

    #####################################################################################################################################################################################################
    #Plot the energy
    #####################################################################################################################################################################################################

    plt.figure(figsize = (30,15))
    plt.xticks(fontsize = fs,fontname = fontname)
    plt.yticks(fontsize = fs,fontname = fontname)

    plt.xlabel("T",fontsize = fs,fontname = fontname)
    plt.ylabel("Energy per spin",fontsize = fs,fontname = fontname)
    plt.title(title,fontsize = fs,fontname = fontname)
    start = 0

    #plot the theoretical prediction
    plt.plot(theory[:,0],theory[:,2],color = "k",label = "theory")

    if reference == True:
        plt.fill_between(x = temps[0],y1 = es[0][0] - es[0][1],y2 = es[0][0] + es[0][1],color = "orange",label = r"$1\sigma$"+" interval")
        plt.plot(temps[0],es[0][0],ls = ":",linewidth = line_width,color = "k",label = labels_es[0])
        start = 1

    for i in range(start,len(es)):
        plt.errorbar(x = temps[i],y = es[i][0],yerr = es[i][1],ls = "",marker = markers[i-start],markersize = marker_size, capsize = capsize, markeredgewidth = markeredgewidth,label = labels_es[i],color = colors[i-start])

    plt.legend(prop = font,loc='lower right',ncol=2)
    plt.savefig(target_path+"energy.jpg")
    plt.close()

    #####################################################################################################################################################################################################
    #Plot the correlation times
    #####################################################################################################################################################################################################

    plt.figure(figsize = (30,15 * n_correlation_times))

    for j in range(n_correlation_times):
        plt.subplot(n_correlation_times,1,j+1)

        plt.xticks(fontsize = fs,fontname = fontname)
        plt.yticks(fontsize = fs,fontname = fontname)
        plt.title(title,fontsize = fs,fontname = fontname)

        plt.xlabel("T",fontsize = fs,fontname = fontname)
        plt.ylabel("Correlation time [iter.]",fontsize = fs,fontname = fontname)

        if reference == True:
            for i in range(1):
                plt.plot(temps_tau[j][i],taus[j][i],ls = ":",linewidth = line_width,label = labels_taus[j][i])
            start = 1

        for i in range(start,len(taus[j])):
            plt.plot(temps_tau[j][i],taus[j][i],ls = "",marker = markers[i-start],markersize = marker_size,label = labels_taus[j][i],color = colors[i-start])

        plt.legend(prop = font)
    plt.savefig(target_path+"correlation_times.jpg")
    plt.close()

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

if __name__ == "__main__":
    get_theoretical_prediction(T_min = 0.1,T_max = 5.0,dT = 0.0025,calculate=False)

    
