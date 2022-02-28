![Alt text](imagnetizationIsing.jpg)

# Applying Energy Based-Models on the Ising model and a scalar lattice field theory in two dimensions

This repository contains the code used in by Bachelor thesis 'Applying Energy Based-Models on the Ising model and a scalar lattice field theory in two dimensions'. It consists of three main parts:

## Toy Examples

The toy examples used to illustrate the concept of Energy Based Models and Bridge Sampling is contained in the folder 'ToyExamples'. To train EBMs on three two-dimensional toy data distributions, run 'toy_example.py'. To train simple INNs on these data distributions, run 'train_inn.py'. The partition function of the trained EBMs can be approximated using Bridge Sampling with the INNs as proposal distributions. This can be done running 'bridge_sampling.py'. 'tail_behaviour.py' compares Bridge Sampling and Importance Sampling for one-dimensional Normal distributions.

## Ising Model

The code used to evaluate the two-dimensional Ising Model with no external magnetic field is contained in the folder 'IsingModel'. To generate training data for the EBM or to obtain a reference simulation based on the true Hamiltonian, run 'Discrete_Ising_Model_Metropolis.py'. To train an EBM on the data set obtained using the true Hamiltonian, run 'Training.py'. 'eval_trained_models.py' compares the EBM to the reference simulation based on the true Hamiltonian.

## Scalar Theory

The code used to evaluate the two-dimensional, real valued scalar theory is contained in the folder 'ScalarTheory'. To generate training data for the EBM or to obtain a reference simulation based on the true action function, run 'Simulation_Scalar_Theory.py'. To train an EBM on the data set obtained using the true action, run 'Training_EBM.py'. 'eval_trained_models.py' compares the EBM to the reference simulation based on the true action function. To approximate the partition function of the Gibbs distribution defined by the true action function and the EBM, normalized proposal distributions are required. In the course of this project, INNs and multivariate Normal distributions have been used for this purpose. The INNs can be trained using 'Training_INN.py' and the Normal distributions using 'get_covariance_matrix.py'. To obtain an approximation of the partition funciton for different hopping parameters use 'Importance_Sampling.py' to use Importance Sampling and 'Bridge_Sampling.py' to use Bridge Sampling. To compare the approximation of the partition function obtained from the different combinations of proposal distribution and sampling method, use 'eval_partition_function.py'. The approximations for the partition functions for different hopping parameters found using Bridege and Importance Sampling can be used to train a Neural Network to obtain an approximation of the partition function as a function of the hopping parameter. This can be done via 'Train_NN_Partition_function.py'.
