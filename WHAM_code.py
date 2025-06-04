#----------------------------------#
#-----------WHAM CODE--------------#
#-----------Written By-------------#
#-------Avijeet Kulshrestha--------#
#---avijeetkulshrestha@gmail.com---#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="WHAM implementation with argument parsing")
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to input file (e.g. wham_input.dat)')
    parser.add_argument('-cv', '--cv_range', nargs=2, type=float, required=True, help='Collective variable range: min max')
    parser.add_argument('-bins', '--N_bins', type=int, default=100, help='Number of histogram bins')
    parser.add_argument('-temp', '--TEMP', type=float, default=120, help='Temperature in Kelvin')
    parser.add_argument('-k', '--k', type=float, default=10, help='Force constant')
    parser.add_argument('-reweight', '--rew',type=str,choices=['yes', 'no'],default='no',help='Enable temperature reweighting (yes or no).')
    parser.add_argument('--target_temp','-target_temp',type=float,default=None,help="Target temperature for reweighting. If not provided, the code will identify the folding temperature.")
    parser.add_argument('-kb', '--kb', type=float, default=8.314e-3, help='Boltzmann constant in kJ/mol/K (default: 8.314e-3)')
    parser.add_argument('-tol', '--tolerance', type=float, default=1e-5, help='Convergence tolerance (default: 1e-7)')
    parser.add_argument('-ite', '--max_iter', type=int, default=10000, help='Maximum WHAM iterations (default: 10000)')
    return parser.parse_args()

# ----- PARAMETERS -----
args = parse_args()
print("Running WHAM with parameters:")
print(f"Input file: {args.file}")
print(f"CV range: {args.cv_range}")
print(f"Number of bins: {args.N_bins}")
print(f"Temperature: {args.TEMP} K", ", or taken from the input file")
print(f"Force constant: {args.k}", ", or taken from the input file")
print(f"Boltzmann constant: {args.kb}")
print(f"Tolerance: {args.tolerance}")
print(f"Max iterations: {args.max_iter}")
print(f"User specified the temperature weighting as {args.rew}")
print("NOTE: When specified in both places, the input file values for temperature and force constant take precedence over the -temp and -k options.")

CV_range = args.cv_range
N_bins = args.N_bins
TEMP = args.TEMP
k = args.k
kb = args.kb
beta = 1 / (kb * TEMP)
tolerance = args.tolerance
max_iter = args.max_iter

if args.rew == 'yes':
    if args.target_temp is None:
        print("Reweighting enabled but no target temperature provided. identifying the folding temperature.\n")
        print("This part of code is under construction. Please manually define the target temperature.")
        # You can set args.target_temp = folding_temp here if needed
    else:
        print(f"Reweighting to target temperature: {args.target_temp}")
        target_temp=args.target_temp
        beta_target=1/(kb*target_temp)
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import cpu_count
        
        def extract_min_max_energy(file_path):
            try:
                data = pd.read_csv(file_path, sep=r'\s+', header=None, usecols=[2], engine='c', comment='#')
                energies = data[2].values
                return np.min(energies), np.max(energies)
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                return float('inf'), float('-inf')  # Use extremes for proper global min/max reduction

        def get_energy_range(file_list, max_workers=None):
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(extract_min_max_energy, file_list))

            min_values, max_values = zip(*results)
            global_min = min(min_values)
            global_max = max(max_values)
            return global_min, global_max
        
        # ----- READ INPUT -----
        FILE = pd.read_csv(args.file, sep=r'\s+', header=None, engine='python', comment='#')
        N_simulation = FILE.shape[0]

        # ----- CREATE BINS -----
        file_paths = FILE[0].tolist()
        total_cores = cpu_count()
        if (total_cores>2):
            max_workers = max(1, total_cores - 2)
            print("Using ", max_workers, " number of processors")
        else:
            max_workers = 1
            print("Using ", max_workers, " number of processors since you have less than 2 processors available")
        
        PE_range = (get_energy_range(file_paths, max_workers=max_workers))

        N_bins_PE = 100
        pe_bin_edges = np.linspace(PE_range[0], PE_range[1], N_bins_PE + 1)
        pe_bin_centers = 0.5 * (pe_bin_edges[1:] + pe_bin_edges[:-1])

        bin_size = (CV_range[1] - CV_range[0]) / N_bins
        bin_edges = np.linspace(CV_range[0], CV_range[1], N_bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Bin centers for 2D histograms
        bin_centers_2d = np.array(np.meshgrid(bin_centers, pe_bin_centers)).reshape(2, -1).T

        #---FUnctions to parallize
        def process_window(i, file_info, bin_edges, pe_bin_edges, beta_target, k, beta, kb):
            file_path = file_info[0]
            x0_i = file_info[1]
            k_i = k if len(file_info) < 3 else file_info[2]
            beta_i = beta if len(file_info) < 4 else 1 / (kb * file_info[3])

            data = pd.read_csv(file_path, sep=r'\s+', header=None, engine='python', comment='#')
            Qi = data[1].values
            Ei = data[2].values

            H, xedges, yedges = np.histogram2d(Qi, Ei, bins=[bin_edges, pe_bin_edges])
            bin_centers = 0.5 * (xedges[1:] + xedges[:-1])
            pe_bin_centers = 0.5 * (yedges[1:] + yedges[:-1])

            c_ijk_i = np.zeros((len(bin_centers), len(pe_bin_centers)))
            for j in range(len(bin_centers)):
                for k_pe in range(len(pe_bin_centers)):
                    cv = bin_centers[j]
                    pe = pe_bin_centers[k_pe]
                    bias_term = np.exp(-(beta_i - beta_target) * pe) * np.exp(-0.5 * beta_i * k_i * (cv - x0_i)**2)
                    c_ijk_i[j, k_pe] = bias_term

            return i, len(Qi), H, c_ijk_i

        def compute_histograms_and_bias_weights_parallel(FILE, bin_edges, pe_bin_edges, beta_target, k, beta, kb, max_workers=None):
            N_simulation = len(FILE)
            N_bins = len(bin_edges) - 1
            N_bins_PE = len(pe_bin_edges) - 1

            n_ij = np.zeros((N_bins, N_bins_PE))
            c_ijk = np.zeros((N_simulation, N_bins, N_bins_PE))
            Ni = [0] * N_simulation

            file_infos = [FILE.iloc[i].tolist() for i in range(N_simulation)]

            if max_workers is None:
                max_workers = max(1, cpu_count() - 4)

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_window, i, file_infos[i],
                        bin_edges, pe_bin_edges,
                        beta_target, k, beta, kb
                    ) for i in range(N_simulation)
                ]

                for f in futures:
                    i, Ni_i, H_i, c_ijk_i = f.result()
                    Ni[i] = Ni_i
                    n_ij += H_i
                    c_ijk[i] = c_ijk_i

            return n_ij, c_ijk, Ni

        #----
        n_ij, c_ijk, Ni = compute_histograms_and_bias_weights_parallel(FILE, bin_edges, pe_bin_edges, beta_target, k, beta, kb, max_workers=max_workers)

        fi = np.ones(N_simulation)
        for it in range(max_iter):
            p_ij = np.zeros((N_bins, N_bins_PE))
            
            for j in range(N_bins):
                for k_pe in range(N_bins_PE):
                    denom = 0
                    for i in range(N_simulation):
                        denom += Ni[i] * (c_ijk[i, j, k_pe] / fi[i])
                    if n_ij[j, k_pe] != 0 and denom != 0:
                        p_ij[j, k_pe] = n_ij[j, k_pe] / denom
                    else:
                        p_ij[j, k_pe] = 1e-15
            
            new_fi = np.zeros(N_simulation)
            eps = 0
            for i in range(N_simulation):
                fn = np.sum(c_ijk[i] * p_ij)
                eps += (1 - fn / fi[i]) ** 2
                new_fi[i] = fn
            
            fi = new_fi
            if eps < tolerance:
                print(f"Converged at iteration {it}")
                break
            print(f"Iteration {it}, error = {eps}")


        p_cv = np.sum(p_ij, axis=1)
        p_cv /= np.sum(p_cv)

        FE = -kb * target_temp * np.log(p_cv)
        FE -= np.min(FE)

        np.savetxt(f"wham_output_{target_temp:.4f}K.dat", np.c_[bin_centers, FE], fmt='%8.5f', delimiter="\t", header="CV_center\tFreeEnergy")

        plt.title(f"WHAM Free Energy at {target_temp:.4f} K", fontsize=16)
        plt.xlabel("CV", fontsize=14)
        plt.ylabel("Free Energy (kJ/mol)", fontsize=14)
        plt.plot(bin_centers, FE, color='blue', linewidth=2.0)
        plt.grid(True)
        plt.savefig(f"wham_output_{target_temp:.4f}K.png", bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
        plt.tight_layout()
        plt.show()
        
else:
    print("Reweighting is not enabled. This code will run serially.")    
    # ----- READ INPUT -----
    FILE = pd.read_csv(args.file, sep=r'\s+', header=None, engine='python', comment='#')
    N_simulation = FILE.shape[0]
    
    # ----- CREATE BINS -----
    bin_size = (CV_range[1] - CV_range[0]) / N_bins
    bin_edges = np.linspace(CV_range[0], CV_range[1], N_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    # ----- COMPUTE Cij and Hij -----
    hij = [] #used in numerator of unbiased probability
    cij = []
    Ni = []
    
    for i in range(N_simulation):
        data = pd.read_csv(FILE[0][i], sep=r'\s+', header=None, engine='python', comment='#')
        Qi = data[1]
        hist, _ = np.histogram(Qi, bins=bin_edges)
        hij.append(hist)
        Ni.append(len(Qi))
        
        k_i = k
        if FILE.shape[1] > 2:
            k_i = FILE[2][i]
        
        x0_i = FILE[1][i]
        beta_i = beta
        if FILE.shape[1] > 3:
            TEMP_i = FILE[3][i]
            beta_i = 1 / (kb * TEMP_i)
    
        cij_i = [math.exp(-0.5 * beta_i * k_i * (x - x0_i) ** 2) for x in bin_centers]
        cij.append(cij_i)
    
    hij = np.array(hij).T  # shape: (N_bins, N_simulations)
    cij = np.array(cij).T  # shape: (N_bins, N_simulations)
    Ni = np.array(Ni)
    nj = np.sum(hij, axis=1)
    
    # ----- WHAM ITERATIONS -----
    fi = np.ones(N_simulation)
    wi = N_simulation #number of windows
    z = N_bins
    
    for it in range(max_iter):
        pj = np.zeros(z)
        for j in range(z):
            denom = 0.0
            for i in range(wi):
                denom += Ni[i] * (cij[j][i] / fi[i])
            if nj[j] != 0 and denom != 0:
                pj[j] = nj[j] / denom
            else:
                pj[j] = 1e-15
        
        new_fi = np.zeros(wi)
        eps = 0.0
        for i in range(wi):
            fn = 0.0
            for j in range(z):
                fn += cij[j][i] * pj[j]
            eps += (1 - (fn / fi[i])) ** 2
            new_fi[i] = fn
    
        fi = new_fi
        if eps < tolerance:
            print(f"Converged at iteration {it}")
            break
        print(f"Iteration {it}, error = {eps}")
    
    # ----- FREE ENERGY -----
    pj /= np.sum(pj)
    FE = -kb*TEMP* np.log(pj)
    #FE -= FE[np.searchsorted(bin_centers, CV_range[1] * 0.6)]  # shift reference
    FE -= np.min(FE)  # shift reference
    
    # ----- OUTPUT -----
    dg = np.c_[bin_centers, FE]
    np.savetxt("wham_output.dat", dg, fmt='%8.5f', delimiter="\t", header="CV_center\tFreeEnergy")
    
    plt.title("WHAM Free Energy", fontsize=16)
    plt.xlabel("CV", fontsize=14)
    plt.ylabel("Free Energy (kJ/mol)", fontsize=14)
    plt.plot(bin_centers, FE, color='blue', linewidth=2.0)
    plt.grid(True)
    plt.savefig("wham_output.png", bbox_inches = 'tight', pad_inches = 0.1, dpi=300)
    plt.tight_layout()
    plt.show()
