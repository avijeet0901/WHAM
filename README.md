# WHAM
This code applies the Weighted Histogram Analysis Method (WHAM) in both temperature and one collective variable (CV) space. The biased potential used in the simulation is harmonic, as in umbrella sampling, and requires two inputs: the center of the harmonic potential and the spring constant.

# Package requirement
```
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install argparse
pip3 install concurrent
pip3 install multiprocessing
```
# Usage
The help options can be printed out with the -h option.

The wham_input file should have the following format, including all the COLVAR files:
/location/to/file/COLVAR biased_value spring_constant temperature
...
A wham input file demo file is included in the repository. 

where the file COLVAR should have at least three columns, including the time step, CV value, and the potential energy of the system. The potential energy should not contain the biased energy contribution. A COLVAR demo file is included in the repository. 

CV argument requires two inputs - the minimum and the maximum range of the biased CV. NOTE: The code is not customized to handle the periodic CV and will be added in the future. 

bins argument defines the number of bins for the CV. NOTE: The number of bins for the temperature space is 100 by default, and the minimum and maximum range of the energy value is computed from the input files. 

The temp argument defines the temperature for the simulations. If the temperature is also defined in the wham_input file, the code uses the wham_input file temperature for each simulation.

k argument defines the spring constant for the simulations. Again, if it is also defined in the wham_input file, the code uses the wham_input file value for each simulation.

NOTE: It is suggested to define the temperature and spring constant in the wham_input file. 

The reweight argument switches on the reweighting in the temperature space along with the CV space. If the user has performed the simulations at different temperatures and wants to reweight the probability (free energy) at a given target temperature, this segment switches on the reweighting at a target temperature. 

If reweight input is provided as "yes", the user must provide the target_temp argument. If left undefined, the code will identify and report the folding temperature. However, this part of the code is not complete. Hence, the user must provide the target_tempetaure. 

The kb argument is to change the units from kJ to kcal. It is suggested to work with kJ units. 

The tol argument is to change the level of accuracy for the WHAM convergence. 

The iteration argument helps in changing the number of iterations allowed in the WHAM convergence. 

--------
Feel free to contact me at avijeetkulshrestha@gmail.com if you find any bugs or need help running the code

# Citations
Please cite the following paper if you are using the code or any segment of code:

```
@article{kulshrestha2022finite,
  title={Finite temperature string method with umbrella sampling using path collective variables: application to secondary structure change in a protein},
  author={Kulshrestha, Avijeet and Punnathanam, Sudeep N and Ayappa, K Ganapathy},
  journal={Soft Matter},
  volume={18},
  number={39},
  pages={7593--7603},
  year={2022},
  publisher={Royal Society of Chemistry}
}
```
# Future updates 
The following part will be added in the future
1. Multiple CV unbiasing: Starting from 2 CV, the aim is to write code for N numbers of CV. 
2. Bootstraping for error analysis: At present, users can make chunks of the input data and rerun the code multiple times to identify the error in estimation.
3. Identification of folding temperature: At present, users need to rerun the code at multiple temperatures until they find an equal weightage to folded and unfolded states. 

Best Regards.  
Avijeet Kulshrestha



