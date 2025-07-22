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
The help options can be printed out with -h option.

The wham_input file should have the following format, including all the COLVAR files:
/location/to/file/COLVAR biased_value spring_constant temperature
...

where the file COLVAR should have at least three column, including the time step, CV value, and the potential energy of the system. The potential energy should not contain the biased enegy contribution.

cv argument require two input - the minimum and the maximum range of the biased CV. NOTE: code is not customized to handle the periodic CV, and will be added in the future. 

bins argument defines the number of bins for the CV. NOTE: the number of bins for the temperature space is 100 by default and the minium and maximum range of the energy value is computed from the input files. 

temp argument defines the temperature fo the simulations. If the temperature is also defined in the wham_input file, the code uses the wham_input file temperature for each simulation.

k argument defines the spring constant for the simulations. Again, if it is also defines in the wham_input file, , the code uses the wham_input file value for each simulation.

NOTE: It is suggested to define the temperature and spring constant in the wham_input file. 

The reweight argument switch on the reweighting in the temperature space along with the CV space. If the user has performed the simulations at different temperature and wants to reweight the probability (free energy) at a given target temperature, this segment switchs on the reweighting at a target temperature. 

If reweight input provided as "yes", the user must provide the target_temp argument. If left undefined, the code will identify and report the folding temperature. However, this part of the code is not complete. Hence, the user must provide the target_tempetaure. 

The kb argument is to change the units from kJ to kcal. Is is suggested to work with kJ units. 

The tol argument is to change the level of accuracy for the WHAM convergence. 

the ite argument helps in changing the number of iteration allowed in the WHAM convergence. 

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

Best Regards.  
Avijeet Kulshrestha



