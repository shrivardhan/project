

# to run, python dr.py [i|s|n] : i = iid model,s = sliding window model when target = host, n = sliding window model when target != host. python dr.py for results of sliding window simulations.





Redesign of code - 

# 2 parts - generate sample, simulation
# a) Generate samples -
# inputs: number of samples, length of new sequence, input sequence/probability distribution, IID/markov
# output: each sequence in a file, all file names in a file
# b) i) Simulation - sliding window
# input - base sequence file, File containing list of file names, max window length
# output - graph, values
# b) ii) Simulation - IID
# input - base sequence file, max window length
# output - graph, values

To improve performance - 
Each test run in the simulation is independent of each other and can be parallized.
