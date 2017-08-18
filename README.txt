Sample Compilation and Execution command

mpicc  pkmeans.c
mpirun -n 8 ./a.out 1000 2 3

where 8 is the number of processors(p)
1000 is the synthetic data size(n)
2 is the feature size(f)
3 is the cluster size (k)

Same applies to loadpkmeans.c 
for dynamic load balancing