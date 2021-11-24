import h5py 

h5f = h5py.File("data/dataset_test_0.h5", "r")

list(h5f["X1"])