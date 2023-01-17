import multiprocessing

cores = multiprocessing.cpu_count()
print(f"Number of cpu cores: {cores}, Number of threads: {2*cores}")
