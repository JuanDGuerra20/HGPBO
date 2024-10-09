import concurrent.futures
import synthetic_dataset as sd
import changing_inputs_scripts as changing
import indep_changing_script as indep_changing
import full_indep_script as full_indep
import efficient_synthetic_script as efficient
from dataset_actions import *
import time


def func1(n, m):
    print(f"func1: starting {m}")
    for i in range(n):
        pass

    print("func1: finishing")


def func2(n, m):
    print(f"func2: starting {m}")
    for i in range(n):
        pass

    print("func2: finishing")

def parallel():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        dimension = 10
        nbr_query = 5
        training_iter = 5
        nbr_repetition = 5
        nbr_rand_init = 5
        k_vals = [1, 2]
        futures = []
        procedures = [sd.training_procedure, changing.training_procedure, indep_changing.training_procedure,
                      efficient.training_procedure, full_indep.training_procedure]

        t0 = time.time()

        for dataset_num in [1, 2, 3]:
            data_name, data_creation_func, eps = get_dataset_info(dataset_num)

            for train in procedures:
                futures.append(executor.submit(
                    train, nbr_query, nbr_repetition, nbr_rand_init, dimension, training_iter, k_vals, data_name,
                          data_creation_func, eps))

        concurrent.futures.wait(futures)
        t1 = time.time()

        print(f"Parallel time: {t1 - t0}")


if __name__ == '__main__':

    parallel()