import numpy as np
import pickle


def create_testdata(path_to_txt_file, path_to_pickle_file):
    num_lines = sum(1 for line in open(path_to_txt_file, "r"))
    kicks = None
    prng_seed = None
    n_part = 0
    if num_lines > 1:
        n_part = num_lines - 1
        kicks = np.empty([n_part, 3], dtype=np.float64)
        with open(path_to_txt_file, "r") as fp:
            for ii, line in enumerate(fp):
                if ii < n_part:
                    line = line.strip("\n")
                    line = line.split(" ")
                    line = list(map(float, line))
                    kicks[ii] = np.array(line)
                else:
                    prng_seed = float(line.strip())

    with open(path_to_pickle_file, "wb") as fp:
        pickle.dump((n_part, prng_seed, kicks), fp)


if __name__ == "__main__":
    create_testdata("./precomputed_kicks.txt", "./precomputed_kicks.pickle")
