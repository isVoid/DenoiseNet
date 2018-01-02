import numpy as np
from tqdm import tqdm

import os
import argparse

def sample(Clean_dir = "../Images/CleanPatches/", Noisy_dir = "../Images/NoisyPatches", Clean_out = "../Images/Clean_Dev/", Noisy_out = "../Images/Noisy_Dev", take = 0.01):
    """Utility to generate dev set from the patch pool.

        Input: Clean_dir, Noisy_dir: directory of clean and noisy patches pool
            take: the percentage of dev set count in the entire pool

        File Output:
            Sampled dev set of clean and nosiy Images

        Return:
            None

    """

    if not os.path.exists(Clean_dir):
        raise IOError("Clean DIR not exist.")
    if not os.path.exists(Noisy_dir):
        raise IOError("Noisy DIR not exist.")
    if not os.path.exists(Clean_out):
        raise IOError("Clean dev dir not exist.")
    if not os.path.exists(Noisy_out):
        raise IOError("Noisy dev dir not exist.")

    clean = []
    noisy = []

    for root, _, files in os.walk(Clean_dir):
        files = sorted(files)
        for f in files:
            clean.append((root + f, f))

    for root, _, files in os.walk(Noisy_dir):
        files = sorted(files)
        for f in files:
            noisy.append((root + f, f))

    if not len(clean) == len(noisy):
        raise ValueError("Clean(%s) and Noisy(%s) Patches number mismatch" % (len(clean), len(noisy)))

    L = len(clean)
    pick = np.random.choice(L, int(L * take), replace = False)

    print ("Sampling %d frames" % L)
    for i in tqdm(pick):
        os.rename(clean[i][0], Clean_out+clean[i][1])
        os.rename(noisy[i][0], Noisy_out+noisy[i][1])

def main(args):

    c = args.Clean
    n = args.Noisy
    co = args.Clean_out
    no = args.Noisy_out

    take = args.take

    sample(c, n ,co, no, take)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Utility to sample dev set from patch pool.")

    parser.add_argument("--Clean", '-c', type = str, help = "Clean Patches Pool", default = "../Images/CleanPatches/")
    parser.add_argument("--Noisy", '-n', type = str, help = "Noisy Patches Pool", default = "../Images/NoisyPatches/")
    parser.add_argument("--Clean_out", '-co', type = str, help = "Clean dev output", default = "../Images/Clean_dev/")
    parser.add_argument("--Noisy_out", '-no', type = str, help = "Noisy dev output", default = "../Images/Noisy_dev/")
    parser.add_argument("--take", '-t', type = float, help = "Percentage of sample", default = 0.01)

    args = parser.parse_args()

    main(args)
