import time
import argparse


import numpy as np
import cupy as cp
import scipy


def main(backend: str = "numpy", size: int = 100):
    rng = np.random.default_rng(0)
    arr = rng.integers(10, size=size)
    if backend == "numpy":
        arr = arr
    elif backend == "cupy":
        arr = cp.array(arr)
    start = time.perf_counter()
    out = scipy.stats.rankdata(arr)
    # crude check for correctness (which requires
    # op to be sync'd/completed):
    assert out.size == arr.size
    end = time.perf_counter()
    print("elapsed time (s):", end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b",
                        "--backend",
                        type=str,
                        default="numpy",
                        help="array provider to use")
    parser.add_argument("-s",
                        "--size",
                        type=int,
                        default=100,
                        help="size of array fed to rankdata")
    args = parser.parse_args()
    main(backend=args.backend, size=args.size)
