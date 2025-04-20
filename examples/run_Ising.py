import argparse
import json
import os
from pathlib import Path

import numpy as np

from ComputablePhaseMetrics.cid import InterlacedTime, SequentialTime
from ComputablePhaseMetrics.models.ising.ising2d import Kawasaki, Metropolis, Wolff


def get_decimated_lattice(data):
    # number of nsteps to skip
    skip = max(1, int(nsteps / base))
    # the region to save
    # window = slice(0, 2*L) # the entire region
    lattice = np.array(data).reshape(nsteps, L, L)
    lattice = np.where(lattice == 1, "1", "-1")
    return lattice, lattice[
        ::skip, :, :
    ]  # lattice[:, window, window], lattice[::skip, window, window]


def get_lattice(data):
    """Entire lattice from data"""
    lattice = data.pop("lattice")
    # lattice = [int(value) for value in latice]
    lattice = np.array(lattice).reshape(nsteps, L, L)
    # lattice = np.where(latice==1, '1', '-1')
    return lattice


# def get_CIDlattice(data):
#     """Reduced lattice size"""
#     # the region to save
#     window = slice(L // 4, 3 * L // 4)  # only the center region
#     return data[:, window, window]


def get_CIDlattice(data):
    """Extract the center region for CID calculations for stability"""
    # Extract the center region that matches the expected size based on order
    expected_size = 1 << order  # 2^order
    start_idx = (L - expected_size) // 2
    end_idx = start_idx + expected_size

    # If L is already the correct size, just return the data
    if L == expected_size:
        return data

    # Otherwise extract the center region
    return data[:, start_idx:end_idx, start_idx:end_idx]


def main():
    global base, L, nsteps, order

    # ** Parse. Get the input for the simulation. **
    parser = argparse.ArgumentParser(
        description="Parameters for Monte-Carlo simulation:"
    )
    parser.add_argument(
        "-L",
        dest="L",
        help="Number of spin sites in one direction.",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-nsteps",
        dest="nsteps",
        help="Number of temporal measurements to be outputted.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "-T",
        dest="temp",
        help="Temperature for the 2D Ising Model.",
        type=str,
        default=0.1,
    )
    parser.add_argument(
        "-o",
        dest="path",
        help="Path to output directory.",
        type=Path,
        default="\output",
    )
    parser.add_argument(
        "-m",
        dest="method",
        help="either kawasaki, metropolis, wolff or metropolis-wolff hybrid",
        type=str,
        default="Metropolis",
    )
    parser.add_argument(
        "-lat",
        dest="lat",
        help="starting lattice",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    basepath = args.path
    L = args.L
    nsteps = args.nsteps  # number of measurements

    base = 4096  # decimation base
    order = int(np.log2(L))
    print("order", order)

    T = args.temp  # float(temp.replace(',','.'))
    beta = 1.0 / T if T > 0 else 1.1050349147204485e-116
    print(f"order: {order}, Temperature: {T}")

    # ** Initialization of chosen algortihm of method and CID **
    it = InterlacedTime(
        order=order,
        alpha=2,
        nshuff=8,
    )

    st = SequentialTime(order=order, alpha=2, nshuff=8)

    method = eval(args.method)
    ising = method(L, T)

    # if starting lattice is given
    if args.lat != None:
        file = open(args.lat)
        data = json.load(file)
        lattice = get_lattice(data)
        input_state = [int(value) for value in lattice[-1, :, :].flatten()]
        ising.state = input_state

    print("using " + str(method) + " at temperature {}".format(T))

    print("performing thermalisation")
    # 1024 is the number of thermalisation steps and was chosen to be large enough
    # to ensure that the system is in equilibrium.
    ising(1024)

    # initialize output dict
    res = {
        "temperature": T,
        "system_size": L,
        "sweeps": nsteps,
        "lattice": [],
        "lattice_decimated": [],
        "energy_mean": [],
        "energy_std": [],
        "magnetization_mean": [],
        "magnetization_std": [],
        "susceptibility": [],
        "specific_heat": [],
        "CIDit": [],
        "CIDit_decimated": [],
        "Qit": [],
        "Qit_decimated": [],
        "CIDst_mean": [],
        "CIDst_mean_decimated": [],
        "CIDst_std": [],
        "CIDst_std_decimated": [],
        "Qst_mean": [],
        "Qst_mean_decimated": [],
        "Qst_std": [],
        "Qst_std_decimated": [],
    }

    # collect data
    E = []
    M = []
    lattice_data = []
    print("nsteps")
    for n in range(args.nsteps):
        ising(1)
        lattice_data.extend(ising.state)
        E.append(ising.energy() / ising.L2)
        M.append(ising.magnetization() / ising.L2)

    print("writing lattice")
    lattice, lattice_decimated = get_decimated_lattice(lattice_data)

    print("Interlaced time")
    CID_it, Q_it = it(get_CIDlattice(lattice))
    CID_it_deci, Q_it_deci = it(get_CIDlattice(lattice_decimated))

    print("Sequential time")
    CID_st, CIDstd_st, Q_st, Qstd_st = st(lattice[:: 1 << (order - 1)])
    CID_st_deci, CIDstd_st_deci, Q_st_deci, Qstd_st_deci = st(
        lattice_decimated[:: 1 << (order - 1)]
    )

    res["lattice"] = lattice.flatten().tolist()
    res["lattice_decimated"] = lattice.flatten().tolist()

    res["energy_mean"] = np.mean(E)
    res["energy_std"] = np.std(E)
    res["specific_heat"] = np.mean(
        beta**2
        * (np.mean(np.array(E) ** 2) - np.mean(np.array(E)) ** 2)
        / np.array(L) ** 2
    )

    res["magnetization_mean"] = np.mean(M)
    res["magnetization_std"] = np.std(M)
    res["susceptibility"] = np.mean(
        beta
        * (np.mean(np.array(M) ** 2) - np.mean(np.array(M)) ** 2)
        / (np.array(L) ** 2)
    )

    # Interlaced
    res["CIDit"] = CID_it
    res["CIDit_decimated"] = CID_it_deci
    res["Qit"] = Q_it
    res["Qit_decimated"] = Q_it_deci
    # Sequential
    res["CIDst_mean"] = CID_st
    res["CIDst_mean_decimated"] = CID_st_deci
    res["CIDst_std"] = CIDstd_st
    res["CIDst_std_decimated"] = CIDstd_st_deci
    res["Qst_mean"] = Q_st
    res["Qst_mean_decimated"] = Q_st_deci
    res["Qst_std"] = Qstd_st
    res["Qst_std_decimated"] = Qstd_st_deci

    # save data
    outpath = basepath.joinpath("out.json")
    os.makedirs(outpath.parent, exist_ok=True)
    with open(outpath, "w") as outfile:
        json.dump(res, outfile)


if __name__ == "__main__":
    main()
