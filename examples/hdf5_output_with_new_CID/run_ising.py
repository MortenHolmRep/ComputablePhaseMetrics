"""
Performs Monte Carlo simulations of the 2D Ising model using various algorithms
(Kawasaki, Metropolis, Wolff). It calculates physical and complexity metrics for the system
and saves the results in an HDF5 file.

Classes:
    - IsingSimulation: Handles the simulation, data processing, and result saving for the Ising model.

Functions:
    - setup_logging(): Configures logging for the script.
    - parse_arguments(): Parses command-line arguments for the simulation parameters.
    - main(): Main entry point for the script, orchestrating the simulation workflow.

Usage:
    Run the script with the desired parameters for the Ising model simulation. Example:
    ```
    python run_isingv2.py -L 64 -nsteps 2048 -T 2.5 -o output/simulation.h5 -m Metropolis
    ```

Command-line Arguments:
    - -L: Number of spin sites in one direction (default: 64).
    - -nsteps: Number of temporal measurements (default: 2048).
    - -T: Temperature for the 2D Ising Model (default: 0.1).
    - -o: Path to output directory (default: "output/simulation.h5").
    - -m: Algorithm to use (Kawasaki, Metropolis, Wolff) (default: "Kawasaki").
    - -lat: Path to starting lattice HDF5 file (optional).

"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ComputablePhaseMetrics.ComputableInformationDensity.cid import (
    interlaced_time,
    sequential_time,
)
from ComputablePhaseMetrics.models.ising.ising2d import (
    Kawasaki,
    Metropolis,
    Wolff,
)


class HDF5Handler:
    """Handles HDF5 file operations for saving simulation results."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        os.makedirs(self.output_path.parent, exist_ok=True)

    def save_results(
        self,
        lattice: np.ndarray,
        lattice_decimated: np.ndarray,
        energy: list[float],
        magnetization: list[float],
        physical_metrics: dict[str, float],
        cid_metrics: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        logging.info(f"Saving results to {self.output_path}")

        try:
            with h5py.File(self.output_path, "w") as f:
                # Save metadata
                metadata_group = f.create_group("metadata")
                for key, value in metadata.items():
                    metadata_group.attrs[key] = value

                # Save lattices
                lattices_group = f.create_group("lattices")
                lattices_group.create_dataset("full", data=lattice, compression="gzip")
                lattices_group.create_dataset(
                    "decimated", data=lattice_decimated, compression="gzip"
                )

                # Save observables
                observables_group = f.create_group("observables")
                observables_group.create_dataset(
                    "energy", data=energy, compression="gzip"
                )
                observables_group.create_dataset(
                    "magnetization", data=magnetization, compression="gzip"
                )
                for key, value in physical_metrics.items():
                    observables_group.attrs[key] = value

                # Save CID metrics
                cid_group = f.create_group("cid")
                for key, value in cid_metrics.items():
                    if isinstance(value, (list, np.ndarray)) and (
                        not isinstance(value, np.ndarray) or value.size > 1
                    ):
                        cid_group.create_dataset(key, data=value, compression="gzip")
                    else:
                        cid_group.attrs[key] = value

                logging.info("Results saved successfully")
        except Exception as e:
            logging.error(f"Error saving results to HDF5: {e}")
            raise


class IsingSimulation:
    """Class to handle Ising model simulations and data processing."""

    @staticmethod
    def _convert_to_native(value):
        if isinstance(value, np.ndarray):
            return value
        elif isinstance(value, np.generic):
            return value.item()
        return value

    def __init__(
        self,
        L: int,
        nsteps: int,
        temperature: float,
        method: str = "Metropolis",
        base_decimation: int = 4096,
        initial_lattice_path: str | None = None,
    ):
        self.L = L
        self.nsteps = nsteps
        self.temperature = float(temperature)
        self.beta = (
            1.0 / self.temperature if self.temperature > 0 else 1.1050349147204485e-116
        )
        self.base_decimation = base_decimation
        self.order = int(np.log2(L) - 1)
        self.method_name = method
        self.method_class = eval(method)
        self.ising = self.method_class(L, self.temperature)

        if initial_lattice_path:
            self._load_initial_lattice(initial_lattice_path)

        self.interlaced_time = interlaced_time(nbits=self.order, nshuff=4)
        self.sequential_time = sequential_time(nbits=self.order, nshuff=4)

        self.lattice_data = []
        self.energy = []
        self.magnetization = []

        logging.info(
            f"Initialized {self.method_name} simulation at temperature {self.temperature}"
        )
        logging.info(f"System size: {L}x{L}, order: {self.order}")

    def _load_initial_lattice(self, lattice_path: str) -> None:
        try:
            with h5py.File(lattice_path, "r") as f:
                if "lattice" in f:
                    lattice = f["lattice"][:]
                    input_state = [int(value) for value in lattice[-1, :, :].flatten()]
                    self.ising.state = input_state
                    logging.info(f"Loaded initial lattice from {lattice_path}")
                else:
                    logging.warning(f"No lattice data found in {lattice_path}")
        except Exception as e:
            logging.error(f"Failed to load initial lattice: {e}")
            raise

    def thermalize(self, steps: int = 1024) -> None:
        logging.info(f"Performing {steps} thermalization steps")
        self.ising(steps)

    def run_simulation(self) -> None:
        logging.info(f"Running simulation for {self.nsteps} steps")

        for n in range(self.nsteps):
            if n % max(1, self.nsteps // 10) == 0:
                logging.info(f"Step {n}/{self.nsteps}")

            self.ising(1)
            self.lattice_data.extend(self.ising.state)
            self.energy.append(self.ising.energy() / self.ising.L2)
            self.magnetization.append(self.ising.magnetization() / self.ising.L2)

    def get_decimated_lattice(self) -> tuple[np.ndarray, np.ndarray]:
        skip = max(1, int(self.nsteps / self.base_decimation))
        lattice = np.array(self.lattice_data).reshape(self.nsteps, self.L, self.L)
        lattice = np.where(lattice == 1, 1, -1)
        return lattice, lattice[::skip, :, :]

    def get_cid_lattice(self, data: np.ndarray) -> np.ndarray:
        window = slice(self.L // 4, 3 * self.L // 4)
        return data[:, window, window]

    def calculate_cid_metrics(
        self, lattice: np.ndarray, lattice_decimated: np.ndarray
    ) -> dict[str, Any]:
        logging.info("Calculating Interlaced Time CID metrics")
        cid_it, q_it = self.interlaced_time(self.get_cid_lattice(lattice))
        cid_it_deci, q_it_deci = self.interlaced_time(
            self.get_cid_lattice(lattice_decimated)
        )

        logging.info("Calculating Sequential Time CID metrics")
        skip_factor = 1 << (self.order - 1)
        expected_size = 1 << self.order

        if self.L == expected_size:
            seq_lattice = lattice[::skip_factor]
            seq_lattice_deci = lattice_decimated[::skip_factor]
        else:
            window = slice(0, expected_size)
            if self.L > expected_size:
                seq_lattice = lattice[::skip_factor, window, window]
                seq_lattice_deci = lattice_decimated[::skip_factor, window, window]
            else:
                logging.warning(
                    f"Lattice size {self.L} smaller than expected {expected_size}. Padding with zeros."
                )
                padded = np.zeros(
                    (lattice.shape[0], expected_size, expected_size),
                    dtype=lattice.dtype,
                )
                padded[:, : self.L, : self.L] = lattice
                seq_lattice = padded[::skip_factor]

                padded_deci = np.zeros(
                    (lattice_decimated.shape[0], expected_size, expected_size),
                    dtype=lattice_decimated.dtype,
                )
                padded_deci[:, : self.L, : self.L] = lattice_decimated
                seq_lattice_deci = padded_deci[::skip_factor]

        cid_st, cidstd_st = self.sequential_time(seq_lattice)
        cid_st_deci, cidstd_st_deci = self.sequential_time(seq_lattice_deci)

        # cid_st, cidstd_st, q_st, qstd_st = self.sequential_time(seq_lattice)
        # cid_st_deci, cidstd_st_deci, q_st_deci, qstd_st_deci = self.sequential_time(
        #     seq_lattice_deci
        # )

        metrics = {
            "CIDit": self._convert_to_native(cid_it),
            "CIDit_decimated": self._convert_to_native(cid_it_deci),
            "Qit": self._convert_to_native(q_it),
            "Qit_decimated": self._convert_to_native(q_it_deci),
            "CIDst_mean": self._convert_to_native(cid_st),
            "CIDst_mean_decimated": self._convert_to_native(cid_st_deci),
            "CIDst_std": self._convert_to_native(cidstd_st),
            "CIDst_std_decimated": self._convert_to_native(cidstd_st_deci),
            # "Qst_mean": self._convert_to_native(q_st),
            # "Qst_mean_decimated": self._convert_to_native(q_st_deci),
            # "Qst_std": self._convert_to_native(qstd_st),
            # "Qst_std_decimated": self._convert_to_native(qstd_st_deci),
        }

        logging.info(
            f"Generated CID metrics with types: {[(k, type(v)) for k, v in metrics.items()]}"
        )

        return metrics

    def calculate_physical_metrics(self) -> dict[str, float]:
        E = np.array(self.energy)
        M = np.array(self.magnetization)

        return {
            "energy_mean": float(np.mean(E)),
            "energy_std": float(np.std(E)),
            "magnetization_mean": float(np.mean(M)),
            "magnetization_std": float(np.std(M)),
            "specific_heat": float(
                self.beta**2 * (np.mean(E**2) - np.mean(E) ** 2) / self.L**2
            ),
            "susceptibility": float(
                self.beta * (np.mean(M**2) - np.mean(M) ** 2) / self.L**2
            ),
        }

    def save_results(self, output_path: Path, args) -> None:
        lattice, lattice_decimated = self.get_decimated_lattice()
        physical_metrics = (
            self.calculate_physical_metrics() if args.save_physical_metrics else {}
        )
        cid_metrics = (
            self.calculate_cid_metrics(lattice, lattice_decimated)
            if args.save_cid_metrics
            else {}
        )

        metadata = (
            {
                "temperature": self.temperature,
                "system_size": self.L,
                "sweeps": self.nsteps,
                "method": self.method_name,
            }
            if args.save_metadata
            else {}
        )

        hdf5_handler = HDF5Handler(output_path)
        hdf5_handler.save_results(
            lattice=lattice if args.save_lattices else None,
            lattice_decimated=lattice_decimated if args.save_lattices else None,
            energy=self.energy if args.save_physical_metrics else [],
            magnetization=self.magnetization if args.save_physical_metrics else [],
            physical_metrics=physical_metrics,
            cid_metrics=cid_metrics,
            metadata=metadata,
        )


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parameters for Ising Model Monte-Carlo simulation"
    )
    parser.add_argument(
        "-L",
        dest="L",
        help="Number of spin sites in one direction",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-nsteps",
        dest="nsteps",
        help="Number of temporal measurements",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "-T",
        dest="temp",
        help="Temperature for the 2D Ising Model",
        type=str,
        default="0.1",
    )
    parser.add_argument(
        "-o",
        dest="path",
        help="Path to output directory",
        type=Path,
        default="output/simulation.h5",
    )
    parser.add_argument(
        "-m",
        dest="method",
        help="Algorithm: Kawasaki, Metropolis, Wolff",
        type=str,
        default="Kawasaki",
    )
    parser.add_argument(
        "-lat",
        dest="lat",
        help="Path to starting lattice HDF5 file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save-lattices",
        dest="save_lattices",
        help="Save lattice data to HDF5",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--save-physical-metrics",
        dest="save_physical_metrics",
        help="Save physical metrics to HDF5",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save-cid-metrics",
        dest="save_cid_metrics",
        help="Save CID metrics to HDF5",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save-metadata",
        dest="save_metadata",
        help="Save metadata to HDF5",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_arguments()

    simulation = IsingSimulation(
        L=args.L,
        nsteps=args.nsteps,
        temperature=args.temp,
        method=args.method,
        initial_lattice_path=args.lat,
    )

    logging.info("Starting simulation")
    simulation.thermalize()
    simulation.run_simulation()
    simulation.save_results(args.path, args)
    logging.info("Simulation completed successfully")


if __name__ == "__main__":
    main()
