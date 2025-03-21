#!/usr/bin/env python3
import argparse
import importlib
import logging
import multiprocessing
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Callable, Sequence, TypedDict

import numpy as np
import openfermion as of
import pandas as pd
import scipy
import tequila as tq
from numpy import eye, floating
from numpy.typing import NDArray
from tequila.quantumchemistry import QuantumChemistryBase
from tqdm import tqdm

from quanti_gin.shared import (
    generate_min_global_distance_edges,
)

logger = logging.getLogger(__name__)


# static seed for development
seed = None
np_random = np.random.default_rng(seed)
rand = Random(seed)


class CustomData(TypedDict):
    name: str
    data: "list | str | float | int | np.number"


class OptimizationResult(TypedDict):
    energy: float
    orbital_coefficients: NDArray | None
    orbital_transformation: NDArray | None
    variables: dict | None
    circuit: Any | None
    molecule: Any | None
    custom_data: Sequence[CustomData] | None


@dataclass
class Job:
    id: int
    geometry: str
    optimization_algorithm: Callable[..., OptimizationResult]
    coordinates: NDArray = field(default_factory=np.array)

    edge_distances: Sequence[floating] = field(default_factory=list)
    coordinate_distances: Sequence[floating] = field(default_factory=list)
    # custom optimization algorithm
    custom_job_data: Sequence[CustomData] = field(default_factory=list)

    # Flag to determine whether fidelity should be calculated
    calculate_fidelity: bool = False

    # custom arguments that one might want to pass to the optimization execution
    kwargs: dict = field(default_factory=dict)


class DataGenerator:
    @classmethod
    def parse_geometry_string(cls, geometry_string: str) -> NDArray[np.float64]:
        lines = geometry_string.split("\n")
        coordinates = []
        for line in lines:
            if not line:
                continue
            line = line.strip()
            parts = line.split(" ")
            coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(coordinates)

    @classmethod
    def generate_geometry_string(cls, coordinates: NDArray[np.float64]):
        geom = ""
        for coordinate in coordinates:
            geom += f"h {coordinate[0]:f} {coordinate[1]:f} {coordinate[2]:f}\n"

        return geom

    @classmethod
    def generate_coordinates(
        cls, count: int, max_distance: int, existing_coordinates=None
    ) -> NDArray:
        if count % 2 != 0:
            print("warning: you should be using multiples of two")
        coordinates = existing_coordinates

        if coordinates is None:
            coordinates = np.array([[0, 0, 0]], dtype=np.float64, ndmin=2)
        # randomize the prev_coord out of the coordinates
        for i in range(1, count):
            prev_coord = np_random.choice(coordinates)
            min_distance = 0
            # make sure we have a minimum of 0.5 angstrom between the atom cores
            while min_distance <= 0.5:
                new_coord = prev_coord + (np_random.random(3) * max_distance)

                min_distance = np.min(np.linalg.norm(new_coord - coordinates, axis=1))
            coordinates = np.append(coordinates, new_coord.reshape(1, 3), axis=0)
        return coordinates

    @classmethod
    def get_pair_distances(cls, coordinates):
        pairs = []
        for i in range(0, len(coordinates), 2):
            pairs.append((coordinates[i], coordinates[i + 1]))

        distances = [pair[0].distance(pair[1]) for pair in pairs]
        return distances

    @classmethod
    def generate_initial_guess_from_edges(
        cls, vertices: NDArray[np.float64], edges: list[tuple[int, int]]
    ) -> NDArray:
        count = len(vertices)
        if count % 2 != 0:
            raise ValueError("count needs to be a multipe of two")
        initial_guess = eye(count)
        for edge in edges:
            initial_guess[edge[0]][edge[0]] = 1.0
            initial_guess[edge[0]][edge[1]] = 1.0
            initial_guess[edge[1]][edge[0]] = -1.0
            initial_guess[edge[1]][edge[1]] = 1.0
        return initial_guess

    @classmethod
    def run_fci_optimization(cls, molecule: QuantumChemistryBase, *args, **kwargs):
        return OptimizationResult(
            energy=molecule.compute_energy("fci"),
            orbital_coefficients=molecule.integral_manager.orbital_coefficients,
            orbital_transformation=None,
            variables=None,
            custom_data=None,
            circuit=None
        )

    @classmethod
    def run_spa_optimization(
        cls, molecule: QuantumChemistryBase, *args, coordinates, **kwargs
    ):
        edges = generate_min_global_distance_edges(coordinates)
        initial_guess = cls.generate_initial_guess_from_edges(
            edges=edges, vertices=coordinates
        ).T

        U = molecule.make_ansatz(name="HCB-SPA", edges=edges) # Hardcore Boson Circuit

        opt = tq.chemistry.optimize_orbitals(
            molecule.use_native_orbitals(),
            U,
            initial_guess=initial_guess,
            silent=True,
            use_hcb=True,
        )

        mol = opt.molecule
        H = mol.make_hardcore_boson_hamiltonian()

        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)

        U_x = mol.hcb_to_me(U)

        return OptimizationResult(
            energy=result.energy,
            orbital_coefficients=opt.molecule.integral_manager.orbital_coefficients,
            orbital_transformation=opt.mo_coeff,
            variables=result.variables,
            circuit=U_x,
            molecule=mol,
            custom_data=None
        )

    @classmethod
    def get_ground_states(cls, molecule, num_states=10):
        mol = molecule

        H = mol.make_hamiltonian().to_openfermion()
        H_sparse = of.linalg.get_sparse_operator(H)
        v, vv = scipy.sparse.linalg.eigsh(H_sparse, k=num_states, ncv=100, sigma=mol.compute_energy("fci"))

        wfns = [tq.QubitWaveFunction.from_array(vv[:, i]) for i in range(num_states)]
        energies = v.tolist()

        return wfns, energies

    @classmethod
    def calculate_fidelity(cls, true_state, optimized_state):
        inner_product = true_state.inner(optimized_state)
        fidelity = abs(inner_product) ** 2

        return fidelity

    @classmethod
    def generate_jobs(
        cls,
        number_of_atoms,
        number_of_jobs,
        size=None,
        method="spa",
        custom_method=None,
        compare_to=[],
        calculate_fidelity=False, # parameter for fidelity between methods
    ):
        def get_algorithm_from_method(method) -> Callable:
            if callable(method):
                return method
            if method == "spa":
                return cls.run_spa_optimization
            if method == "fci":
                return cls.run_fci_optimization
            raise ValueError(f"invalid method {method}")

        jobs = []
        sizes = [2, 3, 4]
        size_ratios = [0.3, 0.3, 0.3]

        for i in tqdm(range(number_of_jobs), desc="generating jobs"):
            coordinates = []
            if size is None:
                size = rand.choices(sizes, weights=size_ratios)[0]
            coordinates = cls.generate_coordinates(
                count=number_of_atoms, max_distance=size
            )
            geometry = cls.generate_geometry_string(coordinates)

            custom_job_data = []

            fidelity_flag = calculate_fidelity and method != "fci"

            if custom_method:
                job = Job(
                    id=i,
                    geometry=geometry,
                    coordinates=coordinates,
                    optimization_algorithm=custom_method,
                    custom_job_data=custom_job_data,
                    calculate_fidelity=fidelity_flag
                )
                jobs.append(job)
            else:
                job = Job(
                    id=i,
                    geometry=geometry,
                    coordinates=coordinates,
                    optimization_algorithm=get_algorithm_from_method(method),
                    custom_job_data=custom_job_data,
                    calculate_fidelity=fidelity_flag
                )
                jobs.append(job)

            if compare_to:
                for compare in compare_to:
                    # do not duplicate methods
                    if compare == method and not custom_method:
                        continue
                    fidelity_flag = calculate_fidelity and compare != "fci"
                    job = Job(
                        id=i,
                        geometry=geometry,
                        coordinates=coordinates,
                        optimization_algorithm=get_algorithm_from_method(compare),
                        custom_job_data=custom_job_data,
                        calculate_fidelity=fidelity_flag
                    )
                    jobs.append(job)

        return jobs

    @classmethod
    def execute_job(cls, job: Job, basis_set="sto-3g"):
        mol = tq.Molecule(geometry=job.geometry, basis_set=basis_set)

        result = job.optimization_algorithm(
            mol, coordinates=job.coordinates, **job.kwargs
        )

        if job.calculate_fidelity:
            if "circuit" in result:
                mol = result["molecule"] # use same molecule as in the optimization
                optimized_variables = result["variables"]
                U = result["circuit"]

                state = tq.simulate(U, optimized_variables)

                wfns, energies = cls.get_ground_states(molecule=mol)
                fidelity = 0.0
                k = 0

                while k < len(wfns) - 1:
                    fidelity += cls.calculate_fidelity(state, wfns[k])
                    print(fidelity)
                    if fidelity < 1e-6:
                        k += 1
                        continue
                    if abs(energies[k] - energies[k + 1]) < 1e-6:
                        break

                    k += 1

                return {"result": result, "fidelity": fidelity}
        else:
            return {"result": result, "fidelity": None}



    @classmethod
    def create_result_df(
        cls, jobs: Sequence[Job], job_results: Sequence[dict[str, Any]]
    ) -> pd.DataFrame:
        max_variable_count = max(
            [0]
            + [
                len(result["variables"])
                for result in job_results
                if "variables" in result and result["variables"]
            ]
        )
        df = pd.DataFrame(
            {
                "job_id": [job.id for job in jobs],
                "method": [
                    f"{job.optimization_algorithm.__module__}.{job.optimization_algorithm.__name__}"
                    for job in jobs
                ],
                "optimized_energy": [entry["result"]["energy"] for entry in job_results],
                "fidelity": [entry["fidelity"] for entry in job_results],
                "optimized_variable_count": max_variable_count,
                "atom_count": [len(job.coordinates) for job in jobs],
                "edge_count": [len(job.coordinates) // 2 for job in jobs],
            }
        )


        # apply custom data to data frame
        for i, entry in enumerate(job_results):
            result = entry["result"]
            if "custom_data" not in result:
                continue
            custom_data = result["custom_data"]
            if not custom_data:
                continue

            for datapoint in custom_data:
                datapoint_name = datapoint["name"]
                name = f"custom_data_{datapoint_name}"
                data = datapoint["data"]
                if name not in df:
                    df[name] = pd.Series()

                df.loc[i, name] = data

        # apply custom job data to data frame
        for i, job in enumerate(jobs):
            custom_data = job.custom_job_data
            if not custom_data:
                continue

            for datapoint in custom_data:
                datapoint_name = datapoint["name"]
                name = f"custom_job_data_{datapoint_name}"
                data = datapoint["data"]
                if name not in df:
                    df[name] = pd.Series()

                df.loc[i, name] = data

        # add optimized and variables
        if max_variable_count > 0:
            for i in range(max_variable_count):
                df[f"optimized_variable_{i}"] = pd.Series()

            for job_index, job_result in enumerate(job_results):
                if not ("variables" in job_result and job_result["variables"]):
                    continue

                for i, variable in enumerate(job_result["variables"].values()):
                    df.loc[job_index, f"optimized_variable_{i}"] = variable

        # add coordinates
        for i in range(len(jobs[0].coordinates)):
            df[f"x_{i}"] = pd.Series()
            df[f"y_{i}"] = pd.Series()
            df[f"z_{i}"] = pd.Series()

        for i, job in enumerate(jobs):
            for coordinate_idx, coordinate in enumerate(job.coordinates):
                df.loc[i, f"x_{coordinate_idx}"] = coordinate[0]
                df.loc[i, f"y_{coordinate_idx}"] = coordinate[1]
                df.loc[i, f"z_{coordinate_idx}"] = coordinate[2]

        return df

    @classmethod
    def execute_jobs(
        cls,
        jobs: Sequence[Job],
    ):
        job_results = []
        for job in tqdm(jobs, desc="calculating energies"):
            result = cls.execute_job(job)
            job_results.append(result)
        return job_results

    @classmethod
    def execute_jobs_in_parallel(cls, jobs: Sequence[Job], threads=None):
        with multiprocessing.Pool(threads) as p:
            job_results = list(tqdm(p.imap(cls.execute_job, jobs), total=len(jobs)))
        return job_results


def _import_custom_method(path: Path) -> Callable:
    # append module to path
    sys.path.append(str(path.parent))

    # import optimization method
    custom_module = importlib.import_module(path.stem)
    if not hasattr(custom_module, "run_optimization"):
        raise ValueError(
            f"invalid file {path} given for custom method, does not contain run_optimization method"
        )
    optimization_method = custom_module.run_optimization
    if not isinstance(optimization_method, Callable):
        raise ValueError(
            f"invalid file {path} given for custom method, run_optimization is not callable"
        )
    return optimization_method


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "number_of_atoms", type=int, help="number of atoms (even number)"
    )
    parser.add_argument("number_of_jobs", type=int, help="number of jobs to generate")

    parser.add_argument(
        "--method",
        "-M",
        type=str,
        choices=["fci", "spa"],
        required=False,
        default="spa",
        help="method to use for optimization, is overridden by custom-method, 'SPA' by default",
    )

    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        required=False,
        default=None,
        help="Number of threads to execute jobs in",
    )

    parser.add_argument(
        "--output", "-O", type=str, required=False, help="output file name"
    )
    parser.add_argument(
        "--evaluate",
        "-E",
        action="store_true",
        help="evaluate the results based on the error class distribution",
    )

    parser.add_argument(
        "--custom-method",
        type=Path,
        help="custom method to run the optimization",
    )

    parser.add_argument(
        "--compare-to",
        type=str,
        nargs="*",
        choices=["fci", "spa"],
        help="what to compare the primary method to",
    )

    parser.add_argument(
        "--size",
        "-S",
        type=float,
        default=2.75,
        help="parameter for the job generator: influences the distance between atoms",
    )

    parser.add_argument(
        "--fidelity",
        "-F",
        action="store_true",
        default=False,
        help="calculate the fidelity between the true ground state and the optimized state",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="increase output verbosity",
    )

    args = parser.parse_args()
    opt_method = None

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.custom_method:
        opt_method = _import_custom_method(args.custom_method)

    number_of_atoms = args.number_of_atoms

    if number_of_atoms % 2 != 0:
        raise ValueError("number of atoms needs to be a multiple of two")

    number_of_jobs = args.number_of_jobs

    job_generator = DataGenerator.generate_jobs

    if number_of_atoms >= 10 and args.fidelity==True:
        warnings.warn("The calculations may be very slow due to the high number of atoms and high fidelity setting.", UserWarning, 2)

    jobs = job_generator(
        number_of_atoms=number_of_atoms,
        number_of_jobs=number_of_jobs,
        size=args.size,
        method=args.method,
        custom_method=opt_method,
        compare_to=args.compare_to,
        calculate_fidelity=args.fidelity,
    )

    job_results = []

    if args.jobs:
        job_results = DataGenerator.execute_jobs_in_parallel(jobs, threads=args.jobs)
    else:
        job_results = DataGenerator.execute_jobs(jobs)

    result_df = DataGenerator.create_result_df(jobs, job_results)

    path = args.output
    if not path:
        filename = f"h{number_of_atoms}_{number_of_jobs}"
        if args.compare_to:
            method = args.method
            if opt_method:
                method = opt_method.__module__
            filename = f"{filename}_{method}-vs-{str(args.compare_to)}.csv"
        else:
            filename = f"{filename}_{args.method}.csv"
        path = filename

    result_df.to_csv(path)


if __name__ == "__main__":
    main()
