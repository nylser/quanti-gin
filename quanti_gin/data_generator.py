#!/usr/bin/env python3
import importlib
import math
import argparse
import logging
from pathlib import Path
import pprint
import sys
from typing import Callable, Sequence, TypedDict
import tequila as tq
from tequila.quantumchemistry import QuantumChemistryBase
import pandas as pd
from tqdm import tqdm
from numpy import eye, floating
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass, field
from random import Random

from quanti_gin.shared import (
    generate_min_local_distance_edges,
    generate_min_global_distance_edges,
)

logger = logging.getLogger(__name__)


# static seed for development
seed = None
np_random = np.random.default_rng(seed)
rand = Random(seed)


class OptimizationResult(TypedDict):
    energy: float
    orbital_coefficients: NDArray | None
    variables: dict | None


@dataclass
class Job:
    id: int
    geometry: str
    optimization_algorithm: Callable[..., OptimizationResult]
    coordinates: NDArray = field(default_factory=np.array)

    edge_distances: Sequence[floating] = field(default_factory=list)
    coordinate_distances: Sequence[floating] = field(default_factory=list)
    # custom optimization algorithm


class DataGenerator:
    @classmethod
    def parse_geometry_string(cls, geometry_string: str) -> NDArray[np.float64]:
        lines = geometry_string.split("\n")
        coordinates = []
        for line in lines:
            if not line:
                continue
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
            # make sure we have a minimum of 0.25 angstrom between the atom cores
            while min_distance <= 0.25:
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
    def generate_initial_guess(cls, count: int):
        if count % 2 != 0:
            raise ValueError("count needs to be a multipe of two")
        initial_guess = eye(count)
        for i in range(0, count, 2):
            initial_guess[i][i] = 1.0
            initial_guess[i][i + 1] = 1.0
            initial_guess[i + 1][i] = 1.0
            initial_guess[i + 1][i + 1] = -1.0
        return initial_guess

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
    def generate_edges(cls, atom_count: int):
        if atom_count % 2 != 0:
            raise ValueError("count needs to be a multipe of two")
        edges = []
        for i in range(0, atom_count, 2):
            edges.append((i, i + 1))
        return edges

    @classmethod
    def run_fci_optimization(cls, molecule: QuantumChemistryBase, *args, **kwargs):
        return OptimizationResult(
            energy=molecule.compute_energy("fci"),
            orbital_coefficients=molecule.integral_manager.orbital_coefficients,
            variables=None,
        )

    @classmethod
    def run_spa_optimization(cls, molecule: QuantumChemistryBase, *args, **kwargs):
        coordinates = cls.parse_geometry_string(molecule.parameters.geometry)
        edges = generate_min_global_distance_edges(coordinates)
        initial_guess = cls.generate_initial_guess_from_edges(
            edges=edges, vertices=coordinates
        ).T

        U = molecule.make_ansatz(name="HCB-SPA", edges=edges)

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
        return OptimizationResult(
            energy=result.energy,
            orbital_coefficients=opt.molecule.integral_manager.orbital_coefficients,
            variables=result.variables,
        )

    @classmethod
    def generate_jobs(
        cls,
        number_of_atoms,
        number_of_jobs,
        size=None,
        method="spa",
        custom_method=None,
        compare_to=[],
    ):

        def get_algorithm_from_method(method):
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

            # edge_distances = [
            #     np.linalg.norm(coordinates[edge[0]] - coordinates[edge[1]])
            #     for edge in edges
            # ]
            # # fully connected graph, all distances
            # coordinate_distances = [
            #     [np.linalg.norm(other - coordinate) for other in coordinates]
            #     for coordinate in coordinates
            # ]

            if custom_method:
                job = Job(
                    id=i,
                    geometry=geometry,
                    coordinates=coordinates,
                    # edge_distances=edge_distances,
                    # coordinate_distances=coordinate_distances,
                    optimization_algorithm=custom_method,
                )
                jobs.append(job)
            else:
                job = Job(
                    id=i,
                    geometry=geometry,
                    coordinates=coordinates,
                    # edge_distances=edge_distances,
                    # coordinate_distances=coordinate_distances,
                    optimization_algorithm=get_algorithm_from_method(method),
                )
                jobs.append(job)

            if compare_to:
                for compare in compare_to:
                    # do not duplicate methods
                    if compare == method and not custom_method:
                        continue
                    job = Job(
                        id=i,
                        geometry=geometry,
                        coordinates=coordinates,
                        # edge_distances=edge_distances,
                        # coordinate_distances=coordinate_distances,
                        optimization_algorithm=get_algorithm_from_method(compare),
                    )
                    jobs.append(job)

        return jobs

    @classmethod
    def execute_job(cls, job: Job, basis_set="sto-3g", compare_to=[]):
        mol = tq.Molecule(geometry=job.geometry, basis_set=basis_set)
        return job.optimization_algorithm(
            mol,
            coordinates=job.coordinates,
        )

    @classmethod
    def create_result_df(
        cls, jobs: Sequence[Job], job_results: Sequence[OptimizationResult]
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
                "optimized_energy": [result["energy"] for result in job_results],
                "atom_count": [len(job.coordinates) for job in jobs],
                "edge_count": [len(job.coordinates) // 2 for job in jobs],
                "optimized_variable_count": max_variable_count,
                "method": [
                    f"{job.optimization_algorithm.__module__}.{job.optimization_algorithm.__name__}"
                    for job in jobs
                ],
            }
        )
        if max_variable_count > 0:
            for i in range(max_variable_count):
                df[f"optimized_variable_{i}"] = pd.Series()

            for job_index, job_result in enumerate(job_results):
                if not ("variable" in job_result and job_result["variables"]):
                    continue

                for i, variable in enumerate(job_result["variables"].values()):
                    # normalize variable into range -2pi to 2pi
                    if variable < 0:
                        normalized_variable = variable % (-math.pi * 2)
                    normalized_variable = variable % (math.pi * 2)
                    normalized_variable = normalized_variable / math.pi
                    df.loc[job_index, f"optimized_variable_{i}"] = normalized_variable

        for i in range(len(jobs[0].coordinates)):
            df[f"x_{i}"] = pd.Series()
            df[f"y_{i}"] = pd.Series()
            df[f"z_{i}"] = pd.Series()

        if hasattr(jobs[0], "edges"):
            for i in range(len(jobs[0].edges)):
                df[f"edge_{i}_start"] = pd.Series()
                df[f"edge_{i}_end"] = pd.Series()

        for job_index, job in enumerate(jobs):
            for i, coordinate in enumerate(job.coordinates):
                df.loc[job_index, f"x_{i}"] = coordinate[0]
                df.loc[job_index, f"y_{i}"] = coordinate[1]
                df.loc[job_index, f"z_{i}"] = coordinate[2]
            if hasattr(job, "edges"):
                for i, edge in enumerate(job.edges):
                    df.loc[job_index, f"edge_{i}_start"] = edge[0]
                    df.loc[job_index, f"edge_{i}_end"] = edge[1]

        return df

    @classmethod
    def execute_jobs(cls, jobs: Sequence[Job], compare_to=[]):
        job_results = []
        for job in tqdm(jobs, desc="calculating energies"):
            result = cls.execute_job(job, compare_to=compare_to)
            job_results.append(result)
        return job_results


def custom_generate_jobs_v1(*args, **kwargs):
    """
    Generates jobs with two groups of atoms:
    - groups are at a distance of constant d
    - atoms in the groups are at a distance of a
    - a is in the range of 0.5 to 5
    """
    d = 0.5
    job_count = 250
    start_a = d / 2
    stop_a = 5
    step_a = abs(start_a - stop_a) / job_count
    jobs = []

    for job_num in range(job_count):
        coordinates = []
        a = step_a * job_num + start_a

        first_coordinate = np.array([0, 0, 0])
        second_coordinate = np.array(first_coordinate + [a, 0, 0])

        third_coordinate = np.array(second_coordinate + [d, 0, 0])
        fourth_coordinate = np.array(third_coordinate + [a, 0, 0])

        coordinates = np.array(
            [first_coordinate, second_coordinate, third_coordinate, fourth_coordinate]
        )

        geometry = DataGenerator.generate_geometry_string(coordinates)
        edges = generate_min_global_distance_edges(coordinates)

        guess = DataGenerator.generate_initial_guess_from_edges(
            vertices=coordinates, edges=edges
        )
        guess = guess.T

        edge_distances = [
            np.linalg.norm(coordinates[edge[0]] - coordinates[edge[1]])
            for edge in edges
        ]

        # fully connected graph, all distances
        coordinate_distances = [
            [np.linalg.norm(other - coordinate) for other in coordinates]
            for coordinate in coordinates
        ]

        pprint.pprint(coordinate_distances)

        job = Job(
            geometry=geometry,
            coordinates=coordinates,
            edges=edges,
            guess=guess,
            edge_distances=edge_distances,
            coordinate_distances=coordinate_distances,
            optimization_algorithm=DataGenerator.run_spa_optimization,
        )
        jobs.append(job)
    return jobs


def custom_generate_jobs_v2(*args, **kwargs):
    """
    Generates jobs with two groups of atoms:
    - groups are at a distance a
    - atoms in the groups are at a distance of constant d
    - a is in the range of 0.5 to 5
    """
    d = 1.5
    job_count = 250
    if "number_of_jobs" in kwargs:
        job_count = kwargs["number_of_jobs"]
    start_a = d / 2
    stop_a = 5
    step_a = abs(start_a - stop_a) / job_count
    jobs = []

    for job_num in range(job_count):
        coordinates = []
        a = step_a * job_num + start_a

        first_coordinate = np.array([0, 0, 0])
        second_coordinate = np.array(first_coordinate + [d, 0, 0])

        third_coordinate = np.array(second_coordinate + [a, 0, 0])
        fourth_coordinate = np.array(third_coordinate + [d, 0, 0])

        coordinates = np.array(
            [
                first_coordinate,
                second_coordinate,
                third_coordinate,
                fourth_coordinate,
            ]
        )

        geometry = DataGenerator.generate_geometry_string(coordinates)
        edges = generate_min_local_distance_edges(coordinates)

        guess = DataGenerator.generate_initial_guess_from_edges(
            vertices=coordinates, edges=edges
        )
        guess = guess.T

        edge_distances = [
            np.linalg.norm(coordinates[edge[0]] - coordinates[edge[1]])
            for edge in edges
        ]

        # fully connected graph, all distances
        coordinate_distances = [
            [np.linalg.norm(other - coordinate) for other in coordinates]
            for coordinate in coordinates
        ]

        job = Job(
            geometry=geometry,
            edges=edges,
            guess=guess,
            coordinates=coordinates,
            edge_distances=edge_distances,
            coordinate_distances=coordinate_distances,
        )
        jobs.append(job)
    return jobs


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

    # TODO: currently not working due to structure changes, this still needs some refactoring
    # parser.add_argument(
    #     "--custom_job_generator",
    #     type=str,
    #     choices=["v1", "v2"],
    #     required=False,
    #     help="choose a custom job generator with different heuristics for H4 evaluation",
    # )

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

    # if args.custom_job_generator:
    #     if args.custom_job_generator == "v1":
    #         job_generator = custom_generate_jobs_v1
    #     elif args.custom_job_generator == "v2":
    #         job_generator = custom_generate_jobs_v2

    jobs = job_generator(
        number_of_atoms=number_of_atoms,
        number_of_jobs=number_of_jobs,
        size=args.size,
        method=args.method,
        custom_method=opt_method,
        compare_to=args.compare_to,
    )

    job_results = []

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
        path = filename

    result_df.to_csv(path)


if __name__ == "__main__":
    main()
