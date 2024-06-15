#!/usr/bin/env python3
import os
import math
import argparse
import pprint
from typing import Sequence
import tequila as tq
import pandas as pd
from tqdm import tqdm
from numpy import eye, float64, floating
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass, field
from random import Random

from quanti_gin.shared import (
    generate_min_local_distance_edges,
    generate_min_global_distance_edges,
)


# static seed for development
seed = None
np_random = np.random.default_rng(seed)
rand = Random(seed)


@dataclass
class Job:
    geometry: str
    edges: NDArray = field(default_factory=np.array)
    guess: Sequence[NDArray[float64]] = field(default_factory=list)
    coordinates: NDArray = field(default_factory=np.array)

    edge_distances: Sequence[floating] = field(default_factory=list)
    coordinate_distances: Sequence[floating] = field(default_factory=list)


class DataGenerator:
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
    def compute_reference_value(
        cls, geometry: str, edges: list[tuple[int, int]], guess, basis_set
    ):
        mol = tq.Molecule(geometry=geometry, basis_set=basis_set)
        return mol.compute_energy("fci")

    @classmethod
    def run_optimization(
        cls, geometry: str, edges: list[tuple[int, int]], *args, **kwargs
    ):
        basis_set = "sto-3g"
        guess = None
        if "basis_set" in kwargs:
            basis_set = kwargs["basis_set"]
        if "guess" in kwargs:
            guess = kwargs["guess"]

        mol = tq.Molecule(geometry=geometry, basis_set=basis_set)

        U = mol.make_ansatz(name="HCB-SPA", edges=edges)

        opt = tq.chemistry.optimize_orbitals(
            mol.use_native_orbitals(), U, initial_guess=guess, silent=True, use_hcb=True
        )

        mol = opt.molecule
        H = mol.make_hardcore_boson_hamiltonian()

        # this line is valid, although my intellisense complains about it
        v = np.linalg.eigvalsh(H.to_matrix())
        hcb_energy = v[0]

        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)

        return (
            result.energy,
            opt.molecule.integral_manager.orbital_coefficients,
            result.variables,
            cls.compute_reference_value(geometry, edges, guess, basis_set),
            hcb_energy,
        )

    @classmethod
    def generate_jobs(cls, number_of_atoms, number_of_jobs, size=None):
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
            edges = generate_min_global_distance_edges(coordinates)

            guess = cls.generate_initial_guess_from_edges(
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

    @classmethod
    def execute_job(cls, job):
        return cls.run_optimization(
            geometry=job.geometry,
            edges=job.edges,
            guess=job.guess,
            coordinates=job.coordinates,
        )

    @classmethod
    def create_result_df(cls, jobs, job_results) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "optimized_energy": [result[0] for result in job_results],
                # "optimized_variables": [result[2] for result in job_results],
                "exact_energy": [result[3] for result in job_results],
                "hcb_energy": [result[4] for result in job_results],
                "atom_count": [len(job.coordinates) for job in jobs],
                "edge_count": [len(job.edges) for job in jobs],
                "optimized_variable_count": [
                    len(job_results[0][2].keys()) for job in jobs
                ],
            }
        )
        for i in range(len(job_results[0][2].keys())):
            df[f"optimized_variable_{i}"] = pd.Series()

        for job_index, job_result in enumerate(job_results):
            for i, variable in enumerate(job_result[2].values()):
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

        for i in range(len(jobs[0].edges)):
            df[f"edge_{i}_start"] = pd.Series()
            df[f"edge_{i}_end"] = pd.Series()

        for job_index, job in enumerate(jobs):
            for i, coordinate in enumerate(job.coordinates):
                df.loc[job_index, f"x_{i}"] = coordinate[0]
                df.loc[job_index, f"y_{i}"] = coordinate[1]
                df.loc[job_index, f"z_{i}"] = coordinate[2]

            for i, edge in enumerate(job.edges):
                df.loc[job_index, f"edge_{i}_start"] = edge[0]
                df.loc[job_index, f"edge_{i}_end"] = edge[1]

        return df

    @classmethod
    def execute_jobs(cls, jobs):
        job_results = []
        for job in tqdm(jobs, desc="calculating energies"):
            result = cls.execute_job(job)
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
            edges=edges,
            guess=guess,
            coordinates=coordinates,
            edge_distances=edge_distances,
            coordinate_distances=coordinate_distances,
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


def evaluate_data(hcb_energy, exact_energy, edge_distances):
    def map_error_to_class(error):
        if error < 20:
            return 0
        if error < 100:
            return 1
        return 2

    error = abs(hcb_energy - exact_energy) * 1000
    error = map_error_to_class(error)

    return error


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "number_of_atoms", type=int, help="number of atoms (even number)"
    )
    parser.add_argument("number_of_jobs", type=int, help="number of jobs to generate")
    parser.add_argument(
        "--custom_job_generator",
        type=str,
        choices=["v1", "v2"],
        required=False,
        help="choose a custom job generator with different heuristics for H4 evaluation",
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

    args = parser.parse_args()

    number_of_atoms = args.number_of_atoms

    if number_of_atoms % 2 != 0:
        raise ValueError("number of atoms needs to be a multiple of two")

    number_of_jobs = args.number_of_jobs

    job_generator = DataGenerator.generate_jobs

    if args.custom_job_generator:
        if args.custom_job_generator == "v1":
            job_generator = custom_generate_jobs_v1
        elif args.custom_job_generator == "v2":
            job_generator = custom_generate_jobs_v2

    evaluations = []

    jobs = job_generator(
        number_of_atoms=number_of_atoms, number_of_jobs=number_of_jobs, size=2.75
    )

    job_results = []

    job_results = DataGenerator.execute_jobs(jobs)

    if args.evaluate:
        for job, result in zip(jobs, job_results):
            evaluations.append(evaluate_data(result[3], result[4], job.edge_distances))
        print("error class distribution:")
        print(pd.Series(evaluations).value_counts())

    result_df = DataGenerator.create_result_df(jobs, job_results)

    path = args.output
    if not path:
        filename = f"h{number_of_atoms}_{len(jobs)}.csv"
        if args.custom_job_generator:
            filename = (
                f"custom_{args.custom_job_generator}_{number_of_atoms}_{len(jobs)}.csv"
            )
        path = filename

    result_df.to_csv(path)


if __name__ == "__main__":
    main()
