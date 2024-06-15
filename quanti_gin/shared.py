from dataclasses import dataclass
from os import PathLike
import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class DataFile:
    file_path: PathLike
    df: pd.DataFrame
    hcb_energy: pd.Series
    exact_energy: pd.Series
    spa_energy: pd.Series
    coordinates: NDArray[np.float64]
    edges: list[list[tuple[int, int]]]
    optimized_variables: list[list[float]]


def read_data_file(file_path: PathLike):
    # load the csv file into a pandas dataframe
    df = pd.read_csv(file_path)

    atom_counts = df["atom_count"]
    coordinate_cols = [
        [(f"x_{i}", f"y_{i}", f"z_{i}") for i in range(atom_count)]
        for atom_count in atom_counts
    ]

    optimized_variable_counts = df["optimized_variable_count"]
    optimized_variable_cols = [
        [f"optimized_variable_{i}" for i in range(optimized_variable_count)]
        for optimized_variable_count in optimized_variable_counts
    ]

    hcb_energy = df["hcb_energy"]
    exact_energy = df["exact_energy"]
    spa_energy = df["optimized_energy"]

    coordinates = []
    for i, cols in enumerate(coordinate_cols):
        sample_coordinates = [
            (df[col[0]][i], df[col[1]][i], df[col[2]][i]) for col in cols
        ]
        coordinates.append(sample_coordinates)
    coordinates = np.array(coordinates)

    optimized_variables = []

    for i, cols in enumerate(optimized_variable_cols):
        sample_variables = [df[col][i] for col in cols]
        optimized_variables.append(sample_variables)

    edge_counts = df["edge_count"]

    edge_cols = [
        [(f"edge_{i}_start", f"edge_{i}_end") for i in range(edge_count)]
        for edge_count in edge_counts
    ]
    edges = []

    for i, cols in enumerate(edge_cols):
        sample_edges = [(df[col[0]][i], df[col[1]][i]) for col in cols]
        edges.append(sample_edges)

    return DataFile(
        file_path=file_path,
        df=df,
        hcb_energy=hcb_energy,
        exact_energy=exact_energy,
        spa_energy=spa_energy,
        coordinates=coordinates,
        optimized_variables=optimized_variables,
        edges=edges,
    )


def generate_min_local_distance_edges(vertices: np.ndarray):
    """A locally optimal solution to the edge generation problem where it always considers the next possible shortest edge.

    Warning: This always depends on the order of the vertices, so the result may vary.

    Args:
        vertices: All vertices in the graph

    Returns
        A list of edges that make sure each vertex is connected to another vertex with the shortest possible edge.
    """
    all_edges = []
    done = set()
    our_vertices = vertices
    while len(done) < len(vertices):
        # find shortest edgee
        shortest_edge = None
        shortest_edge_indices = (0, 0)
        for a, vertex_a in enumerate(our_vertices):
            if tuple(vertex_a.tolist()) in done:
                continue
            for b, vertex_b in enumerate(our_vertices):
                if (
                    np.array_equal(vertex_a, vertex_b)
                    or tuple(vertex_b.tolist()) in done
                ):
                    continue
                if shortest_edge is None or shortest_edge > np.linalg.norm(
                    vertex_a - vertex_b
                ):
                    shortest_edge = np.linalg.norm(vertex_a - vertex_b)
                    shortest_edge_indices = (a, b)
        all_edges.append(tuple(sorted(shortest_edge_indices)))
        done.add(tuple(our_vertices[shortest_edge_indices[0]].tolist()))
        done.add(tuple(our_vertices[shortest_edge_indices[1]].tolist()))
    return all_edges


def generate_min_global_distance_edges(vertices: np.ndarray):
    """
    input: vertices: np.ndarray
    A globally optimal solution to the edge generation problem.
    It will consider all possible start points for the optimization and return the one with the total minimum edge length.
    """
    all_edges = []
    all_edge_lengths = []
    for i in range(len(vertices)):
        edges, length_sum = generate_local_optimal_edges_from_vertices(
            vertices, start=i
        )
        all_edges.append(edges)
        all_edge_lengths.append(length_sum)

    min_edge_lengths = min(all_edge_lengths)
    max_edge_lengths = max(all_edge_lengths)
    if max_edge_lengths > min_edge_lengths:
        pass
        # print("FOUND MORE OPTIMAL EDGES", min_edge_lengths, max_edge_lengths)
    return all_edges[np.argmin(all_edge_lengths)]


def generate_local_optimal_edges_from_vertices(vertices: np.ndarray, start=0):
    """this is a locally optimal solution to the edge generation problem, when inputting the start parameter, the greedy search will start from that index"""
    first_vertice = vertices[start]
    our_vertices = np.delete(vertices, start, axis=0)
    our_vertices = np.insert(our_vertices, 0, first_vertice, axis=0)
    done = set()
    edges = []
    edge_length_sum = 0
    for a, vertex_a in enumerate(our_vertices):
        if tuple(vertex_a.tolist()) in done:
            continue
        nearest_vertex = None
        nearest_vertex_index = None
        for b, vertex_b in enumerate(our_vertices):
            if np.array_equal(vertex_a, vertex_b) or tuple(vertex_b.tolist()) in done:
                continue
            if nearest_vertex is None:
                nearest_vertex_index = b
                nearest_vertex = vertex_b
            elif np.linalg.norm(vertex_a - nearest_vertex) > np.linalg.norm(
                vertex_a - vertex_b
            ):
                nearest_vertex_index = b
                nearest_vertex = vertex_b

        edges.append(tuple(sorted([a, nearest_vertex_index])))
        edge_length_sum += np.linalg.norm(vertex_a - nearest_vertex)
        done.update([tuple(vertex_a.tolist()), tuple(nearest_vertex.tolist())])
    return (edges, edge_length_sum)
