import tequila as tq
import numpy as np
from quanti_gin.data_generator import DataGenerator
import multiprocessing


class CustomizedGenerator(DataGenerator):
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

        coordinates = []
        if "coordinates" in kwargs:
            coordinates = kwargs["coordinates"]
        else:
            return

        mol = tq.Molecule(geometry=geometry, basis_set="sto-3g")
        H = mol.make_hamiltonian()

        v = np.linalg.eigvalsh(H.to_matrix())
        hcb_energy = v[0]

        # U = mol.make_ansatz(name="HCB-SPA", edges=edges)
        U = tq.gates.QCircuit()

        # structure of the ansatz:
        # 1. Ry rotations on all qubits
        for i in range(len(coordinates) * 2):
            U += tq.gates.Ry(target=i, angle="a" + str(i))
        depth = 2
        # 2. Entangling layer => Alternating: CNOTs on even and odd qubits
        # 3. Ry rotations on all qubits
        for d in range(depth):
            offset = d % 2
            iterations = len(coordinates) * 2
            for i in range(0, iterations - 1 - offset, 2):
                U += tq.gates.CNOT(control=offset + i, target=offset + i + 1)
            for i in range(iterations):
                U += tq.gates.Ry(target=i, angle=f"b{d}_{i}")
        # U.export_to(filename="output-circuit.png")

        opt = tq.chemistry.optimize_orbitals(
            mol.use_native_orbitals(),
            U,
            initial_guess=guess,
            silent=True,  # use_hcb=True
        )
        # U += tq.gates.CNOT(target=i, control=i + 1)

        E = tq.ExpectationValue(H=H, U=U)
        result = tq.minimize(E, silent=True)
        print(result.energy, hcb_energy)
        return (
            result.energy,
            opt.molecule.integral_manager.orbital_coefficients,
            result.variables,
            cls.compute_reference_value(geometry, edges, guess, basis_set),
            hcb_energy,
        )

    @classmethod
    def execute_jobs(cls, jobs):
        results = []
        with multiprocessing.Pool() as pool:
            results = pool.map(cls.execute_job, jobs)
        return results

        # return super().run_optimization(geometry, edges, *args, **kwargs)


if __name__ == "__main__":
    generator = CustomizedGenerator()
    jobs = generator.generate_jobs(4, 10)
    results = generator.execute_jobs(jobs)
    # df = generator.create_result_df(results)
