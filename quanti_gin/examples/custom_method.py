import tequila as tq
from tequila.quantumchemistry import QuantumChemistryBase
import logging

logger = logging.getLogger(__name__)


def run_optimization(mol: QuantumChemistryBase, *args, **kwargs):
    logger.info("running custom optimization method")
    H = mol.make_hamiltonian()

    # U = mol.make_ansatz(name="HCB-SPA", edges=edges)
    U = tq.gates.QCircuit()

    # structure of the ansatz:
    # 1. Ry rotations on all qubits
    for i in range(mol.n_electrons * 2):
        U += tq.gates.Ry(target=i, angle="a" + str(i))
    depth = 2
    # 2. Entangling layer => Alternating: CNOTs on even and odd qubits
    # 3. Ry rotations on all qubits
    for d in range(depth):
        offset = d % 2
        iterations = mol.n_electrons * 2
        for i in range(0, iterations - 1 - offset, 2):
            U += tq.gates.CNOT(control=offset + i, target=offset + i + 1)
        for i in range(iterations):
            U += tq.gates.Ry(target=i, angle=f"b{d}_{i}")

    E = tq.ExpectationValue(H=H, U=U)
    result = tq.minimize(E, silent=True)
    logger.info(f"{result.energy}")
    return {
        "energy": result.energy,
    }
