# Generate a test dataset composed of 500 random cliffords (125 for each topology).

import numpy as np
from qiskit.quantum_info import Clifford, random_clifford
from qiskit.transpiler import CouplingMap, StagedPassManager, PassManager
from qiskit.circuit.library.standard_gates import HGate, SGate, CXGate
import pickle
import random
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler import Target
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler.passes.routing.sabre_swap import SabreSwap


def generate_circuit(d, num_qubits, coupling_map, gates_set):
    edges = coupling_map.graph.edge_list()
    qc = QuantumCircuit(num_qubits)
    for i in range(d):
        gate = random.choice(gates_set)
        # print("gate: ", gate)
        nq_gate = gate.num_qubits
        if nq_gate == 1:
            q = random.randint(0, num_qubits-1)
            qc.append(gate, [q])
        else:
            q1, q2 = random.choice(edges)
            qc.append(gate, [q1, q2])
    return qc


maps = [[[0, 1], [1, 2], [2, 3], [3, 4]], [[0, 1], [1, 2], [2, 3], [3, 4], [
    0, 4]], [[0, 2], [2, 1], [2, 3], [3, 4]], [[0, 2], [2, 1], [2, 3], [2, 4]]]
gates_set = [HGate(), SGate(), CXGate()]
n_qubits = 5
i_map = 0
test_dataset_routing = []
test_dataset_norouting = []

pm_staged = StagedPassManager()

equiv_lib = SessionEquivalenceLibrary
target = Target()
target.add_instruction(SGate(), name='s')
target.add_instruction(HGate(), name='h')
target.add_instruction(CXGate(), name='cx')

pass_manager = PassManager(BasisTranslator(equiv_lib, target))

for i in range(5):

    if i_map > len(maps) - 1:
        i_map = 0

    cliff = random_clifford(n_qubits)

    cliff_qc = cliff.to_circuit()

    pm_staged.routing = PassManager(SabreSwap(CouplingMap(maps[i_map])))
    cliff_qc_routed = pm_staged.run(cliff_qc)

    target_qc = pass_manager.run(cliff_qc_routed)

    target_qc_size = 200

    cliff_state = Clifford(target_qc)

    target_tableau_state = cliff_state.tableau

    state = target_tableau_state

    test_dict_routing = {}
    test_dict_norouting = {}
    test_dict_routing['obs'] = state
    test_dict_norouting['obs'] = cliff.tableau  # no routing
    test_dict_routing['map'] = maps[i_map]
    test_dict_norouting['map'] = maps[i_map]
    test_dict_routing['target_qc_size'] = target_qc_size
    test_dict_norouting['target_qc_size'] = target_qc_size
    test_dict_routing['cliff_state'] = cliff_state
    test_dict_norouting['cliff_state'] = cliff  # no routing
    test_dataset_routing.append(test_dict_routing)
    test_dataset_norouting.append(test_dict_norouting)

    i_map += 1

    file_path_routing = 'datasets/test_dataset_routing_prova'
    file_path_norouting = 'datasets/test_dataset_norouting_prova'

    file = open(file_path_routing, 'wb')
    pickle.dump(test_dataset_routing, file)
    file.close()

    file = open(file_path_norouting, 'wb')
    pickle.dump(test_dataset_norouting, file)
    file.close()
