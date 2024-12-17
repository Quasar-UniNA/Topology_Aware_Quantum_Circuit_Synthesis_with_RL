
from synthesis_env import SynthesisEnv
from sb3_contrib import MaskablePPO
from qiskit.transpiler import CouplingMap


def synthesize(clifford, coupling_map):
    n_qubits = clifford.num_qubits
    env = SynthesisEnv(n_qubits)
    if n_qubits == 3:
        model = MaskablePPO.load('3q_model')
    else:
        model = MaskablePPO.load('5q_model')

    obs = clifford.tableau
    env.set_state(obs, CouplingMap(coupling_map), 200, clifford)
    truncated = False
    terminated = False

    while not terminated and not truncated:

        env.render()
        
        action, _ = model.predict(obs, action_masks=env.action_masks())
        obs, reward, terminated, truncated, info = env.step(action)

    rl_synth_qc = env.info["synthetized_qc"].inverse()
    print("terminated: ", terminated)
    print("truncated: ", truncated)
    
    return rl_synth_qc

