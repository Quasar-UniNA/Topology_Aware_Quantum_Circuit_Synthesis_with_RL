import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import random
from statistics import mean
import numpy as np
from gymnasium.spaces import Discrete, Box
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford, random_clifford
from qiskit.transpiler import CouplingMap, PassManager, StagedPassManager
from qiskit.circuit.library.standard_gates import HGate, SGate, CXGate
from sb3_contrib import MaskablePPO
from qiskit.transpiler.passes.routing.sabre_swap import SabreSwap
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler import Target
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
import pickle
import time
from qiskit.synthesis import synth_clifford_greedy, synth_clifford_ag
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from qiskit_sat_synthesis.synthesize_clifford import synthesize_clifford_depth


class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.difficulty = 1
        self.maps = [[[0, 1], [1, 2], [2, 3], [3, 4]], [[0, 1], [1, 2], [2, 3], [3, 4], [0, 4]], [[0, 2], [2, 1], [2, 3], [3, 4]], [[0, 2], [2, 1], [2, 3], [2, 4]]]
        
        self.tot_dict = {}
        i = 0
        for map in self.maps:
            dict = self._mapping__coupling_to_dict(
            CouplingMap(map), [HGate(), SGate(), CXGate()])
            for action in dict.values():
                if action not in self.tot_dict.values():
                    self.tot_dict[str(i)] = action
                    i += 1
        self.dict = {}

        self.action_space = Discrete(len(self.tot_dict.keys()))
        self.n_qubits = 5        
        self.observation_space = Box(low=0, high=1, shape=(
            2*self.n_qubits, 2*self.n_qubits+1), dtype=int)

        self.i = random.randint(0, len(self.maps) - 1)
        self.dict = self._mapping__coupling_to_dict(
        CouplingMap(self.maps[self.i]), [HGate(), SGate(), CXGate()])

        self.target_qc = self.generate_circuit(self.difficulty, self.n_qubits, CouplingMap(self.maps[self.i]), [HGate(), SGate(), CXGate()])

        self.target_qc_size = self.target_qc.size()

        self.cliff_state = Clifford(self.target_qc)

        self.target_tableau_state = self.cliff_state.tableau       

        self.state = self.target_tableau_state
        
        self.qc = QuantumCircuit(self.n_qubits)
        self.info = {}        
        self.seed()
        self.success_count = 0
        self.total_episodes_rollout = 0
        self.qc_cx_size = 0
        self.qc_cx_depth = 0
        self.qc_size = 0
        self.qc_depth = 0      


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _mapping__coupling_to_dict(self, coupling_map, gates_set):
        i = 0
        dict_map = {}

        edges = coupling_map.graph.edge_list()
        nodes_indexes = coupling_map.graph.node_indexes()
        for gate in gates_set:
            num_qubits = gate.num_qubits

            if num_qubits == 1:
                for node_index in nodes_indexes:
                    dict_map[str(i)] = (gate, [node_index])
                    i += 1
            if num_qubits == 2:
                for edge in edges:
                    dict_map[str(i)] = (gate, list(edge))
                    i += 1
                    e = list(edge)
                    e.reverse()
                    dict_map[str(i)] = (gate, e)
                    i += 1
        return dict_map

    def count_q_qubit_gates(self, circuit, q):
        count = 0
        for gate in circuit:
            if gate[0].num_qubits == q:
                count += 1
        return count

    def generate_circuit(self, d, num_qubits, coupling_map, gates_set):
        edges = coupling_map.graph.edge_list()
        qc = QuantumCircuit(num_qubits)
        for i in range(d):
            gate = random.choice(gates_set)
            #print("gate: ", gate)
            nq_gate = gate.num_qubits
            if nq_gate == 1:
                q = random.randint(0, num_qubits-1)
                qc.append(gate, [q])
            else:
                q1, q2 = random.choice(edges)
                qc.append(gate, [q1, q2])
        return qc

    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        self.reset()
    

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.i = random.randint(0, len(self.maps) - 1)
        self.dict = self._mapping__coupling_to_dict(
        CouplingMap(self.maps[self.i]), [HGate(), SGate(), CXGate()])

        pm_staged = StagedPassManager()
        pm_staged.routing = PassManager(SabreSwap(CouplingMap(self.maps[self.i])))

        equiv_lib = SessionEquivalenceLibrary
        target = Target()
        target.add_instruction(SGate(), name='s')
        target.add_instruction(HGate(), name='h')
        target.add_instruction(CXGate(), name='cx')

        pass_manager = PassManager(BasisTranslator(equiv_lib, target))

        if self.difficulty > 1024:

            cliff = random_clifford(self.n_qubits)            

            cliff_qc = cliff.to_circuit()

            cliff_qc_routed = pm_staged.run(cliff_qc)

            self.target_qc = pass_manager.run(cliff_qc_routed)

            self.target_qc_size = self.target_qc.size()  

            self.cliff_state = Clifford(self.target_qc)      
                                
            self.target_tableau_state = self.cliff_state.tableau            
            
            self.state = self.target_tableau_state
            
        else:
            self.target_qc = self.generate_circuit(self.difficulty, self.n_qubits, CouplingMap(self.maps[self.i]), [HGate(), SGate(), CXGate()])

            self.target_qc_size = self.target_qc.size()

            self.cliff_state = Clifford(self.target_qc)

            self.target_tableau_state = self.cliff_state.tableau       

            self.state = self.target_tableau_state

        self.qc = QuantumCircuit(self.n_qubits)
        self.info = {}

        return (self.state, self.info)
    


    def step(self, action):

        gate, q = self.tot_dict[str(action)]      
        qc_tmp = QuantumCircuit(self.n_qubits)
        qc_tmp.append(gate, q)

        c = Clifford(qc_tmp)
        self.cliff_state = self.cliff_state.compose(c)

        self.target_tableau_state = self.cliff_state.tableau        

        depth_before = self.qc.depth()
        oneq_gates_before = self.count_q_qubit_gates(self.qc, 1)
        twoq_gates_before = self.count_q_qubit_gates(self.qc, 2)

        self.qc.append(gate, q)        

        depth_after = self.qc.depth()

        if gate.num_qubits == 1:
            oneq_gates_after = oneq_gates_before + 1
            twoq_gates_after = twoq_gates_before
        else:
            oneq_gates_after = oneq_gates_before
            twoq_gates_after = twoq_gates_before + 1

        if oneq_gates_after + twoq_gates_after >= self.target_qc_size:
            truncated = True
        else: truncated = False
    
        identity = np.identity(2 * self.n_qubits)

        identity_distance = 1 - \
            abs(identity - self.target_tableau_state.astype(int)[:, 0:-1]).mean()

        depth_coeff = 0.5
        twoq_gate_coeff = 0.15
        oneq_gate_coeff = 0.1

        reward = identity_distance - depth_coeff * \
            (depth_after - depth_before) - twoq_gate_coeff * (twoq_gates_after - twoq_gates_before) - oneq_gate_coeff * (oneq_gates_after - oneq_gates_before)

        if identity_distance == 1:
            reward = 1000 - depth_after * (depth_after - depth_before) - 10 * twoq_gates_after
            terminated = True
        else:
            terminated = False

        if truncated or terminated:
            self.total_episodes_rollout += 1
            self.qc_cx_size += self.count_q_qubit_gates(self.qc, 2)
            self.qc_cx_depth += self.qc.depth(lambda gate: gate[0].name in ['cx'])
            self.qc_size += self.qc.size()
            self.qc_depth += self.qc.depth()

        if terminated: self.success_count += 1

        self.info["synthetized_qc"]= self.qc
        self.info["tableau"] = self.target_tableau_state

        self.state = self.target_tableau_state

        return self.state, reward, terminated, truncated, self.info

    def reset_success_data(self):
        print('reset total episodes rollout ', self.total_episodes_rollout)
        self.success_count = 0
        self.total_episodes_rollout = 0
        self.qc_cx_size = 0
        self.qc_cx_depth = 0
        self.qc_size = 0
        self.qc_depth = 0

    def render(self, mode='human'):
        pass
    def close(self):
        pass

    def action_masks(self):
        l = []
        for action in self.tot_dict.values():
            if action in self.dict.values():
                l.append(True)
            else:
                l.append(False)
        return l
    
    def set_state(self, obs, map, target_qc_size, cliff_state):
        self.state = obs
        self.dict = self._mapping__coupling_to_dict(CouplingMap(map), [HGate(), SGate(), CXGate()])
        self.target_qc_size = target_qc_size
        self.cliff_state = cliff_state

        self.qc = QuantumCircuit(self.n_qubits)
        self.info = {}


'''class ProgressiveLearningCallback(BaseCallback):
    def __init__(self, success_threshold=0.5, verbose=0):
        super(ProgressiveLearningCallback, self).__init__(verbose)
        self.success_threshold = success_threshold
        self.step_count = 0

    def _on_step(self):
        self.step_count += 1
        env = self.training_env
        #self.logger.dump(self.step_count)
        return True

    def _on_rollout_end(self):
        env = self.training_env

        qc_cx_size = sum(env.get_attr('qc_cx_size'))
        qc_cx_depth = sum(env.get_attr('qc_cx_depth'))
        qc_size = sum(env.get_attr('qc_size'))
        qc_depth = sum(env.get_attr('qc_depth'))

        success_count = sum(env.get_attr('success_count'))
        total_episodes = sum(env.get_attr('total_episodes_rollout'))        

        success_rate = success_count / total_episodes if total_episodes > 0 else 0
        qc_cx_size_average = qc_cx_size / total_episodes if total_episodes > 0 else 0
        qc_cx_depth_average = qc_cx_depth / total_episodes if total_episodes > 0 else 0
        qc_size_average = qc_size / total_episodes if total_episodes > 0 else 0
        qc_depth_average = qc_depth / total_episodes if total_episodes > 0 else 0   

        current_difficulty = env.get_attr('difficulty', 0)[0]  

        if current_difficulty <= 1024: 
            if success_rate >= self.success_threshold:
                env.env_method('reset_success_data')
                print('increasing difficulty')
                # Increase difficulty
                for env_idx in range(env.num_envs):
                    env.env_method('set_difficulty', current_difficulty + 1, indices=env_idx)
        
        
        self.logger.record('tot_episodes', total_episodes)
        self.logger.record('success_rate', success_rate)
        self.logger.record('cx_size', qc_cx_size)
        self.logger.record('cx_size_average', qc_cx_size_average)
        self.logger.record('cx_depth', qc_cx_depth)
        self.logger.record('cx_depth_average', qc_cx_depth_average)
        self.logger.record('size', qc_size)
        self.logger.record('size_average', qc_size_average)
        self.logger.record('depth', qc_depth)
        self.logger.record('depth_average', qc_depth_average)

        self.logger.record('difficulty', mean(env.get_attr('difficulty')))

        self.model.save('models/5q_model')
        env.env_method('reset_success_data')
        return True'''
    

if __name__ == "__main__":

    env = CustomEnv()

    model = MaskablePPO.load('5q_model')

    # input_type can be "routing" or "norouting" depending on whether you want to use the test dataset that has already been routed or not
    input_type = "routing"

    rl_successes = 0
    sat_successes = 0

    greedy_times = []
    ag_times = []
    sat_times = []
    rl_times = []

    greedy_depths = []
    greedy_cx_depths = []

    ag_depths = []
    ag_cx_depths = []

    sat_depths = []
    sat_cx_depths = []

    rl_depths = []
    rl_cx_depths = []

    success_maps = {'0': 0, '1': 0, '2': 0, '3': 0}
    error_maps = {'0': 0, '1': 0, '2': 0, '3': 0}

    pm_staged = StagedPassManager()
    

    equiv_lib = SessionEquivalenceLibrary
    target = Target()
    target.add_instruction(SGate(), name='s')
    target.add_instruction(HGate(), name='h')
    target.add_instruction(CXGate(), name='cx')

    pass_manager = PassManager(BasisTranslator(equiv_lib, target))

    

    file_path = 'datasets/test_dataset_'+str(input_type)  
    file = open(file_path, 'rb')
    test_dataset = pickle.load(file)
    file.close()
    print("len test dataset: ", len(test_dataset))

    for data in test_dataset:
        obs = data['obs']
        map = data['map']
        target_qc_size = data['target_qc_size']
        cliff_state = data['cliff_state']

        pm_staged.routing = PassManager(SabreSwap(CouplingMap(map)))

        #greedy
        greedy_t1 = time.time()
        greedy_qc_synth = synth_clifford_greedy(Clifford(obs))
        greedy_qc_synth_routed = pm_staged.run(greedy_qc_synth)
        greedy_qc_synth_routed_resynth = pass_manager.run(greedy_qc_synth_routed)
        greedy_t2 = time.time()

        #ag
        ag_t1 = time.time()
        ag_qc_synth = synth_clifford_ag(Clifford(obs))
        ag_qc_synth_routed = pm_staged.run(ag_qc_synth)
        ag_qc_synth_routed_resynth = pass_manager.run(ag_qc_synth_routed)
        ag_t2 = time.time()

        #sat
        sat_t1 = time.time()
        res = synthesize_clifford_depth(Clifford(obs), coupling_map=CouplingMap(map), verbosity=0)
        if res.solutions[0].is_sat:
            sat_qc_synth_routed_resynth = pass_manager.run(res.circuit)
        sat_t2 = time.time()

        #rl
        truncated = False
        terminated = False

        env.set_state(obs, map, target_qc_size, cliff_state)

        rl_t1 = time.time()
        while not terminated and not truncated:

            env.render()
            
            action, _ = model.predict(obs, action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)

        rl_t2 = time.time()    

        rl_synth_qc = env.info["synthetized_qc"]


        greedy_times.append(greedy_t2-greedy_t1)
        ag_times.append(ag_t2-ag_t1)
        

        greedy_depth = greedy_qc_synth_routed_resynth.depth()
        greedy_depths.append(greedy_depth)
        greedy_cx_depth = greedy_qc_synth_routed_resynth.depth(lambda gate: gate[0].name in ['cx'])
        greedy_cx_depths.append(greedy_cx_depth)

        ag_depth = ag_qc_synth_routed_resynth.depth()
        ag_depths.append(ag_depth)            
        ag_cx_depth = ag_qc_synth_routed_resynth.depth(lambda gate: gate[0].name in ['cx'])
        ag_cx_depths.append(ag_cx_depth)

        if res.solutions[0].is_sat:
            sat_successes += 1
            sat_depths.append(sat_qc_synth_routed_resynth.depth())
            sat_cx_depths.append(sat_qc_synth_routed_resynth.depth(lambda gate: gate[0].name in ['cx']))
            sat_times.append(sat_t2-sat_t1)

        if terminated == True:
            rl_successes += 1            
            rl_times.append(rl_t2 - rl_t1)
            rl_depth = rl_synth_qc.depth()
            rl_depths.append(rl_depth)
            rl_cx_depth = rl_synth_qc.depth(lambda gate: gate[0].name in ['cx'])
            rl_cx_depths.append(rl_cx_depth)
            
            
    env.close()
    '''print("rl_successes: ", rl_successes)
    print("sat_successes: ", sat_successes)
    print("rl success rate: ", rl_successes/len(test_dataset))
    print("sat success rate: ", sat_successes/len(test_dataset))
    print("success_maps: ", success_maps)
    print("error_maps: ", error_maps)'''
        

#-----------------------------------------
    #saving results lists

    results = {}
    results['greedy_times'] = greedy_times
    results['greedy_depths'] = greedy_depths
    results['greedy_cx_depths'] = greedy_cx_depths

    results['ag_times'] = ag_times
    results['ag_depths'] = ag_depths
    results['ag_cx_depths'] = ag_cx_depths

    results['sat_times'] = sat_times
    results['sat_depths'] = sat_depths
    results['sat_cx_depths'] = sat_cx_depths

    results['rl_times'] = rl_times
    results['rl_depths'] = rl_depths
    results['rl_cx_depths'] = rl_cx_depths

    file_path = 'results/'+str(input_type)+'/test_'+str(input_type)  

    file = open(file_path, 'wb')
    pickle.dump(results, file)
    file.close()

    time_averages = {}
    time_averages['rl'] = sum(rl_times) / 500
    time_averages['greedy'] = sum(greedy_times) / 500
    time_averages['ag'] = sum(ag_times) / 500
    time_averages['sat'] = sum(sat_times) / 500

    s1 = ""

    for key in time_averages.keys():
        s1 += key + " time average: " + str(time_averages[key]) + "\n"

    with open("results/"+str(input_type)+"/time_averages_"+str(input_type)+".txt", "a") as f1:
        f1.write(s1)

    #--------------------------------------------------
    #--------------------------HISTS--------------------

    #plotting depths and cx depths over time - HISTS - multiple plots
    plt.clf()
    min_bin = min(min(greedy_depths), min(rl_depths), min(ag_depths), min(sat_depths), min(greedy_cx_depths), min(rl_cx_depths), min(ag_cx_depths), min(sat_cx_depths))
    max_bin = max(max(greedy_depths), max(rl_depths), max(ag_depths), max(sat_depths), max(greedy_cx_depths), max(rl_cx_depths), max(ag_cx_depths), max(sat_cx_depths))

    bins = np.arange(min_bin - 0.5, max_bin + 1.5, 1)
    

    max_y = max(
    np.histogram(greedy_depths, bins=bins)[0].max(),
    np.histogram(rl_depths, bins=bins)[0].max(),
    np.histogram(ag_depths, bins=bins)[0].max(),
    np.histogram(sat_depths, bins=bins)[0].max(),
    np.histogram(greedy_cx_depths, bins=bins)[0].max(),
    np.histogram(rl_cx_depths, bins=bins)[0].max(),
    np.histogram(ag_cx_depths, bins=bins)[0].max(),
    np.histogram(sat_cx_depths, bins=bins)[0].max()
)

    bar_width = 0.75
    bin_centers = bins[:-1] + 0.5
    xlim_min = 0
    xlim_max = max_bin

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 6))

    #rl
    hist_rl, _ = np.histogram(rl_depths, bins=bins)
    hist_rl_cx, _ = np.histogram(rl_cx_depths, bins=bins)

    labels_added = {'Depth': False,
                    'CX depth': False}

    for i in range(len(bins) - 1):
        height_rl = hist_rl[i]
        height_rl_cx = hist_rl_cx[i]
        overlap_height = min(height_rl, height_rl_cx)
        if overlap_height > 0:
            ax1.bar(bin_centers[i], overlap_height, color='pink',
                edgecolor='blue', hatch='//')
        if height_rl > overlap_height:
            ax1.bar(bin_centers[i], height_rl - overlap_height, color='blue', edgecolor='blue', bottom=overlap_height,
                label='Depth' if not labels_added['Depth'] else "")
            labels_added['Depth'] = True
        if height_rl_cx > overlap_height:
            ax1.bar(bin_centers[i], height_rl_cx - overlap_height, color='pink', edgecolor='pink', bottom=overlap_height,
                label='CX depth' if not labels_added['CX depth'] else "")
            labels_added['CX depth'] = True

    ax1.set_title("RL")        
    ax1.set_ylabel("Frequency")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    ax1.set_xlim(xlim_min, xlim_max)
    ax1.set_ylim(0, max_y)    
    ax1.legend(fontsize="x-small")


    #greedy
    hist_greedy, _ = np.histogram(greedy_depths, bins=bins)
    hist_greedy_cx, _ = np.histogram(greedy_cx_depths, bins=bins)

    labels_added = {'Depth': False,
                    'CX depth': False}

    for i in range(len(bins) - 1):
        height_greedy = hist_greedy[i]
        height_greedy_cx = hist_greedy_cx[i]
        overlap_height = min(height_greedy, height_greedy_cx)
        if overlap_height > 0:
            ax2.bar(bin_centers[i], overlap_height, width=bar_width, color='pink',
                edgecolor='blue', hatch='//')
        if height_greedy > overlap_height:
            ax2.bar(bin_centers[i], height_greedy - overlap_height, width=bar_width, color='blue', edgecolor='blue', bottom=overlap_height,
                label='Depth' if not labels_added['Depth'] else "")
            labels_added['Depth'] = True
        if height_greedy_cx > overlap_height:
            ax2.bar(bin_centers[i], height_greedy_cx - overlap_height, width=bar_width, color='pink', edgecolor='pink', bottom=overlap_height,
                label='CX depth' if not labels_added['CX depth'] else "")
            labels_added['CX depth'] = True

    ax2.set_title("Greedy")
    ax2.set_ylabel("Frequency")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    ax2.set_xlim(xlim_min, xlim_max)
    ax2.set_ylim(0, max_y) 
    ax2.legend(fontsize="x-small")


    #ag
    hist_ag, _ = np.histogram(ag_depths, bins=bins)
    hist_ag_cx, _ = np.histogram(ag_cx_depths, bins=bins)

    labels_added = {'Depth': False,
                    'CX depth': False}

    for i in range(len(bins) - 1):
        height_ag = hist_ag[i]
        height_ag_cx = hist_ag_cx[i]
        overlap_height = min(height_ag, height_ag_cx)
        if overlap_height > 0:
            ax3.bar(bin_centers[i], overlap_height, width=bar_width, color='pink',
                edgecolor='blue', hatch='//')
        if height_ag > overlap_height:
            ax3.bar(bin_centers[i], height_ag - overlap_height, width=bar_width, color='blue', edgecolor='blue', bottom=overlap_height,
                label='Depth' if not labels_added['Depth'] else "")
            labels_added['Depth'] = True
        if height_ag_cx > overlap_height:
            ax3.bar(bin_centers[i], height_ag_cx - overlap_height, width=bar_width, color='pink', edgecolor='pink', bottom=overlap_height,
                label='CX depth' if not labels_added['CX depth'] else "")
            labels_added['CX depth'] = True

    ax3.set_title("AG")
    ax3.set_ylabel("Frequency")
    ax3.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    ax3.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    ax3.set_xlim(xlim_min, xlim_max)
    ax3.set_ylim(0, max_y) 
    ax3.legend(fontsize="x-small")


    #sat
    hist_sat, _ = np.histogram(sat_depths, bins=bins)
    hist_sat_cx, _ = np.histogram(sat_cx_depths, bins=bins)

    labels_added = {'Depth': False,
                    'CX depth': False}

    for i in range(len(bins) - 1):
        height_sat = hist_sat[i]
        height_sat_cx = hist_sat_cx[i]
        overlap_height = min(height_sat, height_sat_cx)
        if overlap_height > 0:
            ax4.bar(bin_centers[i], overlap_height, width=bar_width, color='pink',
                edgecolor='blue', hatch='//')
        if height_sat > overlap_height:
            ax4.bar(bin_centers[i], height_sat - overlap_height, width=bar_width, color='blue', edgecolor='blue', bottom=overlap_height,
                label='Depth' if not labels_added['Depth'] else "")
            labels_added['Depth'] = True
        if height_sat_cx > overlap_height:
            ax4.bar(bin_centers[i], height_sat_cx - overlap_height, width=bar_width, color='pink', edgecolor='pink', bottom=overlap_height,
                label='CX depth' if not labels_added['CX depth'] else "")
            labels_added['CX depth'] = True

    ax4.set_title("SAT")
    ax4.set_ylabel("Frequency")
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True)) 
    ax4.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    ax4.set_xlim(xlim_min, xlim_max)
    ax4.set_ylim(0, max_y) 
    ax4.legend(fontsize="x-small")

    plt.tight_layout()

    fig.savefig("plots/5q_test_" + str(input_type)+".pdf")
