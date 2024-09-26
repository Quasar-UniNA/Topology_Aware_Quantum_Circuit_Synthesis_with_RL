# Topology_Aware_Quantum_Circuit_Synthesis_with_RL
Generalizing Reinforcement Learning-based Quantum Circuit Synthesis across Multiple Topologies

## Description
The optimization of the limited resources of current quantum devices hinges on the implementation of quantum-compiling techniques. The process of transforming high-level quantum algorithms into executable quantum circuits has a significant impact on the performance and fidelity of computations. One of the most critical aspects of the compiling stack is the synthesis of quantum circuits, which aims to minimize the depth and number of gates, as well as the error rate. Nevertheless, the optimal synthesis algorithms are extremely computationally expensive. This is why synthesis algorithms based on reinforcement learning (RL) have recently been introduced. However, such algorithms are constrained by the topology of quantum processors, necessitating retraining of the RL agent for different topologies. To achieve a balance between adaptability, optimality, and computational cost in RL-based quantum circuit synthesis techniques, this work proposes an RL-based quantum circuit synthesis algorithm that addresses different topologies simultaneously. Experimental tests demonstrate that our synthesizer outperforms state-of-the-art greedy techniques while being significantly faster than SAT solvers.

## Installation
Follow these steps to set up the environment and install the necessary packages to run the project:
1. Clone the repository
2. Create and activate a virtual environment (anaconda was used for this project)
3. Install the required packages using requirements.txt

## Contents
The decription of the files and directories contained in the project is provided below:
- `qiskit_sat_synthesis_main`: directory containing a collection of SAT-based synthesis methods for various Qiskit objects
- `datasets`: directory containing the datasets used for test the model. It contains both the routed and non-routed datasets
- `results`: directory containing the files where the evaluated performance obtained from the test are saved. The performance values are the depth and CX depths of the circuits synthesized by the RL trained model, the state-of-the-art greedy techniques and the SAT solver
- `plots`: directory containing th plots related to the comparison of the performance with that of the state-of-the-art
- `5q_model.zip`: RL trained model used for the test 
- `generate_tst_dataset.py`: file containing the code to generate a new test dataset and save it in the `datasets` directory
- `test.py`: file containing the code to test the trained model: it saves the results in the "results" directory and plots the performance in the "plots" directory

## Usage
In the `results` and `plots` directories the values related to the test that has already been conducted are present.
If you want to conduct a new test, you can create a new dataset by running the `generate_test_dataset.py` file with the following command:
`python generate_test_dataset.py`
and visualize the new results and plots by running the `test.py` file with the following command:
`python test.py`




