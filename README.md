# Topology_Aware_Quantum_Circuit_Synthesis_with_RL
Generalizing Reinforcement Learning-based Quantum Circuit Synthesis across Multiple Topologies

## Description
This project introduces a topology-aware reinforcement learning-based quantum circuit synthesis which addresses different topologies simultaneously. Quantum circuit synthesis represents one of the most critical aspects of the compiling stack is the synthesis of quantum circuits, which aims to minimize the depth and number of gates, as well as the error rate. this work proposes an RL-based quantum circuit synthesis algorithm that addresses different topologies simultaneously. Performance is evaluated in terms of the depth and CX depth values of the synthesized circuits and is compared to the state-of-the-art greedy techniques and a SAT solver. Experimental tests demonstrate that our synthesizer outperforms state-of-the-art greedy techniques while being significantly faster than SAT solvers. The trained models have been tests both on routed and non-routed datasets.

## Installation
Follow these steps to set up the environment and install the necessary packages to run the project:
1. Clone the repository
2. Create and activate a virtual environment (anaconda was used for this project)
3. Install the required packages using requirements.txt

## Contents
The directories and files related to the test experiments in the 5 qubit-case are provided.
The decription of them is provided below:
- `qiskit_sat_synthesis_main`: directory containing a collection of SAT-based synthesis methods for various Qiskit objects
- `datasets`: directory containing the datasets used for test the model. It contains both the routed and non-routed datasets
- `results`: directory containing the files where the evaluated performance obtained from the test are saved. In particular, there are two objects related to the depths and CX depths of the circuits synthesized by the RL trained model, the state-of-the-art greedy techniques and the SAT solver, and a text file related to the time averages of the algorithms
- `plots`: directory containing the plots related to the comparison of the performance with that of the state-of-the-art
- `5q_model.zip`: RL trained model used for the test 
- `generate_tst_dataset.py`: file containing the code to generate a new test dataset and save it in the `datasets` directory
- `test.py`: file containing the code to test the trained model: it saves the results in the `results` directory and plots the performance in the `plots` directory

## Usage
In the `results` and `plots` directories the values related to the test that has already been conducted are present.
If you want to conduct a new test, you can create new datasets, routed and not, by running the `generate_test_dataset.py` file with the following command:
```
python generate_test_dataset.py
```
and test the model by running the `test.py` file with the following command:
```
python test.py
```
In the `test.py` code you can specify if the test dataset you are using is routed or not by filling the `input_type` variable.
At the end of the test you will be able to visualize the new results and plots in the respective folders.



