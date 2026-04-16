# Sequence Dependent Setup Time Scheduling using UVFA and DRL

This repository contains the implementation of a **Utility Function Approximation (UVFA)** based Deep Reinforcement Learning (DRL) agent for solving the **Dynamic Flexible Job Shop Scheduling Problem (DFJSP)**. The model specifically addresses scenarios with **Sequence Dependent Setup Times (SDST)** and aims to optimize a weighted combination of `Time-to-Due` and `Setup Time`.

The architecture utilizes a Two-Stream UVFA network to handle both environmental states and dynamic reward weights, combined with a DQN (Deep Q-Network) agent. Additionally, the code includes post-hoc explainability using **SHAP (SHapley Additive exPlanations)** to interpret the agent's policy.

## 📁 Project Structure

- `main_UVFA.py`: The main script containing the simulation environment, RL agent, and training loop.
- `dqn_policy_net_wc{N}.pth` (Output): Saved PyTorch model weights for each Work Center after training.
- `simulation.log`: Log file recording simulation events and metrics.
- `makespan_per_episode.png`, `shap_*.png` (Output): Generated plots for performance metrics and feature importance.

## 🛠️ Requirements

The code is written in Python and requires the following libraries. You can install them using pip:

```bash
pip install torch numpy scipy matplotlib pandas tqdm shap

⚙️ Configuration and Parameters
The simulation parameters are defined in the if __name__ == "__main__": block at the bottom of main_UVFA.py. Key parameters include:
| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `num_episodes` | 20000 | Total number of training episodes. |
| `total_num_jobs` | 100 | Number of jobs to release per episode. |
| `num_work_centers` | 5 | Fixed number of work centers in the shop. |
| `num_machines_per_wc_range` | [1, 5] | Random range for machines per work center. |
| `mean_interval_range` | [5, 15] | Random range for job inter-arrival times. |
| `update_target_every` | 100 | Frequency (episodes) for updating the target network. |

🚀 How to Run
python main_UVFA.py

🧠 Algorithm Overview
1. State Representation
The state vector (size 43) includes statistical features of the queue such as:
Mean/Min/Max/Std of Processing Time, Waiting Time, and Time-to-Due.
Machine availability statistics.
Global system information (Number of Work Centers, Job Types).
Reward Weights: The last 2 features are the dynamic weights for Time-to-Due and Setup Time.
2. Action Space
The agent selects from a combination of 12 Job Priority Rules (e.g., SPT, EDD, FIFO) and 1 Machine Priority Rule (Least_Utilized), resulting in 12 discrete actions.
3. Reward Function
The reward is a weighted sum of the improvement in average Time-to-Due and reduction in Setup Time compared to a baseline (First-Come-First-Served) sequence.
4. Network Architecture
The TwoStreamUVFA module separates the input into:
Environment Stream: Processes the 41-dimensional state.
Weight Stream: Processes the 2-dimensional reward weights.
The features are fused to predict Q-values for the 12 actions.
📊 Output and Results
After execution, you will find the following outputs in your directory:
Model Checkpoints: dqn_policy_net_wc{N}.pth files for each Work Center (N=0 to 4).
Training Logs: simulation.log contains detailed logs of each episode's metrics.
Performance Plots: PNG files showing trends in Makespan, Flow Time, Tardiness, and Loss.
Explainability Plots: shap_feature_importance_wc{N}.png and shap_summary_action{A}_wc{N}.png, which visualize which features (e.g., waiting time, due date) are most influential for the agent's decisions.
