# -*- coding: utf-8 -*-
"""
Test Trained RL Agents Performance
- No training, only exploitation (epsilon=0.0)
- Uses random seeds 1 to 100 to generate random problem instances
- Loads previously trained models for each work center
- Collects performance metrics
"""

from scipy.stats import expon
import numpy as np
import heapq
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import logging
import shap
import time
import copy

# Set logging to file only (no console output)
logging.basicConfig(
    filename='test_trained_agents.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# ----------------------------
# Classes and Functions (Same as in your original code)
# ----------------------------
# Include or import all classes and functions needed from your original code 
# (Job, Machine, WorkCenter, Event, DQNAgent (without training), 
# initialization functions, generate_job_arrivals, etc.)
# Make sure your code for simulation and metrics logging is included.

# For brevity here, we assume these are defined similarly as in your training code:
# - Job, Machine, WorkCenter, Operation, Event classes
# - system_empty, schedule_event, handle_job_arrival, handle_operation_completion, assign_jobs_to_idle_machines
# - extract_state, decode_action, apply_job_priority, apply_machine_priority, assign_job_to_machine
# - decision_making (unchanged except we do not store transitions since no training)
# - run_simulation
# - log_metrics
# - etc.

# IMPORTANT MODIFICATION:
# In the decision_making function, do not store transitions or learn, since we are only testing.
# Set rl_agent.epsilon = 0.0 after loading models.

# ----------------------------
# Configure Logging
# ----------------------------

logging.basicConfig(
    filename='simulation.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite the log file each run
)

# ----------------------------
# Global Variables and Constants
# ----------------------------

SIMULATION_END_TIME = None  # Will be set after generating all job arrivals

# ----------------------------
# Core Classes
# ----------------------------

class Job:
    def __init__(self, job_id, job_type, routing, arrival_time, due_date):
        self.job_id = job_id
        self.job_type = job_type  # String, e.g., 'Type1'
        self.routing = routing    # Ordered list of work_center_ids to visit
        self.current_operation = -1  # Start before the first operation
        self.arrival_time = arrival_time
        self.arrival_time_at_current_wc = None  # Time when job arrives at current work center
        self.due_date = due_date
        self.completion_time = None
        self.assigned_state = None
        self.assigned_action = None
        

    def next_work_center(self):
        if self.current_operation + 1 < len(self.routing):
            return self.routing[self.current_operation + 1]
        return None

class Machine:
    def __init__(self, machine_id, work_center_id,mean_time_between_failures):
        self.machine_id = machine_id
        self.work_center_id = work_center_id
        self.is_busy = False
        self.current_job = None
        self.available_time = 0  # Time when the machine becomes available
        self.total_busy_time = 0  # Total busy time (could extend beyond simulation time)
        self.total_busy_time_within_simulation = 0  # Busy time within simulation time
        # Optionally, keep track of all operations assigned
        self.assigned_operations = []  # List to store operations
        self.total_setup_time = 0  # New attribute to track accumulative setup time
        self.last_job_type = None
        self.mean_time_between_failures = mean_time_between_failures
        self.last_repair_time = 0

class WorkCenter:
    def __init__(self, work_center_id, machines, num_job_types):
        self.work_center_id = work_center_id
        self.machines = machines
        self.waiting_jobs = deque()  # Queue of jobs waiting to be processed
        self.setup_times = {
            f'Type{jt1}': {f'Type{jt2}': random.randint(1, 50) if jt1 != jt2 else 0 for jt2 in range(1, num_job_types + 1)}
            for jt1 in range(1, num_job_types + 1)
        }
        self.first_job_setup_time = {
            f'Type{jt}': random.randint(1, 50) for jt in range(1, num_job_types + 1)
        }
        self.selected_rule = None

class Operation:
    def __init__(self, job, work_center_id, processing_time):
        self.job = job
        self.work_center_id = work_center_id
        self.processing_time = processing_time
        self.start_time = None
        self.end_time = None

class Event:
    def __init__(self, time, event_type, data):
        self.time = time
        self.event_type = event_type  # 'job_arrival' or 'operation_complete'
        self.data = data              # Dictionary containing relevant data

    def __lt__(self, other):
        return self.time < other.time

class TwoStreamUVFA(nn.Module):
    def __init__(self, env_state_size, weight_state_size, hidden_size, action_size):
        """
        env_state_size : int
            Number of features in the environment portion of the state (excluding weights).
        weight_state_size : int
            Number of features used for the reward weights (e.g., 2).
        hidden_size : int
            A base hidden size used in MLP layers (you can adjust).
        action_size : int
            Number of possible actions (dispatching rules) to output Q-values for.
        """
        super(TwoStreamUVFA, self).__init__()

        # ----- Environment Stream -----
        self.env_stream = nn.Sequential(
            nn.Linear(env_state_size, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.01),
        )

        # ----- Weight Stream -----
        self.weight_stream = nn.Sequential(
            nn.Linear(weight_state_size, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 32),
            nn.LeakyReLU(0.01),
        )

        # ----- Fusion + Output for Q-Values -----
        # After streaming, we can fuse them by concatenation. 
        # If we have env_stream->128 output and weight_stream->64 output,
        # combined dimension = 128 + 32 = 160.
        fusion_input_dim = 128 + 32
        self.post_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, action_size)
        )

    def forward(self, state):
        """
        state: Tensor of shape [batch_size, env_state_size + weight_state_size]
        We must split the state into:
          - environment portion
          - weight portion
        """
        # 1) Separate environment features from weights
        #    We'll assume the *last* weight_state_size features are the reward weights.
        #    e.g. if total state_size = 43, weight_state_size=2 => env_state_size=41
        batch_size, total_dim = state.shape
        # Hard-coded approach: environment features = everything except the last 'weight_state_size' columns
        env_part = state[:, : -2]       # or use env_state_size if known
        weight_part = state[:, -2 :]    # last 2 features are the weights

        # 2) Pass each portion through its respective stream
        env_emb = self.env_stream(env_part)       # shape [batch_size, 128]
        weight_emb = self.weight_stream(weight_part)  # shape [batch_size, 64]

        # 3) Fuse (concatenate) embeddings
        fused = torch.cat([env_emb, weight_emb], dim=-1)  # shape [batch_size, 128+64=192]

        # 4) Post-fusion feed-forward to get Q-values
        q_values = self.post_fusion(fused)  # shape [batch_size, action_size]

        return q_values

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=3e-5, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.999998, epsilon_min=0.01, memory_size=20000, batch_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # ---- Split the state_size into environment portion vs. weight portion
        # We assume the last 2 features in state are the 2 reward weights:
        self.weight_state_size = 2
        self.env_state_size = state_size - self.weight_state_size

        # ---- Two-Stream Networks ----
        self.policy_net = TwoStreamUVFA(
            env_state_size=self.env_state_size,
            weight_state_size=self.weight_state_size,
            hidden_size=hidden_size,
            action_size=action_size
        )

        self.target_net = TwoStreamUVFA(
            env_state_size=self.env_state_size,
            weight_state_size=self.weight_state_size,
            hidden_size=hidden_size,
            action_size=action_size
        )

        self.update_target_network()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay Memory
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Loss Tracking
        self.loss_history = []
        self.cumulative_reward = 0.0


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        # if random.random() < self.epsilon:
        #     action = random.randint(0, self.action_size - 1)
        #     return action

        # Convert to tensor and add batch dimension
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # shape [1, state_size]
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)  # shape [1, action_size]
            action = q_values.argmax(dim=1).item()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
    
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)       # [batch, state_size]
        actions_tensor = torch.LongTensor(actions).unsqueeze(1)                  # [batch, 1]
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)                 # [batch, 1]
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)  # [batch, state_size]
        dones_tensor = torch.FloatTensor(dones).unsqueeze(1)                     # [batch, 1]

        # Current Q
        current_q = self.policy_net(states_tensor)              # [batch, action_size]
        current_q = current_q.gather(1, actions_tensor)         # [batch, 1]

        # Target Q
        with torch.no_grad():
            max_next_q = self.target_net(next_states_tensor).max(dim=1)[0].unsqueeze(1)  # [batch, 1]
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        # Compute MSE loss
        loss = nn.MSELoss()(current_q, target_q)

        self.loss_history.append(loss.item())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
            

# ----------------------------
# Initialize Work Centers and Machines
# ----------------------------

def initialize_work_centers(num_work_centers, num_machines_per_wc, num_job_types):
    work_centers = {}
    for wc_id in range(num_work_centers):
        num_machines = num_machines_per_wc[wc_id]
        machines = []
        mean_time_between_failures = random.randint(500, 1500)
        for m in range(num_machines):
            
            machine_id = f"WC{wc_id}_M{m+1}"
            machine = Machine(
                machine_id=machine_id,
                work_center_id=wc_id,
                mean_time_between_failures=mean_time_between_failures
            )
            machines.append(machine)

        # We will assign setup times after we generate them
        work_centers[wc_id] = WorkCenter(wc_id, machines, num_job_types)
    return work_centers

# ----------------------------
# Job Generation
# ----------------------------
def job_type_processing_time(job_type, wc_id, processing_times):
    try:
        return processing_times[wc_id]['processing_time']
    except KeyError:
        print(f"Error: Missing processing time for Job Type '{job_type}', Work Center ID {wc_id}.")
        sys.exit(1)
        
def generate_job_arrivals(num_job_types, job_type_distributions, total_num_jobs, job_type_routing, job_types):
    job_arrivals = []
    job_id_counter = 1

    # 初始化每种 job type 的下一个 arrival_time
    arrival_heap = []  # heap of (arrival_time, jt)
    arrival_time_tracker = {}

    for jt in range(1, num_job_types + 1):
        arrival_time = max(job_type_distributions[jt].rvs(), 1)
        heapq.heappush(arrival_heap, (arrival_time, jt))
        arrival_time_tracker[jt] = arrival_time

    while len(job_arrivals) < total_num_jobs:
        arrival_time, jt = heapq.heappop(arrival_heap)
        routing = job_type_routing[jt]

        # 计算 due date
        total_work_content = sum([
            job_type_processing_time(
                job_type=f"Type{jt}",
                wc_id=wc_id,
                processing_times=job_types[f"Type{jt}"]['operations']
            )
            for wc_id in routing
        ])
        factor = random.uniform(1, 2)
        due_date = int(arrival_time + total_work_content * factor)

        job = Job(job_id_counter, f"Type{jt}", routing, arrival_time, due_date)
        job_arrivals.append((arrival_time, job))
        job_id_counter += 1

        # 安排下一个同类型作业到达
        inter_arrival = max(job_type_distributions[jt].rvs(), 1)
        next_arrival = arrival_time + inter_arrival
        heapq.heappush(arrival_heap, (next_arrival, jt))

    return job_arrivals


# ----------------------------
# Event Handling
# ----------------------------
def system_empty(work_centers):
    for wc in work_centers.values():
        if wc.waiting_jobs:
            return False
        for machine in wc.machines:
            if machine.is_busy:
                return False
    return True

def schedule_event(time, event_type, data, event_queue):
    event = Event(time, event_type, data)
    heapq.heappush(event_queue, event)

def schedule_machine_breakdown(machine, current_time, event_queue):
    mean_time_between_failures = machine.mean_time_between_failures
    time_until_failure = np.random.exponential(mean_time_between_failures)
    failure_time = current_time + time_until_failure

    schedule_event(
        failure_time,
        'machine_breakdown',
        {'machine': machine},
        event_queue
    )

def handle_job_arrival(job, work_centers, processing_times, rl_agents, simulation_clock, event_queue, work_centers_machines,completed_jobs):
    next_wc_id = job.next_work_center()
    if next_wc_id is not None:
        job.current_operation += 1  # Move to the first operation
        job.arrival_time_at_current_wc = simulation_clock  # Set arrival time at current work center
        work_centers[next_wc_id].waiting_jobs.append(job)
        
        # Make scheduling decision upon job arrival
        decision_making(
            work_centers[next_wc_id],
            rl_agents[next_wc_id],
            simulation_clock,
            total_jobs_in_system(work_centers),
            processing_times,
            event_queue,
            work_centers_machines,
            rl_agents,
            completed_jobs,
            work_centers
        )
        # After decision making, try to assign jobs to idle machines
        assign_jobs_to_idle_machines(work_centers[next_wc_id], simulation_clock, processing_times, event_queue, work_centers)
    

def handle_operation_completion(machine, work_centers, processing_times, rl_agents, simulation_clock, event_queue, completed_jobs, work_centers_machines):
    job = machine.current_job
    machine.last_job_type = job.job_type
    machine.current_job = None
    machine.is_busy = False
    
    
    # Move to next operation
    next_wc_id = job.next_work_center()
    if next_wc_id is not None:
        job.current_operation += 1
        job.arrival_time_at_current_wc = simulation_clock  # Set arrival time at next work center
        work_centers[next_wc_id].waiting_jobs.append(job)
        
        # Make scheduling decision upon job arrival
        decision_making(
            work_centers[next_wc_id],
            rl_agents[next_wc_id],
            simulation_clock,
            total_jobs_in_system(work_centers),
            processing_times,
            event_queue,
            work_centers_machines,
            rl_agents,
            completed_jobs,
            work_centers
        )
        # After decision making, try to assign jobs to idle machines
        assign_jobs_to_idle_machines(work_centers[next_wc_id], simulation_clock, processing_times, event_queue, work_centers)
    else:
        # Job has completed all operations
        job.completion_time = simulation_clock
        completed_jobs.append(job)
        
        
        # Calculate tardiness
        tardiness = max(0, job.completion_time - job.due_date)
        
        
        
    
    # After freeing the machine, check if there are waiting jobs
    wc_id = machine.work_center_id
    assign_jobs_to_idle_machines(work_centers[wc_id], simulation_clock, processing_times, event_queue, work_centers)

def handle_machine_breakdown(machine, simulation_clock, work_centers, event_queue,
                             rl_agents, processing_times, completed_jobs, work_centers_machines):
    machine.is_broken = True
    machine.is_busy = False
    machine.last_job_type = None
    machine.last_breakdown_time = simulation_clock

    current_wc_id = machine.work_center_id
    work_center = work_centers[current_wc_id]

    if machine.current_job:
        interrupted_job = machine.current_job
        work_center.waiting_jobs.append(interrupted_job)
        machine.current_job = None

        # Remove any operation_complete events for this machine
        event_queue[:] = [
            ev for ev in event_queue
            if not (ev.event_type == 'operation_complete' and ev.data['machine'] == machine)
        ]
        heapq.heapify(event_queue)

    # Schedule repair
    repair_duration = max(1, int(np.random.exponential(50)))
    repair_completion_time = simulation_clock + repair_duration
    machine.available_time = repair_completion_time
    schedule_event(
        repair_completion_time,
        'machine_repaired',
        {'machine': machine},
        event_queue
    )

    # Trigger new decisions
    decision_making(
        work_center,
        rl_agents[current_wc_id],
        simulation_clock,
        total_jobs_in_system(work_centers),
        processing_times,
        event_queue,
        work_centers_machines,
        rl_agents,
        completed_jobs,
        work_centers
    )

    assign_jobs_to_idle_machines(
        work_center,
        simulation_clock,
        processing_times,
        event_queue,
        work_centers
    )

def handle_machine_repaired(machine, work_centers, simulation_clock, rl_agents,
                            event_queue, processing_times, completed_jobs, work_centers_machines):
    machine.is_broken = False
    machine.last_repair_time = simulation_clock

    # Schedule next breakdown
    schedule_machine_breakdown(machine, simulation_clock, event_queue)

    current_wc_id = machine.work_center_id
    work_center = work_centers[current_wc_id]

    # Decision-making
    decision_making(
        work_center,
        rl_agents[current_wc_id],
        simulation_clock,
        total_jobs_in_system(work_centers),
        processing_times,
        event_queue,
        work_centers_machines,
        rl_agents,
        completed_jobs,
        work_centers
    )
    assign_jobs_to_idle_machines(work_center, simulation_clock, processing_times, event_queue, work_centers)
    machine.last_job_type = None

def assign_jobs_to_idle_machines(work_center, simulation_clock, processing_times, event_queue, work_centers):
    current_wc_id = work_center.work_center_id
    for machine in work_center.machines:
        if not machine.is_busy and work_center.waiting_jobs:
            job = work_center.waiting_jobs.popleft()  # Get the next job in the sorted queue
            assign_job_to_machine(job, machine, simulation_clock, processing_times, event_queue, work_centers)
            work_center.waiting_jobs = apply_job_priority(work_center.waiting_jobs, work_center.selected_rule, processing_times, current_wc_id, work_center)
        
def total_jobs_in_system(work_centers):
    total = 0
    for wc in work_centers.values():
        total += len(wc.waiting_jobs)
        for machine in wc.machines:
            if machine.is_busy:
                total += 1
    return total

# ----------------------------
# State Extraction and Action Decoding
# ----------------------------

def extract_state(work_centers, simulation_clock, total_jobs, simulation_end_time, current_wc_id, num_machines_per_wc_current, processing_times):
    state = []
    current_work_center = work_centers[current_wc_id]
    num_waiting_jobs = len(current_work_center.waiting_jobs)

    # 1. Number of work centers
    num_work_centers = len(work_centers)
    MAX_WORK_CENTERS = 20
    state.append(num_work_centers / MAX_WORK_CENTERS)
    
    # 2. Number of machines in the work center
    num_machines = num_machines_per_wc_current
    MAX_MACHINES_PER_WC = 5
    state.append(num_machines / MAX_MACHINES_PER_WC)
    
    # 3. Number of job types in the system
    num_job_types = len(processing_times)
    MAX_JOB_TYPES = 10
    state.append(num_job_types / MAX_JOB_TYPES)
    
    # 4. Number of jobs waiting in the work center
    MAX_WAITING_JOBS = 100
    state.append(num_waiting_jobs / MAX_WAITING_JOBS)
    
    # 5. Time to due date statistics
    if num_waiting_jobs > 0:
        time_to_due_dates = [job.due_date - simulation_clock for job in current_work_center.waiting_jobs]
        mean_time_to_due = np.mean(time_to_due_dates)
        min_time_to_due = np.min(time_to_due_dates)
        max_time_to_due = np.max(time_to_due_dates)
        std_time_to_due = np.std(time_to_due_dates)
    else:
        mean_time_to_due = min_time_to_due = max_time_to_due = std_time_to_due = 0.0
    
    state.extend([
        mean_time_to_due / simulation_end_time,
        min_time_to_due / simulation_end_time,
        max_time_to_due / simulation_end_time,
        std_time_to_due / simulation_end_time
    ])
    
    # 6. Waiting time statistics
    if num_waiting_jobs > 0:
        waiting_times = [simulation_clock - job.arrival_time_at_current_wc for job in current_work_center.waiting_jobs]
        mean_waiting_time = np.mean(waiting_times)
        min_waiting_time = np.min(waiting_times)
        max_waiting_time = np.max(waiting_times)
        std_waiting_time = np.std(waiting_times)
    else:
        mean_waiting_time = min_waiting_time = max_waiting_time = std_waiting_time = 0.0
    
    state.extend([
        mean_waiting_time / simulation_end_time,
        min_waiting_time / simulation_end_time,
        max_waiting_time / simulation_end_time,
        std_waiting_time / simulation_end_time
    ])
    
    # 7. Time since release into system statistics
    if num_waiting_jobs > 0:
        time_since_release = [simulation_clock - job.arrival_time for job in current_work_center.waiting_jobs]
        mean_time_since_release = np.mean(time_since_release)
        min_time_since_release = np.min(time_since_release)
        max_time_since_release = np.max(time_since_release)
        std_time_since_release = np.std(time_since_release)
    else:
        mean_time_since_release = min_time_since_release = max_time_since_release = std_time_since_release = 0.0
    
    state.extend([
        mean_time_since_release / simulation_end_time,
        min_time_since_release / simulation_end_time,
        max_time_since_release / simulation_end_time,
        std_time_since_release / simulation_end_time
    ])
    
    # 8. Remaining operations statistics
    if num_waiting_jobs > 0:
        remaining_ops = [len(job.routing) - job.current_operation for job in current_work_center.waiting_jobs]
        mean_remaining_ops = np.mean(remaining_ops)
        min_remaining_ops = np.min(remaining_ops)
        max_remaining_ops = np.max(remaining_ops)
        std_remaining_ops = np.std(remaining_ops)
    else:
        mean_remaining_ops = min_remaining_ops = max_remaining_ops = std_remaining_ops = 0.0
    
    MAX_OPERATIONS = 10
    state.extend([
        mean_remaining_ops / MAX_OPERATIONS,
        min_remaining_ops / MAX_OPERATIONS,
        max_remaining_ops / MAX_OPERATIONS,
        std_remaining_ops / MAX_OPERATIONS
    ])
    
    # 9. Processing time statistics in the current work center
    if num_waiting_jobs > 0:
        processing_times_wc = [processing_times[job.job_type][current_wc_id]['processing_time'] for job in current_work_center.waiting_jobs]
        mean_pt_wc = np.mean(processing_times_wc)
        min_pt_wc = np.min(processing_times_wc)
        max_pt_wc = np.max(processing_times_wc)
        std_pt_wc = np.std(processing_times_wc)
    else:
        mean_pt_wc = min_pt_wc = max_pt_wc = std_pt_wc = 0.0
    
    MAX_PROCESSING_TIME = 100.0
    state.extend([
        mean_pt_wc / MAX_PROCESSING_TIME,
        min_pt_wc / MAX_PROCESSING_TIME,
        max_pt_wc / MAX_PROCESSING_TIME,
        std_pt_wc / MAX_PROCESSING_TIME
    ])
    
    # 10. Remaining processing time statistics
    if num_waiting_jobs > 0:
        remaining_processing_times = []
        for job in current_work_center.waiting_jobs:
            remaining_pt = sum(
                processing_times[job.job_type][wc_id]['processing_time']
                for wc_id in job.routing[job.current_operation:]
            )
            remaining_processing_times.append(remaining_pt)
        mean_remaining_pt = np.mean(remaining_processing_times)
        min_remaining_pt = np.min(remaining_processing_times)
        max_remaining_pt = np.max(remaining_processing_times)
        std_remaining_pt = np.std(remaining_processing_times)
    else:
        mean_remaining_pt = min_remaining_pt = max_remaining_pt = std_remaining_pt = 0.0
    
    MAX_TOTAL_PROCESSING_TIME = 1000.0
    state.extend([
        mean_remaining_pt / MAX_TOTAL_PROCESSING_TIME,
        min_remaining_pt / MAX_TOTAL_PROCESSING_TIME,
        max_remaining_pt / MAX_TOTAL_PROCESSING_TIME,
        std_remaining_pt / MAX_TOTAL_PROCESSING_TIME
    ])
    
    # 11. Earliest machine available time statistics
    available_times = []
    for machine in current_work_center.machines:
        if not machine.is_busy:
            available_times.append(0.0)
        else:
            available_time = machine.available_time - simulation_clock
            available_times.append(available_time)
    
    if available_times:
        mean_available_time = np.mean(available_times)
        min_available_time = np.min(available_times)
        max_available_time = np.max(available_times)
        std_available_time = np.std(available_times)
    else:
        mean_available_time = min_available_time = max_available_time = std_available_time = 0.0
    
    state.extend([
        mean_available_time / simulation_end_time,
        min_available_time / simulation_end_time,
        max_available_time / simulation_end_time,
        std_available_time / simulation_end_time
    ])
    
    # 12. Number of Operations statistics
    if num_waiting_jobs > 0:
        ops = [len(job.routing) for job in current_work_center.waiting_jobs]
        mean_ops = np.mean(ops)
        min_ops = np.min(ops)
        max_ops = np.max(ops)
        std_ops = np.std(ops)
    else:
        mean_ops = min_ops = max_ops = std_ops = 0.0
    
    state.extend([
        mean_ops / MAX_OPERATIONS,
        min_ops / MAX_OPERATIONS,
        max_ops / MAX_OPERATIONS,
        std_ops / MAX_OPERATIONS
    ])

    # 13. Potential job type number in this work center (new feature)
    if num_waiting_jobs > 0:
        unique_job_types = set(job.job_type for job in current_work_center.waiting_jobs)
        potential_job_type_number = len(unique_job_types)
    else:
        potential_job_type_number = 0
    
    MAX_POTENTIAL_JOB_TYPES = 10  # Adjust as necessary
    state.append(potential_job_type_number / MAX_POTENTIAL_JOB_TYPES)
    
    # 14. setup time of the first available machine
    first_available_machine = min(current_work_center.machines, key=lambda m: m.available_time)
    previous_job_type = (
        first_available_machine.current_job.job_type if first_available_machine.current_job else None
    )

    # Calculate setup times for all waiting jobs
    setup_times = []
    for job in current_work_center.waiting_jobs:
        job_type = job.job_type
        if previous_job_type is not None:
            # Use setup time relative to the previous job
            setup_time = current_work_center.setup_times[previous_job_type][job_type]
        else:
            # Use first job setup time if no previous job
            setup_time = current_work_center.first_job_setup_time[job_type]
        setup_times.append(setup_time)

    if setup_times:
        mean_setup_time = np.mean(setup_times)
        min_setup_time = np.min(setup_times)
        max_setup_time = np.max(setup_times) 
        std_setup_time = np.std(setup_times)
    else:
        min_setup_time = max_setup_time = mean_setup_time = std_setup_time = 0.0  # Default for no jobs

    # Normalize setup times if required
    MAX_SETUP_TIME = 200  # Adjust if your setup times are bounded differently

    state.extend([mean_setup_time / MAX_SETUP_TIME,
                  min_setup_time / MAX_SETUP_TIME,
                  max_setup_time / MAX_SETUP_TIME,
                  std_setup_time / MAX_SETUP_TIME
    ])
    state.extend([
        weights['time_to_due'],  # Weight for time-to-due
        weights['setup_time']    # Weight for setup-time
    ])
    return np.array(state)


def decode_action(action, machine_priority_rules=['Least_Utilized']):
    job_priority_rules = [
        'FIFO', 'LIFO', 'EDD', 'LDD', 'SPT', 'LPT', 'LOR', 'ERD', 'Shortest_Setup_Time', 'Longest_Setup_Time','Shortest_Setup_Adjusted_Processing_Time','Longest_Setup_Adjusted_Processing_Time'
        ]
    machine_priority_rules = machine_priority_rules  # Only 'Least_Utilized' in this case
    num_machine_rules = len(machine_priority_rules)
    job_rule = job_priority_rules[action // num_machine_rules]
    machine_rule = machine_priority_rules[action % num_machine_rules]
    return job_rule, machine_rule

# ----------------------------
# Dispatching Rules and Decision Making
# ----------------------------

DEFAULT_PROCESSING_TIME = 50  # Define a default processing time

def apply_job_priority(jobs, rule, processing_times, current_wc_id, current_wc):
    logging.info(f"Applying '{rule}' priority at Work Center ID {current_wc_id}")
    valid_jobs = list(jobs)  # Convert deque to list for easier sorting
    
    # Determine the first available machine
    first_available_machine = min(current_wc.machines, key=lambda m: m.available_time)
    previous_job_type = (
        first_available_machine.current_job.job_type if first_available_machine.current_job else None
    )
    
    try:
        if rule == 'FIFO':  # First In, First Out
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.arrival_time_at_current_wc)
        elif rule == 'LIFO':  # Last In, First Out
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.arrival_time_at_current_wc, reverse=True)
        elif rule == 'EDD':  # Earliest Due Date
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.due_date)
        elif rule == 'LDD':  # Latest Due Date
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.due_date, reverse=True)
        elif rule == 'SPT':  # Shortest Processing Time
            sorted_jobs = sorted(valid_jobs, key=lambda job: processing_times[job.job_type][current_wc_id]['processing_time'])
        elif rule == 'LPT':  # Longest Processing Time
            sorted_jobs = sorted(valid_jobs, key=lambda job: processing_times[job.job_type][current_wc_id]['processing_time'], reverse=True)
        elif rule == 'LOR':  # Minimal Operation Remaining
            sorted_jobs = sorted(valid_jobs, key=lambda job: len(job.routing) - (job.current_operation + 1))
        elif rule == 'ERD':  # Earliest Release Date
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.arrival_time)
        elif rule == 'Shortest_Setup_Time':  # Shortest Setup Time to First Available Machine
            if previous_job_type is not None:
                sorted_jobs = sorted(
                    valid_jobs, key=lambda job: current_wc.setup_times[previous_job_type][job.job_type]
                )
            else:
                sorted_jobs = sorted(
                    valid_jobs, key=lambda job: current_wc.first_job_setup_time[job.job_type]
                )
        elif rule == 'Longest_Setup_Time':  # Longest Setup Time to First Available Machine
            if previous_job_type is not None:
                sorted_jobs = sorted(
                    valid_jobs, key=lambda job: current_wc.setup_times[previous_job_type][job.job_type], reverse=True
                )
            else:
                sorted_jobs = sorted(
                    valid_jobs, key=lambda job: current_wc.first_job_setup_time[job.job_type], reverse=True
                )
        elif rule == 'Shortest_Setup_Adjusted_Processing_Time':  # Shortest Setup Time to First Available Machine
            if previous_job_type is not None:
                sorted_jobs = sorted(
                    valid_jobs, key=lambda job: current_wc.setup_times[previous_job_type][job.job_type] + processing_times[job.job_type][current_wc_id]['processing_time']
                )
            else:
                sorted_jobs = sorted(
                    valid_jobs, key=lambda job: current_wc.first_job_setup_time[job.job_type] + processing_times[job.job_type][current_wc_id]['processing_time']
                )
        elif rule == 'Longest_Setup_Adjusted_Processing_Time':  # Shortest Setup Time to First Available Machine
            if previous_job_type is not None:
                sorted_jobs = sorted(
                    valid_jobs, key=lambda job: current_wc.setup_times[previous_job_type][job.job_type] + processing_times[job.job_type][current_wc_id]['processing_time'], reverse=True
                )
            else:
                sorted_jobs = sorted(
                    valid_jobs, key=lambda job: current_wc.first_job_setup_time[job.job_type] + processing_times[job.job_type][current_wc_id]['processing_time'], reverse=True
                )
        else:
            sorted_jobs = list(valid_jobs)
    except KeyError as e:
        logging.error(f"Error during sorting with rule '{rule}': {e}")
        sorted_jobs = list(valid_jobs)  # Fallback to unsorted jobs
    
    return deque(sorted_jobs)  # Return as deque for efficient pops from the left


def apply_machine_priority(machines, rule, wc_id):
    if rule == 'Least_Utilized':
        # Sort machines by their available time (earliest available first)
        sorted_machines = sorted(machines, key=lambda m: m.available_time)
    else:
        sorted_machines = machines.copy()
    
    return sorted_machines

def assign_job_to_machine(job, machine, simulation_clock, processing_times, event_queue, work_centers):
    current_wc = work_centers[job.routing[job.current_operation]]
    previous_job_type = machine.last_job_type

    if previous_job_type is not None:
        setup_time = current_wc.setup_times[previous_job_type][job.job_type]
    else:
        setup_time = current_wc.first_job_setup_time[job.job_type]

    base_processing_time = processing_times[job.job_type][current_wc.work_center_id]['processing_time']
    processing_time = base_processing_time
    operation_start_time = max(simulation_clock, machine.available_time) + setup_time
    operation_end_time = operation_start_time + processing_time

    operation = Operation(job, current_wc.work_center_id, processing_time)
    operation.start_time = operation_start_time
    operation.end_time = operation_end_time

    machine.available_time = operation_end_time
    machine.is_busy = True
    machine.current_job = job
    machine.total_busy_time += processing_time + setup_time
    machine.total_setup_time += setup_time  # Increment accumulative setup time

    machine.assigned_operations.append(operation)
    schedule_event(operation.end_time, 'operation_complete', {'machine': machine, 'job': job}, event_queue)

def calculate_reward(work_center, simulation_clock, processing_times, original_waiting_jobs, sorted_waiting_jobs):
    def calculate_penalty_factor(job, simulation_clock, base_penalty_factor=10):
        """
        Calculate the penalty factor based on the remaining operations of the job.

        Parameters:
            job (Job): The job object.
            simulation_clock (int): Current simulation time.
            base_penalty_factor (float): Base penalty factor for scaling.

        Returns:
            float: Dynamic penalty factor.
        """
        remaining_operations = len(job.routing) - job.current_operation  # Calculate remaining operations
        penalty_factor = base_penalty_factor * (1 / remaining_operations)
        return penalty_factor

    def simulate_sequence(jobs_sequence, use_dynamic_priority, copied_work_center):
        """
        Simulate the job sequence using a copied work center and calculate metrics.
        """
        machine_available_times = [machine.available_time for machine in copied_work_center.machines]
        last_job_types = [machine.current_job.job_type if machine.current_job else None for machine in copied_work_center.machines]
        expected_completion_times = []
        total_setup_time = 0

        jobs_in_sequence = deque(jobs_sequence)

        while jobs_in_sequence:
            job = jobs_in_sequence.popleft()

            # Find the earliest available machine
            earliest_machine_idx = machine_available_times.index(min(machine_available_times))
            machine = copied_work_center.machines[earliest_machine_idx]
            machine_available_time = machine_available_times[earliest_machine_idx]

            # Determine the setup time
            previous_job_type = last_job_types[earliest_machine_idx]
            if previous_job_type is not None:
                setup_time = copied_work_center.setup_times[previous_job_type][job.job_type]
            else:
                setup_time = copied_work_center.first_job_setup_time[job.job_type]

            # Calculate the start time for the job
            start_time = max(simulation_clock, machine_available_time) + setup_time

            # Get processing time
            processing_time = processing_times[job.job_type][copied_work_center.work_center_id]['processing_time']

            # Calculate completion time
            completion_time = start_time + processing_time
            expected_completion_times.append(completion_time)

            # Update the machine's available time and last job type in the copied work center
            machine_available_times[earliest_machine_idx] = completion_time
            last_job_types[earliest_machine_idx] = job.job_type

            # Update the machine object in copied work center
            machine.available_time = completion_time
            machine.last_job_type = job.job_type
            machine.current_job = job
            machine.is_busy = True
            machine.total_busy_time += processing_time + setup_time
            machine.total_setup_time += setup_time

            # Dynamically re-prioritize jobs if required
            if use_dynamic_priority:
                jobs_in_sequence = apply_job_priority(
                    jobs_in_sequence,
                    copied_work_center.selected_rule,
                    processing_times,
                    copied_work_center.work_center_id,
                    copied_work_center
                )

        return expected_completion_times, total_setup_time

    def calculate_time_to_due(jobs_sequence, expected_completion_times):
        """
        Calculate the time-to-due metric with penalty factors for tardiness.
        """
        time_to_due_list = []
        for job, ect in zip(jobs_sequence, expected_completion_times):
            penalty_factor = calculate_penalty_factor(job, simulation_clock)
            if ect <= job.due_date:
                time_to_due = job.due_date - ect
            else:
                time_to_due = penalty_factor * (job.due_date - ect)
            time_to_due_list.append(time_to_due)

        return np.mean(time_to_due_list) if time_to_due_list else 0.0

    # Create a deep copy of the work center
    copied_work_center_original = copy.deepcopy(work_center)
    copied_work_center_sorted = copy.deepcopy(work_center)

    # Simulate the original and sorted sequences
    original_completion_times, original_setup_time = simulate_sequence(
        original_waiting_jobs, use_dynamic_priority=False, copied_work_center=copied_work_center_original
    )
    sorted_completion_times, sorted_setup_time = simulate_sequence(
        sorted_waiting_jobs, use_dynamic_priority=True, copied_work_center=copied_work_center_sorted
    )

    # Calculate time-to-due metrics
    avg_time_to_due_before = calculate_time_to_due(original_waiting_jobs, original_completion_times)
    avg_time_to_due_after = calculate_time_to_due(sorted_waiting_jobs, sorted_completion_times)

    # Calculate the differences
    time_to_due_diff = avg_time_to_due_after - avg_time_to_due_before
    setup_time_diff = original_setup_time - sorted_setup_time

    # Combine the rewards
    reward = setup_time_diff + time_to_due_diff  # Adjust weights if needed
    return reward



def decision_making(work_center, rl_agent, simulation_clock, total_jobs, processing_times, event_queue,
                    work_centers_machines, rl_agents, completed_jobs, work_centers):
    current_wc_id = work_center.work_center_id
    num_machines_per_wc_current = len(work_centers_machines[current_wc_id])

    # Check if there is a pending state and action to store
    if hasattr(rl_agent, 'last_state') and rl_agent.last_state is not None:
        # Extract next state at the next decision point
        next_state = extract_state(work_centers, simulation_clock, total_jobs, SIMULATION_END_TIME,
                                   current_wc_id, num_machines_per_wc_current, processing_times)
        # Determine if episode is done (set appropriately)
        done = False  # Set to True if the episode ends here
        # Store the transition
        rl_agent.store_transition(rl_agent.last_state, rl_agent.last_action, rl_agent.last_reward, next_state, done)
        
        # Learn from experience
        rl_agent.learn()
        # Reset last_state, last_action, last_reward
        rl_agent.last_state = None
        rl_agent.last_action = None
        rl_agent.last_reward = None

    # Now proceed to select the next action
    # Extract current state specific to the current work center
    state = extract_state(work_centers, simulation_clock, total_jobs, SIMULATION_END_TIME,
                          current_wc_id, num_machines_per_wc_current, processing_times)
    # RL Agent selects action
    action = rl_agent.select_action(state)
    

    # Decode action into priority rules
    job_priority_rule, machine_priority_rule = decode_action(action)
    work_center.selected_rule = job_priority_rule

    # Store the original waiting jobs before sorting
    original_waiting_jobs = list(work_center.waiting_jobs)

    # Apply dispatching rules
    sorted_jobs = apply_job_priority(work_center.waiting_jobs, job_priority_rule, processing_times, current_wc_id, work_center)
    work_center.waiting_jobs = sorted_jobs  # Update the waiting jobs with the sorted list

    # Assign the state and action to all waiting jobs (they all share the same state and action)
    for job in work_center.waiting_jobs:
        job.assigned_state = state.copy()
        job.assigned_action = action

    # Collect reward
    reward = calculate_reward(work_center, simulation_clock, processing_times, original_waiting_jobs, list(work_center.waiting_jobs))
    
    rl_agent.cumulative_reward += reward

    # Instead of extracting next_state and storing the transition now, we store state, action, reward in agent
    rl_agent.last_state = state
    rl_agent.last_action = action
    rl_agent.last_reward = reward




# ----------------------------
# Simulation Loop
# ----------------------------

def run_simulation(rl_agents, processing_times, event_queue, work_centers, job_type_routing, completed_jobs, work_centers_machines, total_num_jobs):
    global simulation_clock
    while len(completed_jobs) < total_num_jobs:
        if event_queue:
            event = heapq.heappop(event_queue)
            simulation_clock = int(event.time)  # Ensure integer simulation time
            
            if event.event_type == 'job_arrival':
                handle_job_arrival(event.data['job'], work_centers, processing_times, rl_agents, simulation_clock, event_queue, work_centers_machines,completed_jobs)
            elif event.event_type == 'operation_complete':
                handle_operation_completion(event.data['machine'], work_centers, processing_times, rl_agents, simulation_clock, event_queue, completed_jobs, work_centers_machines)
            elif event.event_type == 'machine_breakdown':
                handle_machine_breakdown(event.data['machine'], simulation_clock, work_centers,
                                         event_queue, rl_agents, processing_times, completed_jobs, work_centers_machines)
            elif event.event_type == 'machine_repaired':
                handle_machine_repaired(event.data['machine'], work_centers, simulation_clock, rl_agents,
                                        event_queue, processing_times, completed_jobs, work_centers_machines)
            # Add other event types as needed
        else:
            # No events are scheduled, but not all jobs are completed
            if system_empty(work_centers):
                # System is empty; all jobs are completed
                break
            else:
                # Advance simulation clock to the next event (e.g., next machine availability)
                next_event_time = None
                # Find the earliest machine available time
                for wc in work_centers.values():
                    for machine in wc.machines:
                        if machine.is_busy:
                            if next_event_time is None or machine.available_time < next_event_time:
                                next_event_time = machine.available_time
                if next_event_time is not None:
                    simulation_clock = int(next_event_time)
                    # Process any machines that become available at this time
                    for wc in work_centers.values():
                        for machine in wc.machines:
                            if machine.is_busy and machine.available_time == next_event_time:
                                handle_operation_completion(machine, work_centers, processing_times, rl_agents, simulation_clock, event_queue, completed_jobs, work_centers_machines)
                else:
                    # No machines are busy; check for waiting jobs
                    for wc in work_centers.values():
                        if wc.waiting_jobs:
                            assign_jobs_to_idle_machines(wc, simulation_clock, processing_times, event_queue, work_centers)
                    # If the system is empty after decision making, we can end the simulation
                    if system_empty(work_centers):
                        break
                    
    # for wc_id, rl_agent in rl_agents.items():
    #     if hasattr(rl_agent, 'last_state') and rl_agent.last_state is not None:
    #         # Extract the final state
    #         next_state = extract_state(work_centers, simulation_clock, total_jobs_in_system(work_centers), SIMULATION_END_TIME,
    #                                    wc_id, len(work_centers_machines[wc_id]), processing_times)
    #         done = True  # Since the simulation has ended
    #         # Store the transition
    #         rl_agent.store_transition(rl_agent.last_state, rl_agent.last_action, rl_agent.last_reward, next_state, done)
            
    #         # Learn from experience
    #         rl_agent.learn()
    #         # Reset last_state, last_action, last_reward
    #         rl_agent.last_state = None
    #         rl_agent.last_action = None
    #         rl_agent.last_reward = None

# Initialize metrics dictionary
metrics = {
    'Episode': [],
    'Makespan': [],
    'Average_Flow_Time': [],
    'Average_Tardiness': [],
    'Machine_Utilization': [],
    'Accumulative_Setup_Time': [],  # New metric
    'Average_Loss': [],
}

def log_metrics(episode, completed_jobs, work_centers, rl_agents):
    total_jobs = len(completed_jobs)

    # Calculate Makespan (time when the last job is completed)
    makespan = max(job.completion_time for job in completed_jobs) if completed_jobs else 0

    # Calculate mean flow time (average time jobs spend in the system)
    mean_flow_time = np.mean([job.completion_time - job.arrival_time for job in completed_jobs]) if completed_jobs else 0

    # Calculate mean tardiness (average tardiness of all jobs)
    mean_tardiness = np.mean([max(0, job.completion_time - job.due_date) for job in completed_jobs]) if completed_jobs else 0

    # Calculate machine utilization
    total_machines = sum(len(wc.machines) for wc in work_centers.values())
    total_machine_time = makespan * total_machines  # Total available machine time
    total_busy_time = sum(machine.total_busy_time_within_simulation for wc in work_centers.values() for machine in wc.machines)
    machine_utilization = total_busy_time / total_machine_time if total_machine_time > 0 else 0
    
    accumulative_setup_time = sum(machine.total_setup_time for wc in work_centers.values() for machine in wc.machines)
    # Calculate average loss for each agent
    average_loss_per_agent = {}
    for wc_id, agent in rl_agents.items():
        if agent.loss_history:
            average_loss = np.mean(agent.loss_history)
            average_loss_per_agent[wc_id] = average_loss
            # Reset the loss history for the next episode
            agent.loss_history = []
        else:
            average_loss_per_agent[wc_id] = 0.0  # If no loss recorded

    # Store metrics
    metrics['Episode'].append(episode)
    metrics['Makespan'].append(makespan)
    metrics['Average_Flow_Time'].append(mean_flow_time)
    metrics['Average_Tardiness'].append(mean_tardiness)
    metrics['Machine_Utilization'].append(machine_utilization)
    metrics['Accumulative_Setup_Time'].append(accumulative_setup_time)
    metrics['Average_Loss'].append(average_loss_per_agent)

    # Reset completed_jobs for the next episode
    completed_jobs.clear()

    # Print and log metrics
    logging.info(f"=== Metrics - Episode {episode} ===")
    logging.info(f"Makespan: {makespan}")
    logging.info(f"Average Flow Time: {mean_flow_time:.2f}")
    logging.info(f"Average Tardiness: {mean_tardiness:.2f}")
    logging.info(f"Machine Utilization: {machine_utilization:.2f}")
    logging.info(f"Accumulative Setup Time: {accumulative_setup_time:.2f}")
    for wc_id, avg_loss in average_loss_per_agent.items():
        logging.info(f"Average Loss for Work Centre {wc_id}: {avg_loss:.4f}")
    print(f"\n=== Metrics - Episode {episode} ===")
    print(f"Makespan: {makespan}")
    print(f"Average Flow Time: {mean_flow_time:.2f}")
    print(f"Average Tardiness: {mean_tardiness:.2f}")
    print(f"Machine Utilization: {machine_utilization:.2f}")
    print(f"Accumulative Setup Time: {accumulative_setup_time:.2f}")
    for wc_id, avg_loss in average_loss_per_agent.items():
        print(f"Average Loss for Work Centre {wc_id}: {avg_loss:.4f}")
    print("===============================\n")


# ----------------------------
# Reset Simulation Function
# ----------------------------

def reset_simulation(num_work_centers, num_job_types, num_machines_per_wc, processing_times,
                     job_type_distributions, job_type_routing, total_num_jobs, job_types):
    global simulation_clock, event_queue, work_centers, completed_jobs, rl_agents, SIMULATION_END_TIME
    simulation_clock = 0
    event_queue = []
    heapq.heapify(event_queue)
    completed_jobs = []
    
    # Initialize work centers
    work_centers = initialize_work_centers(num_work_centers, num_machines_per_wc, num_job_types)
    
    # Generate job arrivals
    job_arrivals = generate_job_arrivals(num_job_types, job_type_distributions, total_num_jobs, job_type_routing, job_types)
    
    # Update SIMULATION_END_TIME based on last job's arrival time plus some buffer
    last_arrival_time = job_arrivals[-1][0]
    SIMULATION_END_TIME = last_arrival_time + 20000  # Add buffer
    
    # Schedule job arrival events
    for arrival_time, job in job_arrivals:
        schedule_event(arrival_time, 'job_arrival', {'job': job}, event_queue)
    
    for wc in work_centers.values():
        for machine in wc.machines:
            schedule_machine_breakdown(machine, simulation_clock, event_queue)
    
    logging.info(f"Simulation Reset: {num_work_centers} work centers, {num_job_types} job types, {len(job_arrivals)} job arrivals.")


# ----------------------------
# Testing the trained agent
# ----------------------------

def test_trained_agents_with_various_weights(
    num_tests,
    weight_step,
    num_work_centers,
    num_job_types_range,
    mean_interval_range,
    num_machines_per_wc_range,
    total_num_jobs
):
    """
    Tests the trained agents at all weight pairs (w, 1-w)
    for w in {0.0, weight_step, 2*weight_step, ..., 1.0}.
    Runs multiple random seeds for each weight pair and stores all results (including the weight pair).
    """

    # 1) Build the weight-pair list
    # For weight_step = 0.1, we get w in {0.0, 0.1, 0.2, ..., 1.0} 
    test_weights_list = []
    steps = int(round(1.0 / weight_step))  # e.g. 1.0 / 0.1 = 10
    for i in range(steps + 1):            # +1 so we include i=10
        w_time_to_due = round(i * weight_step, 2)
        w_setup_time = round(1.0 - w_time_to_due, 2)
        test_weights_list.append((w_time_to_due, w_setup_time))

    print("Testing weight pairs:")
    print(test_weights_list)

    # 2) Prepare RL Agents (load models)
    job_priority_rules = [
        'FIFO', 'LIFO', 'EDD', 'LDD', 'SPT', 'LPT', 'LOR', 'ERD',
        'Shortest_Setup_Time', 'Longest_Setup_Time',
        'Shortest_Setup_Adjusted_Processing_Time', 'Longest_Setup_Adjusted_Processing_Time'
    ]
    machine_priority_rules = ['Least_Utilized']
    action_size = len(job_priority_rules) * len(machine_priority_rules)

    # Make sure state_size matches what you used in training
    # If you appended 2 more features for weights, it could be 43, 45, etc.
    state_size = 43  # adjust if needed

    global weights  # We'll overwrite it each time
    weights = {'time_to_due': 0.5, 'setup_time': 0.5}  # placeholder

    # Load your trained models for each work center
    

    # 3) We'll store results in a list of dicts:
    results_data = []

    # 4) Loop over all weight pairs
    for (w_time_to_due, w_setup_time) in test_weights_list:
        # Overwrite the global weights with the current pair
        weights['time_to_due'] = w_time_to_due
        weights['setup_time'] = w_setup_time
        # weights['time_to_due'] = 0.7
        # weights['setup_time'] = 0.3

        print(f"\n--- Testing with weights = [{weights['time_to_due']}, {weights['setup_time']}] ---")

        # For each weight pair, run multiple seeds
        for seed in range(1, num_tests + 1):
            # Fix seeds for reproducibility
            random.seed(seed)
            np.random.seed(seed)
            rl_agents = {}
            for wc_id in range(num_work_centers):
                agent = DQNAgent(state_size, action_size)
                model_path = f'dqn_policy_net_wc{wc_id}.pth'
                if not os.path.exists(model_path):
                    logging.error(f"Model file {model_path} not found.")
                    sys.exit(1)
                agent.policy_net.load_state_dict(torch.load(model_path))
                agent.policy_net.eval()
                agent.epsilon = 0.0  # Pure exploitation mode
                rl_agents[wc_id] = agent

            # Randomly pick number of job types
            num_job_types = random.randint(num_job_types_range[0], num_job_types_range[1])

            # Generate job_types, routing, etc. (similar to your code)
            job_types = {}
            job_type_routing = {}
            for jt in range(1, num_job_types + 1):
                routing_length = random.randint(2, num_work_centers)
                routing = random.sample(range(num_work_centers), routing_length)
                operations = {}
                for wc_id in routing:
                    pt = random.randint(1, 99)
                    operations[wc_id] = {'processing_time': pt}
                job_type = f"Type{jt}"
                job_types[job_type] = {'operations': operations}
                job_type_routing[jt] = routing

            # Machines per work center
            num_machines_per_wc = {}
            work_centers_machines = {}
            for wc_id in range(num_work_centers):
                num_machines = random.randint(num_machines_per_wc_range[0], num_machines_per_wc_range[1])
                num_machines_per_wc[wc_id] = num_machines
                work_centers_machines[wc_id] = [f"WC{wc_id}_M{m+1}" for m in range(num_machines)]

            # Convert job_types to processing_times structure
            processing_times = {jt: job_types[jt]['operations'] for jt in job_types}

            # Generate arrival distributions
            job_type_mean_intervals = {
                jt: random.randint(mean_interval_range[0], mean_interval_range[1])
                for jt in range(1, num_job_types + 1)
            }
            job_type_lambda = {jt: 1 / mi for jt, mi in job_type_mean_intervals.items()}
            job_type_distributions = {jt: expon(scale=1 / lam) for jt, lam in job_type_lambda.items()}

            # Reset the simulation
            global simulation_clock, event_queue, completed_jobs, work_centers, SIMULATION_END_TIME
            simulation_clock = 0
            event_queue = []
            heapq.heapify(event_queue)
            completed_jobs = []

            work_centers = initialize_work_centers(num_work_centers, num_machines_per_wc, num_job_types)

            # Generate job arrivals
            job_arrivals = generate_job_arrivals(
                num_job_types,
                job_type_distributions,
                total_num_jobs,
                job_type_routing,
                job_types
            )
            last_arrival_time = job_arrivals[-1][0]
            SIMULATION_END_TIME = last_arrival_time + 20000

            # Schedule arrivals
            for arrival_time, job in job_arrivals:
                schedule_event(arrival_time, 'job_arrival', {'job': job}, event_queue)

            # Run the simulation
            run_simulation(
                rl_agents,
                processing_times,
                event_queue,
                work_centers,
                job_type_routing,
                completed_jobs,
                work_centers_machines,
                total_num_jobs
            )

            # Compute metrics
            if completed_jobs:
                makespan = max(job.completion_time for job in completed_jobs)
                mean_flow_time = np.mean([job.completion_time - job.arrival_time for job in completed_jobs])
                mean_tardiness = np.mean([max(0, job.completion_time - job.due_date) for job in completed_jobs])
            else:
                makespan = 0
                mean_flow_time = 0
                mean_tardiness = 0

            total_machines = sum(len(wc.machines) for wc in work_centers.values())
            total_machine_time = makespan * total_machines
            total_busy_time = sum(m.total_busy_time_within_simulation
                                  for wc in work_centers.values() for m in wc.machines)
            machine_utilization = (
                total_busy_time / total_machine_time
                if total_machine_time > 0 else 0.0
            )
            accumulative_setup_time = sum(
                m.total_setup_time for wc in work_centers.values() for m in wc.machines
            )

            # Store metrics (+ weights + seed) into the results
            results_data.append({
                'Weight_Time_To_Due': weights['time_to_due'],
                'Weight_Setup_Time': weights['setup_time'],
                'Seed': seed,
                'Makespan': makespan,
                'Average_Flow_Time': mean_flow_time,
                'Average_Tardiness': mean_tardiness,
                'Machine_Utilization': machine_utilization,
                'Accumulative_Setup_Time': accumulative_setup_time
            })
    # Define feature names based on your state representation
    feature_names = [
    # 1. Number of Work Centres
    'Num_Work_Centers',
    # 2. Number of Parallel Machines in the Work Centre
    'Num_Machines_in_WC',
    # 3. Number of Job Types in the System
    'Num_Job_Types_in_System',
    # 4. Number of Jobs Waiting in the Work Centre
    'Num_Waiting_Jobs_in_WC',
    # 5. Time to Due Date Statistics
    'Mean_Time_to_Due_Date',
    'Min_Time_to_Due_Date',
    'Max_Time_to_Due_Date',
    'Std_Time_to_Due_Date',
    # 6. Waiting Time Statistics
    'Mean_Waiting_Time',
    'Min_Waiting_Time',
    'Max_Waiting_Time',
    'Std_Waiting_Time',
    # 7. Time since release into system statistics (Newly Added)
    'Mean_Flow_Time',
    'Min_Flow_Time',
    'Max_Flow_Time',
    'Std_Flow_Time',
    # 8. Remaining Operation Number Statistics
    'Mean_Remaining_Operations',
    'Min_Remaining_Operations',
    'Max_Remaining_Operations',
    'Std_Remaining_Operations',
    # 9. Processing Time in Work Centre Statistics
    'Mean_Processing_Time_in_WC',
    'Min_Processing_Time_in_WC',
    'Max_Processing_Time_in_WC',
    'Std_Processing_Time_in_WC',
    # 10. Remaining Processing Time Statistics
    'Mean_Remaining_Processing_Time',
    'Min_Remaining_Processing_Time',
    'Max_Remaining_Processing_Time',
    'Std_Remaining_Processing_Time',
    # 11. Earliest Machine Available Time Statistics
    'Mean_Earliest_Machine_Available_Time',
    'Min_Earliest_Machine_Available_Time',
    'Max_Earliest_Machine_Available_Time',
    'Std_Earliest_Machine_Available_Time',
    # 12. Number of Operations statistics
    'Mean_Operations',
    'Min_Operations',
    'Max_Operations',
    'Std_Operations',
    #13. Potential job type number in this work center
    'Potential_Job_Type_Number_in_WC',
    #14. Setup time of the first available machine statistics
    'Mean_Setup_Time',
    'Min_Setup_TIme',
    'Max_Setup_Time',
    'Std_Setup_Time',
    #15. Weights of the rewards
    'Weights_of_time_to_due',
    'Weights_of_Setup_time'
    ]


    # Compute and visualize SHAP values for each agent
    print("\nComputing SHAP values for all agents...")
    for wc_id, agent in rl_agents.items():
        print(f"\nComputing SHAP values for Work Center ID {wc_id}")
        shap_values, test_states = explain_agent_policy(agent, feature_names, wc_id)
        visualize_shap_values(shap_values, test_states, feature_names, wc_id)

def test_trained_agents_with_given_weights(
    num_tests,
    test_weights_list,
    num_work_centers,
    num_job_types_range,
    mean_interval_range,
    num_machines_per_wc_range,
    total_num_jobs
):
    """
    Tests the trained agents at all weight pairs (w, 1-w)
    for w in {0.0, weight_step, 2*weight_step, ..., 1.0}.
    Runs multiple random seeds for each weight pair and stores all results (including the weight pair).
    """

    
    print(test_weights_list)

    # 2) Prepare RL Agents (load models)
    job_priority_rules = [
        'FIFO', 'LIFO', 'EDD', 'LDD', 'SPT', 'LPT', 'LOR', 'ERD',
        'Shortest_Setup_Time', 'Longest_Setup_Time',
        'Shortest_Setup_Adjusted_Processing_Time', 'Longest_Setup_Adjusted_Processing_Time'
    ]
    machine_priority_rules = ['Least_Utilized']
    action_size = len(job_priority_rules) * len(machine_priority_rules)

    # Make sure state_size matches what you used in training
    # If you appended 2 more features for weights, it could be 43, 45, etc.
    state_size = 43  # adjust if needed

    global weights  # We'll overwrite it each time
    weights = {'time_to_due': 0.0, 'setup_time': 1.0}  # placeholder

    # Load your trained models for each work center
    

    # 3) We'll store results in a list of dicts:
    results_data = []

    
    
    # Overwrite the global weights with the current pair
    weights['time_to_due'] = test_weights_list[0]
    weights['setup_time'] = test_weights_list[1]
    # weights['time_to_due'] = 0.7
    # weights['setup_time'] = 0.3

    print(f"\n--- Testing with weights = [{weights['time_to_due']}, {weights['setup_time']}] ---")

    # For each weight pair, run multiple seeds
    for seed in range(100, num_tests + 100):
        # Fix seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        rl_agents = {}
        for wc_id in range(num_work_centers):
            agent = DQNAgent(state_size, action_size)
            model_path = f'dqn_policy_net_wc{wc_id}.pth'
            if not os.path.exists(model_path):
                logging.error(f"Model file {model_path} not found.")
                sys.exit(1)
            agent.policy_net.load_state_dict(torch.load(model_path))
            agent.policy_net.eval()
            agent.epsilon = 0.0  # Pure exploitation mode
            rl_agents[wc_id] = agent

        # Randomly pick number of job types
        num_job_types = random.randint(num_job_types_range[0], num_job_types_range[1])

        # Generate job_types, routing, etc. (similar to your code)
        job_types = {}
        job_type_routing = {}
        for jt in range(1, num_job_types + 1):
            routing_length = random.randint(2, num_work_centers)
            routing = random.sample(range(num_work_centers), routing_length)
            operations = {}
            for wc_id in routing:
                pt = random.randint(1, 99)
                operations[wc_id] = {'processing_time': pt}
            job_type = f"Type{jt}"
            job_types[job_type] = {'operations': operations}
            job_type_routing[jt] = routing

        # Machines per work center
        num_machines_per_wc = {}
        work_centers_machines = {}
        for wc_id in range(num_work_centers):
            num_machines = random.randint(num_machines_per_wc_range[0], num_machines_per_wc_range[1])
            num_machines_per_wc[wc_id] = num_machines
            work_centers_machines[wc_id] = [f"WC{wc_id}_M{m+1}" for m in range(num_machines)]

        # Convert job_types to processing_times structure
        processing_times = {jt: job_types[jt]['operations'] for jt in job_types}

        # Generate arrival distributions
        job_type_mean_intervals = {
            jt: random.randint(mean_interval_range[0], mean_interval_range[1])
            for jt in range(1, num_job_types + 1)
        }
        job_type_lambda = {jt: 1 / mi for jt, mi in job_type_mean_intervals.items()}
        job_type_distributions = {jt: expon(scale=1 / lam) for jt, lam in job_type_lambda.items()}

        # Reset the simulation
        global simulation_clock, event_queue, completed_jobs, work_centers, SIMULATION_END_TIME
        simulation_clock = 0
        event_queue = []
        heapq.heapify(event_queue)
        completed_jobs = []

        work_centers = initialize_work_centers(num_work_centers, num_machines_per_wc, num_job_types)

        # Generate job arrivals
        job_arrivals = generate_job_arrivals(
            num_job_types,
            job_type_distributions,
            total_num_jobs,
            job_type_routing,
            job_types
        )
        last_arrival_time = job_arrivals[-1][0]
        SIMULATION_END_TIME = last_arrival_time + 20000

        # Schedule arrivals
        for arrival_time, job in job_arrivals:
            schedule_event(arrival_time, 'job_arrival', {'job': job}, event_queue)

        # Run the simulation
        run_simulation(
            rl_agents,
            processing_times,
            event_queue,
            work_centers,
            job_type_routing,
            completed_jobs,
            work_centers_machines,
            total_num_jobs
        )

        # Compute metrics
        if completed_jobs:
            makespan = max(job.completion_time for job in completed_jobs)
            mean_flow_time = np.mean([job.completion_time - job.arrival_time for job in completed_jobs])
            mean_tardiness = np.mean([max(0, job.completion_time - job.due_date) for job in completed_jobs])
        else:
            makespan = 0
            mean_flow_time = 0
            mean_tardiness = 0

        total_machines = sum(len(wc.machines) for wc in work_centers.values())
        total_machine_time = makespan * total_machines
        total_busy_time = sum(m.total_busy_time_within_simulation
                                for wc in work_centers.values() for m in wc.machines)
        machine_utilization = (
            total_busy_time / total_machine_time
            if total_machine_time > 0 else 0.0
        )
        accumulative_setup_time = sum(
            m.total_setup_time for wc in work_centers.values() for m in wc.machines
        )

        # Store metrics (+ weights + seed) into the results
        results_data.append({
                'Weight_Time_To_Due': weights['time_to_due'],
                'Weight_Setup_Time': weights['setup_time'],
                'Seed': seed,
                'Makespan': makespan,
                'Average_Flow_Time': mean_flow_time,
                'Average_Tardiness': mean_tardiness,
                'Machine_Utilization': machine_utilization,
                'Accumulative_Setup_Time': accumulative_setup_time
            })
    # Define feature names based on your state representation
    feature_names = [
    # 1. Number of Work Centres
    'Num_Work_Centers',
    # 2. Number of Parallel Machines in the Work Centre
    'Num_Machines_in_WC',
    # 3. Number of Job Types in the System
    'Num_Job_Types_in_System',
    # 4. Number of Jobs Waiting in the Work Centre
    'Num_Waiting_Jobs_in_WC',
    # 5. Time to Due Date Statistics
    'Mean_Time_to_Due_Date',
    'Min_Time_to_Due_Date',
    'Max_Time_to_Due_Date',
    'Std_Time_to_Due_Date',
    # 6. Waiting Time Statistics
    'Mean_Waiting_Time',
    'Min_Waiting_Time',
    'Max_Waiting_Time',
    'Std_Waiting_Time',
    # 7. Time since release into system statistics (Newly Added)
    'Mean_Flow_Time',
    'Min_Flow_Time',
    'Max_Flow_Time',
    'Std_Flow_Time',
    # 8. Remaining Operation Number Statistics
    'Mean_Remaining_Operations',
    'Min_Remaining_Operations',
    'Max_Remaining_Operations',
    'Std_Remaining_Operations',
    # 9. Processing Time in Work Centre Statistics
    'Mean_Processing_Time_in_WC',
    'Min_Processing_Time_in_WC',
    'Max_Processing_Time_in_WC',
    'Std_Processing_Time_in_WC',
    # 10. Remaining Processing Time Statistics
    'Mean_Remaining_Processing_Time',
    'Min_Remaining_Processing_Time',
    'Max_Remaining_Processing_Time',
    'Std_Remaining_Processing_Time',
    # 11. Earliest Machine Available Time Statistics
    'Mean_Earliest_Machine_Available_Time',
    'Min_Earliest_Machine_Available_Time',
    'Max_Earliest_Machine_Available_Time',
    'Std_Earliest_Machine_Available_Time',
    # 12. Number of Operations statistics
    'Mean_Operations',
    'Min_Operations',
    'Max_Operations',
    'Std_Operations',
    #13. Potential job type number in this work center
    'Potential_Job_Type_Number_in_WC',
    #14. Setup time of the first available machine statistics
    'Mean_Setup_Time',
    'Min_Setup_TIme',
    'Max_Setup_Time',
    'Std_Setup_Time',
    #15. Weights of the rewards
    'Weights_of_time_to_due',
    'Weights_of_Setup_time'
    ]


    # Compute and visualize SHAP values for each agent
    print("\nComputing SHAP values for all agents...")
    for wc_id, agent in rl_agents.items():
        print(f"\nComputing SHAP values for Work Center ID {wc_id}")
        shap_values, test_states = explain_agent_policy(agent, feature_names, wc_id)
        # visualize_shap_values(shap_values, test_states, feature_names, wc_id)
        save_shap_importance_table(shap_values, feature_names, wc_id)
    
# ----------------------------
# Feature understanding part
# ----------------------------

def get_background_data(agent, num_samples=2000):
    # Ensure there are enough samples
    num_samples = min(len(agent.memory), num_samples)
    # Sample random experiences from the agent's memory
    samples = random.sample(agent.memory, num_samples)
    states = [sample[0] for sample in samples]  # Extract states from experiences
    background_data = torch.tensor(states, dtype=torch.float32)
    return background_data

def get_test_states(agent, num_samples=2000):
    # Ensure there are enough samples
    num_samples = min(len(agent.memory), num_samples)
    samples = random.sample(agent.memory, num_samples)
    states = [sample[0] for sample in samples]
    test_states = torch.tensor(states, dtype=torch.float32)
    return test_states

def explain_agent_policy(agent, feature_names, work_center_id, num_background_samples=2000, num_test_samples=100):
    agent.policy_net.eval()  # Set the network to evaluation mode

    # Collect background data
    background_data = get_background_data(agent, num_background_samples)
    background_data = background_data.to(torch.device('cpu'))  # Move to CPU for SHAP

    # Initialize SHAP explainer
    explainer = shap.DeepExplainer(agent.policy_net.to(torch.device('cpu')), background_data)

    # Collect test states
    test_states = get_test_states(agent, num_test_samples)
    test_states = test_states.to(torch.device('cpu'))
    print(f"Test states shape: {test_states.shape}")

    # Compute SHAP values
    shap_values = explainer.shap_values(test_states,check_additivity=False)

    # Return the computed SHAP values and test states
    return shap_values, test_states
    

def visualize_shap_values(shap_values, test_states, feature_names, work_center_id):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import shap

    # shap_values shape: (num_samples, num_features, num_outputs)
    # We need to compute mean absolute SHAP values per feature

    # Compute mean absolute SHAP values over samples and outputs
    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))  # Shape: (num_features,)

    # Verify shapes
    print(f"Feature names length: {len(feature_names)}")
    print(f"Mean_abs_shap_values shape: {mean_abs_shap_values.shape}")

    # Ensure that the lengths match
    assert len(feature_names) == mean_abs_shap_values.shape[0], "Mismatch between number of features and SHAP values"

    # Create a DataFrame to associate features with their mean absolute SHAP values
    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP_Value': mean_abs_shap_values
    })

    # Sort the DataFrame by mean absolute SHAP value in descending order
    shap_df_sorted = shap_df.sort_values(by='Mean_Abs_SHAP_Value', ascending=False)

    # Extract the sorted feature names and SHAP values
    sorted_feature_names = shap_df_sorted['Feature'].values
    sorted_mean_abs_shap_values = shap_df_sorted['Mean_Abs_SHAP_Value'].values

    # Bar plot for feature importance with sorted features
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_feature_names[::-1], sorted_mean_abs_shap_values[::-1], color='skyblue')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for Work Centre ID {work_center_id}')
    plt.tight_layout()
    plt.savefig(f'shap_feature_importance_wc{work_center_id}.png')
    plt.close()
    print(f"Feature importance plot saved for Work Centre ID {work_center_id}")

    # Prepare to reorder features for summary plots
    # Create a list of indices that map from sorted features to original indices
    feature_indices = [feature_names.index(f) for f in sorted_feature_names]

    # Optionally, create a summary plot for each action
    for action_index in range(shap_values.shape[2]):  # Adjusted to use actual number of actions
        shap_values_for_action = shap_values[:, :, action_index]  # Shape: (num_samples, num_features)

        # Reorder shap_values_for_action columns based on sorted_feature_names
        shap_values_for_action_sorted = shap_values_for_action[:, feature_indices]

        # Reorder test_states columns
        if isinstance(test_states, torch.Tensor):
            test_states_np = test_states.numpy()
        else:
            test_states_np = test_states

        test_states_sorted = test_states_np[:, feature_indices]

        shap.summary_plot(
            shap_values_for_action_sorted,
            test_states_sorted,
            feature_names=sorted_feature_names,
            show=False,
            plot_size=(10, 6)
        )
        plt.title(f'SHAP Summary Plot for Action {action_index} at Work Centre {work_center_id}')
        plt.tight_layout()
        plt.savefig(f'shap_summary_action{action_index}_wc{work_center_id}.png')
        plt.close()
        print(f"SHAP summary plot saved for Action {action_index} at Work Centre ID {work_center_id}")

def save_shap_importance_table(shap_values, feature_names, work_center_id, output_dir="shap_results"):
    """
    Save SHAP importance as a table: Feature | Mean_Abs_SHAP_Value
    This matches the format used in 'shap_feature_importance_wcX.csv'
    """
    import os
    import pandas as pd
    import numpy as np
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # shap_values shape: (num_samples, num_features, num_actions)
    # Compute mean absolute SHAP over samples and actions
    mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))  # Shape: (num_features,)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP_Value': mean_abs_shap
    })
    
    # Sort by importance (descending)
    df = df.sort_values('Mean_Abs_SHAP_Value', ascending=False)
    
    # Save to CSV
    filename = os.path.join(output_dir, f'shap_feature_importance_wc{work_center_id}.csv')
    df.to_csv(filename, index=False)
    print(f"Saved SHAP importance for WC{work_center_id} to {filename}")

if __name__ == "__main__":
    num_tests = 100
    num_work_centers = 5
    num_job_types_range = (1, 15)
    mean_interval_range = [5, 15]
    num_machines_per_wc_range = [1, 5]
    total_num_jobs = 100  # or desired number of jobs
    weight_step = 0.02
    test_weights_list = (1.0,0.0)

    # test_trained_agents_with_various_weights(
    #     num_tests,
    #     weight_step,
    #     num_work_centers,
    #     num_job_types_range,
    #     mean_interval_range,
    #     num_machines_per_wc_range,
    #     total_num_jobs
    # )

    test_trained_agents_with_given_weights(
        num_tests,
        test_weights_list,
        num_work_centers,
        num_job_types_range,
        mean_interval_range,
        num_machines_per_wc_range,
        total_num_jobs
    )



    