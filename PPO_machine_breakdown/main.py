

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 01:35:22 2025

Machine breakdown with accumulative work time
Using PPO instead of DQN for scheduling decisions.

@author: Administrator
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
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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

        # For debugging / optional logging
        self.assigned_state = None
        self.assigned_action = None

    def next_work_center(self):
        if self.current_operation + 1 < len(self.routing):
            return self.routing[self.current_operation + 1]
        return None

class Machine:
    def __init__(self, machine_id, work_center_id, mean_time_between_failures):
        self.machine_id = machine_id
        self.work_center_id = work_center_id
        self.is_busy = False
        self.is_broken = False
        self.current_job = None
        self.available_time = 0
        self.mean_time_between_failures = mean_time_between_failures
        
        # Track total accumulated work time since last repair
        self.accumulative_working_time_since_repair = 0.0
        
        # Draw a threshold (in total working hours) until the next breakdown
        self.time_to_failure_threshold = np.random.exponential(self.mean_time_between_failures)

        # For utilization tracking
        self.total_busy_time = 0
        self.total_busy_time_within_simulation = 0
        self.assigned_operations = []
        self.last_repair_time = 0
        self.break_counts = 0

class WorkCenter:
    def __init__(self, work_center_id, machines):
        self.work_center_id = work_center_id
        self.machines = machines
        self.waiting_jobs = deque()

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
        self.event_type = event_type  # 'job_arrival', 'operation_complete', 'machine_breakdown', 'machine_repaired'
        self.data = data

    def __lt__(self, other):
        return self.time < other.time

# ----------------------------
# PPO Agent Class
# ----------------------------

class ActorCriticContinuous(nn.Module):
    """
    改为连续动作：对每个维度输出 mu 和一个全局的 log_std（或每维独立）
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        # ---- 共享网络 (common trunk) ----
        self.shared_net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, hidden_size),
            nn.LeakyReLU(0.01),
        ).to(device)

        # ---- Actor 分支 (输出动作均值 mu) ----
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, action_size)   # 输出维度=action_size
        ).to(device)
        # 可学习的对数标准差 (可改成每维独立或用一个标量)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5)).to(device)
        # ---- Critic 分支 (输出价值) ----
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        ).to(device)

    def forward(self, state):
        """
        返回 (mu, log_std, value)
        state: [batch, state_size]
        """
        x = self.shared_net(state)
        mu = self.actor_net(x)
        value = self.critic_net(x)
        return mu, self.log_std, value

class PPOAgentContinuous:
    """
    连续动作版本的PPOAgent
    """
    def __init__(self, state_size, action_size, hidden_size=128,
                 actor_lr=1e-4, critic_lr=1e-4,
                 gamma=0.99, gae_lambda=0.95,
                 epsilon=0.2, K_epochs=5,
                 rollout_capacity=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.K_epochs = K_epochs

        self.policy = ActorCriticContinuous(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)

        # 轨迹缓存
        self.states = []
        self.raw_actions = []  # tanh前的动作
        self.actions = []      # tanh后的动作
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.rollout_capacity = rollout_capacity
        self.cumulative_reward = 0.0
        self.loss_history = []
        self.background_states = deque(maxlen=20000)
        self.logstd_history = []
        self.critic_loss_history = []
        
    def get_log_std(self):
        # 直接访问 policy 里的 log_std
        # 返回 numpy 或 float
        return self.policy.log_std.detach().cpu().numpy()

    def select_action(self, state):
        """
        输出多维连续动作(复合权重 w)，将其约束在[-1,1]，并计算 log_prob 用于训练
        """
        state_t = torch.tensor(state, dtype=torch.float32,device=device).unsqueeze(0)  # shape [1, state_size]
        self.background_states.append(state.copy())
        with torch.no_grad():
            mu, log_std, value = self.policy(state_t)
        std = log_std.exp()     # [action_size]

        # 独立高斯分布(简化)
        dist = Normal(mu[0], std)    # each dimension
        raw_action = dist.sample()   # [-∞, +∞]
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)  # 标量

        # 用tanh限制动作在[-1,1]
        action = torch.tanh(raw_action)
        action = (action-action.mean(dim=-1, keepdim=True))/2

        # 转回numpy
        action_np = action.cpu().numpy()
        raw_action_np = raw_action.cpu().numpy()

        return action_np, raw_action_np, log_prob.item(), value.item()

    def store_transition(self, state, action, raw_action, log_prob, value, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.raw_actions.append(raw_action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

        if len(self.states) >= self.rollout_capacity:
            self.finish_trajectory()

    def finish_trajectory(self):
        if len(self.states) == 0:
            return

        # 转换为tensor
        states_t = torch.tensor(self.states, dtype=torch.float32, device=device)
        raw_actions_t = torch.tensor(self.raw_actions, dtype=torch.float32, device=device)
        log_probs_old_t = torch.tensor(self.log_probs, dtype=torch.float32, device=device).view(-1,1)
        values_t = torch.tensor(self.values, dtype=torch.float32, device=device)
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)

        # 计算GAE优势
        advantages = []
        gae = 0.0
        next_value = 0.0
        for i in reversed(range(len(rewards_np))):
            mask = 1.0 - dones_np[i]
            delta = rewards_np[i] + self.gamma * next_value * mask - values_t[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            next_value = values_t[i]
        advantages = np.array(advantages, dtype=np.float32)
        returns = values_t.numpy() + advantages

        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32).view(-1)

        # 开始多次PPO更新
        for _ in range(self.K_epochs):
            mu, log_std, new_value = self.policy(states_t)
            std = log_std.exp()
            dist = Normal(mu, std)

            new_log_prob = dist.log_prob(raw_actions_t).sum(dim=-1, keepdim=True)  # [N,1]
            ratio = torch.exp(new_log_prob - log_probs_old_t)
            surr1 = ratio * advantages_t.view(-1,1)
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages_t.view(-1,1)
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (new_value.view(-1) - returns_t).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(loss.item())
            self.critic_loss_history.append(critic_loss.item())

        # 清空缓存
        # if len(self.states) >= self.rollout_capacity:
        self.states.clear()
        self.raw_actions.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        current_log_std = self.get_log_std()  # shape=[action_size]
        mean_log_std = np.mean(current_log_std)
        self.logstd_history.append(mean_log_std)

    def maybe_update(self):
        if len(self.states) >= self.rollout_capacity:
            self.finish_trajectory()



# ----------------------------------------------------------------------------
# Simulation / Scheduling Logic
# ----------------------------------------------------------------------------

def initialize_work_centers(num_work_centers, num_machines_per_wc):
    work_centers = {}
    for wc_id in range(num_work_centers):
        num_machines = num_machines_per_wc[wc_id]
        machines = []
        mean_time_between_failures = random.randint(500, 1500)
        for m in range(num_machines):
            # mean_time_between_failures = random.randint(500, 1500)
            machine_id = f"WC{wc_id}_M{m+1}"
            machine = Machine(
                machine_id=machine_id,
                work_center_id=wc_id,
                mean_time_between_failures=mean_time_between_failures
            )
            machines.append(machine)
        wc = WorkCenter(work_center_id=wc_id, machines=machines)
        work_centers[wc_id] = wc
    return work_centers

def job_type_processing_time(job_type, wc_id, processing_times):
    try:
        return processing_times[wc_id]['processing_time']
    except KeyError:
        print(f"Error: Missing processing time for Job Type '{job_type}', Work Center ID {wc_id}.")
        sys.exit(1)

def generate_job_arrivals(num_job_types, job_type_distributions, total_num_jobs, job_type_routing, job_types):
    job_arrivals = []
    job_id = 1
    jobs_generated = 0
    while jobs_generated < total_num_jobs:
        for jt in range(1, num_job_types + 1):
            if jobs_generated >= total_num_jobs:
                break
            routing = job_type_routing[jt]
            # Generate next inter-arrival time
            inter_arrival = max(job_type_distributions[jt].rvs(), 1)  # Ensure at least 1 time unit
            if job_arrivals:
                arrival_time = job_arrivals[-1][0] + inter_arrival
            else:
                arrival_time = inter_arrival

            # Calculate total work content
            total_work_content = sum([
                job_type_processing_time(
                    f"Type{jt}",
                    wc_id,
                    job_types[f"Type{jt}"]['operations']
                )
                for wc_id in routing
            ])
            factor = random.uniform(1, 2)
            due_date = int(arrival_time + total_work_content * factor)
            job = Job(job_id, f"Type{jt}", routing, arrival_time, due_date)
            job_arrivals.append((arrival_time, job))
            
            job_id += 1
            jobs_generated += 1

    job_arrivals.sort(key=lambda x: x[0])
    return job_arrivals

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

def handle_job_arrival(job, work_centers, processing_times, rl_agents, simulation_clock, event_queue, work_centers_machines):
    next_wc_id = job.next_work_center()
    if next_wc_id is not None:
        job.current_operation += 1
        job.arrival_time_at_current_wc = simulation_clock
        work_centers[next_wc_id].waiting_jobs.append(job)

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
            work_centers,
            event_type="job_arrival",   # <--- NEW ARG
        )
        assign_jobs_to_idle_machines(work_centers[next_wc_id], simulation_clock, processing_times, event_queue, work_centers)

def handle_operation_completion(machine, work_centers, processing_times, rl_agents,
                                simulation_clock, event_queue, completed_jobs, work_centers_machines):
    job = machine.current_job
    if job is None:
        logging.warning("Operation completion on a machine with no job.")
        return

    current_wc_id = machine.work_center_id
    processing_time = processing_times[job.job_type][current_wc_id]['processing_time']

    # Since the job completed, add that to total accumulated time
    machine.accumulative_working_time_since_repair += processing_time

    machine.is_busy = False
    machine.current_job = None

    # Continue with normal logic
    next_wc_id = job.next_work_center()
    if next_wc_id is not None:
        job.current_operation += 1
        job.arrival_time_at_current_wc = simulation_clock
        work_centers[next_wc_id].waiting_jobs.append(job)

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
            work_centers,
            event_type="operation_complete_next_wc",  # <--- NEW ARG
        )
        
        assign_jobs_to_idle_machines(
            work_centers[next_wc_id], simulation_clock,
            processing_times, event_queue, work_centers
        )
    else:
        # Final job completion
        job.completion_time = simulation_clock
        completed_jobs.append(job)

    wc_id = machine.work_center_id
    decision_making(
        work_centers[wc_id],
        rl_agents[wc_id],
        simulation_clock,
        total_jobs_in_system(work_centers),
        processing_times,
        event_queue,
        work_centers_machines,
        rl_agents,
        completed_jobs,
        work_centers,
        event_type="operation_complete",  # <--- NEW ARG
    )
    assign_jobs_to_idle_machines(work_centers[wc_id], simulation_clock, processing_times, event_queue, work_centers)

def handle_machine_breakdown(machine, simulation_clock, work_centers, event_queue, rl_agents, processing_times, completed_jobs, work_centers_machines):
    machine.is_broken = True
    machine.is_busy = False
    machine.last_breakdown_time = simulation_clock
    machine.break_counts += 1

    current_wc_id = machine.work_center_id
    work_center = work_centers[current_wc_id]
    
    if machine.current_job:
        interrupted_job = machine.current_job
        work_center.waiting_jobs.append(interrupted_job)
        machine.current_job = None
        # Remove any scheduled operation_complete event for this machine
        event_queue[:] = [
            event for event in event_queue
            if not (
                event.event_type == 'operation_complete'
                and event.data['machine'] == machine
            )
        ]
        heapq.heapify(event_queue)

    # Schedule machine repair event
    repair_duration = max(1, int(np.random.exponential(50)))
    repair_completion_time = simulation_clock + repair_duration
    machine.available_time = repair_completion_time
    schedule_event(
        repair_completion_time, 'machine_repaired',
        {'machine': machine}, event_queue
    )
    
    # Trigger immediate decision-making after breakdown
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
        work_centers,
        event_type="machine_breakdown",  # <--- NEW ARG
    )
    assign_jobs_to_idle_machines(work_center, simulation_clock, processing_times, event_queue, work_centers)

def handle_machine_repaired(machine, work_centers, simulation_clock, rl_agents, event_queue, processing_times, completed_jobs, work_centers_machines):
    machine.is_broken = False
    machine.last_repair_time = simulation_clock

    # Reset accumulative working time and draw a new threshold
    machine.accumulative_working_time_since_repair = 0.0
    machine.time_to_failure_threshold = np.random.exponential(machine.mean_time_between_failures)

    current_wc_id = machine.work_center_id
    work_center = work_centers[current_wc_id]
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
        work_centers,
        event_type="machine_repaired",
    )
    assign_jobs_to_idle_machines(work_center, simulation_clock, processing_times, event_queue, work_centers)

def assign_jobs_to_idle_machines(work_center, simulation_clock, processing_times, event_queue, work_centers):
    current_wc_id = work_center.work_center_id
    for machine in work_center.machines:
        if not machine.is_busy and not machine.is_broken and work_center.waiting_jobs:
            job = work_center.waiting_jobs.popleft()
            assign_job_to_machine(job, machine, simulation_clock, processing_times, event_queue, work_centers)

def total_jobs_in_system(work_centers):
    total = 0
    for wc in work_centers.values():
        total += len(wc.waiting_jobs)
        for machine in wc.machines:
            if machine.is_busy:
                total += 1
    return total

def extract_state(
    work_centers, 
    simulation_clock, 
    total_jobs,
    simulation_end_time,
    current_wc_id,
    num_machines_per_wc_current,
    processing_times,
    event_type  # <--- new argument with default
):
    """
    Same features as your original code. 
    Make sure the dimension matches the agent's state_size.
    """
    current_work_center = work_centers[current_wc_id]
    state = []
    num_waiting_jobs = len(current_work_center.waiting_jobs)

    # 1) Number of Machines in the WC
    MAX_MACHINES_PER_WC = 5
    state.append(num_machines_per_wc_current / MAX_MACHINES_PER_WC)

    # 2) Number of Job Types in system (we won't track exact, just an upper bound)
    MAX_JOB_TYPES = 10
    # Typically you might pass this in, but here we infer from processing_times
    num_job_types = len(processing_times)
    state.append(num_job_types / MAX_JOB_TYPES)

    # 3) Number of waiting jobs
    MAX_WAITING_JOBS = 100
    state.append(num_waiting_jobs / MAX_WAITING_JOBS)

    # 4) Time to due date statistics
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

    # 5) Waiting time statistics
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

    # 6) Time since release into system
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

    # 7) Remaining operations statistics
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

    # 8) Processing time statistics at this WC
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

    # 9) Remaining total processing time
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

    # # 10) Earliest machine available time statistics
    # available_times = []
    # for machine in current_work_center.machines:
    #     if not machine.is_busy and not machine.is_broken:
    #         available_times.append(0.0)
    #     else:
    #         available_time = machine.available_time - simulation_clock
    #         available_times.append(available_time)
    
    # if available_times:
    #     mean_available_time = np.mean(available_times)
    #     min_available_time = np.min(available_times)
    #     max_available_time = np.max(available_times)
    #     std_available_time = np.std(available_times)
    # else:
    #     mean_available_time = min_available_time = max_available_time = std_available_time = 0.0

    # state.extend([
    #     mean_available_time / simulation_end_time,
    #     min_available_time / simulation_end_time,
    #     max_available_time / simulation_end_time,
    #     std_available_time / simulation_end_time
    # ])

    # 11) Number of operations in each job
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

    # 12) Potential job type number in this WC
    if num_waiting_jobs > 0:
        unique_job_types = set(job.job_type for job in current_work_center.waiting_jobs)
        potential_job_type_number = len(unique_job_types)
    else:
        potential_job_type_number = 0
    MAX_POTENTIAL_JOB_TYPES = 10
    state.append(potential_job_type_number / MAX_POTENTIAL_JOB_TYPES)

    # 13) Number of broken machines in current WC
    num_broken_machines = sum(machine.is_broken for machine in current_work_center.machines)
    state.append(num_broken_machines)

    # 14) Time since last repair (mean, min, max, std)
    time_since_last_repair = []
    for machine in current_work_center.machines:
        last_repair = machine.last_repair_time if hasattr(machine, 'last_repair_time') else 0
        interval_since_repair = simulation_clock - last_repair
        time_since_last_repair.append(interval_since_repair)

    mean_time_since_last_repair = np.mean(time_since_last_repair)
    min_time_since_last_repair = np.min(time_since_last_repair)
    max_time_since_last_repair = np.max(time_since_last_repair)
    std_time_since_last_repair = np.std(time_since_last_repair)

    MAX_TIME_SINCE_LAST_REPAIR = 1000
    state.extend([
        mean_time_since_last_repair / MAX_TIME_SINCE_LAST_REPAIR,
        min_time_since_last_repair / MAX_TIME_SINCE_LAST_REPAIR,
        max_time_since_last_repair / MAX_TIME_SINCE_LAST_REPAIR,
        std_time_since_last_repair / MAX_TIME_SINCE_LAST_REPAIR
    ])

    # 15) Accumulative working time since last repair
    accumulative_working_time_since_repair = [
        machine.accumulative_working_time_since_repair
        for machine in current_work_center.machines
    ]
    mean_ac_working_time = np.mean(accumulative_working_time_since_repair)
    min_ac_working_time = np.min(accumulative_working_time_since_repair)
    max_ac_working_time = np.max(accumulative_working_time_since_repair)
    std_ac_working_time = np.std(accumulative_working_time_since_repair)

    MAX_AC_WORKING_TIME = 1000
    state.extend([
        mean_ac_working_time / MAX_AC_WORKING_TIME,
        min_ac_working_time / MAX_AC_WORKING_TIME,
        max_ac_working_time / MAX_AC_WORKING_TIME,
        std_ac_working_time / MAX_AC_WORKING_TIME
    ])
    
    # 16) Time has been repaired
    time_been_repaired = []
    for machine in current_work_center.machines:
        if machine.is_broken:
            time_been_repaired.append(simulation_clock-machine.last_breakdown_time)
        else:
            time_been_repaired.append(0)
    mean_repaired_time = np.mean(time_been_repaired)
    min_repaired_time = np.min(time_been_repaired)
    max_repaired_time = np.max(time_been_repaired)
    std_repaired_time = np.std(time_been_repaired)

    MAX_REPAIRED_TIME = 50
    state.extend([
        mean_repaired_time / MAX_REPAIRED_TIME,
        min_repaired_time / MAX_REPAIRED_TIME,
        max_repaired_time / MAX_REPAIRED_TIME,
        std_repaired_time / MAX_REPAIRED_TIME
    ])
    
    
    #
    event_one_hot = [0.0, 0.0, 0.0, 0.0, 0.0]
    if event_type == "job_arrival":
        event_one_hot[0] = 1.0
    elif event_type == "operation_complete_next_wc":
        event_one_hot[1] = 1.0
    elif event_type == "machine_breakdown":
        event_one_hot[2] = 1.0
    elif event_type == "machine_repaired":
        event_one_hot[3] = 1.0
    elif event_type == "operation_complete":
        event_one_hot[1] = 1.0
    
    # Add these 4 values to the state vector
    state.extend(event_one_hot) 
    # 16. waiting jobs of all other work centers
    other_wcs_waiting_jobs = 0
    for wcid, wc in work_centers.items():
        if wcid != current_wc_id:
            other_wcs_waiting_jobs += len(wc.waiting_jobs)

    # You can pick a normalization factor however you prefer.
    # For instance, if you think there could be up to 1000 waiting jobs total:
    MAX_WAITING_JOBS_SYSTEM = 100
    state.append(other_wcs_waiting_jobs / MAX_WAITING_JOBS_SYSTEM)

    # ------------------------------------------------------------------
    # 17. Number of available machines in *other* work centers
    # ------------------------------------------------------------------
    other_wcs_available_machines = 0
    for wcid, wc in work_centers.items():
        if wcid != current_wc_id:
            for machine in wc.machines:
                if not machine.is_busy and not machine.is_broken:
                    other_wcs_available_machines += 1

    # Suppose total of 25 machines is typical across all WCs as an upper bound:
    MAX_TOTAL_MACHINES_SYSTEM = 25
    state.append(other_wcs_available_machines / MAX_TOTAL_MACHINES_SYSTEM)
    # ------------------------------------------------------------------
    # 18. Number of available machines in *other* work centers
    # ------------------------------------------------------------------
    break_counts = 0
    for machine in current_work_center.machines:
        break_counts += machine.break_counts
    state.append(break_counts/20)

    return np.array(state)


def apply_machine_priority(machines, rule, wc_id):
    if rule == 'Least_Utilized':
        sorted_machines = sorted(machines, key=lambda m: m.available_time)
    else:
        sorted_machines = machines.copy()
    return sorted_machines

def assign_job_to_machine(job, machine, simulation_clock, processing_times, event_queue, work_centers):
    current_wc_id = job.routing[job.current_operation]
    processing_time = processing_times[job.job_type][current_wc_id]['processing_time']
    
    # The machine can’t start until both the simulation clock and its own availability
    operation_start_time = max(simulation_clock, machine.available_time)
    
    # How many working-hours remain before the next failure
    remaining_until_failure = machine.time_to_failure_threshold - machine.accumulative_working_time_since_repair

    if processing_time <= remaining_until_failure:
        # Machine won't fail during this job
        operation_end_time = operation_start_time + processing_time
        machine.is_busy = True
        machine.current_job = job
        machine.available_time = operation_end_time
        schedule_event(
            operation_end_time,
            'operation_complete',
            {'machine': machine, 'job': job},
            event_queue
        )
    else:
        # The machine will break mid-job
        breakdown_time = operation_start_time + remaining_until_failure
        machine.is_busy = True
        machine.current_job = job
        machine.available_time = breakdown_time
        leftover = processing_time - remaining_until_failure
        schedule_event(
            breakdown_time,
            'machine_breakdown',
            {
                'machine': machine,
                'job': job,
                'remaining_ptime': leftover
            },
            event_queue
        )

def calculate_reward(work_center, simulation_clock, processing_times, original_waiting_jobs, sorted_waiting_jobs):
    """
    This function calculates a reward by comparing 'avg time to due' 
    before and after reordering. You can adapt the logic if desired.
    """
    def average_time_to_due(jobs_sequence):
        expected_completion_times = []
        machine_available_times = [machine.available_time for machine in work_center.machines]
        for job in jobs_sequence:
            earliest_machine_idx = machine_available_times.index(min(machine_available_times))
            machine_available_time = machine_available_times[earliest_machine_idx]
            start_time = max(simulation_clock, machine_available_time)
            pt = processing_times[job.job_type][work_center.work_center_id]['processing_time']
            completion_time = start_time + pt
            expected_completion_times.append(completion_time)
            machine_available_times[earliest_machine_idx] = completion_time

        # time to due date with penalty for tardiness
        time_to_due_list = []
        k = 10
        for job, ect in zip(jobs_sequence, expected_completion_times):
            if ect <= job.due_date:
                time_to_due = job.due_date - ect
            else:
                time_to_due = k * (job.due_date - ect)
            time_to_due_list.append(time_to_due)
        return np.mean(time_to_due_list) if time_to_due_list else 0.0

    avg_time_to_due_before = average_time_to_due(original_waiting_jobs)
    avg_time_to_due_after = average_time_to_due(sorted_waiting_jobs)
    # Reward = difference in time-to-due
    reward = avg_time_to_due_after - avg_time_to_due_before
    return reward

def get_job_features(job, wc_id, simulation_clock, processing_times):
    """
    可自行设计特征长度与 PPO 的 action_size 相对应
    例如这里假设 action_size=4
    """
    # 1) Time to due
    max_due = 500.0
    time_to_due = max(0, job.due_date - simulation_clock) / max_due

    # 2) 已等待时间
    max_wait = 500.0
    if job.arrival_time_at_current_wc is not None:
        wait_time = (simulation_clock - job.arrival_time_at_current_wc) / max_wait
    else:
        wait_time = 0.0

    # 3) 剩余工序数
    remaining_ops = float(len(job.routing) - job.current_operation - 1) / 20.0

    # 4) 在本WC的加工时间
    pt = processing_times[job.job_type][wc_id]['processing_time']
    max_pt = 100.0
    scaled_pt = pt / max_pt

    # 返回向量, 与action维度保持一致
    return np.array([time_to_due, wait_time, remaining_ops, scaled_pt], dtype=np.float32)


def decision_making(
    work_center,        # 当前工作中心
    rl_agent,           # PPOAgentContinuous
    simulation_clock,
    total_jobs,
    processing_times,
    event_queue,
    work_centers_machines,
    rl_agents,
    completed_jobs,
    work_centers,
    event_type
):
    wc_id = work_center.work_center_id
    num_machines_per_wc_current = len(work_centers_machines[wc_id])
    idle_machines = [m for m in work_center.machines if (not m.is_busy and not m.is_broken)]
    if not work_center.waiting_jobs or not idle_machines:
        return  # 若无waiting jobs或无idle machine，则直接return
    
    # 1) 提取系统状态(跟原extract_state相同,只要返回一个state向量即可)
    state = extract_state(
        work_centers,
        simulation_clock,
        total_jobs,
        SIMULATION_END_TIME,
        wc_id,
        num_machines_per_wc_current,
        processing_times,
        event_type
    )

    # 2) 从连续版PPO拿到一组权重(动作) w ∈ [-1,1]^action_size
    #    raw_action仅用于后面计算log_prob
    action, raw_action, log_prob, value = rl_agent.select_action(state)
    # action.shape = [action_size], e.g. 4维

    original_jobs = list(work_center.waiting_jobs)
    if len(original_jobs) <= 1:
        return  # 无需排序

    # 3) 计算每个作业的特征 + 与weights做点积
    job_score_pairs = []
    for job in original_jobs:
        feat = get_job_features(job, wc_id, simulation_clock, processing_times)
        priority = np.dot(action, feat)  # composite rule
        job_score_pairs.append((job, priority))

    # 4) 按priority从大到小排序
    job_score_pairs.sort(key=lambda x:x[1], reverse=True)
    new_jobs = [p[0] for p in job_score_pairs]
    work_center.waiting_jobs = deque(new_jobs)

    # 5) 计算奖励(可与原先 “队列重排前后某指标差” 类似)
    reward = calculate_reward(work_center, simulation_clock, processing_times,
                              original_jobs, list(work_center.waiting_jobs))
    rl_agent.cumulative_reward += reward

    # 6) 存储轨迹
    done = False
    rl_agent.store_transition(
        state,
        action,
        raw_action,
        log_prob,
        value,
        reward,
        None,    # next_state先留None
        done
    )
    rl_agent.maybe_update()

def run_simulation(rl_agents, processing_times, event_queue, work_centers, job_type_routing, completed_jobs, work_centers_machines, total_num_jobs):
    global simulation_clock
    while len(completed_jobs) < total_num_jobs:
        if event_queue:
            event = heapq.heappop(event_queue)
            simulation_clock = int(event.time)
            
            if event.event_type == 'job_arrival':
                handle_job_arrival(event.data['job'], work_centers, processing_times, rl_agents, simulation_clock, event_queue, work_centers_machines)
            elif event.event_type == 'operation_complete':
                handle_operation_completion(event.data['machine'], work_centers, processing_times, rl_agents, simulation_clock, event_queue, completed_jobs, work_centers_machines)
            elif event.event_type == 'machine_breakdown':
                handle_machine_breakdown(event.data['machine'], simulation_clock, work_centers, event_queue, rl_agents, processing_times, completed_jobs, work_centers_machines)
            elif event.event_type == 'machine_repaired':
                handle_machine_repaired(event.data['machine'], work_centers, simulation_clock, rl_agents, event_queue, processing_times, completed_jobs, work_centers_machines)
        else:
            if system_empty(work_centers):
                break
            else:
                next_event_time = None
                for wc in work_centers.values():
                    for machine in wc.machines:
                        if machine.is_busy:
                            if next_event_time is None or machine.available_time < next_event_time:
                                next_event_time = machine.available_time
                if next_event_time is not None:
                    simulation_clock = int(next_event_time)
                    for wc in work_centers.values():
                        for machine in wc.machines:
                            if machine.is_busy and machine.available_time == next_event_time:
                                handle_operation_completion(
                                    machine, work_centers, processing_times,
                                    rl_agents, simulation_clock, event_queue,
                                    completed_jobs, work_centers_machines
                                )
                else:
                    for wc in work_centers.values():
                        if wc.waiting_jobs:
                            assign_jobs_to_idle_machines(wc, simulation_clock, processing_times, event_queue, work_centers)
                    if system_empty(work_centers):
                        break

    # End of the simulation => store any leftover transitions
    for wc_id, agent in rl_agents.items():
        agent.finish_trajectory()

def log_metrics(episode, completed_jobs, work_centers, rl_agents):
    total_jobs = len(completed_jobs)

    makespan = max(job.completion_time for job in completed_jobs) if completed_jobs else 0
    mean_flow_time = np.mean([job.completion_time - job.arrival_time for job in completed_jobs]) if completed_jobs else 0
    mean_tardiness = np.mean([max(0, job.completion_time - job.due_date) for job in completed_jobs]) if completed_jobs else 0

    # utilization
    total_machines = sum(len(wc.machines) for wc in work_centers.values())
    total_machine_time = makespan * total_machines
    total_busy_time = sum(m.total_busy_time_within_simulation for wc in work_centers.values() for m in wc.machines)
    machine_utilization = total_busy_time / total_machine_time if total_machine_time > 0 else 0

    # For each agent, gather average loss from PPO
    average_loss_per_agent = {}
    average_critic_loss_per_agent = {}
    for wc_id, agent in rl_agents.items():
        if agent.loss_history:
            average_loss = np.mean(agent.loss_history)
            average_loss_per_agent[wc_id] = average_loss
            agent.loss_history = []
        else:
            average_loss_per_agent[wc_id] = 0.0
        if agent.logstd_history:
            latest_logstd = agent.logstd_history[-1]  # 这一轮结束时的均值
            print(f"  WC {wc_id}: Mean Log Std = {latest_logstd:.4f}")
        
        if agent.critic_loss_history:
            avg_critic_loss = np.mean(agent.critic_loss_history)
            average_critic_loss_per_agent[wc_id] = avg_critic_loss
            agent.critic_loss_history = []
        else:
            average_critic_loss_per_agent[wc_id] = 0.0

    metrics['Episode'].append(episode)
    metrics['Makespan'].append(makespan)
    metrics['Average_Flow_Time'].append(mean_flow_time)
    metrics['Average_Tardiness'].append(mean_tardiness)
    metrics['Machine_Utilization'].append(machine_utilization)
    metrics['Average_Loss'].append(average_loss_per_agent)
    metrics['Average_Critic_Loss'].append(average_critic_loss_per_agent)

    logging.info(f"=== Metrics - Episode {episode} ===")
    logging.info(f"Makespan: {makespan}")
    logging.info(f"Average Flow Time: {mean_flow_time:.2f}")
    logging.info(f"Average Tardiness: {mean_tardiness:.2f}")
    logging.info(f"Machine Utilization: {machine_utilization:.2f}")
    for wc_id, avg_loss in average_loss_per_agent.items():
        logging.info(f"Average Loss for WC {wc_id}: {avg_loss:.4f}")
    print(f"\n=== Metrics - Episode {episode} ===")
    print(f"Makespan: {makespan}")
    print(f"Average Flow Time: {mean_flow_time:.2f}")
    print(f"Average Tardiness: {mean_tardiness:.2f}")
    print(f"Machine Utilization: {machine_utilization:.2f}")
    for wc_id, avg_loss in average_loss_per_agent.items():
        # print(f"Average Loss for Work Centre {wc_id}: {avg_loss:.4f}")
        print(f"Average Critic Loss for  Work Centre {wc_id}: {average_critic_loss_per_agent[wc_id]:.4f}")
    print("===============================\n")

def plot_metrics(metrics):
    import matplotlib
    matplotlib.use('Agg')

    # Makespan
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['Episode'], metrics['Makespan'], label='Makespan')
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    plt.title('Makespan per Episode')
    plt.legend()
    plt.tight_layout()
    plt.savefig('makespan_per_episode.png')
    plt.close()

    # Average Flow Time
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['Episode'], metrics['Average_Flow_Time'], label='Average Flow Time')
    plt.xlabel('Episode')
    plt.ylabel('Average Flow Time')
    plt.title('Average Flow Time per Episode')
    plt.legend()
    plt.tight_layout()
    plt.savefig('average_flow_time_per_episode.png')
    plt.close()

    # Average Tardiness
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['Episode'], metrics['Average_Tardiness'], label='Average Tardiness')
    plt.xlabel('Episode')
    plt.ylabel('Average Tardiness')
    plt.title('Average Tardiness per Episode')
    plt.legend()
    plt.tight_layout()
    plt.savefig('average_tardiness_per_episode.png')
    plt.close()

    # Machine Utilization
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['Episode'], metrics['Machine_Utilization'], label='Machine Utilization')
    plt.xlabel('Episode')
    plt.ylabel('Utilization')
    plt.title('Machine Utilization per Episode')
    plt.legend()
    plt.tight_layout()
    plt.savefig('machine_utilization_per_episode.png')
    plt.close()

    print("Metrics plots have been saved as PNG files.")

def plot_cumulative_rewards(cumulative_rewards_per_agent):
    global_rewards = []
    for wc_id, rewards in cumulative_rewards_per_agent.items():
        global_rewards += rewards
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rewards) + 1), rewards, label=f'Work Center {wc_id}')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Cumulative Reward per Episode for Work Center {wc_id}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'cumulative_reward_wc{wc_id}.png')
        plt.close()
        print(f"Cumulative reward plot saved for Work Center {wc_id} as 'cumulative_reward.png'")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(global_rewards) + 1), rewards, label='Global')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Global Reward')
    plt.title('Cumulative Global Reward per Episode')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cumulative_global_reward.png')
    plt.close()
    print("Cumulative global reward plot saved as 'cumulative_global_reward.png'")

def reset_simulation(num_work_centers, num_job_types, num_machines_per_wc, processing_times,
                     job_type_distributions, job_type_routing, total_num_jobs, job_types):
    global simulation_clock, event_queue, work_centers, completed_jobs, rl_agents, SIMULATION_END_TIME
    simulation_clock = 0
    event_queue = []
    heapq.heapify(event_queue)
    completed_jobs = []
    
    work_centers = initialize_work_centers(num_work_centers, num_machines_per_wc)
    
    job_arrivals = generate_job_arrivals(num_job_types, job_type_distributions, total_num_jobs, job_type_routing, job_types)
    
    last_arrival_time = job_arrivals[-1][0]
    SIMULATION_END_TIME = last_arrival_time + 20000  # Buffer

    for arrival_time, job in job_arrivals:
        schedule_event(arrival_time, 'job_arrival', {'job': job}, event_queue)

    for wc in work_centers.values():
        for machine in wc.machines:
            schedule_machine_breakdown(machine, simulation_clock, event_queue)
    logging.info(f"Simulation Reset: {num_work_centers} WCs, {num_job_types} job types, {len(job_arrivals)} job arrivals.")


def train_rl_agent(num_episodes, total_num_jobs, rl_agents, update_target_every, mean_interval_range, num_machines_per_wc_range):
    global work_centers_machines
    cumulative_rewards_per_agent = {wc_id: [] for wc_id in rl_agents.keys()}
    num_work_centers = 5  # fixed example
    normalized_tardiness_log = []

    for episode in range(1, num_episodes + 1):
        
        for agent in rl_agents.values():
            agent.cumulative_reward = 0.0

        # Randomly select parameters
        min_num_job_types = 5
        max_num_job_types = 20
        num_job_types = random.randint(min_num_job_types, max_num_job_types)

        # Create job_types and job_type_routing
        job_types = {}
        job_type_routing = {}
        for jt in range(1, num_job_types + 1):
            num_operations = random.randint(2, num_work_centers)
            routing = random.sample(range(num_work_centers), num_operations)
            operations = {}
            for wc_id in routing:
                pt = random.randint(1, 99)
                operations[wc_id] = {'processing_time': pt}
            job_types[f'Type{jt}'] = {'operations': operations}
            job_type_routing[jt] = routing

        num_machines_per_wc = {}
        work_centers_machines = {}
        for wc_id in range(num_work_centers):
            num_machines = random.randint(num_machines_per_wc_range[0], num_machines_per_wc_range[1])
            num_machines_per_wc[wc_id] = num_machines
            work_centers_machines[wc_id] = [f"WC{wc_id}_M{m+1}" for m in range(num_machines)]

        processing_times = {jt: job_types[jt]['operations'] for jt in job_types}

        job_type_mean_intervals = {
            jt: random.randint(mean_interval_range[0], mean_interval_range[1])
            for jt in range(1, num_job_types + 1)
        }
        job_type_lambda = {jt: 1 / mean_interval for jt, mean_interval in job_type_mean_intervals.items()}
        job_type_distributions = {jt: expon(scale=1 / lambda_) for jt, lambda_ in job_type_lambda.items()}

        logging.info(f"=== Starting Episode {episode} ===")
        print(f"\n=== Starting Episode {episode} ===")
        print(f"Number of Work Centers: {num_work_centers}")
        print(f"Number of Job Types: {num_job_types}")
        print(f"Job Type Mean Intervals: {job_type_mean_intervals}")

        for wc_id in sorted(num_machines_per_wc.keys()):
            logging.info(f"  WC {wc_id}: {num_machines_per_wc[wc_id]} machines")
            print(f"  WC {wc_id}: {num_machines_per_wc[wc_id]} machines")

        reset_simulation(
            num_work_centers,
            num_job_types,
            num_machines_per_wc,
            processing_times,
            job_type_distributions,
            job_type_routing,
            total_num_jobs,
            job_types
        )

        # Run simulation
        run_simulation(
            rl_agents, processing_times, event_queue,
            work_centers, job_type_routing, completed_jobs,
            work_centers_machines, total_num_jobs
        )
        
    
        # After entire simulation, call PPO update (finish_and_learn) for each agent
        for wc_id, agent in rl_agents.items():
            agent.finish_trajectory() # The core PPO update
            cumulative_rewards_per_agent[wc_id].append(agent.cumulative_reward)
        
        # --- 新增代码：保存每个 Work Center 的 Cumulative Reward 到 CSV ---
        # 构造一个字典，包含 episode 编号和每个 WC 的累计奖励
        reward_record = {'Episode': episode}
        for wc_id, agent in rl_agents.items():
            reward_record[f'WC_{wc_id}_Cumulative_Reward'] = agent.cumulative_reward

        # 转换为 DataFrame
        reward_df = pd.DataFrame([reward_record])

        # 定义 CSV 文件名
        csv_filename = 'work_center_cumulative_rewards.csv'

        # 如果是第一个 episode，则写入表头；否则追加数据
        if episode == 1:
            reward_df.to_csv(csv_filename, index=False, mode='w')
        else:
            reward_df.to_csv(csv_filename, index=False, header=False, mode='a')
    # --- 新增代码结束 ---

        # No "update_target_network" for PPO, so we just log
        log_metrics(episode, completed_jobs, work_centers, rl_agents)
        logging.info(f"=== Episode {episode} Completed ===")
        print(f"=== Episode {episode} Completed ===")
        if episode % 100 == 0:
            for wc_id, agent in rl_agents.items():
                checkpoint_path = f'ppo_policy_net_wc{wc_id}_ep{episode}.pth'
                torch.save(agent.policy.state_dict(), checkpoint_path)

    # Plotting or additional analysis
    # Plot losses if desired, but we mostly log them in log_metrics
    for wc_id, rewards in cumulative_rewards_per_agent.items():
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(rewards) + 1), rewards, label=f'WC {wc_id}')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Cumulative Reward (Work Center {wc_id})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'cumulative_reward_wc{wc_id}.png')
        plt.close()
        print(f"Cumulative reward plot saved for WC {wc_id}.")
    for wc_id, agent in rl_agents.items():
        plt.plot(agent.logstd_history, label=f'WC{wc_id}')
    plt.xlabel("Update times or episodes")
    plt.ylabel("Mean Log Std")
    plt.title("Evolution of log_std over training")
    plt.legend()
    plt.savefig('Log_Std.png')
    plt.close()
    for wc_id, agent in rl_agents.items():
        logstd_history = agent.logstd_history
        filename = f'logstd_history_wc{wc_id}.csv'
        df_logstd = pd.DataFrame({'logstd': logstd_history})
        df_logstd.to_csv(filename, index=False)
        print(f"logstd history saved for Work Center {wc_id} to {filename}")

# ----------------------------
# Feature explanation routines
# (Optional, if needed)
# ----------------------------

def get_background_data(agent, num_samples=2000):
    """Randomly sample from the circular buffer"""
    available_samples = len(agent.background_states)
    if available_samples == 0:
        raise ValueError("No background states collected. Run simulation first.")
        
    num_samples = min(available_samples, num_samples)
    idxs = random.sample(range(available_samples), num_samples)
    states = [agent.background_states[i] for i in idxs]
    return torch.tensor(states, dtype=torch.float32)

def get_test_states(agent, num_samples=2000):
    """Get latest states for explanation"""
    num_samples = min(len(agent.background_states), num_samples)
    states = list(agent.background_states)[-num_samples:]
    return torch.tensor(states, dtype=torch.float32)

def explain_agent_policy(agent, feature_names, work_center_id, 
                        num_background_samples=2000, num_test_samples=100):
    class ModelWrapper(nn.Module):
        def __init__(self, policy_net):
            super().__init__()
            self.policy_net = policy_net
            
        def forward(self, x):
            # policy_net(x) => (mu, log_std, value)
            mu, log_std, value = self.policy_net(x)
            # 只返回 mu
            return mu

    # 下面保持不变
    background_data = get_background_data(agent, num_background_samples)
    wrapped_model = ModelWrapper(agent.policy).to('cpu')
    explainer = shap.DeepExplainer(wrapped_model, background_data)
    test_states = get_test_states(agent, num_test_samples)
    shap_values = explainer.shap_values(test_states)
    return shap_values, test_states

def visualize_shap_values(shap_values, test_states, feature_names, work_center_id):
    import numpy as np
    import pandas as pd
    import shap

    mean_abs_shap_values = np.mean(np.abs(shap_values), axis=(0, 2))

    shap_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP_Value': mean_abs_shap_values
    })
    shap_df_sorted = shap_df.sort_values(by='Mean_Abs_SHAP_Value', ascending=False)
    sorted_feature_names = shap_df_sorted['Feature'].values
    sorted_mean_abs_shap_values = shap_df_sorted['Mean_Abs_SHAP_Value'].values

    plt.figure(figsize=(10, 8))
    plt.barh(sorted_feature_names[::-1], sorted_mean_abs_shap_values[::-1], color='skyblue')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for WC {work_center_id}')
    plt.tight_layout()
    plt.savefig(f'shap_feature_importance_wc{work_center_id}.png')
    plt.close()

    feature_indices = [feature_names.index(f) for f in sorted_feature_names]
    for action_index in range(shap_values.shape[2]):
        shap_values_for_action = shap_values[:, :, action_index]
        shap_values_for_action_sorted = shap_values_for_action[:, feature_indices]

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
        plt.title(f'SHAP Summary Plot for Action {action_index} (WC {work_center_id})')
        plt.tight_layout()
        plt.savefig(f'shap_summary_action{action_index}_wc{work_center_id}.png')
        plt.close()

# ----------------------------
# Global definitions
# ----------------------------

metrics = {
    'Episode': [],
    'Makespan': [],
    'Average_Flow_Time': [],
    'Average_Tardiness': [],
    'Machine_Utilization': [],
    'Average_Loss': [],
    'Average_Critic_Loss': []
}

if __name__ == "__main__":
    # random.seed(0)
    # np.random.seed(0)

    # Training parameters
    num_episodes = 20000  # total episodes
    total_num_jobs = 100  # jobs per simulation
    update_target_every = 100  # not used for PPO, but kept as is to remain consistent
    mean_interval_range = [5, 15]
    num_machines_per_wc_range = [1, 5]

    num_work_centers = 5
    action_size = 4

    # Our final state_size must match the number of features in extract_state
    # Checking carefully, we have 45 features in the original code example.
    state_size = 53

    # Create PPO agents for each work center
    rl_agents = {}
    for wc_id in range(num_work_centers):
        rl_agents[wc_id] = PPOAgentContinuous(
            state_size=state_size,
            action_size=action_size,
            hidden_size=128,
            actor_lr=3e-5,
            critic_lr=1e-4,
            gamma=0.99,
            gae_lambda=0.95,
            epsilon=0.2,
            K_epochs=5,
            rollout_capacity=500  # You can adjust if you want mid-episode updates
        )
    # rl_agents = {}
    # for wc_id in range(num_work_centers):
    #     agent = PPOAgentContinuous(
    #             state_size=state_size,
    #             action_size=action_size,
    #             hidden_size=128,
    #             actor_lr=1e-4,
    #             critic_lr=1e-4,
    #             gamma=0.99,
    #             gae_lambda=0.95,
    #             epsilon=0.2,
    #             K_epochs=5,
    #             rollout_capacity=500  # You can adjust if you want mid-episode updates
    #         )
    #     model_path = f"ppo_policy_net_wc{wc_id}_ep20000.pth"
    #     agent.policy.load_state_dict(torch.load(model_path))
    #     agent.policy.eval()
    #     rl_agents[wc_id] = agent

    simulation_clock = 0
    event_queue = []
    heapq.heapify(event_queue)
    completed_jobs = []

    start_time = time.time()
    train_rl_agent(
        num_episodes,
        total_num_jobs,
        rl_agents,
        update_target_every,
        mean_interval_range,
        num_machines_per_wc_range
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training finished. Elapsed Time: {elapsed_time:.2f} seconds")

    plot_metrics(metrics)

    # Save trained models
    for wc_id, agent in rl_agents.items():
        torch.save(agent.policy.state_dict(), f'ppo_policy_net_wc{wc_id}.pth')
        logging.info(f"Model for Work Center ID {wc_id} saved.")
        print(f"Model for Work Center ID {wc_id} saved.")

    logging.info("PPO training completed and models saved.")
    print("PPO training completed and models saved.")

    # If you wish to compute SHAP values (optional), define feature_names accordingly:
    feature_names = [
    # 1) Number of Machines in WC
    'Num_Machines_in_WC',
    # 2) Number of Job Types in system
    'Num_Job_Types_in_System',
    # 3) Number of waiting jobs
    'Num_Waiting_Jobs_in_WC',
    # 4) Time to due date stats
    'Mean_Time_to_Due_Date', 'Min_Time_to_Due_Date', 'Max_Time_to_Due_Date', 'Std_Time_to_Due_Date',
    # 5) Waiting time stats
    'Mean_Waiting_Time', 'Min_Waiting_Time', 'Max_Waiting_Time', 'Std_Waiting_Time',
    # 6) Time since release stats
    'Mean_Time_Since_Release', 'Min_Time_Since_Release', 'Max_Time_Since_Release', 'Std_Time_Since_Release',
    # 7) Remaining operations stats
    'Mean_Remaining_Operations', 'Min_Remaining_Operations', 'Max_Remaining_Operations', 'Std_Remaining_Operations',
    # 8) Processing time stats at WC
    'Mean_Processing_Time_in_WC', 'Min_Processing_Time_in_WC', 'Max_Processing_Time_in_WC', 'Std_Processing_Time_in_WC',
    # 9) Remaining total processing time stats
    'Mean_Remaining_Processing_Time', 'Min_Remaining_Processing_Time', 'Max_Remaining_Processing_Time', 'Std_Remaining_Processing_Time',
    # # 10) Machine available time stats
    # 'Mean_Earliest_Machine_Available_Time', 'Min_Earliest_Machine_Available_Time', 'Max_Earliest_Machine_Available_Time', 'Std_Earliest_Machine_Available_Time',
    # 11) Number of operations stats
    'Mean_Operations', 'Min_Operations', 'Max_Operations', 'Std_Operations',
    # 12) Potential job type number
    'Potential_Job_Type_Number_in_WC',
    # 13) Number of broken machines
    'Num_Broken_Machines',
    # 14) Time since last repair stats
    'Mean_Time_Since_Last_Repair', 'Min_Time_Since_Last_Repair', 'Max_Time_Since_Last_Repair', 'Std_Time_Since_Last_Repair',
    # 15) Accumulative working time stats
    'Mean_AC_Working_Time', 'Min_AC_Working_Time', 'Max_AC_Working_Time', 'Std_AC_Working_Time',
    # 16) Time been repaired stats
    'Mean_Repaired_Time', 'Min_Repaired_Time', 'Max_Repaired_Time', 'Std_Repaired_Time',
    # Event type one-hot encoding
    'Event_Job_Arrival', 'Event_Operation_Complete_Next_WC', 'Event_Machine_Breakdown', 'Event_Machine_Repaired','Event_Operation_Complete'
    'Jobs_of_other_WC','Machines_of_other_WC',
    'Break_counts'
]

    # Example of how to compute SHAP (if needed)
 
    print("\nComputing SHAP values for all agents (optional demonstration)...")
    for wc_id, agent in rl_agents.items():
        print(f"SHAP for Work Center {wc_id} ...")
        shap_values, test_states = explain_agent_policy(agent, feature_names, wc_id)
        visualize_shap_values(shap_values, test_states, feature_names, wc_id)

