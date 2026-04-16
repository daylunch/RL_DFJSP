# test_trained_agent.py
# -*- coding: utf-8 -*-
"""
Test Trained PPO Agents Performance (Inference Only)
- Loads the PPO models (ppo_policy_net_wc{wc_id}.pth) for each work center
- Uses multiple random seeds to generate random problem instances
- Runs an event-driven simulation using the same composite-rule logic as main.py
- Collects performance metrics and saves them to Excel

No training is performed; the agent does not store transitions.
We only run the schedule decisions using the learned policy.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import heapq
from collections import deque
from scipy.stats import expon
import logging

# =========== Logging Configuration ============
logging.basicConfig(
    filename='test_trained_agents.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# =========== Global Variables ============
SIMULATION_END_TIME = None
simulation_clock = 0
event_queue = []
completed_jobs = []
work_centers = {}

# We'll accumulate test results here
test_results = {
    'Seed': [],
    'Makespan': [],
    'Average_Flow_Time': [],
    'Average_Tardiness': [],
    'Machine_Utilization': []
}

# =========== Classes for the Continuous Actor-Critic ============

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
        )

        # ---- Actor 分支 (输出动作均值 mu) ----
        self.actor_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_size, action_size)   # 输出维度=action_size
        )
        # 可学习的对数标准差 (可改成每维独立或用一个标量)
        self.log_std = nn.Parameter(torch.full((action_size,), -0.5))

        # ---- Critic 分支 (输出价值) ----
        self.critic_net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        """
        返回 (mu, log_std, value)
        state: [batch, state_size]
        """
        x = self.shared_net(state)
        mu = self.actor_net(x)
        value = self.critic_net(x)
        return mu, self.log_std, value

class PPOInferenceAgentContinuous:
    """
    Inference-only PPO agent for continuous actions (composite rule).
    - Loads the trained model
    - Does not store transitions or update
    """
    def __init__(self, state_dim, action_dim, hidden_size=128):
        self.policy = ActorCriticContinuous(state_dim, action_dim, hidden_size)
        self.policy.eval()

    def load_model(self, path):
        if not os.path.isfile(path):
            print(f"Error: Model file not found: {path}")
            sys.exit(1)
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()
        print(f"Loaded PPO model from {path}")

    def select_action(self, state, deterministic=False):
        """
        Given the state, produce continuous actions w in [-1, 1]^action_dim.
        If deterministic=True, we use mu as action (no sampling).
        Otherwise we do Normal sampling + tanh (like training).
        """
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, log_std, _ = self.policy(state_t)
        std = log_std.exp()

        if deterministic:
            raw_action = mu[0]  # shape=[action_dim]
        else:
            dist = torch.distributions.Normal(mu[0], std)
            raw_action = dist.sample()  # shape=[action_dim]
        action = torch.tanh(raw_action)  # in [-1, 1]
        action = (action-action.mean(dim=-1, keepdim=True))/2

        return action.cpu().numpy()

# =========== Classes for Simulation (same as in main.py) ===========

class Job:
    def __init__(self, job_id, job_type, routing, arrival_time, due_date):
        self.job_id = job_id
        self.job_type = job_type
        self.routing = routing
        self.current_operation = -1
        self.arrival_time = arrival_time
        self.arrival_time_at_current_wc = None
        self.due_date = due_date
        self.completion_time = None

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
        self.accumulative_working_time_since_repair = 0.0
        self.time_to_failure_threshold = np.random.exponential(self.mean_time_between_failures)
        self.total_busy_time_within_simulation = 0
        self.last_repair_time = 0
        self.last_breakdown_time = 0
        self.break_counts = 0

class WorkCenter:
    def __init__(self, work_center_id, machines):
        self.work_center_id = work_center_id
        self.machines = machines
        self.waiting_jobs = deque()

class Event:
    def __init__(self, time, event_type, data):
        self.time = time
        self.event_type = event_type
        self.data = data
    def __lt__(self, other):
        return self.time < other.time

# =========== Utility / Helper Functions ===========

def schedule_event(time, event_type, data, event_queue):
    event = Event(time, event_type, data)
    heapq.heappush(event_queue, event)

def system_empty(work_centers):
    for wc in work_centers.values():
        if wc.waiting_jobs:
            return False
        for m in wc.machines:
            if m.is_busy:
                return False
    return True

def assign_job_to_machine(job, machine, simulation_clock, processing_times, event_queue, work_centers):
    """
    Same as main.py
    """
    current_wc_id = job.routing[job.current_operation]
    pt = processing_times[job.job_type][current_wc_id]['processing_time']
    start_t = max(simulation_clock, machine.available_time)
    remain_fail = machine.time_to_failure_threshold - machine.accumulative_working_time_since_repair

    if pt <= remain_fail:
        end_time = start_t + pt
        machine.is_busy = True
        machine.current_job = job
        machine.available_time = end_time
        schedule_event(end_time, 'operation_complete', {'machine': machine}, event_queue)
    else:
        breakdown_time = start_t + remain_fail
        leftover = pt - remain_fail
        machine.is_busy = True
        machine.current_job = job
        machine.available_time = breakdown_time
        schedule_event(breakdown_time, 'machine_breakdown', {'machine':machine, 'job':job, 'remaining_ptime':leftover}, event_queue)

def assign_jobs_to_idle_machines(work_center, simulation_clock, processing_times, event_queue, work_centers):
    """
    Same as main.py
    """
    while True:
        idle_machine = None
        for m in work_center.machines:
            if (not m.is_busy) and (not m.is_broken):
                idle_machine = m
                break
        if idle_machine and work_center.waiting_jobs:
            job = work_center.waiting_jobs.popleft()
            assign_job_to_machine(job, idle_machine, simulation_clock, processing_times, event_queue, work_centers)
        else:
            break

def total_jobs_in_system(work_centers):
    """
    Same as main.py
    """
    total = 0
    for wc in work_centers.values():
        total += len(wc.waiting_jobs)
        for m in wc.machines:
            if m.is_busy:
                total += 1
    return total

# =========== 你 main.py 中的 get_job_features, extract_state, etc. 保持一致 ===========

def get_job_features(job, wc_id, simulation_clock, processing_times):
    """
    与 main.py 中相同，用于将 job 映射为特征向量(与动作维度对应).
    """
    max_due = 500
    time_to_due = max(0, job.due_date - simulation_clock)/max_due

    max_wait = 500
    if job.arrival_time_at_current_wc is not None:
        wait_time = (simulation_clock - job.arrival_time_at_current_wc)/max_wait
    else:
        wait_time = 0.0

    remaining_ops = float(len(job.routing) - job.current_operation - 1)/20.0

    pt = processing_times[job.job_type][wc_id]['processing_time']
    max_pt = 100.0
    scaled_pt = pt/max_pt

    return np.array([time_to_due, wait_time, remaining_ops, scaled_pt], dtype=np.float32)


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

# =========== Inference decision-making ===========

def decision_making_inference(
    work_center,
    rl_agent,
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
    """
    推理阶段决策函数:
    - 不存储轨迹
    - 与训练时类似地用 action w 对队列排序
    """
    wc_id = work_center.work_center_id
    num_machines_per_wc_current = len(work_centers_machines[wc_id])
    idle_machines = [m for m in work_center.machines if (not m.is_busy and not m.is_broken)]
    if not work_center.waiting_jobs or not idle_machines:
        return  # 若无waiting jobs或无idle machine，则直接return

    # 1) 提取系统状态
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

    # 2) 用 inference agent 选出 w
    # 这里可选 deterministic=True 变成“只用mu”
    action_w = rl_agent.select_action(state, deterministic=True)

    original_jobs = list(work_center.waiting_jobs)
    if len(original_jobs) <= 1:
        return

    # 3) 计算优先级 = dot(w, 作业特征)
    job_score_pairs = []
    for job in original_jobs:
        feat = get_job_features(job, wc_id, simulation_clock, processing_times)
        priority = np.dot(action_w, feat)
        job_score_pairs.append((job, priority))

    job_score_pairs.sort(key=lambda x: x[1], reverse=True)
    work_center.waiting_jobs = deque([p[0] for p in job_score_pairs])
    # 不存储 reward、也不更新

# =========== Event Handlers (只用于推理) ===========

def handle_job_arrival(job, work_centers, processing_times, rl_agents, simulation_clock, event_queue, work_centers_machines):
    next_wc_id = job.next_work_center()
    if next_wc_id is not None:
        job.current_operation += 1
        job.arrival_time_at_current_wc = simulation_clock
        work_centers[next_wc_id].waiting_jobs.append(job)

        decision_making_inference(
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
            event_type='job_arrival'
        )
        assign_jobs_to_idle_machines(work_centers[next_wc_id], simulation_clock, processing_times, event_queue, work_centers)
    else:
        # job finished all ops
        job.completion_time = simulation_clock
        completed_jobs.append(job)

def handle_operation_complete(machine, work_centers, processing_times, rl_agents,
                              simulation_clock, event_queue, completed_jobs, work_centers_machines):
    job = machine.current_job
    if job is None:
        return
    wc_id = machine.work_center_id
    pt = processing_times[job.job_type][wc_id]['processing_time']
    machine.accumulative_working_time_since_repair += pt

    machine.is_busy = False
    machine.current_job = None

    next_wc_id = job.next_work_center()
    if next_wc_id is not None:
        job.current_operation += 1
        job.arrival_time_at_current_wc = simulation_clock
        work_centers[next_wc_id].waiting_jobs.append(job)

        decision_making_inference(
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
            event_type='operation_complete_next_wc'
        )
        assign_jobs_to_idle_machines(work_centers[next_wc_id], simulation_clock, processing_times, event_queue, work_centers)
    else:
        # job done
        job.completion_time = simulation_clock
        completed_jobs.append(job)

    decision_making_inference(
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
        event_type='operation_complete'
    )
    assign_jobs_to_idle_machines(work_centers[wc_id], simulation_clock, processing_times, event_queue, work_centers)

def handle_machine_breakdown(machine, simulation_clock, work_centers, event_queue,
                             rl_agents, processing_times, completed_jobs, work_centers_machines):
    machine.is_broken = True
    machine.is_busy = False
    machine.last_breakdown_time = simulation_clock
    machine.break_counts += 1

    wc_id = machine.work_center_id
    wc = work_centers[wc_id]

    if machine.current_job:
        job = machine.current_job
        wc.waiting_jobs.append(job)
        machine.current_job = None
        # remove any scheduled 'operation_complete' event
        event_queue[:] = [
            e for e in event_queue
            if not (e.event_type=='operation_complete' and e.data['machine']==machine)
        ]
        heapq.heapify(event_queue)

    # schedule machine repair
    repair_time = max(1,int(np.random.exponential(50)))
    repair_finish = simulation_clock + repair_time
    machine.available_time = repair_finish
    schedule_event(repair_finish, 'machine_repaired', {'machine':machine}, event_queue)

    decision_making_inference(
        wc,
        rl_agents[wc_id],
        simulation_clock,
        total_jobs_in_system(work_centers),
        processing_times,
        event_queue,
        work_centers_machines,
        rl_agents,
        completed_jobs,
        work_centers,
        event_type='machine_breakdown'
    )
    assign_jobs_to_idle_machines(wc, simulation_clock, processing_times, event_queue, work_centers)

def handle_machine_repaired(machine, work_centers, simulation_clock, rl_agents,
                            event_queue, processing_times, completed_jobs, work_centers_machines):
    machine.is_broken = False
    machine.last_repair_time = simulation_clock
    machine.accumulative_working_time_since_repair = 0.0
    machine.time_to_failure_threshold = np.random.exponential(machine.mean_time_between_failures)

    wc_id = machine.work_center_id
    wc = work_centers[wc_id]

    decision_making_inference(
        wc,
        rl_agents[wc_id],
        simulation_clock,
        total_jobs_in_system(work_centers),
        processing_times,
        event_queue,
        work_centers_machines,
        rl_agents,
        completed_jobs,
        work_centers,
        event_type='machine_repaired'
    )
    assign_jobs_to_idle_machines(wc, simulation_clock, processing_times, event_queue, work_centers)

# =========== The main "run_simulation_test" ===========

def run_simulation_test(
    rl_agents,
    processing_times,
    event_queue,
    work_centers,
    job_type_routing,
    completed_jobs,
    work_centers_machines,
    total_num_jobs
):
    global simulation_clock
    while len(completed_jobs) < total_num_jobs:
        if event_queue:
            event = heapq.heappop(event_queue)
            simulation_clock = int(event.time)
            
            if event.event_type == 'job_arrival':
                handle_job_arrival(event.data['job'], work_centers, processing_times, rl_agents, simulation_clock, event_queue, work_centers_machines)
            elif event.event_type == 'operation_complete':
                handle_operation_complete(event.data['machine'], work_centers, processing_times, rl_agents, simulation_clock, event_queue, completed_jobs, work_centers_machines)
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
                    for m in wc.machines:
                        if m.is_busy:
                            if next_event_time is None or m.available_time < next_event_time:
                                next_event_time = m.available_time
                if next_event_time is not None:
                    simulation_clock = int(next_event_time)
                    for wc in work_centers.values():
                        for m in wc.machines:
                            if m.is_busy and m.available_time == next_event_time:
                                handle_operation_complete(
                                    m, work_centers, processing_times,
                                    rl_agents, simulation_clock, event_queue,
                                    completed_jobs, work_centers_machines
                                )
                else:
                    for wc in work_centers.values():
                        if wc.waiting_jobs:
                            assign_jobs_to_idle_machines(wc, simulation_clock, processing_times, event_queue, work_centers)
                    if system_empty(work_centers):
                        break

# =========== The main test function ===========

def test_trained_agents(
    num_tests,
    num_work_centers,
    state_size,
    action_size,
    hidden_size,
    total_num_jobs,
    num_job_types_range,
    mean_interval_range,
    num_machines_per_wc_range
):
    """
    Loads each WC's trained model (ppo_policy_net_wc{wc_id}.pth),
    runs multiple random seeds, logs results to test_results_trained_agents.xlsx
    """
    global simulation_clock, event_queue, completed_jobs, work_centers, SIMULATION_END_TIME

    # 1) Create inference agents for each WC
    rl_agents = {}
    for wc_id in range(num_work_centers):
        agent = PPOInferenceAgentContinuous(
            state_dim=state_size,
            action_dim=action_size,
            hidden_size=hidden_size
        )
        model_path = f"ppo_policy_net_wc{wc_id}_ep15000.pth"
        agent.load_model(model_path)
        rl_agents[wc_id] = agent

    # 2) Store results
    episodes_data = {
        'Seed': [],
        'Makespan': [],
        'Average_Flow_Time': [],
        'Average_Tardiness': [],
        'Machine_Utilization': []
    }

    # 3) For each random seed, build environment, run simulation, gather metrics
    for seed in range(1, num_tests+1):
        np.random.seed(seed)
        random.seed(seed)

        # random problem
        num_job_types = random.randint(num_job_types_range[0], num_job_types_range[1])

        # Build job_types
        job_types = {}
        job_type_routing = {}
        for jt in range(1, num_job_types+1):
            ops_count = random.randint(2, num_work_centers)
            routing = random.sample(range(num_work_centers), ops_count)
            operations = {}
            for wc_id in routing:
                pt = random.randint(1, 99)
                operations[wc_id] = {"processing_time": pt}
            job_types[f"Type{jt}"] = {"operations": operations}
            job_type_routing[jt] = routing

        # Machines config
        num_machines_per_wc = {}
        work_centers_machines = {}
        for wc_id in range(num_work_centers):
            nm = random.randint(num_machines_per_wc_range[0], num_machines_per_wc_range[1])
            num_machines_per_wc[wc_id] = nm
            work_centers_machines[wc_id] = [f"WC{wc_id}_M{m+1}" for m in range(nm)]

        # Processing times
        processing_times = {}
        for jt in job_types:
            processing_times[jt] = job_types[jt]["operations"]

        # Arrivals
        job_type_mean_intervals = {}
        for jt in range(1, num_job_types+1):
            mean_iat = random.randint(mean_interval_range[0], mean_interval_range[1])
            job_type_mean_intervals[jt] = mean_iat

        job_type_distributions = {}
        for jt, val in job_type_mean_intervals.items():
            lam = 1.0/val
            job_type_distributions[jt] = expon(scale=1.0/lam)

        # Reset sim
        simulation_clock = 0
        event_queue = []
        heapq.heapify(event_queue)
        completed_jobs.clear()
        work_centers.clear()

        # Initialize work_centers
        wc_temp = {}
        for wc_id in range(num_work_centers):
            machines_list = []
            mtbf = random.randint(500, 1500)
            for m_i in range(num_machines_per_wc[wc_id]):
                machine_id = f"WC{wc_id}_M{m_i+1}"
                machine = Machine(machine_id, wc_id, mtbf)
                machines_list.append(machine)
            wc_temp[wc_id] = WorkCenter(wc_id, machines_list)
        
        work_centers = wc_temp

        # Generate arrivals
        job_arrivals = []
        job_id = 1
        jobs_generated = 0
        while jobs_generated < total_num_jobs:
            for jt in range(1, num_job_types+1):
                if jobs_generated >= total_num_jobs:
                    break
                routing = job_type_routing[jt]
                inter_arr = max(job_type_distributions[jt].rvs(),1)
                if job_arrivals:
                    arrival_time = job_arrivals[-1][0] + inter_arr
                else:
                    arrival_time = inter_arr
                total_work = 0
                for wcid in routing:
                    total_work += job_types[f"Type{jt}"]["operations"][wcid]["processing_time"]
                factor = random.uniform(1,2)
                due_date = int(arrival_time + total_work*factor)
                newjob = Job(job_id, f"Type{jt}", routing, arrival_time, due_date)
                job_arrivals.append((arrival_time, newjob))
                job_id+=1
                jobs_generated+=1

        job_arrivals.sort(key=lambda x:x[0])
        last_arrival_time = job_arrivals[-1][0] if job_arrivals else 0
        SIMULATION_END_TIME = last_arrival_time + 20000

        # schedule arrivals + machine breakdown
        for (at, job) in job_arrivals:
            schedule_event(at, 'job_arrival', {'job': job}, event_queue)
        for wcid, wc in work_centers.items():
            for machine in wc.machines:
                mean_tbf = machine.mean_time_between_failures
                fail_t = np.random.exponential(mean_tbf)
                schedule_event(
                    fail_t,
                    'machine_breakdown',
                    {'machine': machine},
                    event_queue
                )

        # 4) run simulation with the new "run_simulation_test"
        run_simulation_test(
            rl_agents,
            processing_times,
            event_queue,
            work_centers,
            job_type_routing,
            completed_jobs,
            work_centers_machines,
            total_num_jobs
        )

        # 5) gather metrics
        if completed_jobs:
            makespan = max(job.completion_time for job in completed_jobs)
            mean_flow_time = np.mean([job.completion_time - job.arrival_time for job in completed_jobs])
            mean_tardiness = np.mean([max(0, job.completion_time - job.due_date) for job in completed_jobs])
        else:
            makespan=0
            mean_flow_time=0
            mean_tardiness=0

        total_machines = sum(len(wc.machines) for wc in work_centers.values())
        total_machine_time = makespan*total_machines
        busy_time = 0
        for wc in work_centers.values():
            for m in wc.machines:
                busy_time += m.total_busy_time_within_simulation
        if total_machine_time>0:
            machine_util = busy_time/total_machine_time
        else:
            machine_util=0

        episodes_data['Seed'].append(seed)
        episodes_data['Makespan'].append(makespan)
        episodes_data['Average_Flow_Time'].append(mean_flow_time)
        episodes_data['Average_Tardiness'].append(mean_tardiness)
        episodes_data['Machine_Utilization'].append(machine_util)

    # after all seeds, save results
    df = pd.DataFrame(episodes_data)
    df.to_excel('test_results_trained_agents.xlsx', index=False)
    print("Inference test results saved to 'test_results_trained_agents.xlsx'.")

    avg_makespan = df['Makespan'].mean()
    avg_flow = df['Average_Flow_Time'].mean()
    avg_tardiness = df['Average_Tardiness'].mean()
    avg_util = df['Machine_Utilization'].mean()

    print("===== Final Test Results (Average) =====")
    print(f"Avg Makespan: {avg_makespan:.2f}")
    print(f"Avg Flow Time: {avg_flow:.2f}")
    print(f"Avg Tardiness: {avg_tardiness:.2f}")
    print(f"Avg Machine Util: {avg_util:.2f}")

# =========== main entry point ===========

if __name__ == "__main__":
    # example usage
    num_tests = 100
    num_work_centers = 5
    state_size = 53         # must match your main.py
    action_size = 4         # e.g. if you used 4-d composite rule
    hidden_size = 128
    total_num_jobs = 100
    num_job_types_range = (5,20)
    mean_interval_range = [5,15]
    num_machines_per_wc_range = [1,5]

    test_trained_agents(
        num_tests,
        num_work_centers,
        state_size,
        action_size,
        hidden_size,
        total_num_jobs,
        num_job_types_range,
        mean_interval_range,
        num_machines_per_wc_range
    )
