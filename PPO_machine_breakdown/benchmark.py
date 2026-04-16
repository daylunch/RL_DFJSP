# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 20:12:55 2025

@author: Administrator
"""

# benchmark.py
# -*- coding: utf-8 -*-
"""
Benchmark Rule Testing Code

- Tests all candidate dispatching (benchmark) rules (job priority and machine priority)
- Uses the same event-driven simulation structure and classes seen in test_trained_agent.py,
  but replaces RL-based decision-making with fixed job/machine priority rules.
- After running multiple episodes (with different random seeds), records performance metrics
  per (job_rule, seed) into an Excel file: test_results_benchmark.xlsx
"""

import random
import numpy as np
import heapq
import pandas as pd
import sys
import logging
from collections import deque
from scipy.stats import expon

# ----------------------------
# Logging Configuration
# ----------------------------
logging.basicConfig(
    filename='benchmark_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# ----------------------------
# Global Variables
# ----------------------------
SIMULATION_END_TIME = None
simulation_clock = 0
event_queue = []
completed_jobs = []
work_centers = {}

# ----------------------------------------------------------------------------
# Classes (matching structure from test_trained_agent.py)
# ----------------------------------------------------------------------------

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
        # Accumulative working time since last repair
        self.accumulative_working_time_since_repair = 0.0
        # Threshold for next breakdown (exponential)
        self.time_to_failure_threshold = np.random.exponential(self.mean_time_between_failures)
        # For tracking machine utilization
        self.total_busy_time_within_simulation = 0
        self.last_repair_time = 0
        self.last_breakdown_time = 0


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


# ----------------------------------------------------------------------------
# System / Event Utilities
# ----------------------------------------------------------------------------

def system_empty(work_centers):
    """Check if no jobs are waiting and no machines are busy."""
    for wc in work_centers.values():
        if wc.waiting_jobs:
            return False
        for m in wc.machines:
            if m.is_busy:
                return False
    return True


def schedule_event(time, event_type, data, event_queue):
    event = Event(time, event_type, data)
    heapq.heappush(event_queue, event)


def schedule_machine_breakdown(machine, current_time, event_queue):
    """Sample a breakdown time for the machine and schedule the event."""
    time_until_failure = np.random.exponential(machine.mean_time_between_failures)
    failure_time = current_time + time_until_failure
    schedule_event(
        failure_time,
        'machine_breakdown',
        {'machine': machine},
        event_queue
    )


def total_jobs_in_system(work_centers):
    total = 0
    for wc in work_centers.values():
        total += len(wc.waiting_jobs)
        for m in wc.machines:
            if m.is_busy:
                total += 1
    return total


# ----------------------------------------------------------------------------
# Priority / Dispatching (Benchmark Rules)
# ----------------------------------------------------------------------------

def apply_job_priority(jobs, job_rule, processing_times, current_wc_id):
    """
    Sort the waiting jobs in a work center queue by the specified job_rule.
    """
    valid_jobs = list(jobs)  # copy
    try:
        if job_rule == 'FIFO':  # First In, First Out
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.arrival_time_at_current_wc)
        elif job_rule == 'LIFO':  # Last In, First Out
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.arrival_time_at_current_wc, reverse=True)
        elif job_rule == 'EDD':  # Earliest Due Date
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.due_date)
        elif job_rule == 'LDD':  # Latest Due Date
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.due_date, reverse=True)
        elif job_rule == 'SPT':  # Shortest Processing Time
            sorted_jobs = sorted(valid_jobs, key=lambda job: processing_times[job.job_type][current_wc_id]['processing_time'])
        elif job_rule == 'LPT':
            sorted_jobs = sorted(valid_jobs, key=lambda job: processing_times[job.job_type][current_wc_id]['processing_time'], reverse=True)
        elif job_rule == 'LOR':  # Minimal (remaining) Operation count
            sorted_jobs = sorted(valid_jobs, key=lambda job: len(job.routing) - (job.current_operation + 1))
        elif job_rule == 'ERD':  # Earliest Release Date
            sorted_jobs = sorted(valid_jobs, key=lambda job: job.arrival_time)
        else:
            # If not recognized or 'NAN', do no reordering
            sorted_jobs = valid_jobs
    except KeyError:
        # If something is missing in processing_times, fallback
        sorted_jobs = valid_jobs
    return deque(sorted_jobs)


def apply_machine_priority(machines, machine_rule):
    """
    If you want to reorder machines themselves. By default, we have only 'Least_Utilized',
    which sorts by available_time. If you had multiple machine rules, you'd pick them here.
    """
    if machine_rule == 'Least_Utilized':
        return sorted(machines, key=lambda m: m.available_time)
    else:
        return machines.copy()


def assign_job_to_machine(job, machine, simulation_clock, processing_times, event_queue):
    """Assign the given job to the specified machine, scheduling its completion or breakdown."""
    wc_id = machine.work_center_id
    processing_time = processing_times[job.job_type][wc_id]['processing_time']

    # The machine can’t start until both the clock and its own availability
    operation_start_time = max(simulation_clock, machine.available_time)

    # How many working-hours remain before next failure
    remain_until_failure = machine.time_to_failure_threshold - machine.accumulative_working_time_since_repair

    if processing_time <= remain_until_failure:
        # The machine won't fail during this operation
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
        breakdown_time = operation_start_time + remain_until_failure
        leftover = processing_time - remain_until_failure

        machine.is_busy = True
        machine.current_job = job
        machine.available_time = breakdown_time

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


def assign_jobs_to_idle_machines(work_center, simulation_clock, processing_times, event_queue, job_rule, machine_rule):
    """
    Assign waiting jobs (already sorted by job_rule) to any idle machines,
    optionally also sorting the machines by machine_rule (if needed).
    """
    # Sort the machines according to the machine_rule:
    sorted_machines = apply_machine_priority(work_center.machines, machine_rule)

    for machine in sorted_machines:
        if not machine.is_busy and not machine.is_broken and work_center.waiting_jobs:
            # Pop the first job from the waiting queue (already sorted by job_rule)
            job = work_center.waiting_jobs.popleft()
            assign_job_to_machine(job, machine, simulation_clock, processing_times, event_queue)


# ----------------------------------------------------------------------------
# Event Handlers
# ----------------------------------------------------------------------------

def handle_job_arrival(job, work_centers, processing_times, simulation_clock, event_queue, job_rule, machine_rule):
    """When a job arrives to the next work center in its routing."""
    next_wc_id = job.next_work_center()
    if next_wc_id is not None:
        job.current_operation += 1
        job.arrival_time_at_current_wc = simulation_clock
        work_centers[next_wc_id].waiting_jobs.append(job)

        # Reorder the waiting queue by job_rule
        sorted_jobs = apply_job_priority(
            work_centers[next_wc_id].waiting_jobs,
            job_rule,
            processing_times,
            next_wc_id
        )
        work_centers[next_wc_id].waiting_jobs = sorted_jobs

        # Assign jobs to idle machines
        assign_jobs_to_idle_machines(
            work_centers[next_wc_id],
            simulation_clock,
            processing_times,
            event_queue,
            job_rule,
            machine_rule
        )


def handle_operation_completion(machine, work_centers, processing_times, simulation_clock, event_queue, completed_jobs, job_rule, machine_rule):
    job = machine.current_job
    if job is None:
        logging.warning("Operation completion on a machine that has no job.")
        return

    wc_id = machine.work_center_id
    proc_time = processing_times[job.job_type][wc_id]['processing_time']

    # The job finished, so add that time to machine's accumulative usage
    machine.accumulative_working_time_since_repair += proc_time
    machine.is_busy = False
    machine.current_job = None

    next_wc_id = job.next_work_center()
    if next_wc_id is not None:
        # Move job to next work center
        job.current_operation += 1
        job.arrival_time_at_current_wc = simulation_clock
        work_centers[next_wc_id].waiting_jobs.append(job)

        # Re-sort queue in that center
        sorted_jobs = apply_job_priority(
            work_centers[next_wc_id].waiting_jobs,
            job_rule,
            processing_times,
            next_wc_id
        )
        work_centers[next_wc_id].waiting_jobs = sorted_jobs

        # Attempt assignment
        assign_jobs_to_idle_machines(
            work_centers[next_wc_id],
            simulation_clock,
            processing_times,
            event_queue,
            job_rule,
            machine_rule
        )
    else:
        # Final completion
        job.completion_time = simulation_clock
        completed_jobs.append(job)

    # Also see if there's any waiting job in the current center (wc_id) that can run now
    assign_jobs_to_idle_machines(
        work_centers[wc_id],
        simulation_clock,
        processing_times,
        event_queue,
        job_rule,
        machine_rule
    )


def handle_machine_breakdown(machine, work_centers, simulation_clock, event_queue, processing_times, completed_jobs, job_rule, machine_rule):
    machine.is_broken = True
    machine.is_busy = False
    machine.last_breakdown_time = simulation_clock

    wc_id = machine.work_center_id
    work_center = work_centers[wc_id]

    if machine.current_job:
        # Put the interrupted job back in the waiting queue
        interrupted_job = machine.current_job
        machine.current_job = None
        work_center.waiting_jobs.append(interrupted_job)

        # Remove any operation_complete events for this machine
        event_queue[:] = [
            e for e in event_queue
            if not (e.event_type == 'operation_complete' and e.data['machine'] == machine)
        ]
        heapq.heapify(event_queue)

    # Schedule the machine repair
    repair_duration = max(1, int(np.random.exponential(50))) # e.g., mean=50
    repair_finish = simulation_clock + repair_duration
    machine.available_time = repair_finish
    schedule_event(
        repair_finish,
        'machine_repaired',
        {'machine': machine},
        event_queue
    )

    # Re-sort the waiting queue at this center and attempt assignment to other machines
    sorted_jobs = apply_job_priority(work_center.waiting_jobs, job_rule, processing_times, wc_id)
    work_center.waiting_jobs = sorted_jobs
    assign_jobs_to_idle_machines(work_center, simulation_clock, processing_times, event_queue, job_rule, machine_rule)


def handle_machine_repaired(machine, work_centers, simulation_clock, event_queue, processing_times, completed_jobs, job_rule, machine_rule):
    machine.is_broken = False
    machine.last_repair_time = simulation_clock

    # Reset accumulative usage and draw a new threshold
    machine.accumulative_working_time_since_repair = 0.0
    machine.time_to_failure_threshold = np.random.exponential(machine.mean_time_between_failures)

    wc_id = machine.work_center_id
    work_center = work_centers[wc_id]

    # Re-sort the queue & try to assign
    sorted_jobs = apply_job_priority(work_center.waiting_jobs, job_rule, processing_times, wc_id)
    work_center.waiting_jobs = sorted_jobs
    assign_jobs_to_idle_machines(work_center, simulation_clock, processing_times, event_queue, job_rule, machine_rule)


# ----------------------------------------------------------------------------
# Simulation Loop
# ----------------------------------------------------------------------------

def run_simulation(
    processing_times,
    event_queue,
    work_centers,
    completed_jobs,
    total_num_jobs,
    job_rule,
    machine_rule
):
    global simulation_clock
    while len(completed_jobs) < total_num_jobs:
        if event_queue:
            event = heapq.heappop(event_queue)
            simulation_clock = int(event.time)

            if event.event_type == 'job_arrival':
                handle_job_arrival(
                    event.data['job'],
                    work_centers,
                    processing_times,
                    simulation_clock,
                    event_queue,
                    job_rule,
                    machine_rule
                )
            elif event.event_type == 'operation_complete':
                handle_operation_completion(
                    event.data['machine'],
                    work_centers,
                    processing_times,
                    simulation_clock,
                    event_queue,
                    completed_jobs,
                    job_rule,
                    machine_rule
                )
            elif event.event_type == 'machine_breakdown':
                handle_machine_breakdown(
                    event.data['machine'],
                    work_centers,
                    simulation_clock,
                    event_queue,
                    processing_times,
                    completed_jobs,
                    job_rule,
                    machine_rule
                )
            elif event.event_type == 'machine_repaired':
                handle_machine_repaired(
                    event.data['machine'],
                    work_centers,
                    simulation_clock,
                    event_queue,
                    processing_times,
                    completed_jobs,
                    job_rule,
                    machine_rule
                )
        else:
            # If no events are left but jobs remain in the system:
            if system_empty(work_centers):
                break
            else:
                # Advance time to the next machine availability if possible
                next_time = None
                for wc in work_centers.values():
                    for m in wc.machines:
                        if m.is_busy:
                            if next_time is None or m.available_time < next_time:
                                next_time = m.available_time
                if next_time is not None:
                    simulation_clock = int(next_time)
                    # Trigger operation completion for any machines that become free at this time
                    for wc in work_centers.values():
                        for m in wc.machines:
                            if m.is_busy and m.available_time == next_time:
                                handle_operation_completion(
                                    m,
                                    work_centers,
                                    processing_times,
                                    simulation_clock,
                                    event_queue,
                                    completed_jobs,
                                    job_rule,
                                    machine_rule
                                )
                else:
                    # Try to assign idle machines if any job is waiting
                    for wc in work_centers.values():
                        if wc.waiting_jobs:
                            assign_jobs_to_idle_machines(
                                wc,
                                simulation_clock,
                                processing_times,
                                event_queue,
                                job_rule,
                                machine_rule
                            )
                    if system_empty(work_centers):
                        break


# ----------------------------------------------------------------------------
# Setup + Benchmark Main
# ----------------------------------------------------------------------------

def initialize_work_centers(num_work_centers, num_machines_per_wc):
    """Create WorkCenter objects with machines, each machine gets random mean_time_between_failures."""
    wcs = {}
    for wc_id in range(num_work_centers):
        machines_list = []
        mtbf = random.randint(500, 1500)
        for i in range(num_machines_per_wc[wc_id]):
            # mtbf = random.randint(500, 1500)
            machine_id = f"WC{wc_id}_M{i+1}"
            machine = Machine(machine_id, wc_id, mtbf)
            machines_list.append(machine)
        wcs[wc_id] = WorkCenter(wc_id, machines_list)
    return wcs


def generate_job_arrivals(num_job_types, job_type_distributions, total_num_jobs, job_type_routing, job_types):
    """Same approach as in test_trained_agent.py."""
    job_arrivals = []
    job_id = 1
    jobs_generated = 0
    while jobs_generated < total_num_jobs:
        for jt in range(1, num_job_types + 1):
            if jobs_generated >= total_num_jobs:
                break
            routing = job_type_routing[jt]
            inter_arrival = max(job_type_distributions[jt].rvs(), 1)  # ensure at least 1
            if job_arrivals:
                arrival_time = job_arrivals[-1][0] + inter_arrival
            else:
                arrival_time = inter_arrival

            # Compute total work
            total_work = 0
            for wc_id in routing:
                total_work += job_types[f"Type{jt}"]['operations'][wc_id]['processing_time']

            factor = random.uniform(1, 2)
            due_date = int(arrival_time + total_work * factor)

            job = Job(job_id, f"Type{jt}", routing, arrival_time, due_date)
            job_arrivals.append((arrival_time, job))

            job_id += 1
            jobs_generated += 1

    job_arrivals.sort(key=lambda x: x[0])
    return job_arrivals


def collect_metrics(completed_jobs, work_centers):
    """Compute makespan, mean flow time, tardiness, machine utilization."""
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
    total_busy_time = sum(m.total_busy_time_within_simulation for wc in work_centers.values() for m in wc.machines)
    machine_utilization = (total_busy_time / total_machine_time) if total_machine_time > 0 else 0

    return makespan, mean_flow_time, mean_tardiness, machine_utilization


def benchmark_test_all_rules(
    num_tests,
    total_num_jobs,
    num_work_centers,
    num_job_types_range,
    mean_interval_range,
    num_machines_per_wc_range
):
    """
    Try multiple benchmark job rules (and a single machine rule for now).
    For each (job_rule, seed), generate a random scenario and measure performance.
    """
    # You can adjust which job rules to test
    job_priority_rules = [
        'FIFO', 'LIFO', 'EDD', 'LDD', 'SPT', 'LPT', 'LOR', 'ERD'
    ]
    machine_priority_rule = 'Least_Utilized'

    results = {
        'Seed': [],
        'Job_Rule': [],
        'Machine_Rule': [],
        'Makespan': [],
        'Average_Flow_Time': [],
        'Average_Tardiness': [],
        'Machine_Utilization': []
    }

    for job_rule in job_priority_rules:
        for seed in range(1, num_tests + 1):
            random.seed(seed)
            np.random.seed(seed)

            num_job_types = random.randint(num_job_types_range[0], num_job_types_range[1])

            # Build random job types, routing, and processing times
            job_types = {}
            job_type_routing = {}
            for jt in range(1, num_job_types + 1):
                route_length = random.randint(2, num_work_centers)
                route = random.sample(range(num_work_centers), route_length)
                operations = {}
                for wc_id in route:
                    pt = random.randint(1, 99)
                    operations[wc_id] = {'processing_time': pt}
                job_type_name = f"Type{jt}"
                job_types[job_type_name] = {'operations': operations}
                job_type_routing[jt] = route

            # Random number of machines per WC
            num_machines_per_wc = {}
            for wc_id in range(num_work_centers):
                nm = random.randint(num_machines_per_wc_range[0], num_machines_per_wc_range[1])
                num_machines_per_wc[wc_id] = nm

            # processing_times = { 'Type1': {wc_id -> {'processing_time': value}}, ... }
            processing_times = {jt: job_types[jt]['operations'] for jt in job_types}

            # Build job arrival distributions
            job_type_mean_intervals = {
                jt: random.randint(mean_interval_range[0], mean_interval_range[1])
                for jt in range(1, num_job_types + 1)
            }
            job_type_lambda = {jt: 1.0 / val for jt, val in job_type_mean_intervals.items()}
            job_type_distributions = {jt: expon(scale=1.0 / lam) for jt, lam in job_type_lambda.items()}

            # Reset global simulation references
            global simulation_clock, event_queue, completed_jobs, work_centers, SIMULATION_END_TIME
            simulation_clock = 0
            event_queue = []
            heapq.heapify(event_queue)
            completed_jobs = []

            # Initialize the work centers and their machines
            work_centers = initialize_work_centers(num_work_centers, num_machines_per_wc)

            # Generate job arrivals
            job_arrivals = generate_job_arrivals(num_job_types, job_type_distributions, total_num_jobs, job_type_routing, job_types)
            if job_arrivals:
                last_arrival_time = job_arrivals[-1][0]
            else:
                last_arrival_time = 0
            SIMULATION_END_TIME = last_arrival_time + 20000  # buffer

            # Schedule arrivals
            for arrival_time, job in job_arrivals:
                schedule_event(arrival_time, 'job_arrival', {'job': job}, event_queue)

            # Schedule breakdown for each machine
            for wc in work_centers.values():
                for machine in wc.machines:
                    schedule_machine_breakdown(machine, simulation_clock, event_queue)

            # Run the simulation with the chosen benchmark rules
            run_simulation(
                processing_times=processing_times,
                event_queue=event_queue,
                work_centers=work_centers,
                completed_jobs=completed_jobs,
                total_num_jobs=total_num_jobs,
                job_rule=job_rule,
                machine_rule=machine_priority_rule
            )

            # Collect results
            makespan, avg_flow, avg_tardiness, utilization = collect_metrics(completed_jobs, work_centers)

            results['Seed'].append(seed)
            results['Job_Rule'].append(job_rule)
            results['Machine_Rule'].append(machine_priority_rule)
            results['Makespan'].append(makespan)
            results['Average_Flow_Time'].append(avg_flow)
            results['Average_Tardiness'].append(avg_tardiness)
            results['Machine_Utilization'].append(utilization)

    # Save to Excel
    df = pd.DataFrame(results)
    df.to_excel('test_results_benchmark.xlsx', index=False)
    print("Benchmark test results saved to 'test_results_benchmark.xlsx'.")

    # Print final averages by job_rule
    print("=== Final Averages by Job Rule ===")
    for job_rule in job_priority_rules:
        df_rule = df[df['Job_Rule'] == job_rule]
        avg_makespan = df_rule['Makespan'].mean()
        avg_flow_time = df_rule['Average_Flow_Time'].mean()
        avg_tardiness = df_rule['Average_Tardiness'].mean()
        avg_utilization = df_rule['Machine_Utilization'].mean()

        print(f"\nJob Rule: {job_rule}")
        print(f"Average Makespan: {avg_makespan:.2f}")
        print(f"Average Flow Time: {avg_flow_time:.2f}")
        print(f"Average Tardiness: {avg_tardiness:.2f}")
        print(f"Average Machine Utilization: {avg_utilization:.2f}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example usage
    num_tests = 100
    total_num_jobs = 100
    num_work_centers = 5
    num_job_types_range = (5, 20)
    mean_interval_range = [5, 15]
    num_machines_per_wc_range = [1, 5]

    benchmark_test_all_rules(
        num_tests,
        total_num_jobs,
        num_work_centers,
        num_job_types_range,
        mean_interval_range,
        num_machines_per_wc_range
    )
