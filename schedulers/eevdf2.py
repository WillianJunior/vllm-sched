# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum

# import os
# import random
# import time
# from collections import deque
# from dataclasses import dataclass, field
from typing import Callable, Optional
from random import seed, randint, uniform
import joblib
import numpy as np

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.sequence import SequenceGroup

from prioritySchedBase import Scheduler


class EEVDF(Scheduler):
    """EEVDF. Based on Linux scheduler.
    Orders tasks based on earliest vdeadline first.
    If there is a free slot for a task, swap-in the waiting task
    with lowest vdeadline first, which is eligible.

    A task is eligible if it has lag>=0. lag is the amount of time
    a task is owed. Owed time is calculated as the ideal partition of
    time among all tasks. If there are 100 tasks, each step owes a
    task 1/100 units.

    Tasks define their chosen timeslice. A timeslice is originally
    non-preemptive. I.e., if a task is running, it will run for the
    entirely of its timeslice. Tasks with smaller timeslices run
    frequently, but for smaller periods (latency-sensitive). Tasks with
    larger timeslices run for longer uninterruptedly, but less frequently
    (throughput-intensive).

    Tasks can be preempted, but only by new tasks, not waiting tasks.
    If a new task arrive it is naturally eligible. Its vdeadline is checked
    against running tasks, from smallest remaining time to the largest
    remaining time. Remaining time here is the timeslice - cur_runtime.
    This heuristic should diminish swapping of long timeslice tasks.

    TODO: Test all cases for victims: (i) shortest remaining time first,
    (ii) longest remaining time first, (iii) just longest vdeadline first.
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        super(EEVDF, self).__init__(
            scheduler_config,
            cache_config,
            lora_config,
            pipeline_parallel_size,
            output_proc_callback,
        )

        # Load estimator model
        self.estimator = joblib.load(
            "/sonic_home/willianjunior/vllm-segment/git/vllm-sched/llm-len-regression/models/random-forest-model-335.pkl.qrf"
        )

        # Just a different random with extra steps
        # Also didn't work ...
        # data = [gen_val for _, _, gen_val in self.oracle.values()]
        # kde = gaussian_kde(data)
        # for key in self.oracle.keys():
        #    self.oracle[key][OracleFields.GENERATE.value] = max(int(kde.resample(1)), 1)

        # === EEVDF stuff... ==================================================
        # [Will]: Monkey patching SequenceGroup to have virtual runtimes.
        # expected_time_slice represents how much time is required to process
        # the sequence. This value can change with the progress of the
        # execution.
        self._base_expected_time_slice = 10
        if self.using_oracle:
            self._base_expected_time_slice = -1

        setattr(SequenceGroup, "total_vtime", 0)
        setattr(SequenceGroup, "cur_vtime", 0)
        setattr(
            SequenceGroup, "expected_time_slice", self._base_expected_time_slice
        )
        setattr(SequenceGroup, "priority", 1)
        setattr(SequenceGroup, "lag", 0)
        setattr(SequenceGroup, "vdeadline", 0)

        setattr(SequenceGroup, "slice_increment", 10)

        # Number of tokens (or time spent) per sched step by each seq
        # If using multi-step scheduling, it would be more than 1
        # per max_num_seqs.
        self.sched_slice = 1
        self.max_num_seqs = self.scheduler_config.max_num_seqs

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        """Overwritten from base class."""
        # Add sequence groups to the waiting queue.
        self.new_seqs.append(seq_group)

    def _total_queue_size(self):
        return len(self.running) + len(self.waiting)

    def _update_running_priority(self, seq_group):
        seq_group.cur_vtime += self.sched_slice
        ideal_slice = self.sched_slice / self._total_queue_size()

        # lag is how much time it used (sched_slice) minus how much
        # time it was owed (ideal_slice). reducing lag means running
        seq_group.lag -= self.sched_slice - ideal_slice
        if seq_group.lag < 0:
            seq_group.vdeadline = float("inf")
        else:
            seq_group.vdeadline = seq_group.lag + seq_group.timeslice
        seq_group.total_vtime += self.sched_slice

    def _update_waiting_priority(self, seq_group):
        # didn't run last time, thus accumulating lag
        ideal_slice = self.sched_slice / self._total_queue_size()
        seq_group.lag += ideal_slice
        if seq_group.lag < 0:
            seq_group.vdeadline = float("inf")
        else:
            seq_group.vdeadline = seq_group.lag + seq_group.expected_time_slice

    def _can_preempt(self, seq_group):
        if seq_group.first_seq.status != SequenceStatus.RUNNING:
            return True
        else:
            return seq_group.cur_vtime > seq_group.timeslice

    def _should_preempt(self, victim, sub):
        # return self._can_preempt(victim) and sub.vdeadline < victim.vdeadline
        return sub.vdeadline < victim.vdeadline

    def _added_sequence_to_running(self, seq_group):
        # Current seq_group is beginning its execution
        # seq_group.timeslice = ?
        seq_group.cur_vtime = 0

    # def _added_sequence_to_running(self, seq_group):
    #     # just executed some tokens and still didn't finished
    #     # if seq_group.total_vtime >= seq_group.expected_time_slice:
    #     # Initial heuristic: double the expected time
    #     # if the current time was not enough.
    #     # TODO: try fibonacci?

    #     # Expected time wasn't enough, add more time
    #     if seq_group.expected_time_slice <= seq_group.total_vtime:
    #         # seq_group.expected_time_slice *= 2
    #         seq_group.expected_time_slice += seq_group.slice_increment
    #         seq_group.slice_increment *= 4
    #     seq_group.cur_vtime = 0
    #     # seq_group.num_steps += 1

    def priority(self, seq_group):
        return seq_group.vdeadline

    def print_seq(self, seq_group):
        return (
            f"lag={seq_group.lag:.2f} - vdeadline={seq_group.vdeadline:.2f} "
            f"- {seq_group.cur_vtime}/{seq_group.total_vtime}/{seq_group.expected_time_slice}"
        )
