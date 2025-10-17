# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# import enum
# import os
# import random
# import time
# from collections import deque
# from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.sequence import SequenceGroup

from prioritySchedBase import Scheduler


class EEVDF(Scheduler):
    """docstring for EEVDF"""

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

        # === EEVDF stuff... ==================================================
        # [Will]: Monkey patching SequenceGroup to have virtual runtimes.
        # expected_time_slice represents how much time is required to process 
        # the sequence. This value can change with the progress of the 
        # execution.
        setattr(SequenceGroup, "total_vtime", 0)
        setattr(SequenceGroup, "expected_time_slice", 10)
        setattr(SequenceGroup, "priority", 1)
        setattr(SequenceGroup, "lag", 0)
        setattr(SequenceGroup, "vdeadline", 0)

        # Number of tokens (or time spent) per sched step by each seq
        # If using multi-step scheduling, it would be more than 1
        # per max_num_seqs.
        self.sched_slice = 1
        self.max_num_seqs = self.scheduler_config.max_num_seqs

    def _update_queue_size(self, n):
        self.queue_size = n

        self.ideal_slice = (
            self.max_num_seqs * self.sched_slice
        ) / self.queue_size

    def _update_finished_priority(self, seq_group):
        pass

    def _update_running_priority(self, seq_group):
        # lag is how much time it used (sched_slice) minus how much
        # time it was owed (ideal_slice). reducing lag means running
        seq_group.lag -= self.sched_slice - self.ideal_slice
        seq_group.vdeadline = seq_group.lag + seq_group.expected_time_slice
        seq_group.total_vtime += self.sched_slice

    def _update_waiting_priority(self, seq_group):
        # didn't run last time, thus accumulating lag
        seq_group.lag += self.ideal_slice
        seq_group.vdeadline = seq_group.lag + seq_group.expected_time_slice

    def _priosched_should_update_waiting_1(self):
        return True

    def _can_preempt(self, seq_group):
        # preemption condition
        # return seq_group.cur_vtime >= self.min_vtime_run
        return True

    def _should_preempt(self, victim, sub):
        return sub.vdeadline < victim.vdeadline

    def _added_sequence_to_running(self, seq_group):
        # just executed some tokens and still didn't finished
        if seq_group.total_vtime >= seq_group.expected_time_slice:
            # Initial heuristic: double the expected time
            # if the current time was not enough.
            # TODO: try fibonacci?
            seq_group.expected_time_slice *= 2

    def priority(self, seq_group):
        return seq_group.vdeadline
