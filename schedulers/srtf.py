# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum

# import os
# import random
# import time
# from collections import deque
# from dataclasses import dataclass, field
from typing import Callable, Optional

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.sequence import SequenceGroup

from prioritySchedBase import Scheduler


class OracleFields(enum.Enum):
    KEY = 0
    PROMPT = 1
    GENERATE = 2

KEY_TOKEN_IDX = 3

class SRTF(Scheduler):
    """docstring for SRTF"""

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        super(SRTF, self).__init__(
            scheduler_config,
            cache_config,
            lora_config,
            pipeline_parallel_size,
            output_proc_callback,
        )

        self.oracle = dict()
        with open("oracle_costs_sharegpt200.txt", "r") as file:
            for line in file:
                parts = line.strip().split()
                assert len(parts) == 3, "malformed lines"
                key = int(parts[OracleFields.KEY.value])
                value1 = int(parts[OracleFields.PROMPT.value])
                value2 = int(parts[OracleFields.GENERATE.value])
                self.oracle[key] = (key, value1, value2)

        # [Will]: Monkey patching SequenceGroup to have virtual runtimes.
        # Total is across all schedulings and cur is for the current execution.
        setattr(SequenceGroup, "total_time", 0)
        setattr(SequenceGroup, "cur_time", 0)
        setattr(SequenceGroup, "remaining_time", 1)

        # [Will]: Minimum time a sequence must execute. It cannot be preempted
        # if seq.cur_vtime < min_vtime_run. Unless, OOM.
        self.min_time_run = 10

        # [Will]: How many tokens should be be considered per sched step
        self.base_sched_step_time = 1

    def _get_key(self, seq_group):
        return seq_group.first_seq.inputs['prompt_token_ids'][KEY_TOKEN_IDX]
    
    def _update_queue_size(self, n):
        # required for EEVDF
        pass

    def _update_finished_priority(self, seq_group):
        pass

    def _update_running_priority(self, seq_group):
        seq_group.cur_time += self.base_sched_step_time
        seq_group.total_time += self.base_sched_step_time

    def _update_waiting_priority(self, seq_group):
        pass

    def _priosched_should_update_waiting_1(self):
        return False

    def _can_preempt(self, seq_group):
        # preemption condition
        return seq_group.cur_time >= self.min_time_run

    def _should_preempt(self, victim, sub):
        should_preempt = victim.remaining_time > sub.remaining_time
        return self._can_preempt(victim) and should_preempt

    def _added_sequence_to_running(self, seq_group):
        seq_group.cur_time = 0
        seq_group.remaining_time = (
            self.oracle[self._get_key(seq_group)][OracleFields.GENERATE.value]
            - seq_group.total_time
        )

    def priority(self, seq_group):
        return seq_group.remaining_time

    def print_seq(self, seq_group):
        return f"Rem={seq_group.remaining_time} - {seq_group.cur_time}/{seq_group.total_time}"

