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

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.sequence import SequenceGroup

from prioritySchedBase import Scheduler


class OracleFields(enum.Enum):
    KEY = 0
    PROMPT = 1
    GENERATE = 2


KEY_TOKEN_IDX = 3


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

        # Oracle stuff to test if predicting the len would result in SRTF
        self.using_oracle = True
        noise = 2
        max_expected_time = 1000
        min_expected_time = 1
        seed(0)  # to make it reproducible
        self.oracle = dict()
        with open("oracle_costs_sharegpt200.txt", "r") as file:
            for line in file:
                parts = line.strip().split()
                assert len(parts) == 3, "malformed lines"
                key = int(parts[OracleFields.KEY.value])
                value1 = int(parts[OracleFields.PROMPT.value])
                value2 = int(parts[OracleFields.GENERATE.value])

                # Add some noise
                # this is relative noise, does little harm
                # noise_val = value2 * uniform(-noise, noise)
                # print(f"{key} original {value2} noise: {noise_val}")
                # value2 += noise_val
                # value2 = max(1, value2)  # avoid negative numbers

                # now some absolute noise
                noise_val = uniform(-noise, noise) * randint(
                    min_expected_time, max_expected_time
                )

                # bad :( except for small max_expected_time (400 is ok)
                # noise_val = choice([-1, 1]) * randint(
                #    min_expected_time, max_expected_time
                # )

                print(f"{key} original {value2} noise: {noise_val}")
                value2 += noise_val
                value2 = max(1, value2)  # avoid negative numbers

                # just use a random number
                # value2 = randint(min_expected_time, max_expected_time)

                self.oracle[key] = [key, value1, value2]

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
        setattr(SequenceGroup, "expected_time_slice", self._base_expected_time_slice)
        setattr(SequenceGroup, "priority", 1)
        setattr(SequenceGroup, "lag", 0)
        setattr(SequenceGroup, "vdeadline", 0)

        setattr(SequenceGroup, "slice_increment", 10)

        # Number of tokens (or time spent) per sched step by each seq
        # If using multi-step scheduling, it would be more than 1
        # per max_num_seqs.
        self.sched_slice = 1
        self.max_num_seqs = self.scheduler_config.max_num_seqs

    def _get_oracle(self, seq_group):
        key = seq_group.first_seq.inputs["prompt_token_ids"][KEY_TOKEN_IDX]
        return self.oracle[key][OracleFields.GENERATE.value]

    def _update_queue_size(self, n):
        self.queue_size = n

        if self.queue_size < self.max_num_seqs:
            self.ideal_slice = 1
        else:
            self.ideal_slice = (self.max_num_seqs * self.sched_slice) / self.queue_size

    def _update_finished_priority(self, seq_group):
        pass

    def _update_running_priority(self, seq_group):
        if (
            seq_group.expected_time_slice == self._base_expected_time_slice
            and self.using_oracle
        ):
            seq_group.expected_time_slice = self._get_oracle(seq_group)
        seq_group.cur_vtime += self.sched_slice

        # lag is how much time it used (sched_slice) minus how much
        # time it was owed (ideal_slice). reducing lag means running
        seq_group.lag -= self.sched_slice - self.ideal_slice
        if seq_group.lag < 0:
            seq_group.vdeadline = float("inf")
        else:
            seq_group.vdeadline = seq_group.lag + seq_group.expected_time_slice
        seq_group.total_vtime += self.sched_slice

    def _update_waiting_priority(self, seq_group):
        if (
            seq_group.expected_time_slice == self._base_expected_time_slice
            and self.using_oracle
        ):
            seq_group.expected_time_slice = self._get_oracle(seq_group)

        # didn't run last time, thus accumulating lag
        seq_group.lag += self.ideal_slice
        if seq_group.lag < 0:
            seq_group.vdeadline = float("inf")
        else:
            seq_group.vdeadline = seq_group.lag + seq_group.expected_time_slice

    def _priosched_should_update_waiting_1(self):
        return True

    def _can_preempt(self, seq_group):
        # preemption condition
        # return seq_group.cur_vtime >= self.min_vtime_run
        # return True
        return seq_group.cur_vtime > seq_group.expected_time_slice * 1

    def _should_preempt(self, victim, sub):
        # print(f"--------testing {self.print_seq(victim)} -> {self.print_seq(sub)}")
        # if self._can_preempt(victim):
        #    print("    can preempt")
        return self._can_preempt(victim) and sub.vdeadline < victim.vdeadline

    def _added_sequence_to_running(self, seq_group):
        # just executed some tokens and still didn't finished
        # if seq_group.total_vtime >= seq_group.expected_time_slice:
        # Initial heuristic: double the expected time
        # if the current time was not enough.
        # TODO: try fibonacci?

        # Expected time wasn't enough, add more time
        if seq_group.expected_time_slice <= seq_group.total_vtime:
            # seq_group.expected_time_slice *= 2
            seq_group.expected_time_slice += seq_group.slice_increment
            seq_group.slice_increment *= 4
        seq_group.cur_vtime = 0
        # seq_group.num_steps += 1

    def priority(self, seq_group):
        return seq_group.vdeadline

    def print_seq(self, seq_group):
        return (
            f"lag={seq_group.lag:.2f} - vdeadline={seq_group.vdeadline:.2f} "
            f"- {seq_group.cur_vtime}/{seq_group.total_vtime}/{seq_group.expected_time_slice}"
        )
