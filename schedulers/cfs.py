# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# import enum
# import os
# import random
# import time
# from collections import deque
# from dataclasses import dataclass, field
# from typing import Callable, Deque, Dict, Iterable, List, Optional
# from typing import Sequence as GenericSequence
# from typing import Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
# from vllm.core.interfaces import AllocStatus, BlockSpaceManager
# from vllm.logger import init_logger
# from vllm.lora.request import LoRARequest
# from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (
    #     Sequence,
    #     SequenceData,
    SequenceGroup,
    #     SequenceGroupBase,
    #     SequenceGroupMetadata,
    #     SequenceGroupMetadataDelta,
    #     SequenceStage,
    #     SequenceStatus,
)

# from vllm.utils import Device, PyObjectCache

from prioritySchedBase import Scheduler


class CFS(Scheduler):
    """docstring for CFS"""

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        super(CFS, self).__init__(
            scheduler_config,
            cache_config,
            lora_config,
            pipeline_parallel_size,
            output_proc_callback,
        )

        # === CFS stuff... ====================================================
        # [Will]: Monkey patching SequenceGroup to have virtual runtimes.
        # Total is across all schedulings and cur is for the current execution.
        setattr(SequenceGroup, "total_vtime", 0)
        setattr(SequenceGroup, "cur_vtime", 0)

        # [Will]: Initially all seqs have default priority.
        setattr(SequenceGroup, "sched_priority", 1)

        # [Will]: Minimum time a sequence must execute. It cannot be preempted
        # if seq.cur_vtime < min_vtime_run. Unless, OOM.
        self.min_vtime_run = 80

        # [Will]: How many tokens should be be considered per sched step
        self.base_sched_step_vtime = 1

    def _update_finished_priority(seq_group):
        pass

    def _update_running_priority(seq_group):
        # CFS implementation
        # Need to update vtimes
        next_vtime = self.base_sched_step_vtime * seq_group.sched_priority
        seq_group.cur_vtime += next_vtime
        seq_group.total_vtime += next_vtime

    def _update_waiting_priority(seq_group):
        pass

    def _can_preempt(seq_group):
        # preemption condition
        return seq_group.cur_vtime >= self.min_vtime_run

    def _should_preempt(victim, sub):
        should_preempt = victim.total_vtime > sub.total_vtime
        return _can_preempt(victim) and should_preempt

    def _added_sequence_to_running(seq_group):
        seq_group.cur_vtime = 0
