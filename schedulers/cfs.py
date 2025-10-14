# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (
    Sequence,
    SequenceData,
    SequenceGroup,
    SequenceGroupBase,
    SequenceGroupMetadata,
    SequenceGroupMetadataDelta,
    SequenceStage,
    SequenceStatus,
)
from vllm.utils import Device, PyObjectCache

logger = init_logger(__name__)

# Test-only. If configured, decode is preempted with
# ARTIFICIAL_PREEMPTION_PROB% probability.
ENABLE_ARTIFICIAL_PREEMPT = bool(
    os.getenv("VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT", False)
)  # noqa
ARTIFICIAL_PREEMPTION_PROB = 0.5
ARTIFICIAL_PREEMPTION_MAX_CNT = 500


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """

    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class ScheduledSequenceGroup:
    # A sequence group that's scheduled.
    seq_group: SequenceGroup
    # The total chunk size (number of tokens) to process for next iteration.
    # 1 for decoding. Same as prompt tokens for prefill, but if prefill is
    # chunked, it can be smaller than that.
    token_chunk_size: int


@dataclass
class SchedulerOutputs:
    """The scheduling decision made from a scheduler."""

    # Scheduled sequence groups.
    scheduled_seq_groups: GenericSequence[ScheduledSequenceGroup]
    # Number of prefill groups scheduled.
    num_prefill_groups: int
    # Total number of batched tokens.
    num_batched_tokens: int
    # Blocks to swap in. List of CPU -> GPU block number.
    blocks_to_swap_in: List[Tuple[int, int]]
    # Blocks to swap out. List of GPU -> CPU block number.
    blocks_to_swap_out: List[Tuple[int, int]]
    # Blocks to copy. Source to dest block.
    blocks_to_copy: List[Tuple[int, int]]
    # Sequence groups that are going to be ignored.
    ignored_seq_groups: List[SequenceGroup]
    # The number of slots for lookahead decoding.
    num_lookahead_slots: int
    # The number of requests in the running queue
    running_queue_size: int
    preempted: int

    def __post_init__(self):
        # Swap in and swap out should never happen at the same time.
        # [Will]: in my case, which has no swap for OOM and swap for
        # quantum, a seq can be quantumed out while another is loaded
        # from waiting
        # assert not (self.blocks_to_swap_in and self.blocks_to_swap_out)

        self.num_loras: int = len(self.lora_requests)
        if self.num_loras > 0:
            self._sort_by_lora_ids()

        self.num_prompt_adapters: int = len(self.prompt_adapter_requests)

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (
            not self.scheduled_seq_groups
            and not self.blocks_to_swap_in
            # and not self.blocks_to_swap_out
            and not self.blocks_to_copy
        )

    def _sort_by_lora_ids(self):
        assert 0 <= self.num_prefill_groups <= len(self.scheduled_seq_groups)

        def key_fn(group: ScheduledSequenceGroup):
            key = (group.seq_group.lora_int_id, group.seq_group.request_id)
            if 0 < self.num_prefill_groups < len(self.scheduled_seq_groups):
                # Sort sequence groups so that all prefills come before all
                # decodes as required by chunked prefill.
                return (not group.seq_group.is_prefill(), *key)
            return key

        self.scheduled_seq_groups = sorted(
            self.scheduled_seq_groups, key=key_fn
        )

    @property
    def lora_requests(self) -> Set[LoRARequest]:
        return {
            g.seq_group.lora_request
            for g in self.scheduled_seq_groups
            if g.seq_group.lora_request is not None
        }

    @property
    def prompt_adapter_requests(self) -> Set[PromptAdapterRequest]:
        return {
            g.seq_group.prompt_adapter_request
            for g in self.scheduled_seq_groups
            if g.seq_group.prompt_adapter_request is not None
        }


def seq_group_metadata_builder():
    return SequenceGroupMetadata(
        request_id="",
        is_prompt=False,
        seq_data={},
        sampling_params=None,
        block_tables={},
    )


def scheduled_seq_group_builder():
    return ScheduledSequenceGroup(
        SequenceGroup.__new__(SequenceGroup), token_chunk_size=0
    )
    # return ScheduledSequenceGroup(seq_group=None, token_chunk_size=0)


@dataclass
class PartialPrefillMetadata:
    """Holds information about the partial prefills that are currently running
    during a single iteration of the Scheduler.
    When chunked prefill is enabled, we allow a certain number of seqs to be
    partially prefilled during each iteration. Having multiple partial prefills
    in flight allows us to minimize TTFT and avoid decode starvation in cases
    where a single sequence group with a very large prompt blocks the queue for
    too many iterations.
    The number of long prefill requests is limited so that smaller
    requests may jump the queue in front of them and get to the decode
    phase faster.
    """

    # A minimum bound on the total number of prefills to be scheduled during
    # this iteration
    schedulable_prefills: int

    # The number of long prefill requests currently running
    long_prefills: int

    scheduler_config: SchedulerConfig

    def can_schedule(self, seq_group: SequenceGroup) -> bool:
        """When concurrent partial prefills are enabled,
        we limit the number of long requests and only accept
        shorter requests from the queue while running them
        concurrently"""
        return not (
            seq_group.first_seq.get_num_new_tokens()
            > self.scheduler_config.long_prefill_token_threshold
            and self.long_prefills
            >= self.scheduler_config.max_long_partial_prefills
            and self.scheduler_config.max_num_partial_prefills > 1
        )

    def maybe_increment_partial_prefills(
        self, seq_group: SequenceGroup
    ) -> None:
        # When a new prefill is scheduled, we need to know if it is a
        # long request
        if (
            seq_group.first_seq.get_num_new_tokens()
            > self.scheduler_config.long_prefill_token_threshold
        ):
            self.long_prefills += 1

    @classmethod
    def from_queues(
        cls,
        running: Deque[SequenceGroup],
        waiting: Deque[SequenceGroup],
        scheduler_config: SchedulerConfig,
    ) -> "PartialPrefillMetadata":
        """Create a PartialPrefillMetadata object from the current state of
        the scheduler's queues.
        This accounts for the currently running prefill requests, and peeks into
        the waiting queue to see if there are more prefills to potentially be
        scheduled during this iteration."""
        prefills = 0
        long_prefills = 0

        waiting_long_prefills = 0

        for sg in running:
            if sg.first_seq.data.stage == SequenceStage.PREFILL:
                prefills += 1
                if (
                    sg.first_seq.get_num_new_tokens()
                    > scheduler_config.long_prefill_token_threshold
                ):
                    long_prefills += 1

        for sg in waiting:
            # Don't bother looping through the rest of the queue if we know
            # there are already at
            # least max_partial_prefills requests to fill
            if prefills >= scheduler_config.max_num_partial_prefills:
                break

            # Don't count long requests from the waiting queue if we aren't
            # going to schedule them anyway
            if (
                sg.first_seq.get_num_new_tokens()
                > scheduler_config.long_prefill_token_threshold
            ):
                if (
                    long_prefills + waiting_long_prefills
                    >= scheduler_config.max_long_partial_prefills
                ):
                    continue
                waiting_long_prefills += 1
            prefills += 1

        # NB: long_prefills and waiting_long_prefills are tracked separately.
        # We don't account for the waiting requests here because we need to use
        # this metadata to track how many have actually been scheduled.
        return PartialPrefillMetadata(
            schedulable_prefills=min(
                prefills, scheduler_config.max_num_partial_prefills
            ),
            long_prefills=long_prefills,
            scheduler_config=scheduler_config,
        )


class Scheduler:
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "selfattn"
        if (
            self.scheduler_config.runner_type == "pooling"
            or self.cache_config.is_attention_free
        ):
            version = "placeholder"

        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version
        )

        num_gpu_blocks = cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
        )

        # Sequence groups in the WAITING state.
        # Contain new prefill or preempted requests.
        # self.waiting: List[SequenceGroup] = []
        self.waiting: Deque[SequenceGroup] = deque()

        # Sequence groups in the RUNNING state.
        # Contain decode requests.
        # self.running: List[SequenceGroup] = []
        self.running: Deque[SequenceGroup] = deque()

        # Sequence groups in the SWAPPED state.
        # Contain decode requests that are swapped out.
        self.swapped: Deque[SequenceGroup] = deque()
        # Sequence groups finished requests ids since last step iteration.
        # It lets the model know that any state associated with these requests
        # can and must be released after the current step.
        # This is used to evict the finished requests from the Mamba cache.
        self._finished_requests_ids: List[str] = list()
        # Time at previous scheduling step
        self.prev_time = 0.0
        # Did we schedule a prompt at previous step?
        self.prev_prompt = False
        # Latency of the last prompt step
        self.last_prompt_latency = 0.0

        # The following field is test-only. It is used to inject artificial
        # preemption.
        self.enable_artificial_preemption = ENABLE_ARTIFICIAL_PREEMPT
        self.artificial_preempt_cnt = (
            ARTIFICIAL_PREEMPTION_MAX_CNT
            if self.enable_artificial_preemption
            else 0
        )
        self.num_cumulative_preemption: int = 0

        # Used to cache python objects
        self._seq_group_metadata_cache: List[PyObjectCache] = []
        # self._scheduler_running_outputs_cache: List[PyObjectCache] = []
        self._scheduled_seq_group_cache: List[PyObjectCache] = []

        # For async output processing, we need to swap cache buffers between
        # iterations. I.e. since the output processing is lagged one step,
        # we cannot reuse the cached objects immediately when the schedule()
        # is called again, but only when schedule() is called the second time.
        self.output_proc_callback = output_proc_callback
        self.use_async_output_proc = self.output_proc_callback is not None
        self.num_cache_iters = 2 if self.use_async_output_proc else 1

        self.cache_id = 0
        for i in range(self.num_cache_iters):
            self._seq_group_metadata_cache.append(
                PyObjectCache(seq_group_metadata_builder)
            )
            # self._scheduler_running_outputs_cache.append(
            #     PyObjectCache(scheduler_running_outputs_builder)
            # )
            self._scheduled_seq_group_cache.append(
                PyObjectCache(scheduled_seq_group_builder)
            )

        # For async postprocessor, the extra decode run cannot be done
        # when the request reaches max_model_len. In this case, the request
        # will be stopped during schedule() call and added to this stop list
        # for processing and deallocation by the free_finished_seq_groups()
        self._async_stopped: List[SequenceGroup] = []

        # List with the chunk sizes to hand out to each sequence depending
        # on how many partial prefills are running. This is slightly faster than
        # running an integer division every time a prefill is scheduled.
        # This splits the budget evenly among all prefills.
        self.partial_prefill_budget_lookup_list = [0] * (
            self.scheduler_config.max_num_partial_prefills + 1
        )
        self.partial_prefill_budget_lookup_list[
            0
        ] = scheduler_config.max_num_batched_tokens
        for i in range(1, self.scheduler_config.max_num_partial_prefills + 1):
            self.partial_prefill_budget_lookup_list[i] = (
                scheduler_config.max_num_batched_tokens // i
            )

        # === CFS stuff... ====================================================

        # [Will]: we are only doing recompute on OOM
        # preemption mode, RECOMPUTE or SWAP
        # self.user_specified_preemption_mode = scheduler_config.preemption_mode
        self.user_specified_preemption_mode = PreemptionMode.RECOMPUTE

        # [Will]: Monkey patching SequenceGroup to have virtual runtimes.
        # Total is across all schedulings and cur is for the current execution.
        setattr(SequenceGroup, "total_vtime", 0)
        setattr(SequenceGroup, "cur_vtime", 0)

        # [Will]: Initially all seqs have default priority.
        setattr(SequenceGroup, "priority", 1)

        # [Will]: Acknowledge whether it was swapped due to quantum ending.
        # Required for checking if a sequence is a prefill or if it should
        # be swapped in.
        setattr(SequenceGroup, "was_quantumd_out", False)

        # [Will]: Minimum time a sequence must execute. It cannot be preempted
        # if seq.cur_vtime < min_vtime_run. Unless, OOM.
        self.min_vtime_run = 20

        # [Will]: How many tokens should be be considered per sched step
        self.base_sched_step_vtime = 1

    @property
    def next_cache_id(self):
        return (self.cache_id + 1) % self.num_cache_iters

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    def _add_seq_group_to_running(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the running queue.
        # Only for testing purposes.
        self.running.append(seq_group)

    def _add_seq_group_to_swapped(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the swapped queue.
        # Only for testing purposes.
        self.swapped.append(seq_group)

    def abort_seq_group(
        self,
        request_id: Union[str, Iterable[str]],
        seq_id_to_seq_group: Optional[Dict[str, SequenceGroupBase]] = None,
    ) -> None:
        """Aborts a sequence group with the given ID.

        Check if the sequence group with the given ID
            is present in any of the state queue.
        If present, remove the sequence group from the state queue.
            Also, if any of the sequences in the sequence group is not finished,
                free the sequence with status `FINISHED_ABORTED`.
        Otherwise, do nothing.

        Args:
            request_id: The ID(s) of the sequence group to abort.
            seq_id_to_seq_group: helper for groups with n>1
        """
        if isinstance(request_id, str):
            request_id = (request_id,)
        request_ids = set(request_id)
        seq_id_to_seq_group = seq_id_to_seq_group or {}
        for state_queue in [
            self.waiting,
            self.running,
            self.swapped,
        ]:
            aborted_groups: List[SequenceGroup] = []
            for seq_group in state_queue:
                # When n>1, seq_group.request_id looks like
                # foo_parallel_sample_0, while request_ids is just foo, and we
                # should resolve it as real_request_id to match.
                if seq_group.request_id in seq_id_to_seq_group:
                    real_request_id = seq_id_to_seq_group[
                        seq_group.request_id
                    ].group_id
                else:
                    real_request_id = seq_group.request_id
                if real_request_id in request_ids:
                    # Appending aborted group into pending list.
                    aborted_groups.append(seq_group)
                    # We can't remove real_request_id in request_ids here,
                    # because there may be other seq groups sharing the same
                    # real_request_id
            for aborted_group in aborted_groups:
                # Remove the sequence group from the state queue.
                state_queue.remove(aborted_group)
                # Remove the aborted request from the Mamba cache.
                self._finished_requests_ids.append(aborted_group.request_id)
                for seq in aborted_group.get_seqs():
                    if seq.is_finished():
                        continue
                    seq.status = SequenceStatus.FINISHED_ABORTED
                    self.free_seq(seq)
                if aborted_group.request_id in seq_id_to_seq_group:
                    del seq_id_to_seq_group[aborted_group.request_id]

                self._free_seq_group_cross_attn_blocks(aborted_group)

    def _free_seq_group_cross_attn_blocks(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        """
        Free a sequence group from a cross-attention block table.
        Has no effect on decoder-only models.
        """
        if seq_group.is_encoder_decoder():
            self.block_manager.free_cross(seq_group)

    def has_unfinished_seqs(self) -> bool:
        return (
            len(self.waiting) != 0
            or len(self.running) != 0
            or len(self.swapped) != 0
        )

    def has_waiting_seqs(self) -> bool:
        """If max_num_seqs seqs are running  and there are seqs waiting, or
        if there are OOM swapped-out seqs waiting.
        """
        return (len(self.waiting) + len(self.swapped)) > 0

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.block_manager.get_prefix_cache_hit_rate(device)

    def reset_prefix_cache(self, device: Optional[Device] = None) -> bool:
        return self.block_manager.reset_prefix_cache(device)

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def get_and_reset_finished_requests_ids(self) -> List[str]:
        """Flushes the list of request ids of previously finished seq_groups."""
        finished_requests_ids = self._finished_requests_ids
        self._finished_requests_ids = list()
        return finished_requests_ids

    def _get_prompt_limit(self, seq_group: SequenceGroup) -> int:
        if (
            self.scheduler_config.chunked_prefill_enabled
            and not self.scheduler_config.is_multi_step
        ):
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(
                self.scheduler_config.max_model_len,
                self.scheduler_config.max_num_batched_tokens,
            )

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if seq_group.lora_request and seq_group.lora_request.long_lora_max_len:
            assert prompt_limit <= seq_group.lora_request.long_lora_max_len
            return seq_group.lora_request.long_lora_max_len
        else:
            return prompt_limit

    def _schedule_chunked_prefill(self) -> SchedulerOutputs:
        """Schedule queued requests.

        Chunked prefill allows to chunk prefill requests, batch them together
        with decode requests. This policy 1. schedule as many decoding requests
        as possible. 2. schedule chunked prefill requests that are not
        finished. 3. schedule swapped request. 4. schedule new prefill
        requests.

        The policy can sustain the high GPU utilization because it can put
        prefill and decodes requests to the same batch, while it improves
        inter token latency because decodes requests don't need to be blocked
        by prefill requests.
        """

        # print("xxxxxxxxxxxxxxxxxxxxxxx sched start xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

        # budget = SchedulingBudget(
        #     token_budget=self.scheduler_config.max_num_batched_tokens,
        #     max_num_seqs=self.scheduler_config.max_num_seqs,
        # )
        curr_loras: Set[int] = set()

        # Create partial prefill metadata
        partial_prefill_metadata = PartialPrefillMetadata.from_queues(
            running=self.running,
            waiting=self.waiting,
            scheduler_config=self.scheduler_config,
        )

        # if not self._passed_delay(time.time()):
        #     # ??? do not over schedule
        #     # this should be something like batched scheduling
        #     break

        # running = self.running.copy()
        # waiting = self.waiting.copy()

        # === 1. Step vtime ===================================================
        # === Update vtimes and finish seqs

        max_num_seqs_budget = self.scheduler_config.max_num_seqs

        for seq_group in self.running:
            if seq_group.is_finished():
                # Seq group finished
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(request_id=seq_group.request_id)
                self._free_finished_seq_group(seq_group)
                self.running.remove(seq_group)
            else:
                # Need to update vtimes
                next_vtime = self.base_sched_step_vtime * seq_group.priority
                seq_group.cur_vtime += next_vtime
                seq_group.total_vtime += next_vtime
                max_num_seqs_budget -= 1

        # === 2. Fill-up running ==============================================
        # === Add more seqs to running until max_num_seqs batch is filled.

        new_sched_seqs = []
        should_sched_waiting = self._passed_delay(time.time())

        while should_sched_waiting and self.waiting and max_num_seqs_budget > 0:
            # Get waiting with highest priority
            new_seq = self.waiting.popleft()
            new_sched_seqs.append(new_seq)
            max_num_seqs_budget -= 1

        # === 3. Preemption check =============================================
        # === Preempt and swap seqs, when necessary

        preempted_seqs = []

        def _should_preempt(victim, sub):
            # preemption condition
            can_preempt_victim = victim.cur_vtime >= self.min_vtime_run
            should_preempt = victim.total_vtime < sub.total_vtime
            return can_preempt_victim and should_preempt

        if (
            should_sched_waiting
            and max_num_seqs_budget == 0
            and self.waiting
        ):
            while self.running:
                # Get seq with lowest priority
                seq_group = self.running.peek()

                # Get waiting with highest priority
                waiting_seq_head = self.waiting.peek()
                if _should_preempt(seq_group, waiting_seq_head):
                    self.running.popleft()  # remove lowest priority
                    self.waiting.popleft()  # remove highest priority
                    preempted_seqs.append(seq_group)
                    new_sched_seqs.append(waiting_seq_head)
                else:
                    # If the current seq_group of should not be preempted,
                    # then the following seqs with higher priority
                    # won't be also.
                    break

        # === 4. OOM check ====================================================
        # === Preempt and swap seqs when OOM

        self.running.extend(new_sched_seqs)
        #self.running.sort(key=lambda s: s.total_vtime, reverse=True)  # sorted by priority: total_vtime
        self.running = deque(sorted(self.running, key=lambda s: s.total_vtime, reverse=True))

        # Number of free blocks in the GPU, i.e., the memory budget
        block_allocator = self.block_manager.block_allocator
        cur_blocks_budget = block_allocator.get_num_total_blocks(Device.GPU)

        def cdiv(a: int, b: int) -> int:
            """Ceiling division."""
            return -(a // -b)

        def get_num_required_blocks(seq_group):
            # Worst-case scenario heuristic: not checking for prefix-caching
            # blocks reuse.
            num_required_blocks = 0
            for seq in seq_group.get_seqs():
                # seq.get_len() returns all prefill and decode tokens.
                # Thus, for prefill, allocate space for the prefill tokens
                # and 0 decode (actually just 1). For Decode, it is the
                # size of prefill and its decode tokens (which is > 0).
                seq_tokens = seq.get_len()

                # Decode generate a single step
                # All seqs (decode or prefill) are calculated to have
                # 1 new token.
                new_tokens = 1

                num_required_blocks += cdiv(
                    seq_tokens + new_tokens, seq.block_size
                )

            return num_required_blocks

        # Calculate current budget
        for seq_group in self.running:
            cur_blocks_budget -= get_num_required_blocks(seq_group)

        # TODO:
        # This can create a bad thrashing pattern:
        # 1. Seq A is running and seq B should preempt A
        # 2. Seq B cannot run due to OOM
        # 3. Seq B preempted A, but cannot run, thus A
        # could still be running
        # Add a backfill policy??

        # Remove seqs from running if there is no budget
        for seq_group in self.running:
            # Check if preemption due to OOM is still reguired
            if cur_blocks_budget >= 0:
                break

            # Check if it is possible to preempt seq_group and
            # insert it into the proper queue after preemption
            if seq_group.cur_vtime > 0:
                # seq_group was running
                if seq_group.cur_vtime < self.min_vtime_run:
                    # This seq cannot be preempted:
                    # Thrashing avoidance policy
                    continue
                preempted_seqs.append(seq_group)
            else:
                # seq_group was just scheduled
                # Return it to the waiting queue
                self.waiting.append(seq_group)

            # Remove seq and update budget
            self.running.remove(seq_group)
            cur_blocks_budget += get_num_required_blocks(seq_group)

        # If there still is a budget deficit, remove seqs regardless
        # of the thrashing avoidance policy
        while cur_blocks_budget < 0:
            if seq_group.cur_vtime > 0:
                # seq_group was running
                preempted_seqs.append(seq_group)
            else:
                # seq_group was just scheduled
                # Return it to the waiting queue
                self.waiting.append(seq_group)
            # Remove seq and update budget
            self.running.remove(seq_group)
            cur_blocks_budget += get_num_required_blocks(seq_group)

        # === 5. Finish =======================================================
        # === Reset the queues, preempt seqs, swap back seqs

        # self.running
        # self.waiting
        # new_sched_seqs
        # preempted_seqs

        prefill_seq_groups: List[ScheduledSequenceGroup] = []
        decode_seq_groups: List[ScheduledSequenceGroup] = []

        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        # TODO: nothing done to partial prefill. fix later...
        enable_chunking = False

        # Actually preempt seqs
        for seq_group in preempted_seqs:
            if self.use_async_output_proc:
                assert self.output_proc_callback is not None
                self.output_proc_callback(request_id=seq_group.request_id)

            self._swap_out(seq_group, blocks_to_swap_out)

        self.waiting.extend(preempted_seqs)
        #self.waiting.sort(key=lambda s: s.total_vtime)  # higher priority first
        self.waiting = deque(sorted(self.waiting, key=lambda s: s.total_vtime))

        # Manage blocks for new scheduled seqs
        for seq_group in new_sched_seqs:
            seq_group.cur_vtime = 0
            if seq_group.is_prefill():
                # Prefill need to allocate block space
                self.block_manager.allocate(seq_group)
            else:
                # Otherwise, need to swap-in blocks
                self._swap_in(seq_group, blocks_to_swap_in)

        # No need to sort self.running since it was sorted before OOC check
        # and it could only remove seqs, thus ordering is kept.

        # Go though all running seqs, generating a ScheduledSequenceGroup
        # response to the engine, calculating the total number of tokens
        # and appending a slot to each seq.
        num_batched_tokens = 0
        for seq_group in self.running:
            scheduled_seq_group: ScheduledSequenceGroup = (
                self._scheduled_seq_group_cache[self.cache_id].get_object()
            )
            scheduled_seq_group.seq_group = seq_group

            self._append_slots(seq_group, blocks_to_copy, enable_chunking)

            total_seq_group_tokens = 0
            for seq in seq_group.get_seqs():
                seq.status = SequenceStatus.RUNNING
                total_seq_group_tokens += seq.get_len()

            if seq_group.is_prefill():
                scheduled_seq_group.token_chunk_size = total_seq_group_tokens
                prefill_seq_groups.append(scheduled_seq_group)
            else:
                total_seq_group_tokens += 1
                scheduled_seq_group.token_chunk_size = 1
                decode_seq_groups.append(scheduled_seq_group)
            num_batched_tokens += total_seq_group_tokens

        return SchedulerOutputs(
            scheduled_seq_groups=prefill_seq_groups + decode_seq_groups,
            num_prefill_groups=len(prefill_seq_groups),
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],  # TODO: check later for these cases
            num_lookahead_slots=0,  # TODO: check later for these cases
            running_queue_size=len(self.running),
            preempted=len(preempted_seqs),
        )

    def _schedule(self) -> SchedulerOutputs:
        """Schedule queued requests."""
        if self.scheduler_config.chunked_prefill_enabled:
            return self._schedule_chunked_prefill()
        else:
            return self._schedule_default()

    def _allow_async_output_proc(self, seq_group: SequenceGroup) -> bool:
        # async_output_proc is allowed only when we have a single sequence
        # in the sequence group
        no_single_seq = seq_group.sampling_params is None or (
            seq_group.sampling_params.n == 1
        )
        return no_single_seq

    def schedule(
        self,
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(
            scheduler_outputs.scheduled_seq_groups
        ):
            seq_group = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size
            seq_group.maybe_set_first_scheduled_time(now)

            seq_group_metadata = self._seq_group_metadata_cache[
                self.cache_id
            ].get_object()
            seq_group_metadata.seq_data.clear()
            seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            block_tables: Dict[int, List[int]] = {}

            if seq_group.is_encoder_decoder():
                # Encoder associated with SequenceGroup
                encoder_seq = seq_group.get_encoder_seq()
                assert encoder_seq is not None
                encoder_seq_data = encoder_seq.data
                # Block table for cross-attention
                # Also managed at SequenceGroup level
                cross_block_table = self.block_manager.get_cross_block_table(
                    seq_group
                )
            else:
                encoder_seq_data = None
                cross_block_table = None

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            if self.cache_config.enable_prefix_caching:
                common_computed_block_nums = (
                    self.block_manager.get_common_computed_block_ids(
                        seq_group.get_seqs(status=SequenceStatus.RUNNING)
                    )
                )

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (
                    token_chunk_size + num_computed_tokens
                    < seqs[0].data.get_len()
                ):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=seq_data,
                    sampling_params=seq_group.sampling_params,
                    block_tables=block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=common_computed_block_nums,
                    encoder_seq_data=encoder_seq_data,
                    cross_block_table=cross_block_table,
                    state=seq_group.state,
                    token_type_ids=seq_group.token_type_ids,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=(
                        seq_group.multi_modal_data
                        if scheduler_outputs.num_prefill_groups > 0
                        else None
                    ),
                    multi_modal_placeholders=(
                        seq_group.multi_modal_placeholders
                        if scheduler_outputs.num_prefill_groups > 0
                        else None
                    ),
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                seq_data_delta = {}
                for id, data in seq_data.items():
                    seq_data_delta[id] = data.get_delta_and_reset()
                seq_group_metadata = SequenceGroupMetadataDelta(
                    seq_data_delta,
                    seq_group.request_id,
                    block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=common_computed_block_nums,
                )
            seq_group_metadata_list.append(seq_group_metadata)

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(
                    seq_group
                )

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group,
                scheduled_seq_group.token_chunk_size,
            )

        self._seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.metrics is not None:
                if seq_group.metrics.scheduler_time is not None:
                    seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # Return results
        return (
            seq_group_metadata_list,
            scheduler_outputs,
            allow_async_output_proc,
        )

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        """Free a sequence from a block table."""
        self.block_manager.free(seq)

    def remove_seq_from_computed_blocks_tracker(
        self, seq_group: SequenceGroup, status: Optional[SequenceStatus]
    ) -> None:
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            self._remove_seq_from_computed_blocks_tracker(seq)

    def _remove_seq_from_computed_blocks_tracker(self, seq: Sequence) -> None:
        """
        Free a sequence computed blocks tracker _seq_id_to_blocks_hashes
        and _seq_id_to_num_tokens_computed.
        """
        self.block_manager.remove_seq_from_computed_blocks_tracker(seq)

    def _free_finished_seqs(self, seq_group: SequenceGroup) -> None:
        """Free finished seqs in a sequence group."""
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                self.free_seq(seq)

    def _free_finished_seq_group(self, seq_group: SequenceGroup) -> None:
        if seq_group.is_finished():
            # Free cross-attention block table, if it exists
            self._free_seq_group_cross_attn_blocks(seq_group)

            # Add the finished requests to the finished requests list.
            # This list will be used to update the Mamba cache in the
            # next step.
            self._finished_requests_ids.append(seq_group.request_id)

        # Free finished seqs
        self._free_finished_seqs(seq_group)

    def free_finished_seq_groups(self) -> None:
        remaining: Deque[SequenceGroup] = deque()
        for seq_group in self.running:
            self._free_finished_seq_group(seq_group)
            if not seq_group.is_finished():
                remaining.append(seq_group)

        self.running = remaining

        # Handle async stopped sequence groups
        # (ones that reached max model len)
        if self._async_stopped:
            for seq_group in self._async_stopped:
                self._free_seq_group_cross_attn_blocks(seq_group)
                self._finished_requests_ids.append(seq_group.request_id)

                # Free finished seqs
                self._free_finished_seqs(seq_group)

            self._async_stopped.clear()

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: List[Tuple[int, int]],
        enable_chunking: bool = False,
    ) -> None:
        """Appends new slots to the sequences in the given sequence group.

        Args:
            seq_group (SequenceGroup): The sequence group containing the
                sequences to append slots to.
            blocks_to_copy (List[Tuple[int, int]]): A list of tuple of two
                ints, the first int is the source block index, and the second
                int is the destination block index. This list is updated with
                the new source and destination block indices for the appended
                slots.
            enable_chunking (bool): True if chunked prefill is enabled.
        """
        is_prefill: bool = seq_group.is_prefill()
        num_lookahead_slots: int = self._get_num_lookahead_slots(
            is_prefill, enable_chunking
        )

        seq_group.init_multi_step_from_lookahead_slots(
            num_lookahead_slots,
            num_scheduler_steps=self.scheduler_config.num_scheduler_steps,
            is_multi_step=self.scheduler_config.is_multi_step,
            enable_chunking=enable_chunking,
        )

        seq_status: Optional[SequenceStatus] = SequenceStatus.RUNNING
        if self.scheduler_config.is_multi_step and enable_chunking:
            # In multi-step chunked-prefill any sequence type can have
            # slots appended.
            seq_status = None

        for seq in seq_group.get_seqs(status=seq_status):
            cows = self.block_manager.append_slots(seq, num_lookahead_slots)
            if len(cows) > 0:
                blocks_to_copy.extend(cows)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        num_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            num_blocks += len(
                self.block_manager.block_tables[seq.seq_id].blocks
            )
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error."
            )
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _passed_delay(self, now: float) -> bool:
        if self.prev_prompt:
            self.last_prompt_latency = now - self.prev_time
        self.prev_time, self.prev_prompt = now, False
        # Delay scheduling prompts to let waiting queue fill up
        if self.scheduler_config.delay_factor > 0 and self.waiting:
            earliest_arrival_time = min(
                [e.metrics.arrival_time for e in self.waiting]
            )
            passed_delay = (now - earliest_arrival_time) > (
                self.scheduler_config.delay_factor * self.last_prompt_latency
            ) or not self.running
        else:
            passed_delay = True
        return passed_delay

    def _get_num_lookahead_slots(
        self, is_prefill: bool, enable_chunking: bool
    ) -> int:
        """The number of slots to allocate per sequence per step, beyond known
        token ids. Speculative decoding uses these slots to store KV activations
        of tokens which may or may not be accepted.

        Speculative decoding does not yet support prefill, so we do not perform
        lookahead allocation for prefill.

        When chunking is enabled with multi-step, we allocate lookahead slots
        for the prefills for when the prefills turn into decodes in the first
        step.
        """
        if is_prefill:
            if self.scheduler_config.is_multi_step and enable_chunking:
                # num_lookahead_slots was introduced in the context of decodes,
                # in Speculative Decoding.
                # When the num_scheduler_steps is 8, say, then the
                # num_lookahead_slots is 7. Meaning, we are doing a 1-step of
                # decode anyways and we wish to do 7 more.
                #
                # "lookaheads" for prefills, is introduced in support for
                # Chunked-Prefill in Multi-Step.
                return self.scheduler_config.num_lookahead_slots + 1
            else:
                return 0

        return self.scheduler_config.num_lookahead_slots
