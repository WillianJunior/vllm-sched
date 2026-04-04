import time
# from collections import defaultdict, deque
# from collections.abc import Iterable
# from dataclasses import replace
# from typing import Any

from functools import cmp_to_key

# import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

class Scheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # Base Scheduler init...
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            mm_registry,
            include_finished_set,
            log_stats,
        )

        # In number of generated decode tokens
        self.quantum = 60

        setattr(Request, "cur_time", 0)


    def schedule(self) -> SchedulerOutput:
        # NOTE(will) just basic decoder-only scheduling
        # no spec decoding
        # no encoders
        # no mamba
        # no LoRA
        # no remote KVS
        # no structured output wif FSM
        # always with chunked prefill
        # no P/D Disaggregation

        # Usefull attributes:
        # self.running
        # self.waiting

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []     # preempted and recompute
        swapped_out_reqs: list[Request] = []   # preempted and swapped kv to ram
        stopped_reqs: list[Request] = []       # preempted, but still in vram

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()
        debug_rr = True

        all_submitted_reqs = len(self.running) + len(self.waiting)
        #if debug_rr: print(f"[rr] all_reqs_num {all_submitted_reqs}")
        #if debug_rr: print(f"[rr] all_reqs {self.requests}")

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            # Check for quantum preemption
            # This just stops the execution of the request, i.e., will 
            # not generate tokens this turn.
            # Request remains in gpu memory
            if all_submitted_reqs > self.max_num_running_reqs and request.cur_time >= self.quantum:
                self.running.remove(request)
                stopped_reqs.append(request)
                req_index += 1
                if debug_rr: print(f"[rr] stopping {request.request_id}")
                continue

            num_new_tokens = 1
            num_new_tokens = min(num_new_tokens, token_budget)

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # Not implementing this yet...
                    # However, ideas:
                    # 1. Simple, preempt last on waiting queue. All
                    #    reqs are running at the same time, regardless of
                    #    memory pressure. Preempted to RAM.
                    # 2. FCFS + RR. RR eligible reqs. Reqs are scheduled
                    #    FCFS, and are kept in memory. On OOM: swap-out
                    #    the youngest req and block new reqs until the end
                    #    of another req. In this case, reinstante the 
                    #    preempted req, loading the req state from RAM.
                    raise Exception("Not implementing a OOM preemption policy...")

                    # Try preempt stopped reqs before preempting running ones
                    # for block space.
                    if stopped_reqs:
                        preempted_req = stopped_reqs.pop()
                        if debug_rr: print(f"[rr] preempting stopped {preempted_req.request_id}")
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_req.cur_time = 0
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1
            request.cur_time += 1

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Try to fill the token budget if there is memory available
        # i.e., no preemptions
        if not preempted_reqs:
            # Next, schedule the WAITING requests.
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()
                request_id = request.request_id

                # Check if request was stopped (i.e., still in memory)
                if request.cur_time > 0:
                    # Tokens already in memory, just a stopped request
                    num_new_tokens = 1
                    num_computed_tokens = request.num_computed_tokens


                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                else:
                    # Get already-cached tokens.
                    if request.num_computed_tokens == 0:
                        # Get locally-cached tokens.
                        new_computed_blocks, num_new_local_computed_tokens = (
                            self.kv_cache_manager.get_computed_blocks(request)
                        )

                        # Total computed tokens (local + external).
                        num_computed_tokens = num_new_local_computed_tokens
                    else:
                        # KVTransfer: WAITING reqs have num_computed_tokens > 0
                        # after async KV recvs are completed.
                        new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                        num_new_local_computed_tokens = 0
                        num_computed_tokens = request.num_computed_tokens

                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    num_encoder_tokens = 0
                    effective_lookahead_tokens = 0
                    num_external_computed_tokens = 0
                    load_kv_async = False
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_new_computed_tokens=num_new_local_computed_tokens,
                        new_computed_blocks=new_computed_blocks,
                        num_lookahead_tokens=effective_lookahead_tokens,
                        num_external_computed_tokens=num_external_computed_tokens,
                        delay_cache_blocks=load_kv_async,
                        num_encoder_tokens=num_encoder_tokens,
                    )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                request = self.waiting.pop_request()
                self.running.append(request)

                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.cur_time > 0:
                    scheduled_running_reqs.append(request)
                    if debug_rr: print(f"[rr] resumed {request.request_id}")
                elif request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                    if debug_rr: print(f"[rr] starting from waiting {request.request_id}")
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")
                
                # The first token is being processed in this step
                # TODO: Use num_new_tokens? Watch for prefill.
                request.cur_time = 1
                
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )

                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens

            # Sort by request.cur_time: 
            # i.e., requests with longer cur_time should remain stopped
            # compared to stopped reqs which are more recent.
            def smaller_runtime_comparator(lhs, rhs):
                return lhs.cur_time - rhs.cur_time
            stopped_reqs.sort(key=cmp_to_key(smaller_runtime_comparator))

            # Try to fill any remaining budget with requests that could 
            # should be stopped, but there is still some budget.
            # stopped_reqs is sorted by cur_time, ascending
            while stopped_reqs and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = stopped_reqs[0] # peek head
                num_new_tokens = 1
                num_new_tokens = min(num_new_tokens, token_budget)

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Will not preempt other reqs to schedule this
                    # stopped req.
                    break

                # Get the request from the stopped queue and return
                # it to the running queue
                request = stopped_reqs.pop()
                self.running.append(request)

                if debug_rr: print(f"[rr] resuming same step {request.request_id}")

                # Schedule the request.
                scheduled_running_reqs.append(request)
                request_id = request.request_id
                req_to_new_blocks[request_id] = new_blocks
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.cur_time += 1

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Put stopped requests in the waiting queue
        if stopped_reqs:
            # Now stopped reqs are trully waiting
            for req in stopped_reqs:
                req.status = RequestStatus.WAITING

            # Stopped requests must go to the end of the waiting queue.
            self.waiting.extend(stopped_reqs) # add in FCFS ordering
            # self.waiting.prepend_requests(stopped_reqs)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs

        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
