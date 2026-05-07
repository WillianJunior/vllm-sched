import time
# from collections import defaultdict, deque
# from collections.abc import Iterable
# from dataclasses import replace
# from typing import Any

from functools import cmp_to_key

import numpy as np

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
        self.quantum = 20

        self.sched_step = 0

        setattr(Request, "cur_time", 0)
        
        # Will: bad bug fix...
        # waiting_remote_kvs changes to PREEMPTED when available for execution.
        # however, it was on self.running before, not self.waiting.
        # When scheduling a PREEMPTED req we self.waiting.pop(req).
        # This only skips the self.waiting.pop(req).
        setattr(Request, "was_waiting_remote_kvs", False)

     def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = [] # waiting -> running
        scheduled_resumed_reqs: list[Request] = [] # preempted -> running
        scheduled_running_reqs: list[Request] = [] # running - >running
        preempted_reqs: list[Request] = [] # running/waiting -> preempted

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

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Prepare eligeble running requests
        # These are requests which should keep running
        # Assumption: self.running is sorted by priority. If a running
        # request cannot run due to OOM, self.running.pop() can return
        # a preemption victim.
        all_reqs_queue = []
        remaining_reqs = []
        req_idx = 0
        for request in self.running:
            if self._can_preempt(request):
                remaining_reqs.append(request)
            else:
                all_reqs_queue.append(request)

        # Get requests which were scheduled, but are still waiting
        # E.g., KV, encoder, FSM, ...
        # Assumption: self.waiting is ordered by priority
        for request in self.waiting:
            if request.state == RequestStatus.WAITING:
                remaining_reqs.append(request)
            else:
                all_reqs_queue.append(request)

        all_reqs_queue.extend(remaining_reqs)

        num_sched_reqs = 0
        while all_reqs_queue:
            if num_sched_reqs >= self.max_num_running_reqs:
                print(f"[rr][all_reqs] reached MNS")
                break

            if token_budget <= 0:
                print(f"[rr][all_reqs] exausted token_budget")
                break

            request = all_reqs_queue.pop(0)
            request_id = request.request_id

            print(f"[rr][all_reqs_queue] trying req {request_id}, {len(all_reqs_queue)} remaining in the queue, num_sched_reqs={num_sched_reqs}")

            if request.status == RequestStatus.RUNNING:
                can_schedule_request, new_blocks = self._try_sched_running(
                    request, requests_queue, preempted_reqs, token_budget, 
                    encoder_compute_budget, scheduled_encoder_inputs)
                
                # Here are behaviors unique to running, vs waiting.
                # Ideally, this should be empty
                if can_schedule_request:
                    scheduled_running_reqs.append(request)
                    
                    # Speculative decode related.
                    if request.spec_token_ids:
                        num_scheduled_spec_tokens = (
                            num_new_tokens
                            + request.num_computed_tokens
                            - request.num_tokens
                            - request.num_output_placeholders
                        )
                        if num_scheduled_spec_tokens > 0:
                            spec_token_ids = request.spec_token_ids
                            if len(spec_token_ids) > num_scheduled_spec_tokens:
                                spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                            scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                        # New spec tokens will be set in `update_draft_token_ids` before the
                        # next step when applicable.
                        request.spec_token_ids = []
                    
            else:
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    print(f"[rr] waiting is wait remote kvs")
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        if request.num_preemptions:
                            # We must be loading for a resumed preemption
                            # rather than a new request.
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                        # Continue scheduling this waiting request
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id,
                        )

                        # Skiping this request
                        continue

                elif request.status == RequestStatus.WAITING_FOR_FSM:
                    print(f"[rr] waiting is wait fsm")
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                        # Continue scheduling this waiting request
                    else:
                        # Skiping this request
                        continue

                elif request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    print(f"[rr] waiting is wait streaming")
                    assert not request.streaming_queue
                    
                    # Skiping this request
                    continue

                    can_schedule_request, new_blocks = self._try_sched_waiting(
                        request, scheduled_loras, token_budget, 
                        encoder_compute_budget, scheduled_encoder_inputs, 
                        preempted_reqs)

                if can_schedule_request and load_kv_async:
                    # If loading async, memory was allocated.
                    # Now need to wait for loading.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS

                    # Still marks this request as scheduled in this step.
                    # This avoids loading too many requests to gpu and
                    # having to preempt them later
                    num_sched_reqs += 1

                    continue

                # Here are behaviors unique to running, vs waiting.
                # Ideally, this should be empty
                if can_schedule_request:
                    self.waiting.remove(request)
                    self.running.append(request)

                    if self.log_stats:
                        request.record_event(
                            EngineCoreEventType.SCHEDULED, scheduled_timestamp
                        )

                    if request.status == RequestStatus.WAITING:
                        print(f"[rr][waiting] resuming WAITING req {request_id}")
                        scheduled_new_reqs.append(request)
                    elif request.status == RequestStatus.PREEMPTED:
                        print(f"[rr][waiting] resuming PREEMPTED req {request_id}")
                        scheduled_resumed_reqs.append(request)
                    else:
                        raise RuntimeError(f"Invalid request status: {request.status}")

                    if self.lora_config and request.lora_request:
                        scheduled_loras.add(request.lora_request.lora_int_id)

                    request.status = RequestStatus.RUNNING
                    request.num_computed_tokens = num_computed_tokens

                    # Count the number of prefix cached tokens.
                    if request.num_cached_tokens < 0:
                        request.num_cached_tokens = num_computed_tokens
            
            if not can_schedule_request:
                # Either the request needs to wait, or it consumed
                # the remaining all_reqs_queue trying to preempt
                continue

            # Finish scheduling
            num_sched_reqs += 1
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.cur_time += 1

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            
            # Allocate for external load encoder cache
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        if all_reqs_queue:
            # If there are still requests in the queue, the batch should full
            assert num_sched_reqs >= self.max_num_running_reqs or token_budget <= 0

        # ===========================================================
        # Priority queue consumed. All schedulable reqs were scheduled

    def _try_sched_running(self, request, requests_queue, preempted_reqs, 
            token_budget, encoder_compute_budget, scheduled_encoder_inputs):

        # Returns:
        # did_schedule, new_blocks
        # did_schedule=False if there is not enough mem for request

        request_id = request.request_id
        print(f"[rr][running] trying {request.request_id}")
        assert len(self.waiting) == len(set(self.waiting))

        # === 1. pre-checks =========================================

        if (
            request.num_output_placeholders > 0
            # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
            # Since output placeholders are also included in the computed tokens
            # count, we subtract (num_output_placeholders - 1) to remove any draft
            # tokens, so that we can be sure no further steps are needed even if
            # they are all rejected.
            and request.num_computed_tokens + 2 - request.num_output_placeholders
            >= request.num_prompt_tokens + request.max_tokens
        ):
            # Async scheduling: Avoid scheduling an extra step when we are sure that
            # the previous step has reached request.max_tokens. We don't schedule
            # partial draft tokens since this prevents uniform decode optimizations.
            
            # Request was already removed from the queue and still is
            # in self.running
            # return token_budget, encoder_compute_budget, False, True
            return False, new_blocks

        num_new_tokens = (
            request.num_tokens_with_spec
            + request.num_output_placeholders
            - request.num_computed_tokens
        )
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens, token_budget)

        # Make sure the input position does not exceed the max model len.
        # This is necessary when using spec decoding.
        num_new_tokens = min(
            num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
        )

        # Schedule encoder inputs.
        encoder_inputs_to_schedule = None
        external_load_encoder_input: list[int] = []
        new_encoder_compute_budget = encoder_compute_budget
        if request.has_encoder_inputs:
            (
                encoder_inputs_to_schedule,
                num_new_tokens,
                new_encoder_compute_budget,
                external_load_encoder_input,
            ) = self._try_schedule_encoder_inputs(
                request,
                request.num_computed_tokens,
                num_new_tokens,
                encoder_compute_budget,
                shift_computed_tokens=1 if self.use_eagle else 0,
            )

        if self.need_mamba_block_aligned_split:
            num_new_tokens = self._mamba_block_aligned_split(
                request, num_new_tokens
            )

        if num_new_tokens == 0:
            # The request cannot be scheduled because one of the following
            # reasons:
            # 1. No new tokens to schedule. This may happen when
            #    (1) PP>1 and we have already scheduled all prompt tokens
            #    but they are not finished yet.
            #    (2) Async scheduling and the request has reached to either
            #    its max_total_tokens or max_model_len.
            # 2. The encoder budget is exhausted.
            # 3. The encoder cache is exhausted.
            # 4. Insufficient budget for a block-aligned chunk in hybrid
            #    models with mamba cache mode \"align\".
            # NOTE(woosuk): Here, by doing `continue` instead of `break`,
            # we do not strictly follow the FCFS scheduling policy and
            # allow the lower-priority requests to be scheduled.
            print(f"[rr][running] num_new_tokens == 0")
            return False, new_blocks

        # === 2. KV blocks scheduling ===============================
        did_allocated_kv_block, encoder_compute_budget, new_blocks = _try_allocate_kv_blocks(
            request, requests_queue, num_new_tokens, 
            scheduled_encoder_inputs, encoder_compute_budget, 
            preempted_reqs, self.num_lookahead_tokens)
        
        if new_blocks:
            print(f"[rr][running] can schedule {request_id}, cur_time={request.cur_time}")
            return True, new_blocks
        else:
            return False, None

    def _try_sched_waiting(self, request, scheduled_loras, token_budget, 
        encoder_compute_budget, scheduled_encoder_inputs, preempted_reqs):

        # Returns:
        # did_schedule, new_blocks

        request_id = request.request_id
        did_alloc = False
        new_blocks = None

        # TODO: this was in the previous code, but i removed
        # check if no issues...
        # if request_id in self.prev_step_scheduled_req_ids:
        #     print(f"[rr][waiting] request scheduled in the last step")
        #     continue

        if (
            self.lora_config
            and request.lora_request
            and (
                len(scheduled_loras) == self.lora_config.max_loras
                and request.lora_request.lora_int_id not in scheduled_loras
            )
        ):
            # Scheduling would exceed max_loras, skip.
            return did_schedule, new_blocks

        # =======================================================
        # Attempting kv cache load.
        # Can be reuse, or load from elsewhere

        num_external_computed_tokens = 0
        load_kv_async = False
        connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

        # Get already-cached tokens.
        if request.num_computed_tokens == 0:
            print(f"[rr][_try_sched_waiting][kv_load] getting cached tokens")
            # Get locally-cached tokens.
            new_computed_blocks, num_new_local_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )

            # Get externally-cached tokens if using a KVConnector.
            if self.connector is not None:
                ext_tokens, load_kv_async = (
                    self.connector.get_num_new_matched_tokens(
                        request, num_new_local_computed_tokens
                    )
                )

                if ext_tokens is None:
                    # The request cannot be scheduled because
                    # the KVConnector couldn't determine
                    # the number of matched tokens.
                    #self.waiting.pop_request()
                    #will #self.waiting.remove(request)
                    #skipped_waiting_requests.prepend_request(request)
                    continue

                request.num_external_computed_tokens = ext_tokens
                num_external_computed_tokens = ext_tokens

                connector_prefix_cache_queries = (
                    request.num_tokens - num_new_local_computed_tokens
                )
                connector_prefix_cache_hits = num_external_computed_tokens

            # Total computed tokens (local + external).
            num_computed_tokens = (
                num_new_local_computed_tokens + num_external_computed_tokens
            )
        else:
            print(f"[rr][_try_sched_waiting][kv_load] no cached tokens")
            # KVTransfer: WAITING reqs have num_computed_tokens > 0
            # after async KV recvs are completed.
            new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
            num_new_local_computed_tokens = 0
            num_computed_tokens = request.num_computed_tokens

        # =======================================================
        # Resolve encoder, kv async, mamba, P/D disaggregation, ...

        encoder_inputs_to_schedule = None
        external_load_encoder_input = []
        new_encoder_compute_budget = encoder_compute_budget

        if load_kv_async:
            # KVTransfer: loading remote KV, do not allocate for new work.
            print(f"[rr][_try_sched_waiting] load_kv_async")
            assert num_external_computed_tokens > 0
            num_new_tokens = 0
        else:
            print(f"[rr][_try_sched_waiting] no need for load_kv_async")
            # Number of tokens to be scheduled.
            # We use `request.num_tokens` instead of
            # `request.num_prompt_tokens` to consider the resumed
            # requests, which have output tokens.
            num_new_tokens = request.num_tokens - num_computed_tokens
            threshold = self.scheduler_config.long_prefill_token_threshold
            if 0 < threshold < num_new_tokens:
                num_new_tokens = threshold

            # chunked prefill has to be enabled explicitly to allow
            # pooling requests to be chunked
            if (
                not self.scheduler_config.enable_chunked_prefill
                and num_new_tokens > token_budget
            ):
                # If chunked_prefill is disabled,
                # we can stop the scheduling here.
                print(f"[rr][_try_sched_waiting] pooling/chunked preffil break")
                # TODO: When does this happens? This ended scheduling in FCFS
                raise Exception("This ended scheduling in FCFS...")
                return did_schedule, new_blocks
            
            num_new_tokens = min(num_new_tokens, token_budget)
            assert num_new_tokens > 0

            # Schedule encoder inputs.
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )
                if num_new_tokens == 0:
                    # The request cannot be scheduled.
                    print(f"[rr][_try_sched_waiting] encoder: num_new_tokens == 0")
                    # TODO: When does this happens? This ended scheduling in FCFS
                    raise Exception("This ended scheduling in FCFS...")
                    return did_schedule, new_blocks

        if self.need_mamba_block_aligned_split:
            num_new_tokens = self._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
            if num_new_tokens == 0:
                print(f"[rr][_try_sched_waiting] mamba: num_new_tokens == 0")
                # TODO: When does this happens? This ended scheduling in FCFS
                raise Exception("This ended scheduling in FCFS...")
                return did_schedule, new_blocks

        # Handles an edge case when P/D Disaggregation
        # is used with Spec Decoding where an
        # extra block gets allocated which
        # creates a mismatch between the number
        # of local and remote blocks.
        effective_lookahead_tokens = (
            0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
        )

        num_encoder_tokens = (
            self._num_encoder_max_input_tokens
            if self.is_encoder_decoder and request.has_encoder_inputs
            else 0
        )

        did_alloc, encoder_compute_budget, new_blocks = _try_allocate_kv_blocks(
            request, requests_queue, num_new_tokens, 
            scheduled_encoder_inputs, encoder_compute_budget, preempted_reqs, 
            effective_lookahead_tokens, num_new_local_computed_tokens, 
            new_computed_blocks, num_external_computed_tokens, 
            load_kv_async, num_encoder_tokens)

        if did_alloc:
            did_schedule = True
        return did_schedule, new_blocks


    def _try_allocate_kv_blocks(self, request, requests_queue, num_new_tokens, 
            scheduled_encoder_inputs, encoder_compute_budget, preempted_reqs, 
            num_lookahead_tokens, num_new_computed_tokens=0, 
            new_computed_blocks=0, num_external_computed_tokens=0, 
            delay_cache_blocks=False, num_encoder_tokens=0):
        # Returns:
        # did_alloc, encoder_compute_budget, new_blocks

        with record_function_or_nullcontext("schedule: allocate_slots"):
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=num_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=delay_cache_blocks,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is not None:
                    # The request can be scheduled.
                    return True, encoder_compute_budget, new_blocks

                # Find a preemptible request. Requests with num_computed_token=0
                # are new requests, and have no kv cache blocks. Thus, cannot
                # be preempted
                while requests_queue:
                    preempted_req = requests_queue.pop()
                    print(f"----testing preemption of {preempted_req.request_id} status={preempted_req.status}")
                    if preempted_req.num_computed_tokens > 0:
                        print(f"----can preempt {preempted_req.request_id}")
                        break

                if not requests_queue or preempted_req.num_computed_tokens == 0:
                    # No other valid requests to preempt. Not enough kv blocks.
                    print("[rr][_try_allocate_kv_blocks] cannot preempt to allocate blocks")
                    break

                # Preempting the last (valid) request in the queue
                preempted_req_id = preempted_req.request_id
                print(f"[rr][_try_allocate_kv_blocks] preempting {preempted_req_id} with status {preempted_req.status}")

                preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                    preempted_req_id, None
                )
                if preempted_encoder_inputs:
                    # Restore encoder compute budget if the preempted
                    # request had encoder inputs scheduled in this step.
                    num_embeds_to_restore = sum(
                        preempted_req.get_num_encoder_embeds(i)
                        for i in preempted_encoder_inputs
                    )
                    encoder_compute_budget += num_embeds_to_restore

                if preempted_req.status == RequestStatus.RUNNING:
                    self.running.remove(preempted_req)
                    self.waiting.append(preempted_req)
                self._preempt_request(preempted_req, scheduled_timestamp)
                preempted_reqs.append(preempted_req)
                
                # If the request cannot run, and is the victim for preemption,
                # it must still be preempted above.
                if preempted_req == request:
                    # This should never happen!!!
                    # request is popped from the queue for scheduling...
                    assert False
                    return False, encoder_compute_budget, None

            # Could not allocate...
            return False, encoder_compute_budget, None


    TODO!!!!! verificar como funciona a movimentação de reqs quando tem preempção
    Ex: running->waiting? waiting(inmem)->waiting(preempted)
    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        # TODO: This is bad... Should add another Status: IN_MEM, which is waiting
        # but can be preempted, i.e., swapped out. Currently, IN_MEM is WAITING
        #assert request.status == RequestStatus.RUNNING, (
        #    "Only running requests can be preempted"
        #)
        print(f"[rr][preemption] self.kv_cache_manager.block_pool.get_num_free_blocks() before: {self.kv_cache_manager.block_pool.get_num_free_blocks()}")
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        #was_waiting = request.status == RequestStatus.WAITING
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        is_RR = True

        print(f"[rr] preemption: {request.request_id}")
        print(f"[rr][preemption] self.kv_cache_manager.block_pool.get_num_free_blocks() after: {self.kv_cache_manager.block_pool.get_num_free_blocks()}")

        if is_RR:
            # Reset current running time
            request.cur_time = 0

            # Will: for RR, reinsert in the back instead of prepending
            # Will: can preempt waiting reqs which were in-mem, but not running.
            # No need to add them again to self.waiting.
            #if not was_waiting:
            #    self.waiting.append(request) # no longer reinsert to waiting. caller must choose where to reinsert...
            #assert len(self.waiting) == len(set(self.waiting))
        else:
            # Put the request back to the waiting queue.
            self.waiting.prepend_request(request)