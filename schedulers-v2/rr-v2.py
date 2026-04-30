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


    def _schedule_running(self, request, requests_queue, token_budget, 
            encoder_compute_budget, scheduled_encoder_inputs, scheduled_running_reqs, 
            preempted_reqs, req_to_new_blocks, num_scheduled_tokens, 
            scheduled_spec_decode_tokens, scheduled_loras, scheduled_timestamp):

        # Steps:
        # 1. Checks for num_tokens, async sched, encoder, ...
        # 2. Schedule KV blocks
        # 3. Schedule request

        # Returns whether scheduling is impossible from now on,
        # i.e., there is no more kv cache available.
        # By blocking further scheduling we avoid inversion of
        # priority issues. I.e., if req which is head of priority
        # queue cannot be scheduled due to lack of memory, no further
        # req can schedule either.

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
            return token_budget, encoder_compute_budget, False, True

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
            return token_budget, encoder_compute_budget, True, False

        # === 2. KV blocks scheduling ===============================
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

                # preempted_req = self.running.pop(0)
                while requests_queue:
                    preempted_req = requests_queue.pop()
                    if preempted_req.num_computed_tokens > 0:
                        # If num_computed_token=0, then this is a new
                        # request. Cannot preempt it. Just ignore
                        break

                if not requests_queue and preempted_req.num_computed_tokens == 0:
                    # No other valid requests to preempt. Not enough kv blocks.
                    print("[rr][running] cannot preempt to allocate blocks")
                    break

                # Preempting the last (valid) request in the queue
                preempted_req_id = preempted_req.request_id
                print(f"[rr][running] preempting {preempted_req_id} with status {preempted_req.status}")

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
                if preempted_req == request:
                    # No more request to preempt. Cannot schedule this request.
                    break

        if new_blocks is None:
            # Cannot schedule this request.
            print(f"[rr][running] cannot schedule: new_blocks = None")
            return token_budget, encoder_compute_budget, False, False

        # === 3. Schedule the request ===============================
        scheduled_running_reqs.append(request)
        request_id = request.request_id
        req_to_new_blocks[request_id] = new_blocks
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens
        request.cur_time += 1

        print(f"[rr][running] scheduled {request_id}")

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

        # Encoder-related.
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
            # Allocate the encoder cache.
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
            encoder_compute_budget = new_encoder_compute_budget
        if external_load_encoder_input:
            for i in external_load_encoder_input:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)

        if self.lora_config and request.lora_request and request.lora_request.lora_int_id > 0:
            scheduled_loras.add(request.lora_request.lora_int_id)
        
        print(f"[rr][running] scheduled")
        return token_budget, encoder_compute_budget, True, True

    def _prepare_waiting_remote_kvs(self, request, skipped_waiting_requests):
        # Returns whether the request can be scheduled
        # as a waiting request

        print(f"[rr][_prepare_waiting_remote_kvs] {request.request_id}")

        is_ready = self._update_waiting_for_remote_kv(request)
        if is_ready:
            if request.num_preemptions:
                # We must be loading for a resumed preemption
                # rather than a new request.
                request.status = RequestStatus.PREEMPTED
                request.was_waiting_remote_kvs = True
            else:
                request.status = RequestStatus.WAITING

            return True
        else:
            logger.debug(
                "%s is still in WAITING_FOR_REMOTE_KVS state.",
                request.request_id,
            )
            #self.waiting.pop_request()
            #will #self.waiting.remove(request)
            #skipped_waiting_requests.prepend_request(request)
            return False

    def _prepare_waiting_fsm(self, request, skipped_waiting_requests):
        # Returns whether the request can be scheduled
        # as a waiting request

        print(f"[rr][_prepare_waiting_fsm] {request.request_id,}")

        structured_output_req = request.structured_output_request
        if structured_output_req and structured_output_req.grammar:
            request.status = RequestStatus.WAITING
            return True
        else:
            #self.waiting.pop_request()
            #will #self.waiting.remove(request)
            #skipped_waiting_requests.prepend_request(request)
            return False


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

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

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

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Assemble all requests in a single queue
        all_reqs_queue = []
        all_reqs_queue.extend(self.running)
        all_reqs_queue.extend(self.waiting)

        if False: # just disable printing
            print("[rr] self.waiting:")
            for r in self.waiting:
                print(f"\t{r.request_id}")

            print("[rr] self.running:")
            for r in self.running:
                print(f"\t{r.request_id}")

        #print(self.waiting)
        #print(self.running)
        #print(all_reqs_queue)

        #if self.sched_step >5:
        #    0/0
        #self.sched_step += 1

        # for req in all_reqs_queue:
        #     req.should_sched = False

        # First, sort all requests acording to the priority metric
        # for rr: O(n), just move all quantumd reqs to the end...
        # This is inefficient: waiting are never quantumd...
        req_index = 0
        req_unchecked_len = len(all_reqs_queue)
        while req_index < req_unchecked_len:
            request = all_reqs_queue[req_index]
            if request.cur_time > self.quantum:
                request = all_reqs_queue.pop(req_index)
                all_reqs_queue.append(request)

                # Don't need to check reqs which were quantumd
                req_unchecked_len -= 1
            else:
                req_index += 1

        # All states used:
        # RequestStatus.RUNNING
        # RequestStatus.WAITING
        # RequestStatus.WAITING_FOR_REMOTE_KVS
        # RequestStatus.WAITING_FOR_FSM
        # RequestStatus.WAITING_FOR_STREAMING_REQ
        # RequestStatus.PREEMPTED # is it possible to get a preempted?

        # Consume the queue and move the requests into the correct lists
        # for generating the scheduler the outputs
        num_sched_reqs = 0
        while all_reqs_queue:
            if num_sched_reqs >= self.max_num_running_reqs:
                print(f"[rr][all_reqs] reached MNS: len(scheduled_running_reqs)={len(scheduled_running_reqs)}, len(self.running)={len(self.running)})")
                break

            if token_budget <= 0:
                print(f"[rr][all_reqs] exausted token_budget")
                break

            request = all_reqs_queue.pop(0)
            request_id = request.request_id

            print(f"[rr][all_reqs] trying req {request_id}, {len(all_reqs_queue)} remaining in the queue, num_sched_reqs={num_sched_reqs}")

            if request.status == RequestStatus.RUNNING:
                # In memory, ran in the previous step

                (token_budget, encoder_compute_budget, can_continue_scheduling, did_schedule) = self._schedule_running(
                        request, all_reqs_queue,
                        token_budget, encoder_compute_budget, scheduled_encoder_inputs, 
                        scheduled_running_reqs, preempted_reqs, req_to_new_blocks,
                        num_scheduled_tokens, scheduled_spec_decode_tokens,
                        scheduled_loras, scheduled_timestamp)

                if not can_continue_scheduling: 
                    # Cannot schedule request. No other request should be scheduled...
                    break
                
                # Scheduled the current running request 
                # Can go to the next request
                num_sched_reqs += 1
                continue

            # Solve specific waiting status
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                print(f"[rr] waiting is wait remote kvs")
                special_waiting_can_schedule = self._prepare_waiting_remote_kvs(request, skipped_waiting_requests)
            elif request.status == RequestStatus.WAITING_FOR_FSM:
                print(f"[rr] waiting is wait fsm")
                special_waiting_can_schedule = self._prepare_waiting_fsm(request, skipped_waiting_requests)
            elif request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                print(f"[rr] waiting is wait streaming")
                assert not request.streaming_queue
                #self.waiting.pop_request()
                #self.waiting.remove(request)
                #will #skipped_waiting_requests.prepend_request(request)
                special_waiting_can_schedule = False
            else:
                print(f"[rr] waiting is regular waiting")
                special_waiting_can_schedule = True

            # Special waiting, but cannot be scheduled at the moment
            if not special_waiting_can_schedule:
                print(f"[rr][waiting] not special_waiting_can_schedule")
                continue

            # =======================================================
            # If reached this point, the request was waiting

            # TODO: this was in the previous code
            # remove to check if it is required in this version
            if request_id in self.prev_step_scheduled_req_ids:
                print(f"[rr][waiting] request scheduled in the last step")
                continue

            if (
                self.lora_config
                and request.lora_request
                and (
                    len(scheduled_loras) == self.lora_config.max_loras
                    and request.lora_request.lora_int_id not in scheduled_loras
                )
            ):
                # Scheduling would exceed max_loras, skip.
                #self.waiting.pop_request()
                #will self.waiting.remove(request)
                #skipped_waiting_requests.prepend_request(request)
                continue

            # =======================================================
            # Attempting kv cache reuse

            num_external_computed_tokens = 0
            load_kv_async = False
            connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

            # Get already-cached tokens.
            if request.num_computed_tokens == 0:
                print(f"[rr][waiting][kv_reuse] getting cached tokens")
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
                print(f"[rr][waiting][kv_reuse] no cached tokens")
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
                print(f"[rr][waiting] load_kv_async")
                assert num_external_computed_tokens > 0
                num_new_tokens = 0
            else:
                print(f"[rr][waiting] no need for load_kv_async")
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
                    print(f"[rr][waiting] pooling/chunked preffil break")
                    break
                
                #if num_new_tokens == 0:
                #    num_new_tokens = 1 # TODO: BAD TEST!!! throughput test breaks. maybe some in_mem req was being executed??

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
                        print(f"[rr][waiting] encoder: num_new_tokens == 0")
                        break

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request,
                    num_new_tokens,
                    num_new_local_computed_tokens,
                    num_external_computed_tokens,
                )
                if num_new_tokens == 0:
                    print(f"[rr][waiting] mamba: num_new_tokens == 0")
                    break

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

            # =======================================================
            # Allocating kv cache blocks
            print(f"[rr][waiting] trying to allocate blocks")
            while True:
                if request.cur_time > 0:
                    # Request waiting but in-memory
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )
                else:
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

                if new_blocks is not None:
                    # The request can be scheduled.
                    print(f"[rr][waiting] req {request.request_id} can be sched")
                    break


                # Preempting the request with the least priority
                # Special case: all_reqs_queue = [R, WiM, W] with R being
                # the current request, WiM being waiting in-mem (i.e., can
                # be preempted for space) and W being waiting (0 blocks).
                # W is popped before WiM. If W could run with R, such
                # priority inversion could never happen. R must be popped
                # from all_reqs_queue before popping WiM. As such, if
                # WiM is preempted, no request with less priority can
                # run, as the were already consumed away from all_reqs_queue.
                while all_reqs_queue:
                    preempted_req = all_reqs_queue.pop()
                    if preempted_req.num_computed_tokens > 0:
                        # If num_computed_token=0, then this is a new
                        # request. Cannot preempt it. Just ignore
                        break

                if not all_reqs_queue and preempted_req.num_computed_tokens == 0:
                    # No other valid requests to preempt. Not enough kv blocks.
                    print("[rr][waiting] cannot preempt to allocate blocks")
                    break

                preempted_req_id = preempted_req.request_id
                if preempted_req_id == request_id:
                    # Cannot preempt itself for running
                    print("[rr][waiting] cannot preempt itself to allocate blocks")
                    break

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

                

                print(f"[rr][waiting] preempting {preempted_req.request_id}")
                if preempted_req.status == RequestStatus.RUNNING:
                    print(f"[rr][wating] preempted {preempted_req_id} from running: {preempted_req in self.running}")
                    self.running.remove(preempted_req)
                else:
                    print(f"[rr][wating] preempted {preempted_req_id} from waiting: {preempted_req in self.waiting}")
                    self.waiting.remove(preempted_req)
                self._preempt_request(preempted_req, scheduled_timestamp)
                assert preempted_req not in self.waiting
                self.waiting.append(preempted_req)
                unique_waiting = len(self.waiting) == len(set(self.waiting))
                assert unique_waiting, f"failed at req {request.request_id}"
                preempted_reqs.append(preempted_req)

            if new_blocks is None:
                # The request cannot be scheduled.

                # NOTE: we need to untouch the request from the encode cache
                # manager
                if request.has_encoder_inputs:
                    self.encoder_cache_manager.free(request)
                break

            # KVTransfer: the connector uses this info to determine
            # if a load is needed. Note that
            # This information is used to determine if a load is
            # needed for this request.
            if self.connector is not None:
                self.connector.update_state_after_alloc(
                    request,
                    self.kv_cache_manager.get_blocks(request_id),
                    num_external_computed_tokens,
                )
                if (
                    self.connector_prefix_cache_stats is not None
                    and connector_prefix_cache_queries != 0
                ):
                    self.connector_prefix_cache_stats.record(
                        num_tokens=connector_prefix_cache_queries,
                        num_hits=connector_prefix_cache_hits,
                        preempted=request.num_preemptions > 0,
                    )


            # =======================================================
            # Scheduling request as running

            # Request is not waiting anymore
            print(f"[rr][waiting] removing from waiting: {request.request_id} status={request.status}")
            if not request.was_waiting_remote_kvs:
                self.waiting.remove(request)
            else:
                # Request need to be added later to self.running. This avoids
                # duplication in self.running.
                self.running.remove(request)

                # request will change state
                request.was_waiting_remote_kvs = False

            if load_kv_async:
                # If loading async, allocate memory and put request
                # into the WAITING_FOR_REMOTE_KV state.
                print(f"[rr][waiting][before_append_running] load_kv_async")
                skipped_waiting_requests.prepend_request(request)
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                continue

            print(f"[rr][waiting] scheduling {request_id}")
            num_sched_reqs += 1

            self.running.append(request)
            if self.log_stats:
                request.record_event(
                    EngineCoreEventType.SCHEDULED, scheduled_timestamp
                )
            if request.status == RequestStatus.WAITING:
                print(f"[rr][waiting] resuming WAITING req {request.request_id}")
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                print(f"[rr][waiting] resuming PREEMPTED req {request.request_id}")
                scheduled_resumed_reqs.append(request)
                print(f'[rr][waiting] self.running:')
                for r in self.running:
                    print(f'\t{r.request_id}')
            else:
                raise RuntimeError(f"Invalid request status: {request.status}")

            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                request_id
            )
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            request.status = RequestStatus.RUNNING
            request.cur_time += 1
            request.num_computed_tokens = num_computed_tokens
            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = num_computed_tokens
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



        # ===========================================================
        # Priority queue consumed. All schedulable reqs were scheduled

        # There may be request in self.running which are not scheduled at this time
        # e.g., when they are quantumd.
        for request in all_reqs_queue:
            if request.status == RequestStatus.RUNNING:
                print(f"[rr][prep_output] req {request.request_id} was running, not anymore...")
                request.status = RequestStatus.WAITING
                assert request not in self.waiting
                self.waiting.append(request)
                self.running.remove(request)
                assert len(self.waiting) == len(set(self.waiting))

        print(f"[rr][prep_output] running: {len(self.running)}, waiting: {len(self.waiting)}")

        assert len(self.waiting) == len(set(self.waiting))
        assert len(self.running) == len(set(self.running))

        if self.lora_config:
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Put back any skipped requests at the head of the waiting queue
        # will: no need for this anymore... waiting is not being consumed anymore
        #if skipped_waiting_requests:
        #    self.waiting.prepend_requests(skipped_waiting_requests)
        #    assert len(self.waiting) == len(set(self.waiting))

        #if stopped_reqs:
        #    self.waiting.extend(stopped_reqs)

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
        ) <= self.max_num_running_reqs

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

        if False: # just disable the prints
            print(f"[rr][cached_reqs] sched_running:")
            for r in scheduled_running_reqs:
                print(f"\t{r.request_id}")
            print(f"[rr][cached_reqs] sched_resumed:")
            for r in scheduled_resumed_reqs:
                print(f"\t{r.request_id}")

            print(f"[rr][cached_reqs] waiting")
            for r in self.waiting:
                print(f"\t{r.request_id}")

            print(f"[rr][cached_reqs] running:")
            for r in self.running:
                print(f"\t{r.request_id}")

        assert len(self.waiting) == len(set(self.waiting))
        assert len(self.running) == len(set(self.running))

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

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)

        print(f"[rr] sched_time {time.monotonic() - scheduled_timestamp}")

        #print(f"[rr] done scheduling")
        return scheduler_output


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




