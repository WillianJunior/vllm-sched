from typing import Any, Optional, Union
from collections import defaultdict
from collections.abc import Iterable
import time

from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.request import Request, RequestStatus
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry


# Types
RequestID = str


class MyFCFSSched(SchedulerInterface):
    """
    Experimenting with vLLM scheduler.
    Should be similar (if not the same) as vLLM base scheduler...
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        # print(f"=============== init")
        # print(f"\t vllm_config: {vllm_config}")
        # print(f"\t kv_cache_config: {kv_cache_config}")
        # print(f"\t structured_output_manager: {structured_output_manager}")
        # print(f"\t mm_registry: {mm_registry}")
        # print(f"\t include_finished_set: {include_finished_set}")
        # print(f"\t log_stats: {log_stats}")

        self.queue = []

        # TODO: initialize...
        self.running
        self.waiting
        self.max_num_running_reqs
        self.max_num_scheduled_tokens
        self.finished_req_ids
        self.scheduler_config
        self.kv_cache_manager
        self.log_stats

    def schedule(self) -> SchedulerOutput:
        """Schedule the requests to process in this scheduling step.

        The scheduling decision is made at the iteration level. Each scheduling
        step corresponds to a single forward pass of the model. Therefore, this
        method is called repeatedly by a busy loop in the engine.

        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        that specifies how many tokens to process for each request in this
        scheduling step. For example, num_tokens can be as large as the number
        of prompt tokens for new requests, or it can be 1 for the requests that
        are auto-regressively generating new tokens one by one. Otherwise, it
        can be somewhere in between in case of chunked prefills, prefix caching,
        speculative decoding, etc.

        Additionally, the scheduler also returns useful data about each request
        or the batch as a whole. The model runner will use this information in
        preparing inputs to the model.

        Returns:
            A SchedulerOutput object containing information about the scheduled
            requests.
        """

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        # Map of a request to the new blocks of kv-cache allocated
        # to run the current step
        req_to_new_block_ids: dict[RequestID, tuple[list[int], ...]] = {}

        # Scheduled tokens to run in the current step in the same batch.
        # For decode, each request run 1 token. For prefill, more can run.
        num_scheduled_tokens: dict[RequestID, int] = {}
        token_budget = self.max_num_scheduled_tokens

        # Unused features...
        scheduled_spec_decode_tokens: dict[RequestID, list[int]] = {}
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        num_common_prefix_blocks = [0]

        # For logging.
        scheduled_timestamp = time.monotonic()

        # =====================================================================
        # === First, schedule RUNNING requests

        long_pf_token_thrs = self.scheduler_config.long_prefill_token_threshold
        chunked_pf_enabled = self.scheduler_config.chunked_prefill_enabled

        for request in self.running:
            if token_budget <= 0:
                break

            # Calculate the number of new tokens for this request.
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )

            if 0 < long_pf_token_thrs < num_new_tokens:
                num_new_tokens = long_pf_token_thrs
            num_new_tokens = min(num_new_tokens, token_budget)

            assert num_new_tokens > 0, (
                "Should num_new_tokens always be >0? "
                "Check original code 'if num_new_tokens == 0'"
            )

            # Attempt to allocate enough kv-cache space for the new tokens
            # of the current request
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    # num_lookahead_tokens=self.num_lookahead_tokens,
                )
                if new_blocks is None:
                    # Not enough kv-cache spece
                    # Preempting lowest priority request
                    preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp
                        )

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)

                    if preempted_req == request:
                        # No more requests to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                # Only happens if there are no more blocks to run the request
                # and it is the request with the lowest priority (end of queue)
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            req_to_new_block_ids[
                request.request_id
            ] = new_blocks.get_block_ids()
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens

        # =====================================================================
        # === Next, schedule WAITING requests if there are no preempted
        # === requests waiting.

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        # skipped_waiting_requests = []

        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                # # KVTransfer: skip request if still waiting for remote kvs.
                # if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                #     is_ready = self._update_waiting_for_remote_kv(request)
                #     if is_ready:
                #         request.status = RequestStatus.WAITING
                #     else:
                #         logger.debug(
                #             "%s is still in WAITING_FOR_REMOTE_KVS state.",
                #             request.request_id,
                #         )
                #         self.waiting.pop_request()
                #         skipped_waiting_requests.prepend_request(request)
                #         continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    (
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                    ) = self.kv_cache_manager.get_computed_blocks(request)

                    # # Get externally-cached tokens if using a KVConnector.
                    # if self.connector is not None:
                    #     (
                    #         num_external_computed_tokens,
                    #         load_kv_async,
                    #     ) = self.connector.get_num_new_matched_tokens(
                    #         request, num_new_local_computed_tokens
                    #     )

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens
                        + num_external_computed_tokens
                    )
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list()
                    )
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if 0 < long_pf_token_thrs < num_new_tokens:
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold
                        )

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not chunked_pf_enabled and num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    # num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # # KVTransfer: the connector uses this info to determine
                # # if a load is needed. Note that
                # # This information is used to determine if a load is
                # # needed for this request.
                # if self.connector is not None:
                #     self.connector.update_state_after_alloc(
                #         request,
                #         new_computed_blocks + new_blocks,
                #         num_external_computed_tokens,
                #     )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                # if load_kv_async:
                #     # If loading async, allocate memory and put request
                #     # into the WAITING_FOR_REMOTE_KV state.
                #     skipped_waiting_requests.prepend_request(request)
                #     request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                #     continue

                # Set the current request as scheduled (running)
                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}"
                    )

                req_to_new_block_ids[
                    request.request_id
                ] = self.kv_cache_manager.get_block_ids(request.request_id)
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

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

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_block_ids[req.request_id]
            )
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_block_ids,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            # Unused features
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            free_encoder_input_ids=[],
            structured_output_request_ids={},
            # grammar_bitmask=grammar_bitmask,
        )

        # # NOTE(Kuntai): this function is designed for multiple purposes:
        # # 1. Plan the KV cache store
        # # 2. Wrap up all the KV cache load / save ops into an opaque object
        # # 3. Clear the internal states of the connector
        # if self.connector is not None:
        #     meta = self.connector.build_connector_meta(scheduler_output)
        #     scheduler_output.kv_connector_metadata = meta

        # events = self.kv_cache_manager.take_events()
        # if events:
        #     batch = KVEventBatch(ts=time.time(), events=events)
        #     self.kv_event_publisher.publish(batch)

        # Update the internal state of all requests
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

        return scheduler_output

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_block_ids: dict[str, tuple[list[int], ...]],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...]] = []
        num_computed_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = num_scheduled_tokens[req_id] - len(
                spec_decode_tokens.get(req_id, ())
            )
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens
                    + num_tokens
                ]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(req_to_new_block_ids[req_id])
            num_computed_tokens.append(req.num_computed_tokens)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        """Update the scheduler state based on the model runner output.

        This method is called after the model runner has processed the scheduled
        requests. The model runner output includes generated token ids, draft
        token ids for next step, etc. The scheduler uses this information to
        update its states, checks the finished requests, and returns the output
        for each request.

        Returns:
            A dict of client index to EngineCoreOutputs object containing the
            outputs for each request originating from that client.
        """
        raise NotImplementedError

    def add_request(self, request: Request) -> None:
        """Add a new request to the scheduler's internal queue.

        Args:
            request: The new request being added.
        """
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[RequestID, Iterable[RequestID]],
        finished_status: RequestStatus,
    ) -> None:
        """Finish the requests in the scheduler's internal queue. If the request
        is not in the queue, this method will do nothing.

        This method is called in two cases:
        1. When the request is aborted by the client.
        2. When the frontend process detects a stop string of the request after
           de-tokenizing its generated tokens.

        Args:
            request_ids: A single or a list of request IDs.
            finished_status: The finished status of the given requests.
        """
        raise NotImplementedError

    def get_num_unfinished_requests(self) -> int:
        """Number of unfinished requests in the scheduler's internal queue."""
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        """Returns True if there are finished requests that need to be cleared.
        NOTE: This is different from `not self.has_unfinished_requests()`.

        The scheduler maintains an internal list of the requests finished in the
        previous step. This list is returned from the next call to schedule(),
        to be sent to the model runner in the next step to clear cached states
        for these finished requests.

        This method checks if this internal list of finished requests is
        non-empty. This information is useful for DP attention.
        """
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        """Reset the prefix cache for KV cache.

        This is particularly required when the model weights are live-updated.
        """
        return self.kv_cache_manager.reset_prefix_cache()

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        raise NotImplementedError

    def make_stats(
        self, spec_decoding_stats: Optional[SpecDecodingStats] = None
    ) -> Optional[SchedulerStats]:
        """Make a SchedulerStats object for logging.

        The SchedulerStats object is created for every scheduling step.
        """
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(
                req.is_output_corrupted for req in self.running
            ),
        )

    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
