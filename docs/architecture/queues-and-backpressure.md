# Queues and Backpressure

To maintain stability and prevent overload, Ray Serve implements queueing mechanisms at multiple levels. Understanding these queues is critical for tuning latency and handling load spikes.

## Simplified Queue Model

There are two main places a request can wait:

1.  **Proxy Handle Queue**: Waiting to be assigned to a replica.
2.  **Replica Execution Queue**: Assigned to a replica, waiting for execution (or batching).

## 1. Proxy-Side Queue (`max_queued_requests`)

When a request arrives at the HTTP Proxy (or via a Deployment Handle), it is routed to a logical deployment. If all specific replicas are busy, the request waits in a queue managed by the proxy/handle.

- **Config**: `max_queued_requests` (in the deployment spec)
- **Behavior**:
  - Controls the maximum number of requests allowed to wait for assignment.
  - If the queue is full, new requests are immediately rejected with a **503 Service Unavailable** error (or a `BackpressureError` in Python).

### Why limit this?

Without a limit, a system under heavy load might accept requests until it runs out of memory or latency becomes unacceptable. Fail-fast behavior is often preferred over unbounded waiting.

## 2. Replica-Side Queue (`max_ongoing_requests`)

Once a request is assigned to a specific replica, it counts as "ongoing" for that replica.

- **Config**: `max_ongoing_requests` (in the deployment spec)
- **Behavior**:
  - Limits how many concurrent requests a single replica can process _or_ have buffered.
  - If a replica is at its limit, the proxy considers it "busy" and will not assign new requests to it (they will wait in the Proxy Queue instead).

### Usage with Batching

If you use `@serve.batch`, requests sitting in the [batching buffer](batching.md) count towards `max_ongoing_requests`.

- **Warning**: If `max_ongoing_requests` is set too low (e.g., lower than `max_batch_size`), you might throttle your own batching mechanism because the replica will never accept enough requests to fill a batch.

## Backpressure flow

1.  **Client** sends a request.
2.  **HTTP Proxy** receives it.
3.  **Check Replica capacity**: Are there replicas with `ongoing_requests < max_ongoing_requests`?
    - **Yes**: Forward request to one of them.
    - **No**: Enqueue request in the Proxy Queue.
4.  **Check Proxy Queue capacity**: Is `current_queue_size < max_queued_requests`?
    - **Yes**: Request waits.
    - **No**: Reject request immediately (Fail).

## Tuning Guidelines

| Scenario                       | Recommendation                                                                                          |
| :----------------------------- | :------------------------------------------------------------------------------------------------------ |
| **High Throughput / Batching** | Increase `max_ongoing_requests` to ensure replicas can buffer enough work to form full batches.         |
| **Latency Sensitive**          | Decrease `max_queued_requests` to fail fast rather than returning stale responses after a long wait.    |
| **Memory Constrained**         | Lower both values to prevent OOM errors by limiting the number of incomplete requests in system memory. |
