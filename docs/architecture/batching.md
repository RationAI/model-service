# Batching (How It Works Under the Hood)

Batching in Ray Serve is a **replica-local request coalescing** mechanism.

It improves throughput when your model can process multiple inputs more efficiently together (common for GPU inference).

## Where batching happens

Batching happens **inside each replica process**.

Requests only become eligible for batching after they:

1. enter through the proxy and handle queueing/backpressure, and
2. get routed to a specific replica

See also: **[Request lifecycle](request-lifecycle.md)**.

## The API surface (what you configure)

In user code, batching is enabled by decorating an **async** method with `@serve.batch`:

- `max_batch_size`: upper bound for how many requests are grouped into one batch execution
- `batch_wait_timeout_s`: maximum time to wait (since the first queued item) before flushing a smaller batch

Serve expects the batched handler to return **one result per input** (same batch length, same order).

## What Serve actually does internally

Conceptually, each replica maintains an internal structure like:

- an in-memory buffer of pending calls
- a background “flush” loop that decides when to execute a batch
- per-request futures/promises that get completed when the batch finishes

### 1. Collection phase (buffering)

Incoming requests that hit the batched method are appended to a replica-local buffer.

Each buffered entry stores:

- the request arguments (or decoded payload)
- a future representing that request’s eventual response

### 2. Flush conditions (size or time)

The buffer is flushed when either condition becomes true:

- **Size trigger**: buffer length reaches `max_batch_size`
- **Time trigger**: `batch_wait_timeout_s` elapses since the **first** item currently in the buffer

This is why batching can increase latency at low QPS: a request may wait up to `batch_wait_timeout_s` for more arrivals.

### 3. Execution phase (single call)

Serve invokes your batched handler **once** with a list of inputs.

This is where you typically vectorize:

- stack/concat tensors
- run one forward pass
- split/scatter outputs back

### 4. Scatter phase (complete futures)

When the batched handler returns a list of outputs, Serve resolves the stored futures in order.

Each original HTTP request then completes independently with its corresponding output.

## Configuration & Tuning

For a deep dive into how batching interacts with concurrency limits (specifically why `max_ongoing_requests` must be larger than `max_batch_size`), see **[Queues and backpressure](queues-and-backpressure.md)**.

Quick tips:

- Increase `max_batch_size` if the model benefits from larger batches and you have headroom.
- Increase `batch_wait_timeout_s` to favor fuller batches; decrease it to favor latency.

## Next

- Request flow including queue points: [Request lifecycle](request-lifecycle.md)
- Queueing and rejection controls: [Queues and backpressure](queues-and-backpressure.md)
- “Knobs” reference and meanings: [Configuration reference](../guides/configuration-reference.md)
