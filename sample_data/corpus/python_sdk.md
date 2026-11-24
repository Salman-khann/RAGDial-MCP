# Python SDK Guide

The `halyard` package is the official Python client. It supports Python 3.9 and later, ships type hints, and offers both a synchronous client and an `asyncio` client with the same method names.

```bash
pip install "halyard[snappy,zstd]>=3.2"
```

The optional extras add native compression codecs. Without them the client falls back to gzip, which uses roughly three times more CPU per megabyte.

## Creating a client

Credentials are read from the `HALYARD_API_KEY` environment variable unless passed explicitly. The region endpoint is derived from the key prefix, so most applications only need to construct the client with no arguments.

```python
from halyard import Client

client = Client(
    region="eu-west-1",
    request_timeout=10.0,
    max_retries=5,
)
```

## Producing messages

Use `publish` for single messages and `publish_batch` for throughput. Batches are sent as one request and consume one rate-limit token per 100 messages.

```python
from halyard import Client, Message

client = Client()
stream = client.stream("orders")

# single message
stream.publish(key="customer-42", value=b'{"total": 99.5}')

# batch of messages, compressed with zstd
batch = [Message(key=f"customer-{i}", value=payload(i)) for i in range(500)]
result = stream.publish_batch(batch, compression="zstd")
print(result.first_offset, result.last_offset)
```

### Idempotent producers

Network timeouts make it impossible to know whether a write landed. An idempotent producer attaches a producer ID and a per-partition sequence number so the broker can drop duplicates.

```python
producer = client.producer(
    "payments",
    enable_idempotence=True,
    acks="all",
    linger_ms=5,
    batch_max_bytes=1_048_576,
)
for event in events:
    producer.send(key=event.account_id, value=event.to_json())
producer.flush(timeout=30)
```

Setting `enable_idempotence=True` forces `acks="all"` and limits in-flight requests to 5 per partition. If you pass a conflicting value the constructor raises `ConfigurationError` instead of silently overriding it.

## Consuming messages

Consumers join a group, receive partition assignments, and commit offsets after processing. The default commit mode is manual so that a crash never skips unprocessed messages.

```python
consumer = client.consumer(
    stream="orders",
    group="billing-service",
    auto_offset_reset="earliest",
    max_poll_records=500,
)

for batch in consumer.poll_batches(timeout=1.0):
    for msg in batch:
        handle(msg)
    consumer.commit(batch)  # commit after processing, never before
```

### Handling rebalances

Register callbacks to flush state when partitions are revoked. With the cooperative strategy only the partitions that move are revoked, so the callback receives a subset of the current assignment.

```python
def on_revoked(partitions):
    state_store.flush(partitions)
    consumer.commit_sync()

def on_assigned(partitions):
    state_store.warm(partitions)

consumer.on_partitions_revoked(on_revoked)
consumer.on_partitions_assigned(on_assigned)
```

## Async client

```python
import asyncio
from halyard.aio import AsyncClient

async def main():
    async with AsyncClient() as client:
        stream = client.stream("clicks")
        await stream.publish(key="session-9", value=b"{}")
        async for msg in client.consumer(stream="clicks", group="analytics").messages():
            await process(msg)
            await msg.ack()

asyncio.run(main())
```

## Retry behaviour

The SDK retries retryable errors (the `Retryable` column in the error catalogue) with exponential backoff: the first delay is 100 ms, each subsequent delay doubles, full jitter is applied, and the delay is capped at 10 seconds. `max_retries` defaults to 5. Non-retryable errors raise immediately as subclasses of `halyard.errors.ClientError`.

## Logging and tracing

- Set `HALYARD_LOG=debug` to log every request and response header.
- Pass `tracer_provider=` to emit OpenTelemetry spans; each publish creates a span named `halyard.publish` with the stream and partition as attributes.
- The `request_id` of every failed call is attached to the raised exception as `exc.request_id`.
