# Rate Limits and Quotas

Halyard Cloud enforces quotas per **project** and per **API key**. Limits protect shared clusters from noisy neighbours; they are not billing caps. When a quota is exceeded the API responds with HTTP 429 and error code `HY-4291` or `HY-4292` and includes a `Retry-After` header measured in seconds.

## Request rate classes

Every endpoint belongs to one rate class. The class determines how many requests per second (RPS) a single API key may send, sustained over a rolling 10-second window, and the burst allowance on top of it.

| Rate class | Endpoints | Sustained RPS | Burst | Window |
|---|---|---|---|---|
| `control` | stream create/delete, ACL changes, key rotation | 5 | 10 | 10s |
| `metadata` | list streams, describe stream, cluster info | 50 | 100 | 10s |
| `produce` | publish, publish-batch | 2000 | 4000 | 10s |
| `consume` | fetch, commit offsets | 2000 | 4000 | 10s |
| `admin` | billing export, audit log export, usage reports | 1 | 2 | 60s |
| `schema` | register schema, check compatibility | 20 | 40 | 10s |

## Throughput quotas by plan

Throughput quotas apply to the sum of all API keys in a project. Ingress counts the compressed payload bytes received by brokers; egress counts the bytes returned to consumers, including replays.

| Plan | Ingress MB/s | Egress MB/s | Max partitions | Max streams | Max retention | Max message size |
|---|---|---|---|---|---|---|
| Developer | 1 | 3 | 50 | 10 | 24 hours | 256 KiB |
| Starter | 10 | 30 | 500 | 100 | 7 days | 1 MiB |
| Growth | 50 | 150 | 2,000 | 500 | 30 days | 4 MiB |
| Business | 200 | 600 | 10,000 | 2,000 | 90 days | 8 MiB |
| Enterprise | custom | custom | 50,000 | 10,000 | unlimited (tiered) | 16 MiB |

Egress is allowed to reach three times ingress so that multiple consumer groups can read the same data. Replays from tiered storage count against egress at the same rate as reads from local disk.

## How bursts work

Bursting uses a token bucket per API key and rate class. The bucket holds `Burst` tokens and refills at `Sustained RPS` tokens per second. A request consumes one token; a batch publish consumes one token per 100 messages, rounded up. If the bucket is empty the request is rejected immediately rather than queued, so clients should implement backoff instead of relying on server-side waiting.

Burst capacity does not carry over between rate classes. A key that is throttled on `produce` can still call `metadata` endpoints at full speed.

## Connection limits

- Each API key may hold at most **500** concurrent connections per region.
- Idle connections are closed after **10 minutes** without traffic.
- Long-poll fetches count as active traffic and keep a connection open for up to **30 seconds** per request.
- TLS handshakes are limited to **50 per second** per source IP to mitigate reconnect storms.

## Requesting a quota increase

Quota increases are granted per project. Open a request in the console under *Project settings > Quotas* and include the expected peak ingress, the number of partitions you need, and the business justification. Increases up to two times the plan default are approved automatically within one business day. Larger increases are reviewed by the capacity team and may require moving the project to a dedicated cluster.

## Client guidance

Respect `Retry-After` and add jitter. The official SDKs implement exponential backoff starting at 100 ms with a factor of 2 and full jitter, capped at 10 seconds. Batching messages is the most effective way to stay under the `produce` limit: one publish-batch call with 500 messages costs 5 tokens, while 500 single publishes cost 500.
