# Error Code Catalogue

All Halyard APIs return errors in a common envelope. The `code` field is stable across versions and safe to match on; the `message` field is human-readable and may change.

```json
{
  "error": {
    "code": "HY-4091",
    "http_status": 409,
    "message": "stream 'orders' already exists",
    "retryable": false,
    "request_id": "req_01HV7Q4M3J8K2Z"
  }
}
```

Always log the `request_id`. Support can trace a request end to end only with this identifier.

## Client errors (4xx)

| Code | HTTP | Meaning | Retryable | Typical fix |
|---|---|---|---|---|
| HY-4001 | 400 | Malformed JSON body | No | Validate the payload against the OpenAPI schema. |
| HY-4002 | 400 | Unknown field in request | No | Remove the field or upgrade the SDK. |
| HY-4003 | 400 | Message exceeds the plan's max message size | No | Split the payload or store it externally and send a reference. |
| HY-4004 | 400 | Invalid partition key encoding | No | Partition keys must be UTF-8, at most 256 bytes. |
| HY-4005 | 400 | Schema compatibility check failed | No | Evolve the schema according to the stream's compatibility mode. |
| HY-4011 | 401 | Missing or malformed API key | No | Send `Authorization: Bearer <key>`. |
| HY-4012 | 401 | API key expired | No | Rotate the key in the console or via the key rotation endpoint. |
| HY-4031 | 403 | Key lacks the required scope | No | Grant the scope listed in the endpoint reference. |
| HY-4032 | 403 | Request blocked by IP allowlist | No | Add the client CIDR to the project allowlist. |
| HY-4041 | 404 | Stream not found | No | Check the stream name and region. |
| HY-4042 | 404 | Consumer group not found | No | Groups are created on first commit; commit an offset first. |
| HY-4091 | 409 | Stream already exists | No | Use the existing stream or choose another name. |
| HY-4092 | 409 | Offset commit conflict (generation mismatch) | Yes | Rejoin the group and retry after the rebalance completes. |
| HY-4093 | 409 | Concurrent ACL modification | Yes | Re-read the ACL version and retry with the new `if_match` value. |
| HY-4131 | 413 | Batch exceeds 1,000 messages or 16 MiB | No | Split the batch. |
| HY-4221 | 422 | Retention exceeds the plan maximum | No | Lower the retention or upgrade the plan. |
| HY-4291 | 429 | Request rate limit exceeded | Yes | Honour `Retry-After`; see Rate Limits. |
| HY-4292 | 429 | Throughput quota exceeded | Yes | Reduce ingress or request a quota increase. |

## Server errors (5xx)

| Code | HTTP | Meaning | Retryable | Typical fix |
|---|---|---|---|---|
| HY-5001 | 500 | Unexpected internal error | Yes | Retry with backoff; report the request_id if it persists. |
| HY-5021 | 502 | Upstream broker unavailable | Yes | Usually transient during a leader election. |
| HY-5031 | 503 | Partition has no leader | Yes | Wait for leader election; typically resolves in under 10 seconds. |
| HY-5032 | 503 | Not enough in-sync replicas | Yes | Writes with `acks=all` are refused until replicas catch up. |
| HY-5033 | 503 | Region in maintenance mode | Yes | Check the status page; writes resume after the window. |
| HY-5041 | 504 | Produce request timed out in purgatory | Yes | The write may or may not have been persisted; use idempotent producers. |

## Idempotency and retries

Retrying a timed-out produce request (`HY-5041`) can duplicate messages unless the producer is idempotent. Enable idempotence in the SDK (`enable_idempotence=True`); the broker then de-duplicates by producer ID and sequence number within a window of the last 5 batches per partition.

Non-retryable errors should surface to the caller immediately. Retrying `HY-4003` or `HY-4131` will never succeed and only consumes rate-limit tokens.
