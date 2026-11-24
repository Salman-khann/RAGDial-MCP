# REST API Reference (v3)

Base URL: `https://api.{region}.halyard.cloud/v3`. All requests and responses use JSON. Authenticate with a bearer token: `Authorization: Bearer <api-key>`. Every endpoint lists the scope the key must hold and the rate class that governs it.

## Endpoint summary

| Method | Path | Scope | Rate class | Description |
|---|---|---|---|---|
| GET | `/streams` | `streams:read` | metadata | List streams in the project. |
| POST | `/streams` | `streams:admin` | control | Create a stream. |
| GET | `/streams/{name}` | `streams:read` | metadata | Describe a stream, including partitions and retention. |
| PATCH | `/streams/{name}` | `streams:admin` | control | Change retention, compaction or partition count. |
| DELETE | `/streams/{name}` | `streams:admin` | control | Delete a stream and all of its data. Irreversible. |
| POST | `/streams/{name}/messages` | `streams:write` | produce | Publish one message. |
| POST | `/streams/{name}/batches` | `streams:write` | produce | Publish up to 1,000 messages in one request. |
| GET | `/streams/{name}/partitions/{p}/messages` | `streams:read` | consume | Fetch messages from an offset. |
| POST | `/groups/{group}/offsets` | `groups:write` | consume | Commit consumer group offsets. |
| GET | `/groups/{group}/lag` | `groups:read` | metadata | Per-partition lag for a consumer group. |
| POST | `/keys/{key_id}/rotate` | `keys:admin` | control | Rotate an API key; the old key stays valid for 24 hours. |
| GET | `/audit-log` | `audit:read` | admin | Export audit events as NDJSON. |

## Create a stream

`POST /streams`

```json
{
  "name": "orders",
  "partitions": 12,
  "retention": {"time": "7d", "bytes": null},
  "cleanup_policy": "delete",
  "schema": {"type": "avro", "subject": "orders-value", "compatibility": "BACKWARD"}
}
```

The partition count can be increased later but never decreased, because doing so would reorder keys. Choose a count that supports your peak consumer parallelism; each partition is consumed by at most one member of a consumer group. Stream names must match `^[a-z][a-z0-9-]{2,62}$`.

Response `201 Created`:

```json
{
  "name": "orders",
  "partitions": 12,
  "replication_factor": 3,
  "created_at": "2026-03-14T09:26:53Z"
}
```

## Publish a batch

`POST /streams/{name}/batches`

```json
{
  "compression": "zstd",
  "messages": [
    {"key": "customer-42", "value": "eyJ0b3RhbCI6IDk5LjV9", "headers": {"source": "web"}},
    {"key": "customer-43", "value": "eyJ0b3RhbCI6IDEyLjB9"}
  ]
}
```

Values are base64-encoded bytes. A batch may contain at most 1,000 messages and 16 MiB after compression. Messages with the same key always land in the same partition, which preserves per-key ordering. The response returns the partition and offset of every message in request order.

## Fetch messages

`GET /streams/{name}/partitions/{p}/messages?offset=1200&max_messages=500&wait_ms=20000`

Fetch is a long-poll: if no messages are available at `offset`, the request waits up to `wait_ms` (maximum 30000) for new data before returning an empty list. Use the `next_offset` value from the response as the `offset` of the next request.

## Pagination

List endpoints return at most 100 items per page and a `next_cursor` field. Pass it as `?cursor=` to fetch the next page. Cursors expire after 15 minutes.

## Versioning

The API version is part of the path. Breaking changes ship only in a new major version, and each major version is supported for at least 18 months after its successor becomes generally available. Additive changes such as new optional fields can appear at any time, so clients must ignore unknown fields in responses.
