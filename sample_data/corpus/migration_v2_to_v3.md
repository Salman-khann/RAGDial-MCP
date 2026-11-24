# Migrating from API v2 to v3

API v3 became generally available in March 2026. API v2 is supported until **30 September 2027**; after that date v2 requests return HTTP 410. This guide lists every breaking change and shows how to update client code.

## Summary of breaking changes

1. Message values are base64-encoded in JSON bodies instead of being sent as UTF-8 strings.
2. Topic terminology was replaced by stream throughout the API.
3. Offsets are returned as strings to avoid precision loss in JavaScript clients.
4. Consumer group commits require the group generation number.
5. Error responses use the `HY-` code catalogue instead of numeric codes.

## Renamed fields and endpoints

| v2 name | v3 name | Notes |
|---|---|---|
| `/topics` | `/streams` | All topic endpoints moved. |
| `topic_name` | `name` | In stream create and describe bodies. |
| `num_partitions` | `partitions` | Same semantics. |
| `retention_hours` | `retention.time` | Now a duration string such as `"7d"`. |
| `max_bytes` | `retention.bytes` | `null` means unlimited. |
| `consumer_id` | `member_id` | Assigned by the coordinator, not the client. |
| `ack_mode` | `acks` | Values are `0`, `1` and `all`. |
| `error_code` (int) | `error.code` (string) | For example 409 became `HY-4091`. |

## Updating message encoding

In v2, values were sent as raw strings, which corrupted binary payloads. In v3 the value must be base64-encoded.

```diff
- {"key": "customer-42", "value": "{\"total\": 99.5}"}
+ {"key": "customer-42", "value": "eyJ0b3RhbCI6IDk5LjV9"}
```

The official SDKs encode values automatically. Only clients that call the REST API directly need to change.

## Committing offsets with a generation

v3 rejects commits from members of an older group generation to prevent zombie consumers from overwriting progress after a rebalance.

```python
# v2
client.commit(group="billing", offsets={0: 1200})

# v3
client.commit(group="billing", generation=assignment.generation, offsets={"0": "1200"})
```

A commit with a stale generation fails with `HY-4092`; the consumer should rejoin the group and retry.

## Recommended migration plan

1. Upgrade the SDK to 3.x in a staging environment. SDK 3.x still speaks v2 when constructed with `api_version="v2"`, so the upgrade can ship separately from the API switch.
2. Switch staging to `api_version="v3"` and run your integration tests. Pay attention to code that parses offsets as integers.
3. Roll out to production one service at a time. Producers and consumers on different API versions interoperate because the underlying streams are the same.
4. Monitor the `api_version` dimension in the usage dashboard until v2 traffic reaches zero.
