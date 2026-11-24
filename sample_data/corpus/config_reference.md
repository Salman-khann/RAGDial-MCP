# Halyard Broker Configuration Reference

This page lists every key accepted in `halyard.toml`. Keys are grouped by subsystem. Unless noted otherwise, a change takes effect after a rolling restart of the broker fleet. Keys marked *dynamic* can be changed at runtime with `halyardctl config set` and are applied within one heartbeat interval.

Values with units accept the suffixes `ms`, `s`, `m`, `h` for durations and `KiB`, `MiB`, `GiB` for sizes. A bare integer is interpreted as milliseconds or bytes respectively.

## Broker keys

| Key | Type | Default | Description |
|---|---|---|---|
| `broker.id` | int | auto | Unique broker identifier. Assigned by the controller when left unset. |
| `broker.rack` | string | "" | Rack or availability-zone label used for replica placement. |
| `broker.listeners` | list | `["plain://0.0.0.0:7400"]` | Listener URIs. Supported schemes are `plain`, `tls` and `mtls`. |
| `broker.advertised_listeners` | list | same as listeners | Addresses returned to clients in metadata responses. |
| `broker.num_io_threads` | int | 8 | Threads that read and write segment files. |
| `broker.num_network_threads` | int | 3 | Threads that accept and parse client requests. |
| `broker.max_connections` | int | 20000 | Hard limit on concurrent client connections per broker. |
| `broker.max_connections_per_ip` | int | 1000 | Limit per client IP; protects against a single runaway producer. |
| `broker.request_timeout` | duration | 30s | Maximum time a request may wait in the purgatory queue. *dynamic* |
| `broker.shutdown_drain_timeout` | duration | 45s | Time allowed to drain in-flight produce requests on SIGTERM. |
| `broker.heartbeat_interval` | duration | 3s | Interval between broker-to-controller heartbeats. |
| `broker.session_timeout` | duration | 18s | Controller fences a broker that misses heartbeats for this long. |

## Storage keys

| Key | Type | Default | Description |
|---|---|---|---|
| `storage.data_dirs` | list | `["/var/lib/halyard"]` | Directories that hold partition segments. Spread across disks for throughput. |
| `storage.segment_bytes` | size | 512MiB | Size at which the active segment is rolled. |
| `storage.segment_ms` | duration | 7d | Age at which the active segment is rolled even if not full. |
| `storage.index_interval_bytes` | size | 4KiB | Bytes between sparse offset-index entries. |
| `storage.flush_interval_messages` | int | unset | Force an fsync after this many messages. Leave unset to rely on replication. |
| `storage.retention_ms` | duration | 7d | Default retention for streams without an explicit policy. *dynamic* |
| `storage.retention_bytes` | size | unset | Per-partition size cap. Oldest segments are deleted first. *dynamic* |
| `storage.compaction_threads` | int | 2 | Threads used by the log compactor for keyed streams. |
| `storage.compaction_min_dirty_ratio` | float | 0.5 | Minimum ratio of dirty bytes before a partition is compacted. |
| `storage.tiered.enabled` | bool | false | Offload closed segments to object storage. |
| `storage.tiered.bucket` | string | "" | Bucket URI, for example `s3://halyard-archive/prod`. |
| `storage.tiered.local_retention_ms` | duration | 6h | How long offloaded segments stay on local disk as a read cache. |
| `storage.tiered.upload_concurrency` | int | 4 | Parallel segment uploads per broker. |

## Replication keys

| Key | Type | Default | Description |
|---|---|---|---|
| `replication.factor` | int | 3 | Default replica count for new streams. |
| `replication.min_insync` | int | 2 | Minimum in-sync replicas required to acknowledge a write with `acks=all`. |
| `replication.lag_max_ms` | duration | 10s | A follower that lags longer than this is removed from the in-sync set. |
| `replication.fetch_max_bytes` | size | 8MiB | Maximum bytes a follower fetches per request. |
| `replication.fetchers` | int | 2 | Fetcher threads per source broker. |
| `replication.unclean_leader_election` | bool | false | Allow an out-of-sync replica to become leader. Trades durability for availability. |
| `replication.throttle_bytes_per_sec` | size | unset | Caps reassignment traffic so rebalances do not starve producers. *dynamic* |

## Consumer group keys

| Key | Type | Default | Description |
|---|---|---|---|
| `groups.rebalance_strategy` | enum | `cooperative` | Either `eager` or `cooperative`. Cooperative rebalancing only moves the partitions that change owner. |
| `groups.session_timeout` | duration | 45s | A consumer is evicted from its group after this long without a heartbeat. |
| `groups.max_poll_interval` | duration | 5m | Maximum gap between polls before the consumer is considered stuck. |
| `groups.offsets_retention` | duration | 14d | Committed offsets of an empty group are deleted after this period. |
| `groups.initial_rebalance_delay` | duration | 3s | Delay before the first rebalance of a new group so that members can join together. |

## Example

```toml
[broker]
rack = "us-east-1b"
listeners = ["mtls://0.0.0.0:7401"]
num_io_threads = 16

[storage]
data_dirs = ["/mnt/nvme0/halyard", "/mnt/nvme1/halyard"]
retention_ms = "3d"

[storage.tiered]
enabled = true
bucket = "s3://halyard-archive/prod"
local_retention_ms = "12h"

[replication]
factor = 3
min_insync = 2
```

## Validation

The broker refuses to start if `replication.min_insync` is greater than `replication.factor`, or if two entries in `storage.data_dirs` resolve to the same filesystem. Unknown keys are logged at WARN level and ignored so that a rolling upgrade can introduce new keys before every broker understands them.
