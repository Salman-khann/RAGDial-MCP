# Changelog

## 3.5.0 (2026-10-02)

### Added
- Cooperative rebalancing is now the default `groups.rebalance_strategy` for new clusters.
- `halyardctl upgrade preflight` checks for renamed configuration keys and incompatible client versions before an upgrade.
- Tiered storage supports Azure Blob Storage in addition to S3 and GCS.

### Changed
- The default `storage.segment_bytes` was reduced from 1GiB to 512MiB to speed up tiered offload.
- Fetch long-poll maximum raised from 10 seconds to 30 seconds.

### Fixed
- A race in leader election could briefly expose uncommitted records to `read_committed` consumers.
- `publish_batch` no longer double-counts rate-limit tokens for batches of exactly 100 messages.

## 3.4.2 (2026-08-19)

### Fixed
- Brokers could leak file descriptors when a tiered-storage upload was cancelled during shutdown.
- `groups/{group}/lag` returned negative lag for partitions that had been truncated by retention.

### Security
- Upgraded the TLS library to address a denial-of-service issue in session resumption.

## 3.4.0 (2026-07-07)

### Added
- Schema registry with Avro, Protobuf and JSON Schema support and per-subject compatibility modes.
- Customer-managed encryption keys for Business plans.
- `replication.throttle_bytes_per_sec` can now be changed at runtime.

### Deprecated
- The `eager` rebalance strategy is deprecated and will be removed in 4.0.

## 3.3.0 (2026-05-12)

### Added
- Idempotent producers in all official SDKs.
- Audit-log export to customer-owned buckets.

### Changed
- API keys rotated through `/keys/{key_id}/rotate` now remain valid for 24 hours instead of 1 hour, giving deploys time to pick up the new secret.

## 3.0.0 (2026-03-03)

### Breaking
- API v3 released. See the migration guide for the full list of changes from v2.
- Minimum supported Kubernetes version raised to 1.29 for the Helm chart.
- Removed the deprecated `storage.legacy_index` key.
