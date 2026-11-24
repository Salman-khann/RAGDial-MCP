# Architecture Overview

Halyard is a partitioned, replicated log. Producers append records to streams; each stream is divided into partitions, and each partition is an ordered, immutable sequence of records stored as segment files on the brokers that host its replicas. Consumers read partitions sequentially and track their position with offsets, which makes replay a first-class operation rather than an afterthought.

## Control plane

The control plane is a Raft quorum of three or five controller nodes. Controllers own the cluster metadata: the list of brokers, the assignment of partition replicas to brokers, the identity of each partition leader, ACLs and quotas. Metadata changes are written to an internal metadata log, and every broker follows that log to keep a local, eventually consistent view of the cluster. Because brokers never query controllers on the request path, a controller failover does not interrupt produce or fetch traffic; it only delays administrative operations such as creating a stream until a new controller leader is elected, which typically takes under two seconds.

## Data plane

Each partition has exactly one leader replica at a time. Producers send records to the leader, which appends them to its local log and serves them to followers that fetch continuously. A record is considered committed once every replica in the in-sync replica set has persisted it. The in-sync set shrinks when a follower falls behind by more than `replication.lag_max_ms` and grows again when it catches up. Producers using `acks=all` receive an acknowledgment only after the record is committed, which guarantees that an acknowledged write survives the loss of any minority of replicas.

Leadership is balanced across brokers so that no single machine handles a disproportionate share of writes. When a leader fails, the controller picks a new leader from the in-sync set. Unclean leader election, which allows an out-of-sync replica to become leader, is disabled by default because it can silently discard acknowledged records.

## Storage engine

Segments are append-only files with a sparse offset index and a time index beside them. Reads are served with zero-copy transfers from the operating system page cache whenever the data is hot, which is why brokers deliberately leave most of their memory to the kernel instead of a large JVM-style heap. Closed segments are immutable, so they can be offloaded to object storage by the tiered storage subsystem without coordination; the broker keeps only the most recent segments locally and fetches older ones on demand when a consumer replays history.

Log compaction is available for keyed streams. The compactor retains at least the latest record for every key and removes older records with the same key, which turns a stream into a durable changelog for a table. Tombstones, which are records with a null value, mark a key for deletion and are themselves removed after the configured delete retention.

## Consumer groups

Consumers that share a group name divide the partitions of a stream among themselves so that each partition is processed by exactly one member. A group coordinator, which is simply the broker leading the group's offset partition, tracks membership and triggers a rebalance when members join or leave. With the cooperative protocol a rebalance happens in two short phases and only the partitions that change owner are paused, which keeps processing latency stable during deploys. Committed offsets are stored in an internal compacted stream so that they are replicated with the same guarantees as user data.

## Delivery semantics

By default Halyard provides at-least-once delivery. Exactly-once processing within Halyard is achieved by combining idempotent producers with transactions: a transactional producer writes output records and the consumer's offset commits atomically, so a crash in the middle of a read-process-write loop neither loses nor duplicates results. Consumers configured with `isolation_level=read_committed` skip records from aborted transactions.
