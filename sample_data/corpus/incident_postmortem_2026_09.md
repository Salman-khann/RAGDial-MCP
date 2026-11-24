# Postmortem: eu-west-1 Produce Latency Incident (2026-09-17)

## Summary

On 17 September 2026 between 13:42 and 15:06 UTC, customers in eu-west-1 experienced elevated produce latency and intermittent `HY-5032` errors. At the peak, p99 produce latency reached 4.8 seconds against a normal baseline of 35 milliseconds, and 2.1% of produce requests with `acks=all` failed. No acknowledged data was lost.

## Timeline

| Time (UTC) | Event |
|---|---|
| 13:30 | Scheduled reassignment starts moving 1,800 partitions to new brokers in zone c. |
| 13:42 | Produce p99 latency alert fires for eu-west-1. |
| 13:47 | On-call engineer acknowledges; initial suspicion is a noisy tenant. |
| 14:05 | Under-replicated partitions climb to 640; `HY-5032` errors begin. |
| 14:18 | Team identifies reassignment traffic saturating inter-zone network links. |
| 14:22 | Reassignment throttle lowered from unlimited to 200MiB per second. |
| 14:51 | Under-replicated partitions drop below 50. |
| 15:06 | Latency back to baseline; incident resolved. |

## Root cause

The reassignment was started without a replication throttle because the operator runbook for the new zone omitted the step. Catch-up replication consumed the full inter-zone bandwidth, followers fell behind by more than `replication.lag_max_ms`, and they were removed from the in-sync set. Partitions whose in-sync set dropped below `replication.min_insync` refused `acks=all` writes with `HY-5032` until followers caught up.

## What went well

- Durability guarantees held: writes were refused rather than acknowledged without enough replicas.
- The under-replicated partitions dashboard made the cause visible within minutes once the team looked at it.

## What went wrong

- The latency alert did not link to the replication dashboard, which delayed diagnosis by about 30 minutes.
- Reassignments of more than 500 partitions were allowed without a throttle.

## Action items

1. Make a reassignment throttle mandatory for any reassignment touching more than 100 partitions. Owner: platform team. Status: done.
2. Add replication health panels to the produce latency alert. Owner: observability team. Status: in progress.
3. Update the zone expansion runbook to include the throttle step. Owner: SRE. Status: done.
