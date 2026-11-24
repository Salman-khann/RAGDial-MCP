# Self-Hosted Deployment Runbook

This runbook covers installing a production Halyard cluster on Kubernetes with the Helm chart, upgrading it, and recovering from the most common failures. It assumes Kubernetes 1.29 or newer and a storage class backed by local NVMe or provisioned-IOPS volumes.

## Prerequisites

1. A dedicated node pool for brokers, labelled `halyard.io/role=broker`, with at least three nodes in three zones.
2. A storage class with `volumeBindingMode: WaitForFirstConsumer` so volumes are created in the same zone as the pod.
3. Cluster sizing that matches the expected load:

   | Profile | Brokers | vCPU per broker | Memory per broker | Disk per broker | Sustained ingress |
   |---|---|---|---|---|---|
   | small | 3 | 4 | 16 GiB | 500 GiB | 50 MB/s |
   | medium | 5 | 8 | 32 GiB | 2 TiB | 200 MB/s |
   | large | 9 | 16 | 64 GiB | 4 TiB | 600 MB/s |
   | xlarge | 15 | 32 | 128 GiB | 8 TiB | 1.5 GB/s |

4. A TLS certificate issuer. The chart integrates with cert-manager and expects a `ClusterIssuer` named in `tls.issuerRef`.

## Installing

```bash
helm repo add halyard https://charts.halyard.io
helm repo update
kubectl create namespace halyard
helm install halyard halyard/halyard \
  --namespace halyard \
  --version 3.4.2 \
  --values values-prod.yaml \
  --wait --timeout 15m
```

After the release is ready, verify that every broker registered with the controller:

```bash
kubectl -n halyard exec deploy/halyard-toolbox -- halyardctl cluster describe
```

The output must list every broker with state `RUNNING` and zero offline partitions before you route traffic to the cluster.

## Rolling upgrades

Upgrades restart one broker at a time and wait for under-replicated partitions to return to zero before moving on.

1. Read the release notes for any configuration keys that were renamed.
2. Run the pre-flight check:

   ```bash
   halyardctl upgrade preflight --target 3.5.0
   ```

3. Raise the reassignment throttle so that catch-up replication finishes quickly but does not starve producers:

   | Cluster profile | Recommended `replication.throttle_bytes_per_sec` |
   |---|---|
   | small | 50MiB |
   | medium | 150MiB |
   | large | 400MiB |

4. Apply the upgrade with `helm upgrade` using the same values file and the new chart version.
5. Watch `halyard_under_replicated_partitions`. The upgrade controller pauses automatically if it stays above zero for more than 10 minutes.

## Recovering a failed broker

When a broker pod is stuck in `CrashLoopBackOff`, first inspect the logs for segment corruption messages. A broker that crashed during a write may have a truncated final segment; it recovers automatically by truncating to the last valid batch, which can take several minutes for large partitions.

If the underlying volume is lost, delete the PersistentVolumeClaim and the pod. The StatefulSet recreates the pod with an empty volume, and the broker re-replicates its partitions from the other replicas. Expect re-replication to run at the configured throttle rate; a 2 TiB broker at 150MiB per second takes roughly four hours.

## Backup and disaster recovery

- Enable tiered storage so that closed segments are durably stored in object storage independent of the cluster.
- Export cluster metadata (streams, ACLs, consumer group offsets) every hour with `halyardctl metadata export --to s3://...`.
- To restore into a new cluster, import the metadata first, then point tiered storage at the same bucket in read-only mode. Consumers resume from the exported offsets.
- Test the restore procedure at least once per quarter. An untested backup is assumed not to work.
