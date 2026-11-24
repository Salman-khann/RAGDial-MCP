# Pricing and Plans

Halyard Cloud is billed monthly in arrears. Each plan has a base fee that includes a monthly allowance of ingress, storage and partitions; usage above the allowance is billed at the overage rates below. Prices are in US dollars and exclude taxes.

## Plan comparison

| Feature | Developer | Starter | Growth | Business | Enterprise |
|---|---|---|---|---|---|
| Base fee per month | $0 | $49 | $399 | $1,990 | contact sales |
| Included ingress | 5 GB | 200 GB | 2 TB | 15 TB | custom |
| Ingress overage per GB | n/a | $0.08 | $0.06 | $0.04 | custom |
| Included storage | 1 GB | 100 GB | 1 TB | 10 TB | custom |
| Storage overage per GB-month | n/a | $0.10 | $0.08 | $0.05 | custom |
| Tiered storage per GB-month | n/a | n/a | $0.023 | $0.02 | custom |
| Included partition-hours | 36,000 | 360,000 | 1,440,000 | 7,200,000 | custom |
| Partition-hour overage | n/a | $0.0015 | $0.0012 | $0.0008 | custom |
| Uptime SLA | none | 99.5% | 99.9% | 99.95% | 99.99% |
| Support response (severity 1) | community | 1 business day | 4 hours | 1 hour | 15 minutes |
| Private networking | no | no | yes | yes | yes |
| Customer-managed encryption keys | no | no | no | yes | yes |
| Audit log retention | none | 7 days | 30 days | 1 year | 7 years |

Egress to consumers in the same cloud region is free. Cross-region and internet egress are billed at $0.09 per GB on all paid plans.

## Partition-hours

A partition-hour is one partition existing for one hour, regardless of traffic. A stream with 12 partitions that exists for a 30-day month consumes 12 x 720 = 8,640 partition-hours. Over-partitioning is the most common cause of unexpected bills: idle partitions still cost money, and every partition also consumes broker memory for its indexes.

## SLA credits

If monthly uptime falls below the plan's SLA, customers receive service credits against the next invoice:

- below the SLA but at least 99.0%: **10%** credit of the monthly base fee
- below 99.0% but at least 95.0%: **25%** credit
- below 95.0%: **50%** credit

Credits must be requested within 30 days of the incident through the support portal. Downtime caused by exceeding quotas, scheduled maintenance announced at least 72 hours in advance, or customer misconfiguration does not count.

## Free tier limits

The Developer plan is free forever but suspends a project that is idle for 30 consecutive days. Data in a suspended project is retained for another 14 days, after which it is permanently deleted. The Developer plan cannot be used for production workloads under the terms of service.

## Changing plans

Upgrades take effect immediately and are prorated to the day. Downgrades take effect at the start of the next billing cycle and are rejected if the project currently exceeds a limit of the target plan, such as the maximum number of partitions.
