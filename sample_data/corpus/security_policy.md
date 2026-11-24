# Security and Compliance

This document describes how Halyard Cloud protects customer data and which controls customers are responsible for under the shared responsibility model.

## Encryption

All data is encrypted in transit with TLS 1.2 or 1.3. Connections that offer older protocol versions are refused during the handshake. At rest, segment files and tiered-storage objects are encrypted with AES-256-GCM using per-project data keys. Data keys are wrapped by a key encryption key held in the cloud provider's key management service and are rotated every 90 days. Business and Enterprise customers may supply their own key encryption key; revoking it renders the project's data unreadable within 15 minutes, which is the intended behaviour for a cryptographic shred.

## Authentication and access control

- API keys are scoped to a single project and carry an explicit list of scopes such as `streams:write` or `groups:read`.
- Keys are shown once at creation. Halyard stores only a salted hash and cannot recover a lost key.
- Console users authenticate through SAML or OIDC single sign-on; password login can be disabled per organisation.
- Fine-grained ACLs restrict a key to specific streams or name prefixes, for example `orders-*`.
- Mutual TLS can be required per listener, in which case both a valid client certificate and an API key are needed.

## Network controls

Projects can restrict access to a list of CIDR ranges. Requests from other addresses receive error `HY-4032`. Private networking via VPC peering or private endpoints is available on Growth plans and above, and once enabled the public endpoint can be switched off entirely.

## Audit logging

Every control-plane action, including stream creation, ACL changes, key rotation and quota requests, emits an audit event with the actor, source IP, timestamp and request ID. Data-plane reads and writes are not individually audited because of their volume; aggregated access metrics are available instead. Audit events are immutable and can be exported through the audit-log endpoint or streamed to a customer-owned bucket.

## Vulnerability management

Production images are rebuilt weekly from patched base images. Critical vulnerabilities with a known exploit are patched within 72 hours, high severity within 14 days, and medium severity within 30 days. An independent firm performs a penetration test every year, and the executive summary is available under NDA.

## Compliance

Halyard Cloud holds SOC 2 Type II and ISO 27001 certifications. The Business and Enterprise plans support HIPAA workloads under a signed business associate agreement. Customer data never leaves the region selected at project creation, except when the customer configures cross-region replication.

## Reporting a vulnerability

Report security issues to the security team through the responsible disclosure form. We acknowledge reports within two business days and do not pursue legal action against researchers who act in good faith and avoid privacy violations or service disruption.
