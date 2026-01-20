# Agent–Tool Gap Orchestrator

The Agent–Tool Gap Orchestrator is a policy-driven control layer designed to
prevent identity dilution in autonomous AI agent workflows. It enforces
runtime authorization and licensing checks before any tool, model, or service
invocation occurs.

This project was developed as part of the NYU ITP MetaML project (Summer 2025).

---

## Problem Statement

As AI agents autonomously invoke tools and services, the original user or
workload identity is often lost at execution time. This creates an
*Agent–Tool Gap*, where traditional RBAC and audit mechanisms fail, leading to
over-privileged or ungoverned execution.

---

## Solution Overview

This orchestrator introduces a centralized, policy-enforced execution layer
that:
- Preserves agent and workload identity across tool calls
- Applies **Licensing-as-Code** and authorization checks at runtime
- Prevents unauthorized or out-of-scope tool usage
- Provides a foundation for sovereign, auditable AI workflows

---

## Key Components

- **Orchestrator Service**
  - Intercepts agent tool requests
  - Evaluates policies before execution

- **Licensing & Security Layer**
  - Enforces usage constraints and permissions
  - Integrates with RBAC / policy engines

- **Service Integrations**
  - Business, data, taxonomy, and messaging services
  - Supports multi-language components (Python + Java)

---

## Repository Structure

```text
agent-tool-gap-orchestrator/
├── src/                     # Java-based components
├── security/                # Security and policy enforcement
├── app.py                   # Main application entry point
├── orchestrator_service.py  # Orchestration logic
├── pom.xml                  # Maven configuration
├── run_licensing.sh         # Licensing bootstrap script
└── README.md

