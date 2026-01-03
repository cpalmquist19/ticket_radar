# Ticket Classifier Observability Hub - Project Plan

## Project Goal

Transition from Senior SDET at $105k to an AI-focused role at $150k+ by building **production-grade AI infrastructure** that demonstrates expertise in Model Quality Assurance, MLOps, and AI integration.

This project bridges SDET expertise with AI Developer and MLOps requirements through an **Automated Model Observability & Quality Pipeline**—a system that monitors itself, detects failures, and integrates with a professional data stack.

---

## Project Overview

The **Ticket Classifier Observability Hub** is not just a model behind an API. It's a complete system that:
- Serves model predictions via a production-grade API
- Streams predictions to a data warehouse in real-time
- Monitors for model drift and performance degradation
- Uses LLM-as-a-Judge for continuous quality assessment
- Provisions infrastructure using Infrastructure as Code (IaC)

---

## Architecture Components

### 1. The Core: FastAPI & Dockerized Inference

**Objective:** Wrap the DistilBERT model in a production-grade API with validation.

**Actions:**
- Build a **FastAPI** service with `/predict` endpoint
- Implement a `/validate` endpoint that runs contract tests using **Pydantic**
- Validate input/output schemas, data types, and constraints
- Containerize the service using **Docker**
- Ensure the model runs identically in test and production environments

**Skills Demonstrated:** API development, containerization, contract testing, production deployment patterns

---

### 2. The Stream: Kafka for Real-time Monitoring

**Objective:** Implement high-throughput event streaming for prediction tracking.

**Actions:**
- Set up **Kafka** broker and topics
- Fire events to Kafka topic after each prediction containing:
  - Input text
  - Predicted category
  - Confidence score
  - Timestamp
  - Request metadata
- Implement async event publishing to avoid blocking API responses
- Design event schema for scalability and backward compatibility

**Skills Demonstrated:** Distributed systems, event-driven architecture, high-throughput systems, message queues

---

### 3. The Data Lake: Snowflake for Drift Analysis

**Objective:** Store prediction data for long-term analysis and drift detection.

**Actions:**
- Configure Kafka-to-Snowflake connector
- Stream prediction events into **Snowflake** tables
- Design schema for efficient querying and analysis
- Build Python scripts (or Streamlit dashboard) to:
  - Query historical predictions
  - Calculate prediction distribution over time
  - Detect **Model Drift** (e.g., category distribution shifts)
  - Generate drift alerts when thresholds are exceeded

**Skills Demonstrated:** Data engineering, data warehousing, SQL, time-series analysis, monitoring dashboards

---

### 4. The "SDET Special": LLM-as-a-Judge for Quality

**Objective:** Apply QA principles to AI model evaluation using LLM-based validation.

**Actions:**
- Set up integration with LLM APIs (GPT-4o or Claude)
- Implement sampling strategy (1% of predictions)
- Build "Judge" service that:
  - Receives ticket text and DistilBERT prediction
  - Queries LLM for its classification
  - Compares outputs to calculate metrics
- Calculate real-time metrics:
  - **Precision** (Judge agreement on positive predictions)
  - **Recall** (Judge agreement on all relevant cases)
  - **Accuracy** (Overall agreement rate)
- Store evaluation results in Snowflake for trending

**Skills Demonstrated:** Model evaluation, QA for AI systems, API integration, statistical analysis, test strategy design

---

### 5. The "Glue": Terraform (Infrastructure as Code)

**Objective:** Provision infrastructure programmatically for reproducibility and scalability.

**Actions:**
- Write **Terraform** configurations for:
  - EC2 instances for Docker containers
  - Networking (VPC, subnets, security groups)
  - Kafka cluster setup (or managed Kafka service)
  - Snowflake resource provisioning (if applicable)
- Implement environment-specific configurations (dev/staging/prod)
- Document infrastructure dependencies and setup

**Skills Demonstrated:** DevOps, Infrastructure as Code, cloud architecture, automation, reproducibility

---

## Skills Demonstrated Summary

| Skill Category | Project Component | Industry Relevance |
|----------------|-------------------|-------------------|
| **AI Development** | Fine-tuned DistilBERT inference with FastAPI | Model deployment, API design |
| **MLOps** | Dockerization and Model Drift monitoring | Production ML systems, observability |
| **Data Engineering** | Real-time streaming with Kafka into Snowflake | High-scale data pipelines |
| **SDET / QA** | "LLM-as-a-Judge" evaluation framework | Model quality assurance, testing |
| **DevOps** | Infrastructure as Code (IaC) via Terraform | Cloud infrastructure, automation |

---

## Success Criteria

1. **FastAPI Service:** Model inference via REST API with input/output validation
2. **Docker Deployment:** Containerized service running consistently across environments
3. **Event Streaming:** All predictions streamed to Kafka with <100ms latency impact
4. **Data Warehouse Integration:** Predictions stored in Snowflake within 1 minute of inference
5. **Drift Detection:** Automated alerts when category distribution shifts >20% from baseline
6. **Quality Monitoring:** LLM Judge evaluations running on 1% sample with metrics tracked
7. **Infrastructure as Code:** All infrastructure provisioned via Terraform with minimal manual steps

---

## Resources & Learning

- [MLOps for Beginners: Deploying Models with FastAPI and Docker](https://www.youtube.com/watch?v=Pc5kdZygCj0) - Essential MLOps roadmap for 2025
- FastAPI documentation: https://fastapi.tiangolo.com/
- Kafka documentation: https://kafka.apache.org/documentation/
- Snowflake documentation: https://docs.snowflake.com/
- Terraform AWS provider: https://registry.terraform.io/providers/hashicorp/aws/latest/docs

---

## Project Philosophy

This project leverages your **12+ years of SDET experience** in framework design, test strategy, and quality assurance—applying these skills to AI systems. The "LLM-as-a-Judge" component is a unique differentiator that demonstrates how traditional QA expertise translates to modern AI infrastructure.

By building a system that monitors itself, validates its outputs, and integrates with enterprise data stacks, you demonstrate the ability to build production-grade AI systems—exactly what $150k+ AI Developer and MLOps roles require.
