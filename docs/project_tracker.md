# Project Tracker: Ticket Classifier Observability Hub

This document tracks the implementation progress of each phase and step in the project.

**Status Legend:**
- ⏳ Not Started
- 🚧 In Progress
- ✅ Completed
- ⚠️ Blocked
- 📝 Needs Review

---

## Phase 1: Core FastAPI & Dockerized Inference

**Status:** ⏳ Not Started  
**Target Skills:** API development, containerization, contract testing

### Steps:

- [ ] **1.1 Setup FastAPI Project Structure**
  - Create `api/` directory with proper structure
  - Initialize FastAPI application with basic configuration
  - Set up logging and error handling
  - Status: ⏳

- [ ] **1.2 Load Model and Tokenizer**
  - Implement model loading from `models/best_model/`
  - Load label encoder for predictions
  - Add model initialization on startup
  - Status: ⏳

- [ ] **1.3 Implement `/predict` Endpoint**
  - Create Pydantic models for request/response
  - Implement prediction logic
  - Add input validation (text length, format)
  - Return prediction + confidence score
  - Status: ⏳

- [ ] **1.4 Implement `/validate` Endpoint**
  - Create contract test validation logic
  - Validate input schema (text, max length, etc.)
  - Validate output schema (category, confidence, etc.)
  - Add comprehensive error messages
  - Status: ⏳

- [ ] **1.5 Add Health Check Endpoint**
  - Implement `/health` endpoint
  - Check model loading status
  - Return service readiness
  - Status: ⏳

- [ ] **1.6 Create Dockerfile**
  - Base image selection (Python slim)
  - Install dependencies from requirements.txt
  - Copy model files and application code
  - Expose port 8000
  - Set CMD for FastAPI server
  - Status: ⏳

- [ ] **1.7 Create Docker Compose (Optional)**
  - For local development
  - Configure volume mounts
  - Set environment variables
  - Status: ⏳

- [ ] **1.8 Test Docker Build**
  - Build Docker image locally
  - Run container and test endpoints
  - Verify model predictions work
  - Status: ⏳

---

## Phase 2: Kafka for Real-time Monitoring

**Status:** ⏳ Not Started  
**Target Skills:** Distributed systems, event-driven architecture, message queues

### Steps:

- [ ] **2.1 Setup Kafka Environment**
  - Install Kafka locally or set up managed service
  - Create Kafka topic for predictions
  - Configure topic settings (partitions, replication)
  - Status: ⏳

- [ ] **2.2 Design Event Schema**
  - Define prediction event structure (JSON schema)
  - Include: input_text, predicted_category, confidence, timestamp, request_id
  - Document schema versioning strategy
  - Status: ⏳

- [ ] **2.3 Implement Kafka Producer**
  - Add Kafka Python client dependency
  - Create producer service/class
  - Implement async event publishing
  - Add error handling and retries
  - Status: ⏳

- [ ] **2.4 Integrate Producer with FastAPI**
  - Modify `/predict` endpoint to publish events
  - Ensure non-blocking async publishing
  - Add producer initialization on startup
  - Handle producer errors gracefully
  - Status: ⏳

- [ ] **2.5 Test Event Streaming**
  - Send test predictions through API
  - Verify events appear in Kafka topic
  - Check event schema and data integrity
  - Measure latency impact
  - Status: ⏳

- [ ] **2.6 Add Kafka Consumer (Optional - for testing)**
  - Create simple consumer to verify events
  - Log events for debugging
  - Status: ⏳

---

## Phase 3: Snowflake Data Lake & Drift Analysis

**Status:** ⏳ Not Started  
**Target Skills:** Data engineering, data warehousing, SQL, monitoring

### Steps:

- [ ] **3.1 Setup Snowflake Account**
  - Create Snowflake account (free tier or trial)
  - Set up database and schema
  - Create service account/user
  - Configure connection credentials
  - Status: ⏳

- [ ] **3.2 Design Snowflake Schema**
  - Create predictions table schema
  - Define columns: id, input_text, predicted_category, confidence, timestamp, metadata
  - Add indexes/partitions for efficient querying
  - Status: ⏳

- [ ] **3.3 Setup Kafka-Snowflake Connector**
  - Install and configure Kafka Connect
  - Set up Snowflake connector (or use Snowpipe)
  - Configure connection parameters
  - Test data flow from Kafka to Snowflake
  - Status: ⏳

- [ ] **3.4 Create Drift Detection Script**
  - Write Python script to query Snowflake
  - Calculate category distribution over time windows
  - Implement baseline distribution calculation
  - Calculate drift metrics (percentage changes)
  - Status: ⏳

- [ ] **3.5 Implement Drift Alerting**
  - Define drift thresholds (e.g., 20% shift)
  - Add alerting logic (email, Slack, or logging)
  - Create scheduled job (cron or task scheduler)
  - Status: ⏳

- [ ] **3.6 Build Monitoring Dashboard (Optional)**
  - Use Streamlit or similar tool
  - Visualize prediction distributions over time
  - Show drift metrics and trends
  - Display recent predictions
  - Status: ⏳

---

## Phase 4: LLM-as-a-Judge Quality System

**Status:** ⏳ Not Started  
**Target Skills:** Model evaluation, QA for AI, API integration, test strategy

### Steps:

- [ ] **4.1 Setup LLM API Integration**
  - Choose LLM provider (OpenAI GPT-4o or Anthropic Claude)
  - Set up API keys and authentication
  - Install API client libraries
  - Test API connectivity
  - Status: ⏳

- [ ] **4.2 Design Judge Prompt**
  - Create prompt template for classification task
  - Include ticket text and available categories
  - Design output format (JSON)
  - Test prompt effectiveness
  - Status: ⏳

- [ ] **4.3 Implement Sampling Strategy**
  - Create function to sample 1% of predictions
  - Query Snowflake for recent predictions
  - Implement sampling logic (random or stratified)
  - Status: ⏳

- [ ] **4.4 Build Judge Service**
  - Create service class for LLM evaluation
  - Implement async API calls to LLM
  - Parse LLM responses
  - Handle API errors and retries
  - Status: ⏳

- [ ] **4.5 Implement Metrics Calculation**
  - Calculate Precision (Judge agreement on positive predictions)
  - Calculate Recall (Judge agreement on all relevant cases)
  - Calculate Accuracy (Overall agreement rate)
  - Calculate F1-score
  - Store metrics in Snowflake
  - Status: ⏳

- [ ] **4.6 Create Evaluation Scheduler**
  - Set up scheduled job (daily or hourly)
  - Run sampling and evaluation automatically
  - Log results and metrics
  - Status: ⏳

- [ ] **4.7 Build Evaluation Dashboard (Optional)**
  - Visualize precision, recall, accuracy over time
  - Show confusion matrix (Judge vs. DistilBERT)
  - Display sample disagreements
  - Status: ⏳

---

## Phase 5: Terraform Infrastructure as Code

**Status:** ⏳ Not Started  
**Target Skills:** DevOps, IaC, cloud architecture, automation

### Steps:

- [ ] **5.1 Setup Terraform Environment**
  - Install Terraform CLI
  - Choose cloud provider (AWS recommended)
  - Configure AWS credentials
  - Create Terraform project structure
  - Status: ⏳

- [ ] **5.2 Design Infrastructure Architecture**
  - Define EC2 instance requirements
  - Plan VPC, subnets, security groups
  - Design network architecture
  - Document resource dependencies
  - Status: ⏳

- [ ] **5.3 Create EC2 Configuration**
  - Write Terraform for EC2 instance
  - Configure instance type and AMI
  - Set up user data script (Docker installation)
  - Add security group rules (port 8000)
  - Status: ⏳

- [ ] **5.4 Create Networking Configuration**
  - Define VPC and subnets
  - Configure security groups
  - Set up internet gateway (if needed)
  - Status: ⏳

- [ ] **5.5 Add Kafka Infrastructure (Optional)**
  - Terraform for managed Kafka (MSK) or EC2-based
  - Configure Kafka networking
  - Status: ⏳

- [ ] **5.6 Create Variables and Outputs**
  - Define Terraform variables for configuration
  - Create outputs for important resource IDs
  - Add variable validation
  - Status: ⏳

- [ ] **5.7 Test Terraform Deployment**
  - Run `terraform init`
  - Run `terraform plan` and review
  - Run `terraform apply` (in test account)
  - Verify resources created correctly
  - Status: ⏳

- [ ] **5.8 Document Deployment Process**
  - Write deployment instructions
  - Document required variables
  - Add cleanup instructions (`terraform destroy`)
  - Status: ⏳

---

## Phase 6: Integration & Testing

**Status:** ⏳ Not Started  
**Target Skills:** System integration, end-to-end testing, production readiness

### Steps:

- [ ] **6.1 End-to-End Integration Test**
  - Test full flow: API → Kafka → Snowflake
  - Verify data integrity throughout pipeline
  - Test error scenarios
  - Status: ⏳

- [ ] **6.2 Performance Testing**
  - Load test API endpoint
  - Measure latency and throughput
  - Test Kafka producer performance
  - Status: ⏳

- [ ] **6.3 Documentation**
  - Write API documentation (OpenAPI/Swagger)
  - Document setup and deployment process
  - Create architecture diagrams
  - Status: ⏳

- [ ] **6.4 Production Readiness Review**
  - Security review (credentials, network)
  - Error handling review
  - Logging and monitoring review
  - Cost optimization review
  - Status: ⏳

---

## Notes & Blockers

### Current Blockers:
_None yet_

### Key Decisions Needed:
- [ ] Choose LLM provider (OpenAI vs Anthropic)
- [ ] Choose Kafka deployment (local, managed, or EC2)
- [ ] Choose cloud provider (AWS, Azure, GCP)
- [ ] Determine Snowflake tier/account type

### Lessons Learned:
_Add insights and learnings as you progress_

---

## Timeline Estimates

- **Phase 1:** 1-2 weeks
- **Phase 2:** 1 week
- **Phase 3:** 1-2 weeks
- **Phase 4:** 1-2 weeks
- **Phase 5:** 1 week
- **Phase 6:** 1 week

**Total Estimated Time:** 6-9 weeks (part-time)

---

## Next Steps

1. Start with Phase 1, Step 1.1: Setup FastAPI Project Structure
2. Set up development environment and dependencies
3. Review existing model structure and training code
