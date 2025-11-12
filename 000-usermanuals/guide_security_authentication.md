# Security & Authentication Guide for AI Agents
**Version:** 1.0
**Last Updated:** 2025-11-12
**Applies To:** Google ADK, Vertex AI Agent Engine, Cloud Run Agents

---

## Overview

This guide covers security best practices and authentication patterns for deploying AI agents in production environments. Based on Google Cloud's security model and enterprise requirements.

---

## Table of Contents

1. [Authentication Methods](#authentication-methods)
2. [Authorization & Access Control](#authorization--access-control)
3. [Data Security](#data-security)
4. [Network Security](#network-security)
5. [API Security](#api-security)
6. [Compliance & Governance](#compliance--governance)
7. [Security Checklist](#security-checklist)

---

## Authentication Methods

### 1. Service Account Authentication (Recommended for Production)

**Use Case:** Agent-to-agent communication, backend services

```python
from google.adk import Runner, Agent
from google.auth import default

# Use Application Default Credentials
credentials, project_id = default(
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Initialize agent with service account
agent = Agent(
    name="secure_agent",
    model="gemini-2.5-flash",
    # ... other config
)
```

**Best Practices:**
- Create dedicated service accounts for each agent
- Grant minimum required permissions (principle of least privilege)
- Rotate service account keys regularly
- Never commit service account keys to version control

**Service Account Setup:**
```bash
# Create service account
gcloud iam service-accounts create agent-service-account \
    --display-name="Agent Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:agent-service-account@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

---

### 2. IAM-Based Authentication

**Use Case:** Internal Google Cloud communication

```python
# Cloud Run service with IAM authentication
from google.auth.transport.requests import Request
from google.auth import default

def get_bearer_token():
    """Fetch Google Cloud bearer token using Application Default Credentials."""
    credentials, project = default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    request = Request()
    credentials.refresh(request)
    return credentials.token

# Use token in requests
headers = {
    'Authorization': f'Bearer {get_bearer_token()}',
    'Content-Type': 'application/json'
}
```

**Best Practices:**
- Use IAM roles for fine-grained access control
- Enable audit logging for all IAM changes
- Implement conditional access policies

---

### 3. API Key Authentication

**Use Case:** External client access, rate limiting

```python
import os

# Store API key securely in Secret Manager
from google.cloud import secretmanager

def get_api_key():
    """Retrieve API key from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/agent-api-key/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode('UTF-8')

# Validate API key in agent executor
class SecureAgentExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        # Validate API key from request headers
        api_key = context.request.headers.get('X-API-Key')
        if not self._validate_api_key(api_key):
            raise UnauthorizedError("Invalid API key")

        # Proceed with execution
        # ...
```

**Best Practices:**
- Never hardcode API keys
- Use Secret Manager for key storage
- Implement key rotation policies
- Rate limit API key usage

---

### 4. OAuth 2.0 / OIDC

**Use Case:** User-facing applications, third-party integrations

```python
# OAuth 2.0 configuration for Cloud Run
import google.auth.transport.requests
from google.oauth2 import id_token

def verify_oauth_token(token):
    """Verify OAuth token from client."""
    try:
        # Verify the token
        request = google.auth.transport.requests.Request()
        id_info = id_token.verify_oauth2_token(
            token, request, AUDIENCE
        )

        # Extract user info
        user_id = id_info['sub']
        email = id_info.get('email')

        return user_id, email
    except ValueError:
        raise UnauthorizedError("Invalid token")
```

**Best Practices:**
- Use industry-standard OAuth 2.0 flows
- Validate tokens on every request
- Implement token refresh mechanisms
- Use HTTPS for all OAuth endpoints

---

## Authorization & Access Control

### Role-Based Access Control (RBAC)

**Agent Permissions Matrix:**

| Role | Permissions | Use Case |
|------|------------|----------|
| `agent.user` | Query agent, view results | End users |
| `agent.admin` | Deploy, update, delete agents | DevOps |
| `agent.developer` | Test locally, view logs | Development |
| `agent.viewer` | View agent card, metrics | Monitoring |

**Implementation:**
```python
from enum import Enum

class AgentRole(Enum):
    USER = "agent.user"
    ADMIN = "agent.admin"
    DEVELOPER = "agent.developer"
    VIEWER = "agent.viewer"

def check_permission(user_role: AgentRole, required_role: AgentRole):
    """Check if user has required permissions."""
    role_hierarchy = {
        AgentRole.ADMIN: [AgentRole.ADMIN, AgentRole.DEVELOPER,
                         AgentRole.USER, AgentRole.VIEWER],
        AgentRole.DEVELOPER: [AgentRole.DEVELOPER, AgentRole.USER,
                             AgentRole.VIEWER],
        AgentRole.USER: [AgentRole.USER, AgentRole.VIEWER],
        AgentRole.VIEWER: [AgentRole.VIEWER]
    }

    return required_role in role_hierarchy.get(user_role, [])
```

---

## Data Security

### 1. Data Encryption

**At Rest:**
- Use Customer-Managed Encryption Keys (CMEK)
- Enable by default for Vertex AI Agent Engine

```bash
# Enable CMEK for Agent Engine
gcloud agent-engines create my-agent \
    --kms-key=projects/PROJECT_ID/locations/LOCATION/keyRings/KEYRING/cryptoKeys/KEY
```

**In Transit:**
- Always use HTTPS/TLS for API communication
- Enable TLS 1.3 minimum

```python
# Configure secure HTTP client
import httpx

client = httpx.AsyncClient(
    verify=True,  # Verify SSL certificates
    timeout=30.0,
    headers={'User-Agent': 'SecureAgent/1.0'}
)
```

---

### 2. Data Privacy

**PII Handling:**
```python
import re

def sanitize_pii(text: str) -> str:
    """Remove PII from agent responses."""
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL]', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

    # Remove SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)

    return text

class PrivacyAwareExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        # ... execute agent logic
        response = await self.agent.run(query)

        # Sanitize before returning
        sanitized_response = sanitize_pii(response)
        await updater.add_artifact([TextPart(text=sanitized_response)])
```

**Data Retention Policies:**
```python
from datetime import datetime, timedelta

class DataRetentionPolicy:
    """Implement data retention for compliance."""

    def __init__(self, retention_days: int = 90):
        self.retention_days = retention_days

    async def cleanup_old_sessions(self, session_service):
        """Delete sessions older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        # Query and delete old sessions
        old_sessions = await session_service.list_sessions(
            created_before=cutoff_date
        )

        for session in old_sessions:
            await session_service.delete_session(session.id)
```

---

## Network Security

### 1. VPC Configuration

**Private Agent Deployment:**
```bash
# Deploy agent in private VPC
gcloud agent-engines create private-agent \
    --network=projects/PROJECT_ID/global/networks/NETWORK_NAME \
    --subnet=projects/PROJECT_ID/regions/REGION/subnetworks/SUBNET_NAME \
    --no-public-ip
```

**VPC Service Controls:**
```python
# Configure VPC perimeter for agents
from google.cloud import accesscontextmanager_v1

def create_vpc_perimeter(project_id: str, perimeter_name: str):
    """Create VPC Service Controls perimeter for agents."""
    client = accesscontextmanager_v1.AccessContextManagerClient()

    perimeter = {
        "name": perimeter_name,
        "resources": [f"projects/{project_id}"],
        "restricted_services": [
            "aiplatform.googleapis.com",
            "storage.googleapis.com"
        ]
    }

    return client.create_service_perimeter(perimeter)
```

---

### 2. Firewall Rules

```bash
# Create firewall rule for agent traffic
gcloud compute firewall-rules create allow-agent-traffic \
    --network=NETWORK_NAME \
    --allow=tcp:443 \
    --source-ranges=10.0.0.0/8 \
    --target-tags=agent-instance
```

---

## API Security

### 1. Rate Limiting

```python
from functools import wraps
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    """Token bucket rate limiter for API endpoints."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = datetime.now()
        self.lock = asyncio.Lock()

    async def acquire(self, user_id: str) -> bool:
        """Acquire token for request."""
        async with self.lock:
            now = datetime.now()
            time_passed = (now - self.last_update).total_seconds()

            # Refill tokens
            self.tokens = min(
                self.requests_per_minute,
                self.tokens + time_passed * (self.requests_per_minute / 60)
            )
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            return False

def rate_limit(requests_per_minute: int = 60):
    """Decorator for rate limiting."""
    limiter = RateLimiter(requests_per_minute)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = kwargs.get('user_id', 'anonymous')

            if not await limiter.acquire(user_id):
                raise RateLimitError("Too many requests")

            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

---

### 2. Input Validation

```python
from pydantic import BaseModel, validator, constr

class AgentRequest(BaseModel):
    """Validated agent request schema."""

    message: constr(min_length=1, max_length=10000)
    user_id: str
    session_id: str | None = None

    @validator('message')
    def validate_message(cls, v):
        """Validate message content."""
        # Check for injection attempts
        dangerous_patterns = ['<script>', 'DROP TABLE', 'UNION SELECT']
        for pattern in dangerous_patterns:
            if pattern.lower() in v.lower():
                raise ValueError("Potentially malicious input detected")
        return v

class SecureAgentExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        # Validate input
        try:
            request_data = AgentRequest(**context.request_data)
        except ValidationError as e:
            raise InvalidInputError(str(e))

        # Proceed with validated input
        # ...
```

---

### 3. Model Armor (Prompt Injection Protection)

```python
def detect_prompt_injection(user_input: str) -> bool:
    """Detect common prompt injection patterns."""
    injection_patterns = [
        r'ignore previous instructions',
        r'system prompt',
        r'you are now',
        r'disregard.*above',
        r'forget.*instructions'
    ]

    for pattern in injection_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True

    return False

class ProtectedAgentExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        query = context.get_user_input()

        # Check for prompt injection
        if detect_prompt_injection(query):
            await updater.update_status(
                TaskState.failed,
                message=new_agent_text_message(
                    "Input rejected for security reasons"
                )
            )
            return

        # Continue with execution
        # ...
```

---

## Compliance & Governance

### HIPAA Compliance

**Requirements for Healthcare Agents:**

1. **BAA (Business Associate Agreement)** - Required with Google Cloud
2. **Audit Logging** - Enable comprehensive audit trails
3. **Data Encryption** - Use CMEK for all data
4. **Access Controls** - Implement strict RBAC
5. **Data Retention** - Follow HIPAA retention policies

```python
# HIPAA-compliant agent configuration
hipaa_agent_config = {
    'encryption': {
        'at_rest': 'CMEK',
        'in_transit': 'TLS 1.3'
    },
    'audit_logging': True,
    'data_retention_days': 2555,  # 7 years
    'access_control': 'RBAC',
    'vpc': 'private',
    'backup': 'encrypted'
}
```

---

### Audit Logging

```python
import logging
from google.cloud import logging as cloud_logging

def setup_audit_logging(project_id: str):
    """Configure Cloud Logging for security audits."""
    client = cloud_logging.Client(project=project_id)
    client.setup_logging()

    logger = logging.getLogger('agent_security')
    logger.setLevel(logging.INFO)

    return logger

class AuditedAgentExecutor(AgentExecutor):
    def __init__(self):
        self.audit_logger = setup_audit_logging(PROJECT_ID)

    async def execute(self, context, event_queue):
        # Log security-relevant events
        self.audit_logger.info('Agent execution started', extra={
            'user_id': context.user_id,
            'session_id': context.session_id,
            'timestamp': datetime.now().isoformat(),
            'ip_address': context.request.client.host
        })

        try:
            # Execute agent logic
            result = await self.agent.run(context.get_user_input())

            self.audit_logger.info('Agent execution completed', extra={
                'user_id': context.user_id,
                'success': True
            })

        except Exception as e:
            self.audit_logger.error('Agent execution failed', extra={
                'user_id': context.user_id,
                'error': str(e)
            })
            raise
```

---

## Security Checklist

### Pre-Deployment

- [ ] Service accounts configured with minimal permissions
- [ ] API keys stored in Secret Manager
- [ ] TLS/HTTPS enabled for all endpoints
- [ ] Input validation implemented
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] VPC/firewall rules configured
- [ ] Data encryption enabled (CMEK if required)
- [ ] PII handling implemented
- [ ] Prompt injection protection in place

### Production

- [ ] Regular security audits scheduled
- [ ] Automated vulnerability scanning enabled
- [ ] Incident response plan documented
- [ ] Service account key rotation automated
- [ ] Access reviews conducted quarterly
- [ ] Compliance requirements met (HIPAA, SOC 2, etc.)
- [ ] Backup and disaster recovery tested
- [ ] Monitoring and alerting configured

### Ongoing

- [ ] Security patches applied monthly
- [ ] Dependencies updated regularly
- [ ] Access logs reviewed weekly
- [ ] Anomaly detection monitored
- [ ] Security training for team
- [ ] Third-party security assessments annually

---

## Resources

- [Google Cloud Security Best Practices](https://cloud.google.com/security/best-practices)
- [Vertex AI Security Documentation](https://cloud.google.com/vertex-ai/docs/general/security)
- [A2A Protocol Security Spec](https://a2aprotocol.ai/security)
- [OWASP API Security Top 10](https://owasp.org/API-Security/)

---

**Next Steps:**
1. Review your current agent security posture
2. Implement missing security controls
3. Test authentication and authorization
4. Document security procedures
5. Train team on security best practices

---

**Related Guides:**
- `tutorial_a2a_on_agent_engine.ipynb` - A2A authentication examples
- `get_started_with_memory_for_adk_in_cloud_run.ipynb` - Cloud Run security
- `guide_production_deployment.md` - Production deployment checklist
