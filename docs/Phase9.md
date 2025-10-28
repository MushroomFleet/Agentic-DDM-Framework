# Phase 9: Production Deployment

## Phase Overview

**Goal:** Prepare the framework for production deployment with Docker, monitoring, scaling, and operational best practices  
**Prerequisites:** 
- Phases 1-8 complete (full framework with advanced features)
- Docker and Docker Compose installed
- Understanding of production deployment concepts
- Optional: Kubernetes knowledge for scaling

**Estimated Duration:** 6-8 hours  

**Key Deliverables:**
- ✅ Docker containerization
- ✅ Docker Compose setup
- ✅ Environment configuration management
- ✅ Logging and monitoring setup
- ✅ Health checks and readiness probes
- ✅ Error tracking (Sentry integration)
- ✅ Performance monitoring
- ✅ Deployment scripts
- ✅ Security best practices
- ✅ Production configuration

**Why This Phase Matters:**  
Production deployment requires containerization, monitoring, and operational tooling to ensure reliability, scalability, and maintainability. Proper deployment practices prevent downtime and enable confident scaling.

---

## Architecture

```
deployment/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── kubernetes/              # Optional
│   ├── deployment.yaml
│   └── service.yaml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana-dashboard.json
├── scripts/
│   ├── deploy.sh
│   ├── health_check.sh
│   └── rollback.sh
└── configs/
    ├── production.env.example
    └── staging.env.example
```

---

## Step-by-Step Implementation

### Step 1: Docker Containerization

**Purpose:** Create production-ready Docker image  
**Duration:** 60 minutes

#### Instructions

1. Create deployment structure:
```bash
mkdir -p deployment/docker
mkdir -p deployment/scripts
mkdir -p deployment/configs
```

2. Create Dockerfile:
```bash
cat > deployment/docker/Dockerfile << 'EOF'
# Multi-stage build for ADDM Framework
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.py

# Set Python path
ENV PYTHONPATH=/app/src
ENV PATH=/root/.local/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 addm && \
    chown -R addm:addm /app
USER addm

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from addm_framework import ADDM_Agent; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "addm_framework.cli"]
EOF
```

3. Create .dockerignore:
```bash
cat > deployment/docker/.dockerignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Documentation
docs/
*.md

# Data
data/
*.csv
*.json
*.sqlite

# Logs
*.log
logs/

# Environment
.env
.env.local
EOF
```

4. Create Docker Compose:
```bash
cat > deployment/docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  addm-framework:
    build:
      context: ../../
      dockerfile: deployment/docker/Dockerfile
    container_name: addm-agent
    restart: unless-stopped
    
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - ADDM_ENV=production
      - ADDM_LOG_LEVEL=INFO
      - ADDM_LOG_DIR=/app/logs
      - ADDM_DATA_DIR=/app/data
    
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./configs:/app/configs:ro
    
    ports:
      - "8000:8000"  # For API server (if implemented)
    
    networks:
      - addm-network
    
    healthcheck:
      test: ["CMD", "python", "-c", "from addm_framework import ADDM_Agent; print('OK')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Optional: Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: addm-prometheus
    restart: unless-stopped
    
    volumes:
      - ../monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    
    ports:
      - "9090:9090"
    
    networks:
      - addm-network
    
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  # Optional: Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: addm-grafana
    restart: unless-stopped
    
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    
    volumes:
      - grafana-data:/var/lib/grafana
    
    ports:
      - "3000:3000"
    
    networks:
      - addm-network
    
    depends_on:
      - prometheus

networks:
  addm-network:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
EOF
```

5. Create production config example:
```bash
cat > deployment/configs/production.env.example << 'EOF'
# ADDM Framework Production Configuration

# Required: OpenRouter API Key
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Environment
ADDM_ENV=production

# Logging
ADDM_LOG_LEVEL=INFO
ADDM_LOG_DIR=/app/logs
ADDM_LOG_FORMAT=json

# Data Storage
ADDM_DATA_DIR=/app/data

# DDM Configuration
ADDM_DDM_THRESHOLD=1.0
ADDM_DDM_TRIALS=100
ADDM_DDM_BASE_DRIFT=1.0

# LLM Configuration
ADDM_LLM_MODEL=x-ai/grok-4-fast
ADDM_LLM_TEMPERATURE=0.7
ADDM_LLM_MAX_TOKENS=1500
ADDM_LLM_TIMEOUT=30

# Performance
ADDM_CACHE_ENABLED=true
ADDM_CACHE_TTL_HOURS=24

# Monitoring
ADDM_METRICS_ENABLED=true
ADDM_SENTRY_DSN=  # Optional: Sentry error tracking

# Security
ADDM_RATE_LIMIT_PER_MINUTE=60
ADDM_MAX_CONCURRENT_REQUESTS=10
EOF
```

#### Verification
- [ ] Dockerfile builds successfully
- [ ] Docker Compose starts containers
- [ ] Health checks pass
- [ ] Configurations work

---

### Step 2: Deployment Scripts

**Purpose:** Automate deployment and operations  
**Duration:** 45 minutes

#### Instructions

```bash
cat > deployment/scripts/deploy.sh << 'EOF'
#!/bin/bash
# Deploy ADDM Framework to production

set -e

echo "🚀 ADDM Framework Deployment Script"
echo "===================================="

# Configuration
ENV=${1:-production}
VERSION=${2:-latest}

echo ""
echo "Environment: $ENV"
echo "Version: $VERSION"
echo ""

# Check prerequisites
if [ ! -f "deployment/configs/${ENV}.env" ]; then
    echo "❌ Configuration file not found: deployment/configs/${ENV}.env"
    echo "   Copy from ${ENV}.env.example and configure"
    exit 1
fi

# Load environment
source "deployment/configs/${ENV}.env"

# Validate API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ OPENROUTER_API_KEY not set in configuration"
    exit 1
fi

echo "✅ Prerequisites validated"
echo ""

# Build Docker image
echo "🔨 Building Docker image..."
cd deployment/docker
docker-compose build

echo "✅ Docker image built"
echo ""

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Start new containers
echo "🚀 Starting containers..."
docker-compose up -d

# Wait for health check
echo "⏳ Waiting for health check..."
sleep 10

# Check health
if docker-compose ps | grep -q "healthy"; then
    echo "✅ Deployment successful!"
    echo ""
    echo "📊 Container status:"
    docker-compose ps
    echo ""
    echo "📝 View logs:"
    echo "   docker-compose logs -f addm-framework"
    echo ""
    echo "🔍 Monitor metrics:"
    echo "   Prometheus: http://localhost:9090"
    echo "   Grafana: http://localhost:3000"
else
    echo "❌ Health check failed"
    echo "Checking logs..."
    docker-compose logs addm-framework
    exit 1
fi
EOF

chmod +x deployment/scripts/deploy.sh
```

2. Create health check script:
```bash
cat > deployment/scripts/health_check.sh << 'EOF'
#!/bin/bash
# Check ADDM Framework health

set -e

echo "🏥 ADDM Framework Health Check"
echo "=============================="

# Check if container is running
if ! docker-compose -f deployment/docker/docker-compose.yml ps | grep -q "Up"; then
    echo "❌ Container not running"
    exit 1
fi

echo "✅ Container is running"

# Check health endpoint
if docker exec addm-agent python -c "from addm_framework import ADDM_Agent; print('OK')" > /dev/null 2>&1; then
    echo "✅ Application is healthy"
else
    echo "❌ Application health check failed"
    exit 1
fi

# Check disk space
DISK_USAGE=$(df -h /app/data 2>/dev/null | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "⚠️  Disk usage high: ${DISK_USAGE}%"
else
    echo "✅ Disk usage OK: ${DISK_USAGE}%"
fi

# Check memory
MEM_USAGE=$(docker stats --no-stream addm-agent --format "{{.MemPerc}}" | sed 's/%//')
if (( $(echo "$MEM_USAGE > 80" | bc -l) )); then
    echo "⚠️  Memory usage high: ${MEM_USAGE}%"
else
    echo "✅ Memory usage OK: ${MEM_USAGE}%"
fi

echo ""
echo "✅ All health checks passed"
EOF

chmod +x deployment/scripts/health_check.sh
```

3. Create rollback script:
```bash
cat > deployment/scripts/rollback.sh << 'EOF'
#!/bin/bash
# Rollback ADDM Framework deployment

set -e

echo "⏪ ADDM Framework Rollback Script"
echo "=================================="

PREVIOUS_VERSION=${1:-previous}

echo ""
echo "Rolling back to: $PREVIOUS_VERSION"
echo ""

# Stop current version
echo "🛑 Stopping current version..."
cd deployment/docker
docker-compose down

# Restore previous version
echo "📦 Restoring previous version..."
# This would pull/restore the previous Docker image
docker pull addm-framework:$PREVIOUS_VERSION || echo "⚠️  Previous image not found"

# Start previous version
echo "🚀 Starting previous version..."
docker-compose up -d

# Verify
sleep 10
if deployment/scripts/health_check.sh; then
    echo "✅ Rollback successful"
else
    echo "❌ Rollback failed - manual intervention required"
    exit 1
fi
EOF

chmod +x deployment/scripts/rollback.sh
```

#### Verification
- [ ] Deploy script works
- [ ] Health check script validates system
- [ ] Rollback script can restore previous version

---

### Step 3: Monitoring & Observability

**Purpose:** Set up production monitoring  
**Duration:** 60 minutes

#### Instructions

1. Create Prometheus configuration:
```bash
mkdir -p deployment/monitoring

cat > deployment/monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'addm-framework'
    static_configs:
      - targets: ['addm-framework:8000']
    metrics_path: '/metrics'
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
```

2. Create logging configuration:
```bash
cat > src/addm_framework/utils/production_logging.py << 'EOF'
"""Production-grade logging configuration."""
import logging
import logging.handlers
import json
from pathlib import Path
from typing import Optional
import sys


class JSONFormatter(logging.Formatter):
    """Format logs as JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record
        
        Returns:
            JSON string
        """
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "decision_id"):
            log_data["decision_id"] = record.decision_id
        
        return json.dumps(log_data)


def setup_production_logging(
    log_dir: Path,
    log_level: str = "INFO",
    json_format: bool = True
) -> None:
    """Set up production logging.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        json_format: Use JSON formatting
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    handlers.append(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "addm.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    handlers.append(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "addm_errors.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    handlers.append(error_handler)
    
    # Set formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    for handler in handlers:
        handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    for handler in handlers:
        root_logger.addHandler(handler)
    
    logging.info("Production logging configured")
EOF
```

3. Create monitoring metrics:
```bash
cat > src/addm_framework/utils/metrics.py << 'EOF'
"""Production metrics tracking."""
from typing import Dict, Any
from datetime import datetime
import json
from pathlib import Path


class MetricsCollector:
    """Collect and export metrics for monitoring."""
    
    def __init__(self, metrics_dir: Path):
        """Initialize metrics collector.
        
        Args:
            metrics_dir: Directory for metrics files
        """
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, list] = {}
    
    def increment(self, name: str, value: int = 1) -> None:
        """Increment counter.
        
        Args:
            name: Counter name
            value: Increment value
        """
        self.counters[name] = self.counters.get(name, 0) + value
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set gauge value.
        
        Args:
            name: Gauge name
            value: Gauge value
        """
        self.gauges[name] = value
    
    def record_histogram(self, name: str, value: float) -> None:
        """Record histogram value.
        
        Args:
            name: Histogram name
            value: Value to record
        """
        if name not in self.histograms:
            self.histograms[name] = []
        self.histograms[name].append(value)
        
        # Keep last 1000 values
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics.
        
        Returns:
            Metrics dict
        """
        import numpy as np
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "counters": self.counters,
            "gauges": self.gauges,
            "histograms": {}
        }
        
        # Compute histogram statistics
        for name, values in self.histograms.items():
            if values:
                metrics["histograms"][name] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        return metrics
    
    def save_metrics(self) -> Path:
        """Save metrics to file.
        
        Returns:
            Path to metrics file
        """
        metrics = self.export_metrics()
        
        metrics_file = self.metrics_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics_file
EOF
```

#### Verification
- [ ] Prometheus configuration created
- [ ] Production logging works
- [ ] Metrics collection works
- [ ] JSON log format valid

---

### Step 4: Security & Best Practices

**Purpose:** Implement production security  
**Duration:** 45 minutes

#### Instructions

```bash
cat > deployment/docs/SECURITY.md << 'EOF'
# ADDM Framework Security Guide

## API Key Management

### Production
- Store `OPENROUTER_API_KEY` in secure secrets manager (AWS Secrets Manager, Vault, etc.)
- Never commit API keys to version control
- Rotate keys regularly (monthly recommended)
- Use separate keys for different environments

### Docker Secrets
```bash
# Create Docker secret
echo "your_api_key" | docker secret create openrouter_api_key -

# Use in docker-compose.yml
secrets:
  openrouter_api_key:
    external: true
```

## Network Security

### Firewall Rules
- Restrict inbound traffic to necessary ports only
- Use HTTPS for all external communication
- Implement rate limiting at network level

### Container Security
- Run containers as non-root user (already configured)
- Use read-only file systems where possible
- Limit container resources (CPU, memory)

## Logging Security

### Sensitive Data
- Never log API keys
- Redact sensitive information in logs
- Use structured logging with JSON format
- Implement log retention policies

### Log Access
- Restrict log file access to authorized users only
- Use centralized logging (ELK, CloudWatch, etc.)
- Monitor logs for security events

## Dependency Security

### Regular Updates
```bash
# Check for vulnerabilities
pip-audit

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Docker Image Security
```bash
# Scan Docker image for vulnerabilities
docker scan addm-framework:latest

# Use minimal base images
# Keep images updated
```

## Monitoring & Alerts

### Security Monitoring
- Monitor for unusual API usage patterns
- Track error rates
- Alert on authentication failures
- Log security events

### Incident Response
1. Immediate: Rotate compromised credentials
2. Investigate: Check logs for breach extent
3. Remediate: Apply security patches
4. Document: Update security policies

## Compliance

### Data Privacy
- No user data is stored by default
- Decision traces can contain sensitive information
- Implement data retention policies
- Comply with GDPR/CCPA if applicable

### Audit Trail
- Log all decision-making activities
- Track API usage
- Maintain audit logs for compliance
EOF
```

#### Verification
- [ ] Security documentation created
- [ ] Best practices documented
- [ ] Secrets management explained

---

## Summary

### What Was Accomplished

✅ **Docker Containerization**: Production-ready images  
✅ **Docker Compose**: Complete orchestration  
✅ **Deployment Scripts**: Automated deployment and rollback  
✅ **Health Checks**: System validation  
✅ **Monitoring**: Prometheus + Grafana integration  
✅ **Production Logging**: JSON structured logging  
✅ **Metrics Collection**: Performance tracking  
✅ **Security Guide**: Production security practices  

### Deployment Components

1. **Docker Image** - Multi-stage build, non-root user
2. **Docker Compose** - Full stack with monitoring
3. **Deployment Scripts** - Automated operations
4. **Health Checks** - System validation
5. **Monitoring Stack** - Prometheus + Grafana
6. **Production Logging** - Structured JSON logs
7. **Security Practices** - Comprehensive guide

### Phase 9 Metrics

- **Files Created**: 12 (Dockerfile, scripts, configs, docs)
- **Deployment Methods**: Docker + Docker Compose
- **Monitoring Tools**: Prometheus, Grafana
- **Security Features**: 10+ best practices

---

**Phase 9 Status:** ✅ COMPLETE  
**Next Phase:** Phase 10 (Documentation)

