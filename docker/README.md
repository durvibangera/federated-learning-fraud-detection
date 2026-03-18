# Docker Deployment Guide

## Overview

This directory contains Docker configurations for deploying the Federated Fraud Detection system with complete MLOps infrastructure.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MLOps Infrastructure                     │
├─────────────────────────────────────────────────────────────┤
│  MLflow (5000)  │  Prometheus (9090)  │  Grafana (3000)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Aggregation Server (8080, 8001)                │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Bank Client 1│      │ Bank Client 2│      │ Bank Client 3│
│  (W) - 8100  │      │ (H,R) - 8200 │      │ (S,C) - 8300 │
└──────────────┘      └──────────────┘      └──────────────┘
```

## Components

### 1. Federated Learning Components

#### Aggregation Server
- **Container:** `aggregation_server`
- **Ports:** 
  - 8080: Flower FL server
  - 8001: Prometheus metrics
- **Purpose:** Coordinates FL rounds and aggregates model weights

#### Bank Clients (3 instances)
- **Bank 1:** ProductCD W, Port 8100
- **Bank 2:** ProductCD H,R, Port 8200
- **Bank 3:** ProductCD S,C, Port 8300
- **Purpose:** Train local models on partitioned data

### 2. MLOps Infrastructure

#### MLflow
- **Port:** 5000
- **Purpose:** Experiment tracking, model versioning
- **UI:** http://localhost:5000

#### Prometheus
- **Port:** 9090
- **Purpose:** Real-time metrics collection
- **UI:** http://localhost:9090

#### Grafana
- **Port:** 3000
- **Purpose:** Metrics visualization
- **UI:** http://localhost:3000
- **Credentials:** admin/admin

## Prerequisites

### Required Software

1. **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
   - Version: 20.10+
   - Download: https://www.docker.com/products/docker-desktop

2. **Docker Compose**
   - Version: 2.0+
   - Included with Docker Desktop
   - Linux: `sudo apt-get install docker-compose-plugin`

3. **System Requirements**
   - RAM: 8GB minimum, 16GB recommended
   - Disk: 20GB free space
   - CPU: 4 cores recommended

### Verify Installation

```bash
# Check Docker
docker --version
# Expected: Docker version 20.10.x or higher

# Check Docker Compose
docker compose version
# Expected: Docker Compose version v2.x.x or higher

# Check Docker is running
docker ps
# Should show empty list or running containers
```

## Quick Start

### 1. Prepare Data

Ensure IEEE-CIS dataset is in the correct location:

```bash
# From project root
ls data/raw/
# Should show:
# - train_transaction.csv
# - train_identity.csv
# - test_transaction.csv
# - test_identity.csv
```

### 2. Build Containers

**⚠️ DOCKER ENGINE MUST BE RUNNING**

```bash
# Navigate to docker directory
cd docker

# Build all containers (takes 5-10 minutes first time)
docker compose build

# Or build specific container
docker compose build bank_client_1
```

### 3. Start Infrastructure

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### 4. Access Services

Once running, access:

- **MLflow UI:** http://localhost:5000
- **Prometheus UI:** http://localhost:9090
- **Grafana UI:** http://localhost:3000 (admin/admin)

### 5. Run Federated Learning

```bash
# Option 1: Execute in aggregation server container
docker compose exec aggregation_server python -m src.federated.aggregation_server

# Option 2: Use docker exec
docker exec -it aggregation_server python -m src.federated.aggregation_server
```

### 6. Stop Services

```bash
# Stop all containers
docker compose down

# Stop and remove volumes (deletes data)
docker compose down -v
```

## Configuration

### Environment Variables

Each component can be configured via environment variables in `docker-compose.yml`:

#### Aggregation Server
```yaml
environment:
  - FL_SERVER_ADDRESS=0.0.0.0:8080
  - MLFLOW_TRACKING_URI=http://mlflow:5000
  - PROMETHEUS_PORT=8001
  - LOG_LEVEL=INFO
```

#### Bank Clients
```yaml
environment:
  - BANK_ID=bank1
  - PRODUCT_CD=W
  - FL_SERVER_ADDRESS=aggregation_server:8080
  - MLFLOW_TRACKING_URI=http://mlflow:5000
  - PROMETHEUS_PORT=8000
```

### Volume Mounts

Persistent data is stored in Docker volumes:

- **MLflow:** `mlflow_artifacts`, `mlflow_backend`
- **Prometheus:** `prometheus_data`
- **Grafana:** `grafana_data`
- **Models:** `server_models`, `bank1_models`, `bank2_models`, `bank3_models`
- **Logs:** `server_logs`, `bank1_logs`, `bank2_logs`, `bank3_logs`

### Network Configuration

All containers run on isolated network `fl_network` (172.20.0.0/16):
- Containers can communicate via service names
- External access only through exposed ports

## Usage Examples

### View Container Logs

```bash
# All containers
docker compose logs -f

# Specific container
docker compose logs -f aggregation_server
docker compose logs -f bank_client_1

# Last 100 lines
docker compose logs --tail=100 bank_client_2
```

### Execute Commands in Containers

```bash
# Open bash shell
docker compose exec aggregation_server bash

# Run Python script
docker compose exec bank_client_1 python verify_setup.py

# Check Python packages
docker compose exec bank_client_2 pip list
```

### Monitor Resources

```bash
# Container resource usage
docker stats

# Specific container
docker stats aggregation_server
```

### Inspect Containers

```bash
# Container details
docker compose ps
docker inspect aggregation_server

# Network details
docker network inspect docker_fl_network

# Volume details
docker volume ls
docker volume inspect docker_mlflow_artifacts
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs aggregation_server

# Check if port is already in use
netstat -ano | findstr :8080  # Windows
lsof -i :8080                 # Mac/Linux

# Rebuild container
docker compose build --no-cache aggregation_server
docker compose up -d aggregation_server
```

### Out of Memory

```bash
# Check Docker memory limit
docker info | grep Memory

# Increase Docker Desktop memory:
# Settings → Resources → Memory → Increase to 8GB+

# Or reduce batch size in config/config.yaml
```

### Network Issues

```bash
# Recreate network
docker compose down
docker network prune
docker compose up -d

# Check network connectivity
docker compose exec bank_client_1 ping aggregation_server
docker compose exec bank_client_1 curl http://mlflow:5000/health
```

### Volume Issues

```bash
# List volumes
docker volume ls

# Remove unused volumes
docker volume prune

# Remove specific volume
docker volume rm docker_mlflow_artifacts

# Backup volume
docker run --rm -v docker_mlflow_artifacts:/data -v $(pwd):/backup \
  alpine tar czf /backup/mlflow_backup.tar.gz /data
```

### Port Conflicts

If ports are already in use, modify `docker-compose.yml`:

```yaml
ports:
  - "5001:5000"  # Change 5000 to 5001 for MLflow
```

### Clean Slate

```bash
# Stop everything
docker compose down -v

# Remove all containers, images, volumes
docker system prune -a --volumes

# Rebuild from scratch
docker compose build --no-cache
docker compose up -d
```

## Development Workflow

### 1. Code Changes

```bash
# Make changes to src/
# Rebuild affected containers
docker compose build bank_client_1
docker compose up -d bank_client_1
```

### 2. Testing

```bash
# Run tests in container
docker compose exec bank_client_1 pytest tests/

# Run specific test
docker compose exec bank_client_1 pytest tests/test_bank_client.py -v
```

### 3. Debugging

```bash
# Interactive Python shell
docker compose exec bank_client_1 python

# Install additional packages (temporary)
docker compose exec bank_client_1 pip install ipdb

# Copy files from container
docker cp aggregation_server:/app/logs/federated.log ./local_logs/
```

## Production Deployment

### Security Considerations

1. **Change default passwords:**
   ```yaml
   # In docker-compose.yml
   - GF_SECURITY_ADMIN_PASSWORD=<strong-password>
   ```

2. **Use secrets for sensitive data:**
   ```yaml
   secrets:
     mlflow_password:
       file: ./secrets/mlflow_password.txt
   ```

3. **Enable TLS/SSL:**
   - Use reverse proxy (nginx, traefik)
   - Configure HTTPS certificates

4. **Network isolation:**
   - Use separate networks for different components
   - Restrict external access

### Scaling

```bash
# Scale bank clients
docker compose up -d --scale bank_client_1=2

# Use Docker Swarm for multi-node deployment
docker swarm init
docker stack deploy -c docker-compose.yml fl_stack
```

### Monitoring

```bash
# Container health
docker compose ps

# Resource usage
docker stats

# Logs aggregation
# Use ELK stack or Loki for centralized logging
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Test Docker

on: [push, pull_request]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build containers
        run: |
          cd docker
          docker compose build
      
      - name: Start services
        run: |
          cd docker
          docker compose up -d
          sleep 30
      
      - name: Run tests
        run: |
          docker compose exec -T bank_client_1 pytest tests/
      
      - name: Stop services
        run: |
          cd docker
          docker compose down -v
```

## Files Reference

- **Dockerfile.bank_client** - Bank client container definition
- **Dockerfile.aggregation_server** - Server container definition
- **Dockerfile.mlflow** - MLflow container definition
- **docker-compose.yml** - Multi-container orchestration
- **prometheus.yml** - Prometheus scrape configuration
- **alert_rules.yml** - Prometheus alerting rules
- **.dockerignore** - Files to exclude from build context

## Next Steps

1. ✅ Build containers: `docker compose build`
2. ✅ Start services: `docker compose up -d`
3. ✅ Access UIs: MLflow, Prometheus, Grafana
4. ✅ Run FL training: Execute in aggregation_server
5. ✅ Monitor metrics: Check Grafana dashboards
6. ✅ Review results: Check MLflow experiments

## Support

For issues:
1. Check logs: `docker compose logs -f`
2. Verify configuration: `docker compose config`
3. Check system resources: `docker stats`
4. Review documentation: This README

## Summary

**DO NOT START DOCKER YET** - These are just configuration files. You'll start Docker when you're ready to actually run the system (Phase 6, Task 21).

Current status: ✅ Docker configurations created, ready to build when needed.
