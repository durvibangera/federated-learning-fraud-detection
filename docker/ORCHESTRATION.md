# Docker Orchestration Guide

## Overview

This guide covers the enhanced docker-compose orchestration with service dependencies, health checks, and graceful shutdown capabilities.

## Key Features

### 1. Service Dependencies

Services start in the correct order with health check conditions:

```
MLflow (healthy)
    ↓
Aggregation Server (healthy)
    ↓
Bank Clients (started) + Prometheus (healthy)
    ↓
Grafana (healthy)
```

### 2. Health Checks

All critical services have health checks:

| Service | Health Check | Interval | Timeout | Start Period |
|---------|-------------|----------|---------|--------------|
| MLflow | HTTP /health | 30s | 10s | 10s |
| Prometheus | HTTP /-/healthy | 30s | 10s | 10s |
| Grafana | HTTP /api/health | 30s | 10s | 15s |
| Aggregation Server | Python check | 30s | 10s | 15s |
| Bank Clients | Python check | 30s | 10s | 15s |

### 3. Graceful Shutdown

All services configured for graceful shutdown:
- **Stop Grace Period:** 30-60 seconds
- **Stop Signal:** SIGTERM (allows cleanup)
- **Restart Policy:** unless-stopped

### 4. Network Isolation

Bank clients are isolated on `fl_network`:
- Subnet: 172.20.0.0/16
- Inter-container communication via service names
- External access only through exposed ports

### 5. Service Labels

All services tagged with labels for identification:
- `com.federated.component` - Component type (fl, mlops)
- `com.federated.service` - Service name
- `com.federated.role` - Role (server, client)
- `com.federated.bank_id` - Bank identifier (clients only)
- `com.federated.product_cd` - Product code (clients only)

## Orchestration Scripts

### orchestrate.sh (Linux/Mac)

Comprehensive orchestration management:

```bash
# Start all services in correct order
./orchestrate.sh start

# Stop all services gracefully
./orchestrate.sh stop

# Restart all services
./orchestrate.sh restart

# Restart specific service
./orchestrate.sh restart bank_client_1

# Show service status and health
./orchestrate.sh status

# Show logs (all services)
./orchestrate.sh logs

# Show logs (specific service)
./orchestrate.sh logs aggregation_server

# Execute command in service
./orchestrate.sh exec bank_client_1 bash

# Check service health
./orchestrate.sh health mlflow_server

# Validate configuration
./orchestrate.sh validate

# Show network information
./orchestrate.sh network

# Show volume information
./orchestrate.sh volumes

# Clean up everything
./orchestrate.sh cleanup
```

### orchestrate.ps1 (Windows PowerShell)

Same functionality for Windows:

```powershell
# Start all services
.\orchestrate.ps1 start

# Stop all services
.\orchestrate.ps1 stop

# Restart specific service
.\orchestrate.ps1 restart bank_client_2

# Show status
.\orchestrate.ps1 status

# Show logs
.\orchestrate.ps1 logs prometheus

# Execute command
.\orchestrate.ps1 exec aggregation_server python --version

# Check health
.\orchestrate.ps1 health grafana

# Validate config
.\orchestrate.ps1 validate

# Cleanup
.\orchestrate.ps1 cleanup
```

## Service Startup Sequence

### Phase 1: MLOps Infrastructure

1. **MLflow** starts first
   - Waits for health check to pass
   - Provides experiment tracking for all components

### Phase 2: FL Server

2. **Aggregation Server** starts
   - Depends on MLflow being healthy
   - Waits for health check to pass
   - Provides FL coordination

### Phase 3: FL Clients

3. **Bank Clients** start in parallel
   - Depend on Aggregation Server being healthy
   - Depend on MLflow being healthy
   - All three start simultaneously

### Phase 4: Monitoring

4. **Prometheus** starts
   - Depends on all FL components being started
   - Waits for health check to pass
   - Begins scraping metrics

5. **Grafana** starts last
   - Depends on Prometheus being healthy
   - Waits for health check to pass
   - Provides visualization

## Service Shutdown Sequence

Reverse order for graceful shutdown:

1. **Grafana** stops first (30s grace period)
2. **Prometheus** stops (30s grace period)
3. **Bank Clients** stop in parallel (60s grace period each)
4. **Aggregation Server** stops (60s grace period)
5. **MLflow** stops last (30s grace period)

## Health Check Details

### MLflow Health Check

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

Checks HTTP endpoint every 30 seconds. Service is healthy when endpoint returns 200.

### Prometheus Health Check

```yaml
healthcheck:
  test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 10s
```

Uses wget to check Prometheus health endpoint.

### Grafana Health Check

```yaml
healthcheck:
  test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:3000/api/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 15s
```

Checks Grafana API health endpoint.

### FL Component Health Check

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 15s
```

Simple Python check to ensure container is responsive.

## Dependency Conditions

### service_healthy

Service must pass health check before dependent services start:

```yaml
depends_on:
  mlflow:
    condition: service_healthy
```

### service_started

Service must be started (but not necessarily healthy):

```yaml
depends_on:
  bank_client_1:
    condition: service_started
```

## Graceful Shutdown Configuration

### Stop Grace Period

Time allowed for graceful shutdown before force kill:

```yaml
stop_grace_period: 60s  # FL components
stop_grace_period: 30s  # MLOps components
```

### Stop Signal

Signal sent to container for shutdown:

```yaml
stop_signal: SIGTERM  # Allows cleanup handlers
```

### Restart Policy

Automatic restart behavior:

```yaml
restart: unless-stopped  # Restart unless explicitly stopped
```

## Network Configuration

### Bridge Network

```yaml
networks:
  fl_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Service Communication

Services communicate via service names:
- `mlflow:5000` - MLflow tracking
- `aggregation_server:8080` - FL server
- `prometheus:9090` - Metrics
- `grafana:3000` - Visualization

### Port Mapping

External access through mapped ports:
- `localhost:5000` → `mlflow:5000`
- `localhost:8080` → `aggregation_server:8080`
- `localhost:9090` → `prometheus:9090`
- `localhost:3000` → `grafana:3000`
- `localhost:8100` → `bank_client_1:8000`
- `localhost:8200` → `bank_client_2:8000`
- `localhost:8300` → `bank_client_3:8000`

## Volume Management

### Persistent Volumes

Data persists across container restarts:

```yaml
volumes:
  mlflow_artifacts:      # MLflow experiment artifacts
  mlflow_backend:        # MLflow database
  prometheus_data:       # Prometheus time-series data
  grafana_data:          # Grafana dashboards
  server_models:         # Server model checkpoints
  server_logs:           # Server logs
  bank1_models:          # Bank 1 models
  bank1_logs:            # Bank 1 logs
  # ... (bank2, bank3)
```

### Volume Inspection

```bash
# List volumes
docker volume ls --filter "name=docker_"

# Inspect specific volume
docker volume inspect docker_mlflow_artifacts

# Backup volume
docker run --rm -v docker_mlflow_artifacts:/data -v $(pwd):/backup \
  alpine tar czf /backup/mlflow_backup.tar.gz /data

# Restore volume
docker run --rm -v docker_mlflow_artifacts:/data -v $(pwd):/backup \
  alpine tar xzf /backup/mlflow_backup.tar.gz -C /
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
./orchestrate.sh logs <service>

# Check health
./orchestrate.sh health <service>

# Validate config
./orchestrate.sh validate

# Restart service
./orchestrate.sh restart <service>
```

### Health Check Failing

```bash
# Check service logs
docker compose logs <service>

# Inspect container
docker inspect <service>

# Check if service is listening on port
docker compose exec <service> netstat -tulpn

# Manual health check
docker compose exec <service> curl http://localhost:<port>/health
```

### Dependency Issues

```bash
# Check startup order
docker compose events

# Check service dependencies
docker compose config | grep -A 5 depends_on

# Start services manually in order
docker compose up -d mlflow
docker compose up -d aggregation_server
docker compose up -d bank_client_1 bank_client_2 bank_client_3
```

### Network Issues

```bash
# Check network
./orchestrate.sh network

# Test connectivity
docker compose exec bank_client_1 ping aggregation_server
docker compose exec bank_client_1 curl http://mlflow:5000/health

# Recreate network
docker compose down
docker network prune
docker compose up -d
```

### Graceful Shutdown Not Working

```bash
# Check stop grace period
docker inspect <service> | grep StopTimeout

# Increase grace period in docker-compose.yml
stop_grace_period: 120s

# Force stop if needed
docker compose kill <service>
```

## Best Practices

### 1. Always Use Orchestration Scripts

```bash
# Good
./orchestrate.sh start

# Avoid
docker compose up -d
```

Scripts ensure correct startup order and health checks.

### 2. Check Health Before Operations

```bash
# Check all services are healthy
./orchestrate.sh status

# Check specific service
./orchestrate.sh health mlflow_server
```

### 3. Graceful Shutdown

```bash
# Good - graceful shutdown
./orchestrate.sh stop

# Avoid - force kill
docker compose kill
```

### 4. Monitor Logs During Startup

```bash
# Watch logs during startup
./orchestrate.sh logs &
./orchestrate.sh start
```

### 5. Validate Before Deployment

```bash
# Always validate configuration
./orchestrate.sh validate

# Check for syntax errors
docker compose config
```

## Advanced Usage

### Scaling Services

```bash
# Scale bank clients (not recommended - use separate configs)
docker compose up -d --scale bank_client_1=2
```

### Custom Startup Order

```bash
# Start specific services only
docker compose up -d mlflow aggregation_server

# Add clients later
docker compose up -d bank_client_1
```

### Rolling Restart

```bash
# Restart clients one at a time
./orchestrate.sh restart bank_client_1
sleep 30
./orchestrate.sh restart bank_client_2
sleep 30
./orchestrate.sh restart bank_client_3
```

### Debug Mode

```bash
# Start with debug logging
LOG_LEVEL=DEBUG docker compose up -d

# Attach to service output
docker compose logs -f <service>
```

## Requirements Satisfied

### Requirement 6.3: Network Isolation
✅ Bank containers isolated on fl_network with controlled communication

### Requirement 6.4: Service Dependencies
✅ Proper dependency chain with health check conditions

### Requirement 6.6: Graceful Shutdown
✅ All services configured with stop grace periods and SIGTERM

### Requirement 6.7: Automatic Initialization
✅ Services auto-initialize on startup with correct dependencies

## Summary

The enhanced docker-compose orchestration provides:
- ✅ Correct service startup order
- ✅ Health check-based dependencies
- ✅ Graceful shutdown with cleanup time
- ✅ Network isolation between components
- ✅ Comprehensive management scripts
- ✅ Production-ready configuration

Use the orchestration scripts for all operations to ensure proper service management.
