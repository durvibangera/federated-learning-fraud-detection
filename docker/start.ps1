# Quick start script for Federated Fraud Detection Docker deployment (PowerShell)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Federated Fraud Detection - Docker Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info | Out-Null
    Write-Host "✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Docker is not running" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again" -ForegroundColor Yellow
    exit 1
}

# Check if docker compose is available
try {
    docker compose version | Out-Null
    Write-Host "✓ Docker Compose is available" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: docker compose not found" -ForegroundColor Red
    Write-Host "Please install Docker Compose" -ForegroundColor Yellow
    exit 1
}

# Check if data files exist
if (-not (Test-Path "../data/raw/train_transaction.csv")) {
    Write-Host "⚠ Warning: IEEE-CIS dataset not found in data/raw/" -ForegroundColor Yellow
    Write-Host "Please download the dataset before running FL training" -ForegroundColor Yellow
}

# Build containers
Write-Host ""
Write-Host "Building Docker containers..." -ForegroundColor Cyan
Write-Host "This may take 5-10 minutes on first run..." -ForegroundColor Yellow
docker compose build

Write-Host ""
Write-Host "✓ Containers built successfully" -ForegroundColor Green

# Start services
Write-Host ""
Write-Host "Starting services..." -ForegroundColor Cyan
docker compose up -d

Write-Host ""
Write-Host "✓ Services started" -ForegroundColor Green

# Wait for services to be ready
Write-Host ""
Write-Host "Waiting for services to be ready..." -ForegroundColor Cyan
Start-Sleep -Seconds 10

# Check service status
Write-Host ""
Write-Host "Service Status:" -ForegroundColor Cyan
docker compose ps

# Display access information
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Services are ready!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access the following UIs:" -ForegroundColor White
Write-Host "  • MLflow:     http://localhost:5000" -ForegroundColor White
Write-Host "  • Prometheus: http://localhost:9090" -ForegroundColor White
Write-Host "  • Grafana:    http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host ""
Write-Host "To run federated learning:" -ForegroundColor Yellow
Write-Host "  docker compose exec aggregation_server python -m src.federated.aggregation_server" -ForegroundColor White
Write-Host ""
Write-Host "To view logs:" -ForegroundColor Yellow
Write-Host "  docker compose logs -f" -ForegroundColor White
Write-Host ""
Write-Host "To stop services:" -ForegroundColor Yellow
Write-Host "  docker compose down" -ForegroundColor White
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
