#!/bin/bash
# Quick start script for Federated Fraud Detection Docker deployment

set -e

echo "=========================================="
echo "Federated Fraud Detection - Docker Setup"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

echo "✓ Docker is running"

# Check if docker-compose is available
if ! command -v docker compose &> /dev/null; then
    echo "❌ Error: docker compose not found"
    echo "Please install Docker Compose"
    exit 1
fi

echo "✓ Docker Compose is available"

# Check if data files exist
if [ ! -f "../data/raw/train_transaction.csv" ]; then
    echo "⚠ Warning: IEEE-CIS dataset not found in data/raw/"
    echo "Please download the dataset before running FL training"
fi

# Build containers
echo ""
echo "Building Docker containers..."
echo "This may take 5-10 minutes on first run..."
docker compose build

echo ""
echo "✓ Containers built successfully"

# Start services
echo ""
echo "Starting services..."
docker compose up -d

echo ""
echo "✓ Services started"

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check service status
echo ""
echo "Service Status:"
docker compose ps

# Display access information
echo ""
echo "=========================================="
echo "Services are ready!"
echo "=========================================="
echo ""
echo "Access the following UIs:"
echo "  • MLflow:     http://localhost:5000"
echo "  • Prometheus: http://localhost:9090"
echo "  • Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo "To run federated learning:"
echo "  docker compose exec aggregation_server python -m src.federated.aggregation_server"
echo ""
echo "To view logs:"
echo "  docker compose logs -f"
echo ""
echo "To stop services:"
echo "  docker compose down"
echo ""
echo "=========================================="
