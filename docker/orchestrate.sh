#!/bin/bash
# Docker Compose Orchestration Management Script

set -e

COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="federated-fraud-detection"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running"
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check service health
check_service_health() {
    local service=$1
    local max_attempts=30
    local attempt=1
    
    print_info "Checking health of $service..."
    
    while [ $attempt -le $max_attempts ]; do
        health=$(docker inspect --format='{{.State.Health.Status}}' $service 2>/dev/null || echo "unknown")
        
        if [ "$health" = "healthy" ]; then
            print_success "$service is healthy"
            return 0
        fi
        
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_warning "$service did not become healthy within timeout"
    return 1
}

# Function to start services
start_services() {
    print_info "Starting services..."
    
    # Start MLflow first
    print_info "Starting MLflow..."
    docker compose up -d mlflow
    check_service_health "mlflow_server"
    
    # Start aggregation server
    print_info "Starting Aggregation Server..."
    docker compose up -d aggregation_server
    check_service_health "aggregation_server"
    
    # Start bank clients
    print_info "Starting Bank Clients..."
    docker compose up -d bank_client_1 bank_client_2 bank_client_3
    sleep 5
    
    # Start monitoring stack
    print_info "Starting Monitoring Stack..."
    docker compose up -d prometheus
    check_service_health "prometheus"
    
    docker compose up -d grafana
    check_service_health "grafana"
    
    print_success "All services started successfully"
}

# Function to stop services gracefully
stop_services() {
    print_info "Stopping services gracefully..."
    
    # Stop monitoring first
    print_info "Stopping Grafana..."
    docker compose stop grafana
    
    print_info "Stopping Prometheus..."
    docker compose stop prometheus
    
    # Stop FL components
    print_info "Stopping Bank Clients..."
    docker compose stop bank_client_1 bank_client_2 bank_client_3
    
    print_info "Stopping Aggregation Server..."
    docker compose stop aggregation_server
    
    # Stop MLflow last
    print_info "Stopping MLflow..."
    docker compose stop mlflow
    
    print_success "All services stopped gracefully"
}

# Function to restart a specific service
restart_service() {
    local service=$1
    print_info "Restarting $service..."
    docker compose restart $service
    
    # Check health if service has healthcheck
    if docker inspect --format='{{.State.Health}}' $service 2>/dev/null | grep -q "Status"; then
        check_service_health $service
    fi
    
    print_success "$service restarted"
}

# Function to show service status
show_status() {
    print_info "Service Status:"
    docker compose ps
    
    echo ""
    print_info "Health Status:"
    for service in mlflow_server aggregation_server prometheus grafana; do
        if docker ps --filter "name=$service" --format "{{.Names}}" | grep -q $service; then
            health=$(docker inspect --format='{{.State.Health.Status}}' $service 2>/dev/null || echo "no healthcheck")
            echo "  $service: $health"
        fi
    done
}

# Function to show logs
show_logs() {
    local service=$1
    if [ -z "$service" ]; then
        docker compose logs -f
    else
        docker compose logs -f $service
    fi
}

# Function to execute command in service
exec_command() {
    local service=$1
    shift
    docker compose exec $service "$@"
}

# Function to clean up everything
cleanup() {
    print_warning "This will remove all containers, networks, and volumes"
    read -p "Are you sure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        print_info "Cleaning up..."
        docker compose down -v
        print_success "Cleanup complete"
    else
        print_info "Cleanup cancelled"
    fi
}

# Function to validate configuration
validate_config() {
    print_info "Validating docker-compose configuration..."
    if docker compose config > /dev/null 2>&1; then
        print_success "Configuration is valid"
        return 0
    else
        print_error "Configuration is invalid"
        docker compose config
        return 1
    fi
}

# Function to show network info
show_network() {
    print_info "Network Information:"
    docker network inspect docker_fl_network 2>/dev/null || print_warning "Network not created yet"
}

# Function to show volume info
show_volumes() {
    print_info "Volume Information:"
    docker volume ls --filter "name=docker_"
}

# Main command handler
case "$1" in
    start)
        check_docker
        validate_config
        start_services
        show_status
        ;;
    stop)
        check_docker
        stop_services
        ;;
    restart)
        if [ -z "$2" ]; then
            print_info "Restarting all services..."
            stop_services
            start_services
        else
            restart_service "$2"
        fi
        ;;
    status)
        check_docker
        show_status
        ;;
    logs)
        check_docker
        show_logs "$2"
        ;;
    exec)
        check_docker
        if [ -z "$2" ]; then
            print_error "Usage: $0 exec <service> <command>"
            exit 1
        fi
        service=$2
        shift 2
        exec_command $service "$@"
        ;;
    health)
        check_docker
        if [ -z "$2" ]; then
            print_error "Usage: $0 health <service>"
            exit 1
        fi
        check_service_health "$2"
        ;;
    validate)
        validate_config
        ;;
    network)
        check_docker
        show_network
        ;;
    volumes)
        check_docker
        show_volumes
        ;;
    cleanup)
        check_docker
        cleanup
        ;;
    *)
        echo "Usage: $0 {start|stop|restart [service]|status|logs [service]|exec <service> <command>|health <service>|validate|network|volumes|cleanup}"
        echo ""
        echo "Commands:"
        echo "  start              - Start all services in correct order"
        echo "  stop               - Stop all services gracefully"
        echo "  restart [service]  - Restart all services or specific service"
        echo "  status             - Show service status and health"
        echo "  logs [service]     - Show logs (all or specific service)"
        echo "  exec <service> <cmd> - Execute command in service"
        echo "  health <service>   - Check health of specific service"
        echo "  validate           - Validate docker-compose configuration"
        echo "  network            - Show network information"
        echo "  volumes            - Show volume information"
        echo "  cleanup            - Remove all containers, networks, and volumes"
        exit 1
        ;;
esac
