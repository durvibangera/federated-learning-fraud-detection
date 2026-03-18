# Docker Compose Orchestration Management Script (PowerShell)

param(
    [Parameter(Position=0)]
    [string]$Command,
    
    [Parameter(Position=1)]
    [string]$Service,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Args
)

$ComposeFile = "docker-compose.yml"
$ProjectName = "federated-fraud-detection"

# Function to print colored output
function Print-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Print-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Print-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Print-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Function to check Docker is running
function Check-Docker {
    try {
        docker info | Out-Null
        Print-Success "Docker is running"
        return $true
    } catch {
        Print-Error "Docker is not running"
        return $false
    }
}

# Function to check service health
function Check-ServiceHealth {
    param([string]$ServiceName)
    
    $maxAttempts = 30
    $attempt = 1
    
    Print-Info "Checking health of $ServiceName..."
    
    while ($attempt -le $maxAttempts) {
        try {
            $health = docker inspect --format='{{.State.Health.Status}}' $ServiceName 2>$null
            
            if ($health -eq "healthy") {
                Print-Success "$ServiceName is healthy"
                return $true
            }
        } catch {
            # Service might not have health check
        }
        
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 2
        $attempt++
    }
    
    Write-Host ""
    Print-Warning "$ServiceName did not become healthy within timeout"
    return $false
}

# Function to start services
function Start-Services {
    Print-Info "Starting services..."
    
    # Start MLflow first
    Print-Info "Starting MLflow..."
    docker compose up -d mlflow
    Check-ServiceHealth "mlflow_server"
    
    # Start aggregation server
    Print-Info "Starting Aggregation Server..."
    docker compose up -d aggregation_server
    Check-ServiceHealth "aggregation_server"
    
    # Start bank clients
    Print-Info "Starting Bank Clients..."
    docker compose up -d bank_client_1 bank_client_2 bank_client_3
    Start-Sleep -Seconds 5
    
    # Start monitoring stack
    Print-Info "Starting Monitoring Stack..."
    docker compose up -d prometheus
    Check-ServiceHealth "prometheus"
    
    docker compose up -d grafana
    Check-ServiceHealth "grafana"
    
    Print-Success "All services started successfully"
}

# Function to stop services gracefully
function Stop-Services {
    Print-Info "Stopping services gracefully..."
    
    # Stop monitoring first
    Print-Info "Stopping Grafana..."
    docker compose stop grafana
    
    Print-Info "Stopping Prometheus..."
    docker compose stop prometheus
    
    # Stop FL components
    Print-Info "Stopping Bank Clients..."
    docker compose stop bank_client_1 bank_client_2 bank_client_3
    
    Print-Info "Stopping Aggregation Server..."
    docker compose stop aggregation_server
    
    # Stop MLflow last
    Print-Info "Stopping MLflow..."
    docker compose stop mlflow
    
    Print-Success "All services stopped gracefully"
}

# Function to restart a specific service
function Restart-Service {
    param([string]$ServiceName)
    
    Print-Info "Restarting $ServiceName..."
    docker compose restart $ServiceName
    
    # Check health if service has healthcheck
    try {
        $hasHealth = docker inspect --format='{{.State.Health}}' $ServiceName 2>$null
        if ($hasHealth) {
            Check-ServiceHealth $ServiceName
        }
    } catch {
        # No health check
    }
    
    Print-Success "$ServiceName restarted"
}

# Function to show service status
function Show-Status {
    Print-Info "Service Status:"
    docker compose ps
    
    Write-Host ""
    Print-Info "Health Status:"
    $services = @("mlflow_server", "aggregation_server", "prometheus", "grafana")
    
    foreach ($service in $services) {
        $running = docker ps --filter "name=$service" --format "{{.Names}}" 2>$null
        if ($running -eq $service) {
            try {
                $health = docker inspect --format='{{.State.Health.Status}}' $service 2>$null
                if (-not $health) { $health = "no healthcheck" }
                Write-Host "  $service`: $health"
            } catch {
                Write-Host "  $service`: no healthcheck"
            }
        }
    }
}

# Function to show logs
function Show-Logs {
    param([string]$ServiceName)
    
    if ([string]::IsNullOrEmpty($ServiceName)) {
        docker compose logs -f
    } else {
        docker compose logs -f $ServiceName
    }
}

# Function to execute command in service
function Exec-Command {
    param(
        [string]$ServiceName,
        [string[]]$Command
    )
    
    docker compose exec $ServiceName $Command
}

# Function to clean up everything
function Cleanup {
    Print-Warning "This will remove all containers, networks, and volumes"
    $confirm = Read-Host "Are you sure? (yes/no)"
    
    if ($confirm -eq "yes") {
        Print-Info "Cleaning up..."
        docker compose down -v
        Print-Success "Cleanup complete"
    } else {
        Print-Info "Cleanup cancelled"
    }
}

# Function to validate configuration
function Validate-Config {
    Print-Info "Validating docker-compose configuration..."
    try {
        docker compose config | Out-Null
        Print-Success "Configuration is valid"
        return $true
    } catch {
        Print-Error "Configuration is invalid"
        docker compose config
        return $false
    }
}

# Function to show network info
function Show-Network {
    Print-Info "Network Information:"
    try {
        docker network inspect docker_fl_network
    } catch {
        Print-Warning "Network not created yet"
    }
}

# Function to show volume info
function Show-Volumes {
    Print-Info "Volume Information:"
    docker volume ls --filter "name=docker_"
}

# Main command handler
switch ($Command) {
    "start" {
        if (-not (Check-Docker)) { exit 1 }
        if (-not (Validate-Config)) { exit 1 }
        Start-Services
        Show-Status
    }
    "stop" {
        if (-not (Check-Docker)) { exit 1 }
        Stop-Services
    }
    "restart" {
        if (-not (Check-Docker)) { exit 1 }
        if ([string]::IsNullOrEmpty($Service)) {
            Print-Info "Restarting all services..."
            Stop-Services
            Start-Services
        } else {
            Restart-Service $Service
        }
    }
    "status" {
        if (-not (Check-Docker)) { exit 1 }
        Show-Status
    }
    "logs" {
        if (-not (Check-Docker)) { exit 1 }
        Show-Logs $Service
    }
    "exec" {
        if (-not (Check-Docker)) { exit 1 }
        if ([string]::IsNullOrEmpty($Service)) {
            Print-Error "Usage: .\orchestrate.ps1 exec <service> <command>"
            exit 1
        }
        Exec-Command $Service $Args
    }
    "health" {
        if (-not (Check-Docker)) { exit 1 }
        if ([string]::IsNullOrEmpty($Service)) {
            Print-Error "Usage: .\orchestrate.ps1 health <service>"
            exit 1
        }
        Check-ServiceHealth $Service
    }
    "validate" {
        Validate-Config
    }
    "network" {
        if (-not (Check-Docker)) { exit 1 }
        Show-Network
    }
    "volumes" {
        if (-not (Check-Docker)) { exit 1 }
        Show-Volumes
    }
    "cleanup" {
        if (-not (Check-Docker)) { exit 1 }
        Cleanup
    }
    default {
        Write-Host "Usage: .\orchestrate.ps1 {start|stop|restart [service]|status|logs [service]|exec <service> <command>|health <service>|validate|network|volumes|cleanup}"
        Write-Host ""
        Write-Host "Commands:"
        Write-Host "  start              - Start all services in correct order"
        Write-Host "  stop               - Stop all services gracefully"
        Write-Host "  restart [service]  - Restart all services or specific service"
        Write-Host "  status             - Show service status and health"
        Write-Host "  logs [service]     - Show logs (all or specific service)"
        Write-Host "  exec <service> <cmd> - Execute command in service"
        Write-Host "  health <service>   - Check health of specific service"
        Write-Host "  validate           - Validate docker-compose configuration"
        Write-Host "  network            - Show network information"
        Write-Host "  volumes            - Show volume information"
        Write-Host "  cleanup            - Remove all containers, networks, and volumes"
        exit 1
    }
}
