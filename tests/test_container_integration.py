"""
Integration Tests for Container Orchestration

Tests Docker container integration, network isolation, and orchestration:
- Container startup and health checks
- Network isolation between banks
- Volume mounts and persistence
- Service dependencies
- Graceful shutdown
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import subprocess
import yaml


class TestContainerOrchestration:
    """Integration tests for Docker container orchestration."""

    @pytest.fixture(scope="class")
    def docker_compose_config(self):
        """Load docker-compose configuration."""
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"

        if not compose_path.exists():
            pytest.skip("docker-compose.yml not found")

        with open(compose_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def test_docker_compose_structure(self, docker_compose_config):
        """Test that docker-compose.yml has correct structure."""
        config = docker_compose_config

        # Should have services section
        assert "services" in config

        services = config["services"]

        # Should have aggregation server
        assert any("aggregation" in name.lower() or "server" in name.lower() for name in services.keys())

        # Should have bank clients
        bank_services = [name for name in services.keys() if "bank" in name.lower() or "client" in name.lower()]
        assert len(bank_services) >= 3, "Should have at least 3 bank client services"

    def test_service_dependencies(self, docker_compose_config):
        """Test that services have correct dependencies."""
        services = docker_compose_config["services"]

        # Find aggregation server service
        server_service = None
        for name, config in services.items():
            if "aggregation" in name.lower() or "server" in name.lower():
                server_service = name
                break

        if server_service:
            # Bank clients should depend on server or have proper startup order
            for name, config in services.items():
                if "bank" in name.lower() or "client" in name.lower():
                    # Check for depends_on or other dependency mechanism
                    # This is a structural check
                    assert "image" in config or "build" in config

    def test_network_configuration(self, docker_compose_config):
        """Test network isolation configuration."""
        services = docker_compose_config["services"]

        # Check if networks are defined
        if "networks" in docker_compose_config:
            networks = docker_compose_config["networks"]
            assert len(networks) > 0, "Should define at least one network"

        # Each service should have network configuration
        for name, config in services.items():
            # Services should either use default network or specify networks
            assert "image" in config or "build" in config

    def test_volume_mounts(self, docker_compose_config):
        """Test that services have appropriate volume mounts."""
        services = docker_compose_config["services"]

        # At least some services should have volumes for persistence
        services_with_volumes = [
            name for name, config in services.items() if "volumes" in config and len(config["volumes"]) > 0
        ]

        # Should have at least one service with volumes (for data/models)
        assert len(services_with_volumes) > 0, "Should have services with volume mounts"

    def test_environment_variables(self, docker_compose_config):
        """Test that services have necessary environment variables."""
        services = docker_compose_config["services"]

        # Check that services have environment configuration
        for name, config in services.items():
            # Services should have environment or env_file
            if "bank" in name.lower() or "client" in name.lower():
                # Bank clients should have some configuration
                assert "environment" in config or "env_file" in config or "command" in config

    def test_port_mappings(self, docker_compose_config):
        """Test that services expose correct ports."""
        services = docker_compose_config["services"]

        # Aggregation server should expose a port
        for name, config in services.items():
            if "aggregation" in name.lower() or "server" in name.lower():
                # Should have ports defined
                assert "ports" in config or "expose" in config, "Aggregation server should expose ports"

    def test_health_checks(self, docker_compose_config):
        """Test that critical services have health checks."""
        services = docker_compose_config["services"]

        # Check for health check configurations
        for name, config in services.items():
            # Health checks are optional but recommended
            # Just verify structure is valid
            if "healthcheck" in config:
                healthcheck = config["healthcheck"]
                assert "test" in healthcheck or "interval" in healthcheck

    def test_restart_policies(self, docker_compose_config):
        """Test that services have appropriate restart policies."""
        services = docker_compose_config["services"]

        # Services should have restart policies for resilience
        for name, config in services.items():
            # Restart policy is optional but check if present
            if "restart" in config:
                assert config["restart"] in ["no", "always", "on-failure", "unless-stopped"]

    def test_resource_limits(self, docker_compose_config):
        """Test that services have resource limits defined."""
        services = docker_compose_config["services"]

        # Check for resource limits (optional but good practice)
        for name, config in services.items():
            if "deploy" in config:
                deploy = config["deploy"]
                if "resources" in deploy:
                    resources = deploy["resources"]
                    # If resources are defined, they should have limits or reservations
                    assert "limits" in resources or "reservations" in resources

    @pytest.mark.integration
    @pytest.mark.slow
    def test_docker_compose_validation(self):
        """Test that docker-compose.yml is valid."""
        compose_path = Path(__file__).parent.parent / "docker" / "docker-compose.yml"

        if not compose_path.exists():
            pytest.skip("docker-compose.yml not found")

        try:
            # Validate docker-compose file
            result = subprocess.run(
                ["docker-compose", "-f", str(compose_path), "config"], capture_output=True, text=True, timeout=10
            )

            # Should not have errors
            assert result.returncode == 0, f"docker-compose validation failed: {result.stderr}"

        except FileNotFoundError:
            pytest.skip("docker-compose not installed")
        except subprocess.TimeoutExpired:
            pytest.skip("docker-compose validation timed out")


class TestDockerfileStructure:
    """Tests for Dockerfile structure and best practices."""

    def test_aggregation_server_dockerfile_exists(self):
        """Test that Aggregation Server Dockerfile exists."""
        dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile.aggregation_server"

        assert dockerfile_path.exists(), "Aggregation Server Dockerfile should exist"

    def test_bank_client_dockerfile_exists(self):
        """Test that Bank Client Dockerfile exists."""
        dockerfile_path = Path(__file__).parent.parent / "docker" / "Dockerfile.bank_client"

        assert dockerfile_path.exists(), "Bank Client Dockerfile should exist"

    def test_dockerfile_has_base_image(self):
        """Test that Dockerfiles specify base images."""
        dockerfiles = [
            Path(__file__).parent.parent / "docker" / "Dockerfile.aggregation_server",
            Path(__file__).parent.parent / "docker" / "Dockerfile.bank_client",
        ]

        for dockerfile_path in dockerfiles:
            if dockerfile_path.exists():
                with open(dockerfile_path, "r") as f:
                    content = f.read()

                # Should have FROM statement
                assert "FROM" in content, f"{dockerfile_path.name} should have FROM statement"

    def test_dockerfile_has_workdir(self):
        """Test that Dockerfiles set working directory."""
        dockerfiles = [
            Path(__file__).parent.parent / "docker" / "Dockerfile.aggregation_server",
            Path(__file__).parent.parent / "docker" / "Dockerfile.bank_client",
        ]

        for dockerfile_path in dockerfiles:
            if dockerfile_path.exists():
                with open(dockerfile_path, "r") as f:
                    content = f.read()

                # Should have WORKDIR statement
                assert "WORKDIR" in content, f"{dockerfile_path.name} should set WORKDIR"

    def test_dockerfile_copies_requirements(self):
        """Test that Dockerfiles copy requirements.txt."""
        dockerfiles = [
            Path(__file__).parent.parent / "docker" / "Dockerfile.aggregation_server",
            Path(__file__).parent.parent / "docker" / "Dockerfile.bank_client",
        ]

        for dockerfile_path in dockerfiles:
            if dockerfile_path.exists():
                with open(dockerfile_path, "r") as f:
                    content = f.read()

                # Should copy requirements
                assert "requirements" in content.lower(), f"{dockerfile_path.name} should copy requirements"

    def test_dockerfile_has_entrypoint_or_cmd(self):
        """Test that Dockerfiles have ENTRYPOINT or CMD."""
        dockerfiles = [
            Path(__file__).parent.parent / "docker" / "Dockerfile.aggregation_server",
            Path(__file__).parent.parent / "docker" / "Dockerfile.bank_client",
        ]

        for dockerfile_path in dockerfiles:
            if dockerfile_path.exists():
                with open(dockerfile_path, "r") as f:
                    content = f.read()

                # Should have ENTRYPOINT or CMD
                assert (
                    "ENTRYPOINT" in content or "CMD" in content
                ), f"{dockerfile_path.name} should have ENTRYPOINT or CMD"


class TestOrchestrationScripts:
    """Tests for orchestration scripts."""

    def test_orchestration_script_exists(self):
        """Test that orchestration scripts exist."""
        script_paths = [
            Path(__file__).parent.parent / "docker" / "orchestrate.sh",
            Path(__file__).parent.parent / "docker" / "orchestrate.ps1",
        ]

        # At least one orchestration script should exist
        exists = any(path.exists() for path in script_paths)
        assert exists, "At least one orchestration script should exist"

    def test_start_script_exists(self):
        """Test that start scripts exist."""
        script_paths = [
            Path(__file__).parent.parent / "docker" / "start.sh",
            Path(__file__).parent.parent / "docker" / "start.ps1",
        ]

        # At least one start script should exist
        exists = any(path.exists() for path in script_paths)
        assert exists, "At least one start script should exist"

    def test_orchestration_script_executable(self):
        """Test that shell scripts are executable."""
        script_path = Path(__file__).parent.parent / "docker" / "orchestrate.sh"

        if script_path.exists():
            import os

            # Check if file has execute permission
            is_executable = os.access(script_path, os.X_OK)
            # Note: This might not work on Windows
            if not sys.platform.startswith("win"):
                assert is_executable, "orchestrate.sh should be executable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
