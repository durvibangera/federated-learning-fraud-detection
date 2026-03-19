"""
Aggregation Server Implementation for Federated Learning

This module implements the central aggregation server that coordinates
federated learning rounds using the FedProx strategy.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import flwr as fl
from flwr.server.strategy import FedProx
from flwr.server import ServerConfig
from typing import Dict, List, Optional, Tuple

from loguru import logger


class Aggregation_Server:
    """
    Central aggregation server for federated learning.

    This class coordinates federated learning rounds across multiple bank clients,
    implements FedProx aggregation strategy, and handles client failures gracefully.

    Attributes:
        num_rounds: Number of federated learning rounds to execute
        min_clients: Minimum number of clients required per round
        min_available_clients: Minimum clients that must be available
        proximal_mu: FedProx proximal term parameter (default: 0.01)
        strategy: Flower aggregation strategy (FedProx)
        server_address: Address for server to listen on
    """

    def __init__(
        self,
        num_rounds: int = 30,
        min_clients: int = 2,
        min_available_clients: int = 2,
        proximal_mu: float = 0.01,
        server_address: str = "[::]:8080",
    ):
        """
        Initialize Aggregation Server with FedProx strategy.

        Args:
            num_rounds: Number of FL rounds to execute (default: 30)
            min_clients: Minimum clients required per round (default: 2)
            min_available_clients: Minimum clients that must be available (default: 2)
            proximal_mu: FedProx proximal term for handling heterogeneous data (default: 0.01)
            server_address: Server address in format "[host]:port" (default: "[::]:8080")
        """
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.min_available_clients = min_available_clients
        self.proximal_mu = proximal_mu
        self.server_address = server_address

        # Initialize FedProx strategy
        self.strategy = FedProx(
            fraction_fit=1.0,  # Use all available clients for training
            fraction_evaluate=1.0,  # Use all available clients for evaluation
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_available_clients,
            proximal_mu=proximal_mu,
            on_fit_config_fn=self._get_fit_config,
            on_evaluate_config_fn=self._get_evaluate_config,
            evaluate_metrics_aggregation_fn=self._aggregate_evaluate_metrics,
            fit_metrics_aggregation_fn=self._aggregate_fit_metrics,
        )

        logger.info(f"Initialized Aggregation Server with FedProx (μ={proximal_mu})")
        logger.info(f"Configuration: {num_rounds} rounds, min {min_clients} clients")

    def _get_fit_config(self, server_round: int) -> Dict[str, str]:
        """
        Generate configuration for client training.

        Args:
            server_round: Current FL round number

        Returns:
            Configuration dictionary sent to clients
        """
        config = {
            "round": str(server_round),
            "local_epochs": "5",  # Number of local epochs per round
            "learning_rate": "0.001",
        }
        logger.debug(f"Round {server_round}: Sending fit config to clients")
        return config

    def _get_evaluate_config(self, server_round: int) -> Dict[str, str]:
        """
        Generate configuration for client evaluation.

        Args:
            server_round: Current FL round number

        Returns:
            Configuration dictionary sent to clients
        """
        config = {"round": str(server_round)}
        return config

    def _aggregate_fit_metrics(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """
        Aggregate training metrics from all clients.

        Args:
            metrics: List of (num_examples, metrics_dict) from each client

        Returns:
            Aggregated metrics dictionary
        """
        if not metrics:
            return {}

        # Calculate weighted average of metrics
        total_examples = sum(num_examples for num_examples, _ in metrics)

        aggregated = {}
        for key in metrics[0][1].keys():
            if key in ["epsilon", "delta"]:  # Privacy metrics - take max
                aggregated[key] = max(m[key] for _, m in metrics if key in m)
            else:  # Other metrics - weighted average
                weighted_sum = sum(num_examples * m.get(key, 0.0) for num_examples, m in metrics)
                aggregated[key] = weighted_sum / total_examples

        logger.info(f"Aggregated training metrics: {aggregated}")
        return aggregated

    def _aggregate_evaluate_metrics(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """
        Aggregate evaluation metrics from all clients.

        Args:
            metrics: List of (num_examples, metrics_dict) from each client

        Returns:
            Aggregated metrics dictionary
        """
        if not metrics:
            return {}

        # Calculate weighted average of metrics
        total_examples = sum(num_examples for num_examples, _ in metrics)

        aggregated = {}
        for key in metrics[0][1].keys():
            weighted_sum = sum(num_examples * m.get(key, 0.0) for num_examples, m in metrics)
            aggregated[key] = weighted_sum / total_examples

        logger.info(f"Aggregated evaluation metrics: {aggregated}")
        return aggregated

    def start_federated_learning(self, num_rounds: Optional[int] = None) -> fl.server.History:
        """
        Start federated learning process.

        This method starts the Flower server and coordinates federated learning
        rounds with connected clients. It uses the FedProx strategy for aggregation.

        Args:
            num_rounds: Number of FL rounds (overrides initialization value if provided)

        Returns:
            Flower History object containing training history
        """
        rounds = num_rounds if num_rounds is not None else self.num_rounds

        logger.info("=" * 70)
        logger.info(f"Starting Federated Learning: {rounds} rounds")
        logger.info(f"Server listening on: {self.server_address}")
        logger.info(f"Strategy: FedProx (proximal_mu={self.proximal_mu})")
        logger.info("=" * 70)

        # Configure server
        config = ServerConfig(num_rounds=rounds)

        # Start Flower server
        try:
            history = fl.server.start_server(server_address=self.server_address, config=config, strategy=self.strategy)

            logger.info("=" * 70)
            logger.info("Federated Learning Complete!")
            logger.info(f"Total rounds completed: {len(history.losses_distributed)}")
            logger.info("=" * 70)

            return history

        except Exception as e:
            logger.error(f"Federated learning failed: {str(e)}")
            raise

    def handle_client_failure(self, client_id: str) -> None:
        """
        Handle client failure during federated learning.

        The FedProx strategy automatically handles client failures by continuing
        with remaining clients if minimum requirements are met.

        Args:
            client_id: Identifier of the failed client
        """
        logger.warning(f"Client {client_id} failed or disconnected")
        logger.info(f"Continuing with remaining clients (min required: {self.min_clients})")


def create_aggregation_server(
    num_rounds: int = 30, min_clients: int = 2, proximal_mu: float = 0.01, server_address: str = "[::]:8080"
) -> Aggregation_Server:
    """
    Factory function to create an Aggregation Server.

    Args:
        num_rounds: Number of federated learning rounds
        min_clients: Minimum number of clients required
        proximal_mu: FedProx proximal term parameter
        server_address: Server address to listen on

    Returns:
        Configured Aggregation_Server instance
    """
    return Aggregation_Server(
        num_rounds=num_rounds,
        min_clients=min_clients,
        min_available_clients=min_clients,
        proximal_mu=proximal_mu,
        server_address=server_address,
    )
