"""
Federated Learning Module

This module contains components for federated learning implementation using
the Flower framework, including bank clients and aggregation server.
"""

from .bank_client import Bank_Client
from .aggregation_server import Aggregation_Server, create_aggregation_server

__all__ = ['Bank_Client', 'Aggregation_Server', 'create_aggregation_server']
