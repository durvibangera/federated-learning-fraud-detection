"""
Bank Client Implementation for Federated Learning

This module implements the Bank_Client class that extends Flower's NumPyClient
to enable federated learning for fraud detection across financial institutions.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import flwr as fl
from collections import OrderedDict

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from loguru import logger
from model.fraud_mlp import FraudMLP

logger = logger


class Bank_Client(fl.client.NumPyClient):
    """
    Flower client implementation representing a financial institution.
    
    This class wraps a PyTorch fraud detection model and implements the Flower
    client interface for federated learning. It handles local training, model
    weight extraction, and evaluation while ensuring no raw data leaves the client.
    
    Attributes:
        bank_id: Unique identifier for this bank client
        model: Local FraudMLP model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: PyTorch device (cpu or cuda)
        local_epochs: Number of local training epochs per FL round
        learning_rate: Learning rate for local optimizer
        privacy_engine: Optional differential privacy engine (Opacus)
    """
    
    def __init__(
        self,
        bank_id: str,
        model: FraudMLP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu',
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        privacy_engine: Optional[object] = None
    ):
        """
        Initialize Bank_Client with model and data.
        
        Args:
            bank_id: Unique identifier for this bank (e.g., "bank_1")
            model: FraudMLP model instance
            train_loader: DataLoader for local training data
            val_loader: DataLoader for local validation data
            device: Device to run training on ('cpu' or 'cuda')
            local_epochs: Number of epochs to train locally per FL round
            learning_rate: Learning rate for local optimizer
            privacy_engine: Optional Opacus privacy engine for differential privacy
        """
        super().__init__()
        
        self.bank_id = bank_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.privacy_engine = privacy_engine
        
        # Training metrics tracking
        self.training_history = {
            'rounds': [],
            'train_loss': [],
            'val_loss': [],
            'val_auprc': [],
            'val_auroc': []
        }
        
        logger.info(f"Initialized {bank_id} with {len(train_loader.dataset)} training samples")
    
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """
        Extract model parameters as numpy arrays.
        
        This method is called by the Flower framework to get the current model
        weights for aggregation. Only model weights are shared, never raw data.
        
        Args:
            config: Configuration dictionary from server (unused)
            
        Returns:
            List of numpy arrays containing model parameters
        """
        logger.debug(f"{self.bank_id}: Extracting model parameters")
        
        # Convert PyTorch parameters to numpy arrays
        parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
        logger.info(f"{self.bank_id}: Extracted {len(parameters)} parameter tensors")
        return parameters
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Update model with new parameters from server.
        
        Args:
            parameters: List of numpy arrays containing updated model weights
        """
        logger.debug(f"{self.bank_id}: Setting model parameters")
        
        # Convert numpy arrays back to PyTorch tensors
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        
        # Load updated parameters into model
        self.model.load_state_dict(state_dict, strict=True)
        
        logger.info(f"{self.bank_id}: Updated model with {len(parameters)} parameter tensors")
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally on bank's data partition.
        
        This method implements local training for one federated learning round.
        It updates the model with global parameters, trains for local_epochs,
        and returns the updated weights along with training metrics.
        
        Args:
            parameters: Global model parameters from aggregation server
            config: Configuration dictionary (can contain FL round number, etc.)
            
        Returns:
            Tuple containing:
                - Updated model parameters as numpy arrays
                - Number of training samples
                - Dictionary of training metrics
        """
        fl_round = config.get('round', 0)
        logger.info(f"{self.bank_id}: Starting local training for FL round {fl_round}")
        
        # Update model with global parameters
        self.set_parameters(parameters)
        
        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Calculate pos_weight for class imbalance
        # This should be calculated from training data
        pos_weight = torch.tensor([15.0]).to(self.device)  # From baseline experiments
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Apply differential privacy if privacy engine is provided
        if self.privacy_engine is not None:
            logger.info(f"{self.bank_id}: Applying differential privacy")
            self.model, optimizer, self.train_loader = self.privacy_engine.make_private(
                self.model, optimizer, self.train_loader
            )
        
        # Local training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for features, targets in self.train_loader:
                # Move data to device
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs.squeeze(), targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            total_loss += epoch_loss / len(self.train_loader)
            
            if (epoch + 1) % 2 == 0:
                logger.debug(f"{self.bank_id}: Epoch {epoch+1}/{self.local_epochs}, "
                           f"Loss={epoch_loss/len(self.train_loader):.4f}")
        
        avg_loss = total_loss / self.local_epochs
        num_examples = len(self.train_loader.dataset)
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config={})
        
        # Prepare metrics dictionary
        metrics = {
            'train_loss': avg_loss,
            'num_examples': num_examples,
            'local_epochs': self.local_epochs
        }
        
        # Add privacy budget if using differential privacy
        if self.privacy_engine is not None:
            epsilon, delta = self.privacy_engine.get_privacy_spent()
            metrics['epsilon'] = epsilon
            metrics['delta'] = delta
            logger.info(f"{self.bank_id}: Privacy spent - ε={epsilon:.2f}, δ={delta:.2e}")
        
        logger.info(f"{self.bank_id}: Completed local training - "
                   f"Loss={avg_loss:.4f}, Samples={num_examples}")
        
        return updated_parameters, num_examples, metrics
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local validation data.
        
        This method evaluates the global model on the bank's local validation
        set to assess performance without sharing data.
        
        Args:
            parameters: Global model parameters to evaluate
            config: Configuration dictionary
            
        Returns:
            Tuple containing:
                - Loss value
                - Number of evaluation samples
                - Dictionary of evaluation metrics (AUPRC, AUROC, etc.)
        """
        logger.info(f"{self.bank_id}: Evaluating model on local validation data")
        
        # Update model with parameters to evaluate
        self.set_parameters(parameters)
        
        # Set model to evaluation mode
        self.model.eval()
        
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in self.val_loader:
                # Move data to device
                features = {k: v.to(self.device) for k, v in features.items()}
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs.squeeze(), targets)
                
                # Collect predictions and targets for metrics
                probs = torch.sigmoid(outputs.squeeze())
                all_predictions.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        num_examples = len(self.val_loader.dataset)
        
        # Calculate AUPRC and AUROC
        from sklearn.metrics import average_precision_score, roc_auc_score
        
        auprc = average_precision_score(all_targets, all_predictions)
        auroc = roc_auc_score(all_targets, all_predictions)
        
        metrics = {
            'loss': avg_loss,
            'auprc': auprc,
            'auroc': auroc,
            'num_examples': num_examples
        }
        
        logger.info(f"{self.bank_id}: Evaluation complete - "
                   f"Loss={avg_loss:.4f}, AUPRC={auprc:.4f}, AUROC={auroc:.4f}")
        
        return avg_loss, num_examples, metrics
