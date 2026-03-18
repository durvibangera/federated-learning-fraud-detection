# Federated Learning Setup Guide

## Overview

You now have a complete federated learning system with:
- **Bank_Client**: Flower client for local training at each bank
- **Aggregation_Server**: Central server coordinating FL rounds with FedProx
- **Example Script**: Complete working example

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Aggregation Server                         │
│                  (FedProx Strategy)                          │
│                   Port: 8080                                 │
└──────────────┬──────────────┬──────────────┬────────────────┘
               │              │              │
               │              │              │
        ┌──────▼──────┐ ┌────▼──────┐ ┌────▼──────┐
        │   Bank 1    │ │  Bank 2   │ │  Bank 3   │
        │  (Client)   │ │ (Client)  │ │ (Client)  │
        │             │ │           │ │           │
        │ Data: W     │ │ Data: H,R │ │ Data: S,C │
        └─────────────┘ └───────────┘ └───────────┘
```

## Components Implemented

### 1. Bank_Client (`src/federated/bank_client.py`)
- Extends Flower's `NumPyClient`
- Methods:
  - `get_parameters()` - Extract model weights
  - `set_parameters()` - Update model weights
  - `fit()` - Local training (5 epochs per round)
  - `evaluate()` - Local validation
- Features:
  - Differential privacy support (Opacus ready)
  - Comprehensive metrics tracking
  - Privacy budget monitoring

### 2. Aggregation_Server (`src/federated/aggregation_server.py`)
- Coordinates federated learning rounds
- FedProx strategy with proximal_mu=0.01
- Features:
  - Handles client failures gracefully
  - Aggregates training/evaluation metrics
  - Configurable minimum clients (default: 2)
  - 30 FL rounds by default

### 3. Example Script (`examples/federated_learning_example.py`)
- Complete working example
- Demonstrates server + 3 clients setup
- Includes data preparation and partitioning

## How to Run Federated Learning

### Option 1: Using the Example Script (Recommended for Testing)

**Terminal 1 - Start Server:**
```bash
python examples/federated_learning_example.py --mode server --rounds 5
```

**Terminal 2 - Start Bank 1:**
```bash
python examples/federated_learning_example.py --mode client --bank-id bank_1
```

**Terminal 3 - Start Bank 2:**
```bash
python examples/federated_learning_example.py --mode client --bank-id bank_2
```

**Terminal 4 - Start Bank 3:**
```bash
python examples/federated_learning_example.py --mode client --bank-id bank_3
```

### Option 2: Custom Implementation

```python
# Server side
from federated import Aggregation_Server

server = Aggregation_Server(
    num_rounds=30,
    min_clients=2,
    proximal_mu=0.01
)
history = server.start_federated_learning()

# Client side
from federated import Bank_Client

client = Bank_Client(
    bank_id="bank_1",
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    local_epochs=5
)

import flwr as fl
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client
)
```

## Key Configuration Parameters

### Server Configuration
- `num_rounds`: Number of FL rounds (default: 30)
- `min_clients`: Minimum clients required per round (default: 2)
- `proximal_mu`: FedProx proximal term (default: 0.01)
- `server_address`: Server listening address (default: "[::]:8080")

### Client Configuration
- `bank_id`: Unique identifier for the bank
- `local_epochs`: Epochs to train locally per round (default: 5)
- `learning_rate`: Learning rate for local optimizer (default: 0.001)
- `privacy_engine`: Optional Opacus privacy engine

## Data Partitioning

Banks are assigned data based on ProductCD:
- **Bank 1**: ProductCD = 'W'
- **Bank 2**: ProductCD = 'H' or 'R'
- **Bank 3**: ProductCD = 'S' or 'C'

This creates heterogeneous data distributions across banks, which is why we use FedProx.

## Expected Workflow

1. **Server starts** and waits for clients
2. **Clients connect** to server
3. **For each FL round:**
   - Server sends global model to clients
   - Clients train locally (5 epochs)
   - Clients send updated weights to server
   - Server aggregates weights using FedProx
   - Server evaluates global model
4. **After 30 rounds:** Training complete

## Metrics Tracked

### Training Metrics (per round)
- Train loss
- Number of training examples
- Local epochs completed
- Privacy budget (ε, δ) if using DP

### Evaluation Metrics (per round)
- Validation loss
- AUPRC (Area Under Precision-Recall Curve)
- AUROC (Area Under ROC Curve)
- Number of validation examples

## Next Steps

1. **Test the system**: Run the example script with 5 rounds
2. **Add differential privacy**: Integrate Opacus privacy engine (Task 9)
3. **Add fault tolerance**: Implement retry logic and checkpointing (Task 7.3)
4. **Add MLOps monitoring**: Integrate MLflow and Prometheus (Tasks 12-13)

## Troubleshooting

**Issue**: Clients can't connect to server
- **Solution**: Ensure server is running first, check firewall settings

**Issue**: "Minimum clients not available"
- **Solution**: Start at least `min_clients` (default: 2) before FL begins

**Issue**: Out of memory errors
- **Solution**: Reduce batch size in dataloaders or use smaller model

## Files Created

- `src/federated/bank_client.py` - Bank client implementation
- `src/federated/aggregation_server.py` - Aggregation server implementation
- `src/federated/__init__.py` - Module exports
- `examples/federated_learning_example.py` - Complete working example
- `tests/test_bank_client.py` - Unit tests

## Status

✅ **Phase 1 Complete**: Foundation and Data Processing
✅ **Task 6.1 Complete**: Bank_Client implementation
✅ **Task 7.1 Complete**: Aggregation_Server implementation

**Ready for**: Testing federated learning end-to-end!
