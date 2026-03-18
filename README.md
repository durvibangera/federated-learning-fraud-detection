# Federated Fraud Intelligence Network

A privacy-preserving federated learning system that enables multiple financial institutions to collaboratively train fraud detection models without sharing raw transaction data. Built with PyTorch, Flower framework, and comprehensive MLOps monitoring.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- IEEE-CIS Fraud Detection dataset from Kaggle

### Setup

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd federated-fraud-detection
python scripts/setup_environment.py
```

2. **Activate virtual environment:**
```bash
# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. **Place dataset files:**
```
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv (optional)
└── test_identity.csv (optional)
```

4. **Verify installation:**
```bash
python -c "from src.config.config_manager import get_config; print('✅ Configuration loaded successfully')"
```

## 📁 Project Structure

```
federated-fraud-detection/
├── data/
│   ├── raw/                    # IEEE-CIS dataset files
│   └── splits/                 # Auto-generated bank data splits
├── src/
│   ├── data/                   # Data preprocessing
│   ├── model/                  # Neural network architecture
│   ├── federated/              # Flower client/server
│   ├── privacy/                # Differential privacy (Opacus)
│   ├── explainability/         # SHAP model interpretation
│   ├── monitoring/             # MLflow, Prometheus, Grafana
│   ├── config/                 # Configuration management
│   └── utils/                  # Utility functions
├── docker/                     # Container configurations
├── monitoring/                 # Grafana dashboards
├── notebooks/                  # Jupyter experimentation
├── tests/                      # Test suite
├── config/                     # YAML configuration
└── scripts/                    # Setup and utility scripts
```

## 🔧 Configuration

The system uses YAML configuration with environment variable overrides:

- **Base config:** `config/config.yaml`
- **Environment overrides:** Use `FFD_` prefix (e.g., `FFD_MODEL__LEARNING_RATE=0.01`)

Key configuration sections:
- `federated_learning`: FL rounds, strategy, client settings
- `model`: Architecture, hyperparameters
- `privacy`: Differential privacy parameters
- `data`: Preprocessing and splitting
- `monitoring`: MLOps and logging

## 🏗️ Development Phases

### Phase 1: Foundation (Tasks 1-5)
- ✅ Project setup and configuration
- 🔄 Data preprocessing and model architecture
- 🔄 Jupyter notebook for experimentation

### Phase 2: Federated Learning (Tasks 6-8)
- 🔄 Flower client implementation
- 🔄 FedProx aggregation server
- 🔄 Fault tolerance and error handling

### Phase 3: Privacy & Compliance (Tasks 9-11)
- 🔄 Opacus differential privacy
- 🔄 SHAP explainability
- 🔄 Privacy-utility analysis

### Phase 4: MLOps (Tasks 12-14)
- 🔄 MLflow experiment tracking
- 🔄 Prometheus metrics
- 🔄 Grafana dashboards

### Phase 5: Infrastructure (Tasks 15-18)
- 🔄 Docker containerization
- 🔄 Configuration management
- 🔄 Resource management

### Phase 6: Testing & Integration (Tasks 19-22)
- 🔄 Property-based testing
- 🔄 CI/CD pipeline
- 🔄 End-to-end validation

## 🧪 Testing

The system uses dual testing approach:

```bash
# Run all tests
pytest tests/

# Run property-based tests only
pytest tests/ -k "property"

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐳 Docker Deployment

```bash
# Build and run the federated system
docker-compose up --build

# Access services:
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090  
# - Grafana: http://localhost:3000
```

## 📊 Key Features

- **Privacy-Preserving:** Differential privacy with configurable ε budgets
- **Federated Learning:** 3-bank simulation with FedProx aggregation
- **MLOps Ready:** Complete monitoring with MLflow, Prometheus, Grafana
- **Explainable AI:** SHAP-based model interpretations for compliance
- **Production Ready:** Docker deployment, comprehensive testing, CI/CD

## 🔒 Privacy Guarantees

The system provides formal differential privacy guarantees:
- Configurable ε values: [0.5, 1.0, 2.0, 4.0, 8.0]
- Privacy budget tracking across FL rounds
- Privacy-utility curve analysis
- Opacus integration for gradient noise injection

## 📈 Performance Metrics

- **Primary:** AUPRC (Area Under Precision-Recall Curve)
- **Secondary:** AUROC (Area Under ROC Curve)
- **Privacy:** ε-spent tracking
- **System:** Round duration, convergence metrics

## 🤝 Contributing

1. Follow the task-based development approach in `tasks.md`
2. Run tests before committing: `pytest tests/`
3. Use pre-commit hooks for code quality
4. Update documentation for new features

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Python version errors:** Ensure Python 3.10+
2. **Dependency conflicts:** Use fresh virtual environment
3. **Memory issues:** Reduce batch size in config
4. **Docker issues:** Check port availability (5000, 8080, 9090, 3000)

### Getting Help

- Check logs in `logs/` directory
- Review configuration in `config/config.yaml`
- Run diagnostics: `python -m src.utils.diagnostics`

---

Built with ❤️ for privacy-preserving machine learning