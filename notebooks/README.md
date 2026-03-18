# Federated Fraud Detection - Experimentation Notebooks

This directory contains Jupyter notebooks for ML experimentation and model refinement for the federated fraud detection system.

## Notebooks Overview

### 1. `baseline_centralized.ipynb` - Main Experimentation
**Purpose**: Establish baseline performance and optimize model architecture

**Contents**:
- Data exploration and visualization of IEEE-CIS dataset
- Centralized model training (no federated learning)
- Model architecture comparison (5 different configurations)
- Hyperparameter tuning (learning rate, weight decay)
- Performance evaluation with AUPRC/AUROC metrics
- Results visualization and analysis

**Prerequisites**: 
- IEEE-CIS dataset in `../data/raw/` directory
- PyTorch installation: `pip install torch torchvision`

### 2. `architecture_experiments.ipynb` - Advanced Architecture Testing
**Purpose**: Deep dive into architecture optimization and validation

**Contents**:
- Embedding dimension optimization strategies
- Normalization layer comparison (GroupNorm vs LayerNorm)
- Loss function analysis (BCE vs Focal Loss)
- Opacus compatibility validation
- Regularization technique testing
- Architecture recommendations for federated learning

**Prerequisites**: 
- Completed baseline experiments
- Understanding of differential privacy requirements

### 3. `privacy_utility_baseline.ipynb` - Privacy-Utility Analysis
**Purpose**: Establish privacy-utility tradeoff baselines

**Contents**:
- Privacy budget framework (ε values: 0.5, 1.0, 2.0, 4.0, 8.0)
- Performance degradation modeling
- Federated learning impact estimation
- Privacy-utility curve visualization
- Evaluation metrics standardization
- Baseline documentation for future comparison

**Prerequisites**: 
- Baseline model performance metrics
- Understanding of differential privacy concepts

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Prepare Data**:
   - Place IEEE-CIS dataset files in `../data/raw/`:
     - `train_transaction.csv`
     - `train_identity.csv`
     - `test_transaction.csv`
     - `test_identity.csv`

3. **Run Notebooks in Order**:
   1. Start with `baseline_centralized.ipynb`
   2. Proceed to `architecture_experiments.ipynb`
   3. Finish with `privacy_utility_baseline.ipynb`

## Expected Outputs

### Performance Benchmarks
- **Centralized AUPRC**: ~0.75-0.85
- **Centralized AUROC**: ~0.85-0.92
- **Optimal Architecture**: [256, 128, 64] hidden layers with GroupNorm
- **Best Hyperparameters**: LR=0.001, WD=1e-5, Dropout=0.3

### Privacy-Utility Targets
- **Optimal Privacy Budget**: ε = 2.0 (85% performance retention)
- **Practical Range**: ε ∈ [1.0, 4.0]
- **Success Thresholds**: AUPRC ≥ 0.65, AUROC ≥ 0.75

### Generated Artifacts
- `../logs/baseline_metrics.json` - Final model performance
- `../logs/privacy_utility_baseline.json` - Privacy analysis results
- `../logs/privacy_utility_baseline.png` - Visualization charts

## Integration with Federated Learning

The optimized architecture and hyperparameters from these experiments will be used in:
- **Phase 2**: Federated learning client implementation
- **Phase 3**: Differential privacy integration with Opacus
- **Phase 4**: MLOps monitoring and evaluation

## Troubleshooting

**Common Issues**:
1. **PyTorch Import Error**: Install with `pip install torch torchvision`
2. **Data Not Found**: Ensure IEEE-CIS files are in `../data/raw/`
3. **Memory Issues**: Reduce batch size or use data sampling
4. **Plotting Issues**: Install with `pip install matplotlib seaborn`

**Performance Notes**:
- Notebooks are designed to work without GPU but will be faster with CUDA
- Use data sampling for faster experimentation during development
- Full experiments may take 30-60 minutes depending on hardware

## Next Steps

After completing these experiments:
1. Proceed to **Task 5**: Checkpoint validation
2. Begin **Phase 2**: Federated learning implementation
3. Implement **Differential Privacy** with validated architecture
4. Deploy **MLOps monitoring** with established metrics