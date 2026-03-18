"""
Verification script for Privacy_Engine implementation.

Tests basic functionality of the Privacy_Engine with a simple model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from privacy import Privacy_Engine
from model.fraud_mlp import FraudMLP

print("=" * 70)
print("PRIVACY ENGINE VERIFICATION")
print("=" * 70)

# Step 1: Create a simple model
print("\n[1/5] Creating test model...")
# FraudMLP expects categorical_embedding_dims as Dict[str, Tuple[vocab_size, embed_dim]]
categorical_embedding_dims = {
    'card1': (100, 16),  # vocab_size=100, embed_dim=16
    'card2': (50, 16)    # vocab_size=50, embed_dim=16
}
numerical_input_dim = 10
model = FraudMLP(
    categorical_embedding_dims=categorical_embedding_dims,
    numerical_input_dim=numerical_input_dim,
    hidden_dims=[64, 32],
    dropout_rate=0.3
)
print(f"✓ Created FraudMLP model with {sum(p.numel() for p in model.parameters())} parameters")

# Step 2: Validate model compatibility
print("\n[2/5] Validating Opacus compatibility...")
is_compatible, errors = Privacy_Engine.validate_model_compatibility(model)
if is_compatible:
    print("✓ Model is compatible with Opacus")
else:
    print(f"⚠ Model has {len(errors)} compatibility issues")
    print("  Applying automatic fixes...")
    model = Privacy_Engine.fix_model_compatibility(model)
    is_compatible, errors = Privacy_Engine.validate_model_compatibility(model)
    if is_compatible:
        print("✓ Model fixed and now compatible")
    else:
        print(f"✗ Model still has issues: {errors}")
        sys.exit(1)

# Step 3: Create dummy data
print("\n[3/5] Creating dummy dataset...")
batch_size = 32
num_samples = 128

# Create dummy features matching FraudMLP expected format
# FraudMLP expects features dict with 'categorical' and 'numerical' keys
card1 = torch.randint(0, 100, (num_samples,))
card2 = torch.randint(0, 50, (num_samples,))
categorical = torch.stack([card1, card2], dim=1)  # Shape: (num_samples, 2)
numerical = torch.randn(num_samples, numerical_input_dim)
targets = torch.randint(0, 2, (num_samples,)).float()

# Create dataset and dataloader
dataset = TensorDataset(categorical, numerical, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)
print(f"✓ Created dataset with {num_samples} samples, batch_size={batch_size}")

# Step 4: Initialize Privacy_Engine
print("\n[4/5] Initializing Privacy_Engine...")
privacy_engine = Privacy_Engine(
    epsilon=1.0,
    delta=1e-5,
    max_grad_norm=1.0,
    noise_multiplier=1.1
)
print(f"✓ Privacy_Engine initialized with ε=1.0, δ=1e-5")

# Step 5: Make model private
print("\n[5/5] Making model private with Opacus...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

try:
    private_model, private_optimizer, private_dataloader = privacy_engine.make_private(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        epochs=1
    )
    print("✓ Successfully attached Opacus privacy engine")
    
    # Get privacy summary
    summary = privacy_engine.get_privacy_summary()
    print(f"\nPrivacy Summary:")
    print(f"  Target ε: {summary['target_epsilon']:.2f}")
    print(f"  Current ε: {summary['epsilon_spent']:.4f}")
    print(f"  Remaining ε: {summary['remaining_epsilon']:.4f}")
    print(f"  δ: {summary['delta']:.2e}")
    print(f"  Max grad norm: {summary['max_grad_norm']:.2f}")
    print(f"  Noise multiplier: {summary['noise_multiplier']:.2f}")
    print(f"  Budget exhausted: {summary['budget_exhausted']}")
    
    # Test a training step
    print("\n[Bonus] Testing training step with DP...")
    private_model.train()
    criterion = nn.BCEWithLogitsLoss()
    
    for batch_idx, (cat_feat, num_feat, target) in enumerate(private_dataloader):
        features = {
            'categorical': cat_feat,
            'numerical': num_feat
        }
        
        private_optimizer.zero_grad()
        outputs = private_model(features)
        loss = criterion(outputs, target)
        loss.backward()
        private_optimizer.step()
        
        if batch_idx == 0:
            print(f"✓ Training step completed - Loss: {loss.item():.4f}")
            break
    
    # Check privacy spent after training
    epsilon_spent, delta = privacy_engine.get_privacy_spent()
    print(f"✓ Privacy spent after 1 batch: ε={epsilon_spent:.4f}, δ={delta:.2e}")
    
    print("\n" + "=" * 70)
    print("✅ PRIVACY ENGINE VERIFICATION SUCCESSFUL!")
    print("=" * 70)
    print("\nKey Components Verified:")
    print("✓ Privacy_Engine initialization")
    print("✓ Model compatibility validation")
    print("✓ Opacus integration")
    print("✓ Privacy budget tracking")
    print("✓ Training with differential privacy")
    
except Exception as e:
    print(f"\n✗ Error during privacy engine verification: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
