#!/usr/bin/env python
"""
Quick integration test: Validate model architecture + batch norm + flow decay
Run: python test_integration.py
"""
import torch
import numpy as np
from omegaconf import OmegaConf
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_forward():
    """Test model forward pass with random data"""
    print("[TEST] Loading config...")
    cfg = OmegaConf.load("configs/config_a1_10k_p64_b64.yml")
    
    print("[TEST] Initializing model...")
    from src.interpolator import Interpolator
    model = Interpolator(cfg)
    model.eval()
    
    # Random input: batch_size=2, channels=1, height=64, width=64
    batch_size, channels, h, w = 2, 1, 64, 64
    x0 = torch.randn(batch_size, channels, h, w)
    x1 = torch.randn(batch_size, channels, h, w)
    time = torch.full((batch_size, 1), 0.5)
    
    print(f"[TEST] Forward pass: x0={x0.shape}, x1={x1.shape}, time={time.shape}")
    with torch.no_grad():
        pred = model(x0, x1, time)
    
    print(f"[TEST] ✓ Output shape: {pred.shape}")
    assert pred.shape == (batch_size, channels, h, w), f"Expected {(batch_size, channels, h, w)}, got {pred.shape}"
    print("[TEST] ✓ Forward pass successful")
    
    # Count batch norm layers
    bn_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d))
    print(f"[TEST] ✓ Batch norm layers: {bn_count}")
    
    return model

def test_loss():
    """Test loss computation"""
    print("\n[TEST] Testing loss function...")
    cfg = OmegaConf.load("configs/config_a1_10k_p64_b64.yml")
    
    from src.loss import CombinedLoss
    loss_fn = CombinedLoss(cfg)
    
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    epoch = 50
    
    loss = loss_fn(pred, target, epoch)
    print(f"[TEST] ✓ Loss value: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    print("[TEST] ✓ Loss computation successful")

def test_data_loader():
    """Test data loader with validation"""
    print("\n[TEST] Testing data loader...")
    from src.data_loader import CustomDataset
    
    # Create minimal test
    cfg = OmegaConf.load("configs/config_a1_10k_p64_b64.yml")
    print(f"[TEST] Config loaded. Image dir: {cfg.dir.image}")
    
    # Just test the class instantiation and validation logic
    dataset = CustomDataset(
        record_file=cfg.file.dataset_dict,
        img_dir=cfg.dir.image,
        img_map=cfg.data.interpolator_images_map,
        shuffle=True,
        train_val_test_ratio=[0.8, 0.1, 0.1]
    )
    print(f"[TEST] ✓ CustomDataset created (ratio validation passed)")

if __name__ == "__main__":
    try:
        print("="*60)
        print("HiCInterpolate Integration Test")
        print("="*60)
        
        test_model_forward()
        test_loss()
        test_data_loader()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        print("\nNext steps:")
        print("1. Prepare small dataset (10-20 samples) with triplets")
        print("2. Run: python hicinterpolate.py --config config_test -train -test")
        print("3. Monitor loss convergence and metrics in logs")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
