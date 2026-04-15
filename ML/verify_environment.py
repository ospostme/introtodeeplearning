#!/usr/bin/env python3
"""Verify PyTorch CUDA environment setup"""

import sys

def check_pytorch():
    """Check PyTorch and CUDA availability"""
    try:
        import torch
        print(f"✓ PyTorch:     {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA avail:  {cuda_available}")
        
        if cuda_available:
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU:          {torch.cuda.get_device_name(0)}")
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU memory:   {gpu_mem:.1f} GB")
            
            # Quick compute test
            x = torch.randn(100, 100).cuda()
            _ = x @ x.T
            print(f"✓ GPU compute:  Working")
        else:
            print("✗ CUDA not available!")
            return False
            
        return True
    except Exception as e:
        print(f"✗ PyTorch error: {e}")
        return False

def check_torchvision():
    """Check TorchVision"""
    try:
        import torchvision
        print(f"✓ TorchVision:  {torchvision.__version__}")
        return True
    except Exception as e:
        print(f"✗ TorchVision error: {e}")
        return False

def check_torchaudio():
    """Check TorchAudio"""
    try:
        import torchaudio
        print(f"✓ TorchAudio:   {torchaudio.__version__}")
        return True
    except Exception as e:
        print(f"✗ TorchAudio error: {e}")
        return False

def check_triton():
    """Check Triton compiler"""
    try:
        import triton
        print(f"✓ Triton:       {triton.__version__}")
        return True
    except ImportError:
        print(f"⚠ Triton:       Not available (optional)")
        return True  # Not critical
    except Exception as e:
        print(f"✗ Triton error: {e}")
        return False

def check_unsloth():
    """Check Unsloth (optional)"""
    try:
        import unsloth
        print(f"✓ Unsloth:      Available")
        return True
    except ImportError:
        print(f"⚠ Unsloth:      Not installed")
        return True  # Not critical for base check
    except Exception as e:
        print(f"✗ Unsloth error: {e}")
        return False

def main():
    print("="*60)
    print("PyTorch CUDA Environment Verification")
    print("="*60)
    
    results = []
    results.append(check_pytorch())
    results.append(check_torchvision())
    results.append(check_torchaudio())
    results.append(check_triton())
    results.append(check_unsloth())
    
    print("="*60)
    
    if all(results):
        print("✓ All checks passed!")
        return 0
    else:
        print("✗ Some checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
