#!/usr/bin/env python3
"""
Test script to verify SNR layer configurations work correctly
"""

import torch
from src.models.constellation_model import ConstellationResNet
from src.models.vision_transformer_model import ConstellationVisionTransformer
from src.models.swin_transformer_model import ConstellationSwinTransformer

def test_model_with_config(model_class, model_name, snr_config):
    """Test a model with a specific SNR configuration"""
    print(f"\nTesting {model_name} with snr_layer_config='{snr_config}'")
    
    try:
        if model_class == ConstellationResNet:
            model = model_class(
                num_classes=17,
                snr_classes=16,
                input_channels=1,
                dropout_prob=0.3,
                model_name=model_name,
                snr_layer_config=snr_config
            )
        elif model_class == ConstellationVisionTransformer:
            model = model_class(
                num_classes=17,
                snr_classes=16,
                input_channels=1,
                dropout_prob=0.3,
                patch_size=32,
                snr_layer_config=snr_config
            )
        else:  # ConstellationSwinTransformer
            model = model_class(
                num_classes=17,
                snr_classes=16,
                input_channels=1,
                dropout_prob=0.3,
                model_variant=model_name,
                use_pretrained=False,
                snr_layer_config=snr_config
            )
        
        # Test forward pass
        dummy_input = torch.randn(2, 1, 224, 224)
        mod_out, snr_out = model(dummy_input)
        
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Modulation output shape: {mod_out.shape}")
        print(f"  ✓ SNR output shape: {snr_out.shape}")
        
        # Count SNR head parameters
        snr_params = sum(p.numel() for p in model.snr_head.parameters())
        print(f"  ✓ SNR head parameters: {snr_params:,}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

def main():
    print("Testing SNR layer configurations across all models")
    print("=" * 60)
    
    snr_configs = ['standard', 'bottleneck_64', 'bottleneck_128', 'dual_layer']
    
    # Test ResNet models
    print("\n### Testing ResNet Models ###")
    for config in snr_configs:
        test_model_with_config(ConstellationResNet, 'resnet18', config)
    
    # Test Vision Transformer models
    print("\n### Testing Vision Transformer Models ###")
    for config in snr_configs:
        test_model_with_config(ConstellationVisionTransformer, 'vit_b_32', config)
    
    # Test Swin Transformer models
    print("\n### Testing Swin Transformer Models ###")
    for config in snr_configs:
        test_model_with_config(ConstellationSwinTransformer, 'swin_tiny', config)
    
    print("\n" + "=" * 60)
    print("All tests completed!")

if __name__ == "__main__":
    main()