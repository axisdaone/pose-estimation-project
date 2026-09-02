#!/usr/bin/env python3
"""
Smoke Test for LST-LA-GCN — test_model.py

Verifies that:
    1. Each sub-module produces the correct output shape.
    2. The full network's forward pass runs without errors.
    3. The output shape is (B, 2) for binary classification.
    4. Gradients flow back through the entire computation graph.

Run from the project root:
    python -m ui_prmd_preprocess.model.test_model
"""

import sys
import torch

# ── Constants matching the UI-PRMD dataset ──────────────────────────────────
BATCH_SIZE = 4
IN_CHANNELS = 3
SEQ_LEN = 150
NUM_JOINTS = 20
NUM_CLASSES = 2


def test_graph_conv():
    """Test SpatialGraphConv in isolation."""
    from ui_prmd_preprocess.model.graph_conv import SpatialGraphConv

    gcn = SpatialGraphConv(
        in_channels=IN_CHANNELS, out_channels=64, num_joints=NUM_JOINTS
    )
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN, NUM_JOINTS)
    A = torch.randn(NUM_JOINTS, NUM_JOINTS)

    out = gcn(x, A)
    assert out.shape == (BATCH_SIZE, 64, SEQ_LEN, NUM_JOINTS), (
        f"SpatialGraphConv output shape mismatch: {out.shape}"
    )
    print(f"  ✓ SpatialGraphConv:        {out.shape}")


def test_temporal_conv():
    """Test MultiScaleTemporalConv with stride=1 and stride=2."""
    from ui_prmd_preprocess.model.temporal_conv import MultiScaleTemporalConv

    # stride=1: preserve temporal length
    tcn1 = MultiScaleTemporalConv(in_channels=64, out_channels=64, stride=1)
    x1 = torch.randn(BATCH_SIZE, 64, SEQ_LEN, NUM_JOINTS)
    out1 = tcn1(x1)
    assert out1.shape == (BATCH_SIZE, 64, SEQ_LEN, NUM_JOINTS), (
        f"TCN stride=1 shape mismatch: {out1.shape}"
    )
    print(f"  ✓ TemporalConv (stride=1): {out1.shape}")

    # stride=2: halve temporal length
    tcn2 = MultiScaleTemporalConv(in_channels=64, out_channels=128, stride=2)
    out2 = tcn2(x1)
    expected_T = SEQ_LEN // 2
    assert out2.shape == (BATCH_SIZE, 128, expected_T, NUM_JOINTS), (
        f"TCN stride=2 shape mismatch: {out2.shape}"
    )
    print(f"  ✓ TemporalConv (stride=2): {out2.shape}")


def test_attention():
    """Test TemporalAttention."""
    from ui_prmd_preprocess.model.attention import TemporalAttention

    attn = TemporalAttention(channels=64, reduction=4)
    x = torch.randn(BATCH_SIZE, 64, SEQ_LEN, NUM_JOINTS)
    out = attn(x)
    assert out.shape == x.shape, (
        f"TemporalAttention shape mismatch: {out.shape} vs {x.shape}"
    )
    print(f"  ✓ TemporalAttention:       {out.shape}")


def test_st_block():
    """Test STBlock with channel change and stride."""
    from ui_prmd_preprocess.model.st_block import STBlock

    block = STBlock(
        in_channels=IN_CHANNELS,
        out_channels=64,
        num_joints=NUM_JOINTS,
        stride=1,
    )
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN, NUM_JOINTS)
    A = torch.randn(NUM_JOINTS, NUM_JOINTS)
    out = block(x, A)
    assert out.shape == (BATCH_SIZE, 64, SEQ_LEN, NUM_JOINTS), (
        f"STBlock output shape mismatch: {out.shape}"
    )
    print(f"  ✓ STBlock (3→64, s=1):     {out.shape}")


def test_full_network():
    """Test the complete LSTLAGCN with dummy data matching the dataset."""
    from ui_prmd_preprocess.model.network import LSTLAGCN

    model = LSTLAGCN(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        num_joints=NUM_JOINTS,
        channels=(64, 128, 256),
        dropout=0.3,
    )

    # Dummy inputs matching UIPRMDDataset output
    joint = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN, NUM_JOINTS)
    bone = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN, NUM_JOINTS)
    A = torch.randn(BATCH_SIZE, NUM_JOINTS, NUM_JOINTS)  # Batched adjacency

    # Forward pass
    logits = model(joint, bone, A)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Network output shape mismatch: {logits.shape}"
    )
    print(f"  ✓ LSTLAGCN full network:   {logits.shape}")

    # Verify gradient flow
    loss = logits.sum()
    loss.backward()
    all_grads_ok = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    assert all_grads_ok, "Some parameters have no gradients!"
    print(f"  ✓ Gradient flow:           OK (all parameters receive gradients)")

    # Parameter count
    n_params = model.count_parameters()
    print(f"  ✓ Total parameters:        {n_params:,}")

    return model


def test_dataset_integration():
    """
    Test that the model works with the exact dict format returned by
    UIPRMDDataset.__getitem__ (simulated).
    """
    from ui_prmd_preprocess.model.network import LSTLAGCN

    model = LSTLAGCN()

    # Simulate a collated batch from the DataLoader
    batch = {
        "joint": torch.randn(BATCH_SIZE, 3, 150, 20),
        "bone": torch.randn(BATCH_SIZE, 3, 150, 20),
        "A": torch.randn(BATCH_SIZE, 20, 20),
        "label": torch.randint(0, 2, (BATCH_SIZE,)),
    }

    logits = model(batch["joint"], batch["bone"], batch["A"])
    assert logits.shape == (BATCH_SIZE, 2)

    # Verify loss computation
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, batch["label"])
    loss.backward()
    print(f"  ✓ Dataset integration:     loss = {loss.item():.4f}")


def main():
    print("=" * 60)
    print("LST-LA-GCN Smoke Test")
    print("=" * 60)
    print()

    tests = [
        ("SpatialGraphConv", test_graph_conv),
        ("MultiScaleTemporalConv", test_temporal_conv),
        ("TemporalAttention", test_attention),
        ("STBlock", test_st_block),
        ("Full LSTLAGCN Network", test_full_network),
        ("Dataset Integration", test_dataset_integration),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"Testing {name}...")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed! The model is ready for training.\n")


if __name__ == "__main__":
    main()
