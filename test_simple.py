#!/usr/bin/env python3
"""
Simple in-process test: Run aggregator in thread and test weight transmission.

This eliminates subprocess overhead and shows console errors directly.
"""

import os
import sys
import threading
import time
import pickle
import json
import requests
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from communication.edge_node.train_local import train_local, get_model_weights
from communication.edge_node.dataset import load_dataset, get_data_loaders


def start_aggregator_thread():
    """Start aggregator Flask app in a thread."""
    print("\n[1/5] Starting aggregator in thread...")

    # Import here to avoid circular imports
    from regional_aggregator.aggregator import RegionalAggregator

    # Create aggregator instance
    aggregator = RegionalAggregator(
        region_id='A',
        farm_ids=[0, 1],
        port=5001,
        num_classes=6
    )

    # Run in thread
    thread = threading.Thread(
        target=lambda: aggregator.app.run(
            host='localhost',
            port=5001,
            debug=False,
            use_reloader=False,
            threaded=True
        ),
        daemon=True
    )
    thread.start()

    print(f"  ✓ Aggregator started in thread")

    # Wait for server to start
    print("\n[2/5] Waiting for aggregator to be ready...")
    for i in range(10):
        try:
            response = requests.get('http://localhost:5001/status', timeout=1)
            if response.status_code == 200:
                print(f"  ✓ Aggregator is ready")
                return aggregator
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)

    print(f"  ✗ Aggregator did not respond")
    return None


def train_farm_model():
    """Train a simple model for Farm 0."""
    print("\n[3/5] Training Farm 0 model (1 epoch)...")

    try:
        # Load dataset
        all_image_paths, all_labels, class_names = load_dataset(
            'data/plantvillage/', num_classes=6
        )

        # Get data loader for Farm 0
        train_loader, val_loader = get_data_loaders(
            all_image_paths, all_labels, farm_id=0, num_farms=6,
            batch_size=32, num_workers=0
        )
        num_samples = len(train_loader.dataset)

        # Train for 1 epoch
        model, metrics = train_local(
            farm_id=0,
            dataset_root='data/plantvillage/',
            num_epochs=1,
            batch_size=32,
            learning_rate=0.001,
            num_classes=6,
            device='cpu'
        )

        print(f"  ✓ Training complete")
        print(f"    Final Val Accuracy: {metrics['final_val_accuracy']:.4f}")
        print(f"    Training samples: {num_samples}")

        return model, num_samples

    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def send_raw_pickle_weights(model, num_samples):
    """Send raw pickle bytes of model weights."""
    print("\n[4/5] Sending raw pickle weights to aggregator...")

    try:
        # Get model weights
        weights = get_model_weights(model)
        print(f"  Model has {len(weights)} parameter tensors")

        # Serialize to pickle bytes
        serialized = pickle.dumps(weights)
        print(f"  Serialized to {len(serialized):,} bytes")

        # Prepare metadata JSON (send as header)
        metadata = {
            'farm_id': 0,
            'round_number': 1,
            'num_samples': num_samples
        }

        # Send raw pickle bytes with metadata headers
        headers = {
            'Content-Type': 'application/octet-stream',
            'X-Farm-Id': '0',
            'X-Round-Number': '1',
            'X-Num-Samples': str(num_samples)
        }

        print(f"  Sending POST request to /upload_weights...")
        print(f"    Body: {len(serialized):,} bytes of pickle data")
        print(f"    Headers: {headers}")

        response = requests.post(
            'http://localhost:5001/upload_weights',
            data=serialized,
            headers=headers,
            timeout=120
        )

        print(f"\n  Response status: {response.status_code}")
        print(f"  Response headers: {dict(response.headers)}")
        print(f"  Response body: {response.text}")

        # Try to parse as JSON
        try:
            json_response = response.json()
            print(f"\n  Response JSON:")
            print(f"    {json.dumps(json_response, indent=6)}")
        except:
            pass

        if response.status_code == 200:
            print(f"\n  ✓ Weights sent successfully!")
            return True
        else:
            print(f"\n  ✗ Send returned status {response.status_code}")
            return False

    except Exception as e:
        print(f"  ✗ Send failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_aggregator_state(aggregator):
    """Check what aggregator received."""
    print("\n[5/5] Checking aggregator state...")

    try:
        response = requests.get('http://localhost:5001/status', timeout=5)
        status = response.json()

        print(f"  ✓ Aggregator status retrieved")
        print(f"    Weights received this round: {status.get('weights_received_this_round', 0)}")
        print(f"    Farms in region: {status.get('farms_in_region', 0)}")
        print(f"    Current round: {status.get('current_round', 0)}")

        # Check if weights were actually stored
        if status.get('weights_received_this_round', 0) > 0:
            print(f"\n  ✓ ✓ ✓ SUCCESS - Weights received! ✓ ✓ ✓")
            return True
        else:
            print(f"\n  ✗ No weights received")
            return False

    except Exception as e:
        print(f"  ✗ Status check failed: {e}")
        return False


def main():
    """Main test execution."""
    print("="*70)
    print("SIMPLE IN-PROCESS WEIGHT TRANSMISSION TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Step 1: Start aggregator in thread
    aggregator = start_aggregator_thread()
    if not aggregator:
        print("\n[ABORTED] Aggregator failed to start")
        return False

    # Step 2: Not needed, already waited above

    # Step 3: Train model
    model, num_samples = train_farm_model()
    if model is None:
        print("\n[ABORTED] Training failed")
        return False

    # Step 4: Send weights
    success = send_raw_pickle_weights(model, num_samples)
    if not success:
        print("\n[ABORTED] Failed to send weights")
        # Don't abort yet, still check state

    # Step 5: Check state
    weights_received = check_aggregator_state(aggregator)

    print("\n" + "="*70)
    if weights_received:
        print("RESULT: ✓ SUCCESS - Weights transmitted and received!")
    else:
        print("RESULT: ✗ FAILURE - Weights did not reach aggregator")
    print("="*70 + "\n")

    return weights_received


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
