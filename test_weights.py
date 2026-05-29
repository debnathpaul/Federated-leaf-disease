#!/usr/bin/env python3
"""
Debug test script: Verify weight transmission from Farm to Aggregator.

This script:
1. Starts Regional Aggregator A on port 5001
2. Trains Farm 0 for 1 epoch
3. Sends weights to aggregator
4. Checks aggregator status
5. Reports SUCCESS or FAILURE with diagnostic info
"""

import subprocess
import time
import requests
import json
import sys
import os
import signal
import pickle
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from communication.edge_node.train_local import train_local, get_model_weights, get_data_loaders
from communication.edge_node.dataset import load_dataset
from communication.edge_node.send_weights import send_weights_to_aggregator


def start_aggregator():
    """Start Region A aggregator as subprocess."""
    print("\n[1/7] Starting Region A aggregator on port 5001...")
    try:
        proc = subprocess.Popen(
            [sys.executable, '-m', 'regional_aggregator.aggregator',
             '--region', 'A', '--farms', '0', '1', '--port', '5001'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"  ✓ Aggregator subprocess started (PID: {proc.pid})")
        return proc
    except Exception as e:
        print(f"  ✗ Failed to start aggregator: {e}")
        return None


def wait_for_aggregator(timeout=5):
    """Wait for aggregator to be ready."""
    print("\n[2/7] Waiting for aggregator to initialize...")
    time.sleep(2)  # Give process time to start

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get('http://localhost:5001/status', timeout=1)
            if response.status_code == 200:
                print(f"  ✓ Aggregator is ready (took {time.time()-start:.1f}s)")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)

    print(f"  ✗ Aggregator did not respond within {timeout}s")
    return False


def train_farm_0():
    """Train Farm 0 for 1 epoch and return model, num_samples."""
    print("\n[3/7] Training Farm 0 for 1 epoch...")
    try:
        # Load dataset to get sample count
        all_image_paths, all_labels, class_names = load_dataset(
            'data/plantvillage/', num_classes=6
        )
        print(f"  Dataset loaded: {len(all_image_paths)} total images")

        # Get data loaders for Farm 0
        train_loader, val_loader = get_data_loaders(
            all_image_paths, all_labels, farm_id=0, num_farms=6,
            batch_size=32, num_workers=0
        )
        num_train_samples = len(train_loader.dataset)
        print(f"  Farm 0 assigned: {num_train_samples} training samples")

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

        return model, num_train_samples

    except Exception as e:
        print(f"  ✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def test_debug_endpoint(weights_dict, num_samples):
    """Send to debug /test_upload endpoint first."""
    print("\n[4a/7] Testing debug endpoint: http://localhost:5001/test_upload...")

    try:
        import pickle

        # Serialize same way as send_weights.py
        serialized_weights = pickle.dumps(weights_dict)
        print(f"  Weights serialized: {len(serialized_weights):,} bytes")

        # Prepare metadata
        metadata = {
            'farm_id': 0,
            'round_number': 1,
            'num_samples': num_samples,
            'timestamp': time.time()
        }

        # Create multipart request like send_weights.py does
        files = {
            'weights': ('weights.pkl', serialized_weights, 'application/octet-stream'),
            'metadata': (None, json.dumps(metadata), 'application/json')
        }

        print(f"  Sending multipart POST to /test_upload...")
        response = requests.post(
            'http://localhost:5001/test_upload',
            files=files,
            timeout=120
        )

        print(f"  Response status: {response.status_code}")
        print(f"  Response body: {response.text}")

        try:
            json_response = response.json()
            print(f"  Response JSON: {json.dumps(json_response, indent=2)}")
        except:
            pass

        return response

    except Exception as e:
        print(f"  ✗ Debug endpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def send_weights_to_agg(model, num_samples):
    """Send weights from Farm 0 to aggregator."""
    print("\n[4/7] Sending weights to http://localhost:5001/upload_weights...")

    try:
        weights = get_model_weights(model)
        print(f"  Model has {len(weights)} parameter tensors")

        response = send_weights_to_aggregator(
            farm_id=0,
            weights_dict=weights,
            aggregator_url='http://localhost:5001',
            round_number=1,
            num_samples=num_samples,
            retries=3,
            timeout=120
        )

        print(f"  ✓ Weights sent successfully!")
        print(f"  Aggregator response: {response}")
        return response

    except Exception as e:
        print(f"  ✗ SEND FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_aggregator_status():
    """Check aggregator status."""
    print("\n[5/7] Checking aggregator status at http://localhost:5001/status...")

    try:
        response = requests.get('http://localhost:5001/status', timeout=5)
        status = response.json()
        print(f"  ✓ Status retrieved")

        return status

    except Exception as e:
        print(f"  ✗ Status check failed: {e}")
        return None


def print_full_response(status):
    """Print full aggregator status response."""
    print("\n[6/7] Full Aggregator Status Response:")
    print("  " + "="*66)
    if status:
        for key, value in status.items():
            print(f"  {key}: {value}")
    else:
        print("  [No response received]")
    print("  " + "="*66)


def determine_success(status):
    """Determine if weights were received."""
    print("\n[7/7] Final Result:")
    print("  " + "="*66)

    if not status:
        print("  ✗ FAILURE - Could not reach aggregator")
        return False

    weights_received = status.get('weights_received_this_round', 0)
    farms_in_region = status.get('farms_in_region', 0)
    current_round = status.get('current_round', 0)

    print(f"  Weights received this round: {weights_received}/{farms_in_region} farms")
    print(f"  Current round: {current_round}")

    if weights_received >= 1:
        print(f"\n  ✓ ✓ ✓ SUCCESS - Weights reached the aggregator! ✓ ✓ ✓")
        return True
    else:
        print(f"\n  ✗ ✗ ✗ FAILURE - Aggregator received 0 weights ✗ ✗ ✗")
        print(f"\n  Diagnostic hints:")
        print(f"    - Check if aggregator is truly listening on :5001")
        print(f"    - Verify /upload_weights endpoint is accessible")
        print(f"    - Check aggregator logs for errors")
        return False


def cleanup_aggregator(proc):
    """Terminate aggregator process."""
    if proc:
        print("\n[Cleanup] Terminating aggregator...")
        try:
            proc.terminate()
            proc.wait(timeout=2)
            print("  ✓ Aggregator terminated")
        except subprocess.TimeoutExpired:
            proc.kill()
            print("  ✓ Aggregator killed (forced)")


def main():
    """Main test execution."""
    print("\n" + "="*70)
    print("FEDERATED LEARNING WEIGHT TRANSMISSION DEBUG TEST")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    aggregator_proc = None

    try:
        # Step 1: Start aggregator
        aggregator_proc = start_aggregator()
        if not aggregator_proc:
            print("\n[ABORTED] Could not start aggregator")
            return False

        # Step 2: Wait for it to be ready
        if not wait_for_aggregator(timeout=5):
            print("\n[ABORTED] Aggregator did not respond")
            return False

        # Step 3: Train Farm 0
        model, num_samples = train_farm_0()
        if model is None:
            print("\n[ABORTED] Training failed")
            return False

        # Step 4a: Test debug endpoint first
        weights = get_model_weights(model)
        debug_response = test_debug_endpoint(weights, num_samples)

        # Step 4: Send weights to actual endpoint
        send_response = send_weights_to_agg(model, num_samples)
        if send_response is None:
            print("\n[ABORTED] Failed to send weights")
            return False

        # Step 5: Check status
        status = check_aggregator_status()

        # Step 6: Print full response
        print_full_response(status)

        # Step 7: Determine success
        success = determine_success(status)

        print("\n" + "="*70)
        print(f"Test ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")

        return success

    finally:
        cleanup_aggregator(aggregator_proc)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
