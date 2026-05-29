"""
Weight transmission module for edge nodes.

After local training, each farm sends its model weights to the regional aggregator
via HTTP POST request. This implements communication in the federated learning loop.

The weights are serialized to bytes and sent with metadata (farm_id, round_number, etc).
The aggregator acknowledges receipt and sends back the aggregated weights for next round.
"""

import sys
import os
import requests
import pickle
import json
import time
from typing import Dict, Any

# Fix Windows terminal encoding issues
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


def serialize_weights(weights_dict):
    """
    Serialize model weights (state dict) to bytes for transmission.

    Args:
        weights_dict (dict): PyTorch model state_dict()

    Returns:
        bytes: Serialized weights
    """
    serialized = pickle.dumps(weights_dict)
    return serialized


def deserialize_weights(serialized_bytes):
    """
    Deserialize weights from bytes back to dictionary.

    Args:
        serialized_bytes (bytes): Serialized weight data

    Returns:
        dict: PyTorch state_dict
    """
    weights_dict = pickle.loads(serialized_bytes)
    return weights_dict


def create_completion_flag(farm_id, round_number):
    """
    Create a flag file to signal that farm has sent weights.

    Args:
        farm_id (int): Farm identifier
        round_number (int): FL round number
    """
    flags_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'flags')
    os.makedirs(flags_dir, exist_ok=True)

    flag_file = os.path.join(flags_dir, f'round_{round_number}_farm_{farm_id}_done.txt')
    with open(flag_file, 'w') as f:
        f.write(f'Farm {farm_id} sent weights for round {round_number}\n')

    print(f"  [OK] Flag created: {flag_file}")


def send_weights_to_aggregator(farm_id, weights_dict, aggregator_url, round_number=1,
                               num_samples=1000, retries=3, timeout=60):
    """
    Send trained weights to regional aggregator as raw pickle bytes.

    Args:
        farm_id (int): Farm identifier
        weights_dict (dict): Model state_dict from training
        aggregator_url (str): URL of regional aggregator (e.g., 'http://localhost:5001')
        round_number (int): Current FL round number
        num_samples (int): Number of samples used in local training
        retries (int): Number of retries if request fails
        timeout (int): Request timeout in seconds

    Returns:
        dict: Response from aggregator
    """
    # Serialize weights to pickle bytes
    serialized_weights = serialize_weights(weights_dict)

    # Send as raw bytes with metadata in headers
    endpoint = f"{aggregator_url}/upload_weights"
    headers = {
        'Content-Type': 'application/octet-stream',
        'X-Farm-Id': str(farm_id),
        'X-Round-Number': str(round_number),
        'X-Num-Samples': str(num_samples)
    }

    attempt = 0
    last_error = None

    while attempt < retries:
        try:
            print(f"\n[Farm {farm_id}] Sending weights to Region aggregator...")
            print(f"  Endpoint: {endpoint}")
            print(f"  Content-Type: application/octet-stream")
            print(f"  Payload: {len(serialized_weights):,} bytes of pickle data")
            print(f"  Metadata: farm_id={farm_id}, round={round_number}, samples={num_samples}")

            response = requests.post(
                endpoint,
                data=serialized_weights,
                headers=headers,
                timeout=timeout
            )

            # Check status
            if response.status_code >= 400:
                print(f"  [FAIL] Server returned {response.status_code}")
                print(f"  Response: {response.text}")
                raise requests.exceptions.HTTPError(
                    f"HTTP {response.status_code}: {response.text}"
                )

            result = response.json()

            print(f"  [OK] Weights sent successfully!")
            print(f"  Response status: {result.get('status', 'unknown')}")
            print(f"  Message: {result.get('message', 'N/A')}")

            # Create completion flag file for orchestrator to detect
            create_completion_flag(farm_id, round_number)

            return result

        except requests.exceptions.Timeout:
            last_error = f"Request timeout ({timeout}s)"
            attempt += 1
            print(f"  [FAIL] Timeout (attempt {attempt}/{retries})")
            if attempt < retries:
                wait_time = 2 ** attempt
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)

        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {str(e)}"
            attempt += 1
            print(f"  [FAIL] Connection error (attempt {attempt}/{retries})")
            print(f"    Is the aggregator running on {endpoint}?")
            if attempt < retries:
                wait_time = 2 ** attempt
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            last_error = str(e)
            attempt += 1
            print(f"  [FAIL] Request failed: {last_error} (attempt {attempt}/{retries})")
            if attempt < retries:
                wait_time = 2 ** attempt
                print(f"    Retrying in {wait_time}s...")
                time.sleep(wait_time)

    # All retries failed
    error_msg = f"Failed to send weights after {retries} attempts: {last_error}"
    print(f"  [FAIL] {error_msg}")
    raise RuntimeError(error_msg)


def receive_weights_from_aggregator(serialized_response):
    """
    Extract weights from aggregator response (deserialization).

    Args:
        serialized_response (bytes or dict): Response containing aggregated weights

    Returns:
        dict: Aggregated model weights for next round
    """
    if isinstance(serialized_response, bytes):
        return deserialize_weights(serialized_response)
    elif isinstance(serialized_response, dict):
        # If response is already JSON with 'weights' field
        if 'weights' in serialized_response:
            return deserialize_weights(serialized_response['weights'])
        return serialized_response
    else:
        raise TypeError(f"Unexpected response type: {type(serialized_response)}")


def broadcast_weights_to_aggregator(farm_ids, weights_dict_list, aggregator_url,
                                   round_number=1, num_samples_list=None):
    """
    Send weights from multiple farms (batch operation for testing).

    Args:
        farm_ids (list): List of farm IDs
        weights_dict_list (list): List of weight dictionaries (one per farm)
        aggregator_url (str): Aggregator endpoint
        round_number (int): FL round number
        num_samples_list (list): Samples per farm (default: equal distribution)

    Returns:
        list: List of responses from aggregator
    """
    if num_samples_list is None:
        num_samples_list = [1000] * len(farm_ids)

    responses = []
    for farm_id, weights, num_samples in zip(farm_ids, weights_dict_list, num_samples_list):
        try:
            response = send_weights_to_aggregator(
                farm_id=farm_id,
                weights_dict=weights,
                aggregator_url=aggregator_url,
                round_number=round_number,
                num_samples=num_samples
            )
            responses.append(response)
        except RuntimeError as e:
            print(f"Failed to send weights from farm {farm_id}: {e}")
            responses.append(None)

    return responses


if __name__ == "__main__":
    # Example usage: test weight transmission
    import argparse
    from communication.edge_node.train_local import get_model_weights
    from model.cnn_model import LeafDiseaseNet

    parser = argparse.ArgumentParser(description="Send weights to aggregator")
    parser.add_argument('--farm_id', type=int, default=0, help='Farm ID')
    parser.add_argument('--aggregator_url', type=str, default='http://localhost:5001',
                       help='Aggregator URL')
    parser.add_argument('--round', type=int, default=1, help='FL round number')

    args = parser.parse_args()

    # Create dummy model
    model = LeafDiseaseNet(num_classes=6)
    weights = get_model_weights(model)

    # Send to aggregator
    try:
        response = send_weights_to_aggregator(
            farm_id=args.farm_id,
            weights_dict=weights,
            aggregator_url=args.aggregator_url,
            round_number=args.round,
            num_samples=1000
        )
        print(f"\nResponse: {response}")
    except RuntimeError as e:
        print(f"\nError: {e}")
