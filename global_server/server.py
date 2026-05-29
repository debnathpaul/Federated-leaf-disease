"""
Global Server orchestrator for 3-tier federated learning.

The global server:
1. Manages overall FL training process (rounds)
2. Communicates with regional aggregators
3. Implements global FedAvg aggregation
4. Coordinates the training loop

3-Tier Hierarchy:
    Global Server (this file)
        └─ Regional Aggregator A (port 5001)
           ├─ Farm 0
           └─ Farm 1
        └─ Regional Aggregator B (port 5002)
           ├─ Farm 2
           └─ Farm 3
        └─ Regional Aggregator C (port 5003)
           ├─ Farm 4
           └─ Farm 5
"""

import os
import sys
import torch
import requests
import time
import base64
from flask import Flask, jsonify
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__) + '/..')

from model import LeafDiseaseNet
from communication.edge_node.send_weights import deserialize_weights, serialize_weights


class GlobalServer:
    """
    Global federated learning server.

    Coordinates training across 3 regions, each with multiple farms.
    Implements FedAvg at global level combining regional aggregates.

    Attributes:
        num_rounds (int): Number of FL rounds
        aggregator_ports (dict): Regional aggregator ports {region: port}
        current_round (int): Current FL round
        global_model (nn.Module): Global model
        global_weights (dict): Global model weights
    """

    def __init__(self, num_rounds=10, num_classes=6):
        """
        Args:
            num_rounds (int): Number of federated learning rounds
            num_classes (int): Number of disease classes
        """
        self.num_rounds = num_rounds
        self.num_classes = num_classes
        self.current_round = 0

        # Regional aggregator configuration
        # Region -> (port, farms in region)
        self.regions = {
            'A': (5001, [0, 1]),
            'B': (5002, [2, 3]),
            'C': (5003, [4, 5])
        }

        # Initialize global model
        self.global_model = LeafDiseaseNet(num_classes=num_classes)
        self.global_weights = self.global_model.state_dict()

        # Storage for aggregated weights from each region
        self.region_weights = {}  # {region: (weights, num_samples)}

        # Flask app for REST API
        self.app = Flask("GlobalServer")
        self._setup_routes()

        print("="*70)
        print("Global Federated Learning Server Initialized")
        print("="*70)
        print(f"Number of FL Rounds: {num_rounds}")
        print(f"Regions: {list(self.regions.keys())}")
        for region, (port, farms) in self.regions.items():
            print(f"  Region {region}: Port {port}, Farms {farms}")
        print()
        print("IMPORTANT: Regional aggregators must be running on ports 5001-5003")
        print("in separate terminals before starting this global server.")

    def _setup_routes(self):
        """Setup Flask routes for global server."""

        @self.app.route('/status', methods=['GET'])
        def status():
            """Return global server status."""
            return jsonify({
                'global_round': self.current_round,
                'total_rounds': self.num_rounds,
                'status': 'training' if self.current_round < self.num_rounds else 'complete'
            })

        @self.app.route('/get_global_weights', methods=['GET'])
        def get_global_weights():
            """Return current global model weights."""
            try:
                serialized = serialize_weights(self.global_weights)
                return {
                    'weights': serialized,
                    'round': self.current_round,
                    'status': 'success'
                }
            except Exception as e:
                return jsonify({'status': 'error', 'message': str(e)}), 400


    def fetch_regional_weights(self, region, port, timeout=120):
        """
        Fetch aggregated weights from regional aggregator.

        Args:
            region (str): Region ID
            port (int): Aggregator port
            timeout (int): Max wait time in seconds

        Returns:
            dict: Aggregated weights from region
        """
        url = f"http://localhost:{port}/get_aggregated_weights"
        start_time = time.time()
        poll_count = 0

        while time.time() - start_time < timeout:
            try:
                print(f"\n  [Fetch {region}] GET {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                result = response.json()
                status = result.get('status', 'unknown')

                print(f"    Status: {status}")

                if status == 'success':
                    # Weights are ready - decode from base64
                    weights_b64 = result.get('weights')
                    if not weights_b64:
                        raise ValueError("Response missing 'weights' field")

                    print(f"    [OK] Decoding base64 weights...")
                    weights_bytes = base64.b64decode(weights_b64)
                    weights_dict = deserialize_weights(weights_bytes)

                    print(f"    [OK] Region {region}: Aggregated weights ready ({len(weights_bytes):,} bytes)")
                    return weights_dict

                elif status == 'waiting':
                    # Farms still training or not all weights received
                    elapsed = time.time() - start_time
                    progress = result.get('progress', '?')
                    total = result.get('total', '?')
                    poll_count += 1
                    message = result.get('message', 'waiting')
                    print(f"    [WAIT] {progress}/{total} - {message}")
                    time.sleep(2)  # Poll every 2 seconds

                else:
                    # Unknown status
                    elapsed = time.time() - start_time
                    print(f"    [WARN] Unknown status '{status}' ({int(elapsed)}s)...")
                    time.sleep(2)

            except requests.exceptions.ConnectionError as e:
                elapsed = time.time() - start_time
                print(f"    [ERROR] Cannot connect to {region} ({int(elapsed)}s): {e}")
                time.sleep(2)
            except requests.exceptions.RequestException as e:
                elapsed = time.time() - start_time
                print(f"    [ERROR] Request error on {region} ({int(elapsed)}s): {e}")
                time.sleep(2)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"    [ERROR] Failed to process {region} response ({int(elapsed)}s): {e}")
                import traceback
                traceback.print_exc()
                time.sleep(2)

        raise TimeoutError(
            f"Timeout waiting for Region {region} weights (max {timeout}s). "
            f"Make sure farms are training and sending weights."
        )

    def global_fedavg(self):
        """
        Perform global FedAvg: aggregate weights from all regions.

        Each region contributes proportionally to number of farms.
        """
        print(f"\n[Round {self.current_round}] Global FedAvg Aggregation")

        region_weights = {}

        # Fetch weights from each region
        for region, (port, farms) in self.regions.items():
            try:
                weights = self.fetch_regional_weights(region, port)
                # Weight by number of farms in region
                region_weights[region] = (weights, len(farms))
            except TimeoutError as e:
                print(f"  [ERROR] {e}")
                return False

        # Simple average across regions (equal weighting)
        # Alternative: weight by number of samples in each region
        num_regions = len(region_weights)
        aggregated_weights = {}

        first_weights = list(region_weights.values())[0][0]
        for param_name in first_weights.keys():
            aggregated_weights[param_name] = torch.zeros_like(first_weights[param_name])

        for region, (weights_dict, num_farms) in region_weights.items():
            weight_ratio = 1.0 / num_regions  # Equal weighting across regions

            for param_name, param in weights_dict.items():
                aggregated_weights[param_name] += weight_ratio * param

        self.global_weights = aggregated_weights

        print(f"  [OK] Global FedAvg complete")
        print(f"  Regions aggregated: {len(region_weights)}")

        return True

    def run_fedavg_round(self):
        """Execute one round of federated averaging."""
        self.current_round += 1

        print(f"\n{'='*70}")
        print(f"FL ROUND {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Wait for farms to train and send weights to aggregators
        print(f"\nWaiting for farms to train and aggregators to process...")
        print(f"Farms should train during this time and send weights to:")
        for region, (port, farms) in self.regions.items():
            print(f"  Region {region} (port {port}): Farms {farms}")
        print(f"\nMonitoring aggregators...")
        time.sleep(10)

        # Step 2: Fetch and aggregate globally
        success = self.global_fedavg()

        if not success:
            print(f"[ERROR] Round {self.current_round} failed")
            return False

        print(f"\n[OK] Round {self.current_round} complete")

        return True

    def run_training_loop(self):
        """Execute full federated learning training loop."""
        print(f"\n{'#'*70}")
        print(f"STARTING FEDERATED LEARNING")
        print(f"Total Rounds: {self.num_rounds}")
        print(f"{'#'*70}\n")

        start_time = time.time()

        for round_num in range(self.num_rounds):
            success = self.run_fedavg_round()

            if not success:
                print(f"\n[ERROR] Training failed at round {self.current_round}")
                break

            time.sleep(5)  # Pause between rounds

        elapsed = time.time() - start_time

        print(f"\n{'#'*70}")
        print(f"FEDERATED LEARNING COMPLETE")
        print(f"Total Time: {elapsed:.2f}s")
        print(f"Rounds Completed: {self.current_round}/{self.num_rounds}")
        print(f"{'#'*70}\n")

    def run(self, port=5000):
        """Start global server Flask app."""
        print(f"\nGlobal Server listening on port {port}")
        self.app.run(host='localhost', port=port, debug=False, use_reloader=False)


def main():
    """Main entry point for global server."""
    import argparse

    parser = argparse.ArgumentParser(description="Global Federated Learning Server")
    parser.add_argument('--rounds', type=int, default=10, help='Number of FL rounds')
    parser.add_argument('--classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--port', type=int, default=5000, help='Server port')

    args = parser.parse_args()

    server = GlobalServer(num_rounds=args.rounds, num_classes=args.classes)

    # Run training loop
    server.run_training_loop()


if __name__ == "__main__":
    main()
