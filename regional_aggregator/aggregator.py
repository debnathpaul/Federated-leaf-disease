"""
Regional Aggregator for 2-tier federated learning.

The regional aggregator sits between farm nodes and the global server.
It receives weights from multiple farms, aggregates them using Federated Averaging (FedAvg),
and sends aggregated weights back to farms for next round.

FedAvg Formula:
    w_agg = sum(n_i / n_total * w_i) for i in farms
    where n_i = number of samples at farm i, w_i = weights from farm i
"""

import os
import sys
import torch
import pickle
import json
import base64
import traceback
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.serving import WSGIRequestHandler
from typing import Dict, List, Tuple
from datetime import datetime

# Increase timeout for large weight uploads (1.6MB+)
WSGIRequestHandler.protocol_version = "HTTP/1.1"

# Fix Windows terminal encoding issues
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import LeafDiseaseNet
from communication.edge_node.send_weights import deserialize_weights, serialize_weights


class RegionalAggregator:
    """
    Aggregates model weights from farm nodes using Federated Averaging.

    Attributes:
        region_id (str): Region identifier (A, B, C)
        farm_ids (list): List of farm IDs in this region
        port (int): Flask server port
        current_round (int): Current FL round number
        aggregated_weights (dict): Current aggregated model weights
        num_classes (int): Number of disease classes
    """

    def __init__(self, region_id, farm_ids, port=5001, num_classes=6):
        """
        Args:
            region_id (str): Region identifier ('A', 'B', or 'C')
            farm_ids (list): Farm IDs that report to this aggregator
            port (int): Flask server port
            num_classes (int): Number of classes
        """
        self.region_id = region_id
        self.farm_ids = farm_ids
        self.port = port
        self.num_classes = num_classes
        self.current_round = 0

        # Initialize global model
        self.model = LeafDiseaseNet(num_classes=num_classes)
        self.aggregated_weights = self.model.state_dict()

        # Storage for this round's received weights
        self.round_weights = {}  # {farm_id: (weights, num_samples)}
        self.round_received_count = 0
        self.weights_aggregated = False  # Flag indicating aggregation is complete

        # Flask app
        self.app = Flask(f"Aggregator-Region-{region_id}")
        self._setup_routes()

        print(f"[Region {region_id}] Aggregator initialized")
        print(f"  Farm IDs: {farm_ids}")
        print(f"  Port: {port}")

    def _setup_routes(self):
        """Setup Flask routes for receiving weights and sending aggregated weights."""

        @self.app.route('/test_upload', methods=['POST'])
        def test_upload():
            """Debug route: Test raw data reception."""
            try:
                print(f"\n[Region {self.region_id}] /test_upload called")
                print(f"  Content-Type: {request.content_type}")
                print(f"  Content-Length: {request.content_length}")
                print(f"  Form keys: {list(request.form.keys())}")
                print(f"  File keys: {list(request.files.keys())}")

                # Try to get raw data
                data = request.get_data()
                print(f"  Raw data received: {len(data)} bytes")
                print(f"  First 100 bytes: {data[:100]}")

                # Try to parse as pickle (just the raw data)
                try:
                    parsed = pickle.loads(data)
                    print(f"  [OK] Successfully parsed as pickle")
                    print(f"  Parsed type: {type(parsed)}")
                    if isinstance(parsed, dict):
                        print(f"  Parsed keys: {list(parsed.keys())}")
                    return jsonify({
                        "status": "success",
                        "bytes_received": len(data),
                        "parsed_type": str(type(parsed))
                    })
                except pickle.UnpicklingError as e:
                    print(f"  [FAIL] Pickle parse failed: {e}")
                    # If raw data isn't pickle, try to parse form/files
                    print(f"  Attempting to parse form/files...")
                    if 'metadata' in request.form:
                        print(f"    [OK] Found metadata in form")
                    if 'weights' in request.files:
                        print(f"    [OK] Found weights file")
                    raise

            except Exception as e:
                print(f"  [FAIL] Test upload failed: {e}")
                traceback.print_exc()
                return jsonify({
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error": str(e)
                }), 500

        @self.app.route('/upload_weights', methods=['POST'])
        def upload_weights():
            """Receive weights from farm as raw pickle bytes."""
            print(f"\n[Region {self.region_id}] /upload_weights called")

            try:
                # Extract metadata from headers
                farm_id = request.headers.get('X-Farm-Id')
                round_number = request.headers.get('X-Round-Number')
                num_samples = request.headers.get('X-Num-Samples')

                print(f"  Metadata from headers:")
                print(f"    farm_id: {farm_id}")
                print(f"    round_number: {round_number}")
                print(f"    num_samples: {num_samples}")

                # Validate metadata
                if not farm_id or not round_number or not num_samples:
                    raise ValueError("Missing required headers: X-Farm-Id, X-Round-Number, X-Num-Samples")

                try:
                    farm_id = int(farm_id)
                    round_number = int(round_number)
                    num_samples = int(num_samples)
                except ValueError as e:
                    raise ValueError(f"Invalid header values (must be integers): {e}")

                # Get raw pickle bytes from request body
                print(f"  Reading raw pickle bytes from request body...")
                weights_bytes = request.get_data()
                print(f"  Received {len(weights_bytes):,} bytes")

                if not weights_bytes:
                    raise ValueError("Request body is empty")

                # Deserialize pickle
                print(f"  Deserializing pickle...")
                weights_dict = pickle.loads(weights_bytes)
                print(f"  [OK] Deserialized successfully")
                print(f"  Weights dict has {len(weights_dict)} keys")

                # Store weights
                self.round_weights[farm_id] = (weights_dict, num_samples)
                self.round_received_count += 1
                print(f"  [OK] Weights stored")

                # Log progress
                print(f"\n[Region {self.region_id}] [OK] Received weights from Farm {farm_id}")
                print(f"  Round: {round_number}, Samples: {num_samples}")
                print(f"  Progress: {self.round_received_count}/{len(self.farm_ids)} farms")

                # Check if all farms reported
                if self.round_received_count == len(self.farm_ids):
                    print(f"[Region {self.region_id}] [OK] All farms reported! Aggregating...")
                    self._aggregate_weights()
                    self.current_round = round_number

                return jsonify({
                    'status': 'success',
                    'message': f'Weights received from farm {farm_id}',
                    'farm_id': farm_id,
                    'round': round_number
                }), 200

            except Exception as e:
                print(f"[Region {self.region_id}] [FAIL] Error receiving weights:")
                print(f"  {type(e).__name__}: {e}")
                traceback.print_exc()
                return jsonify({
                    'status': 'error',
                    'error_type': type(e).__name__,
                    'message': str(e)
                }), 400

        @self.app.route('/receive_weights', methods=['POST'])
        def receive_weights():
            """Legacy route (calls upload_weights)."""
            return upload_weights()

        @self.app.route('/get_aggregated_weights', methods=['GET'])
        def get_aggregated_weights():
            """Return aggregated weights when all farms sent weights."""
            print(f"\n[Region {self.region_id}] /get_aggregated_weights called")
            print(f"  weights_aggregated: {self.weights_aggregated}")
            print(f"  round_received_count: {self.round_received_count}/{len(self.farm_ids)}")
            print(f"  round_weights keys: {list(self.round_weights.keys())}")

            # Check if aggregation is complete
            if not self.weights_aggregated:
                status_msg = f'Waiting for all farms. Received: {self.round_received_count}/{len(self.farm_ids)}'
                print(f"  [WAITING] {status_msg}")
                return jsonify({
                    'status': 'waiting',
                    'message': status_msg,
                    'region': self.region_id,
                    'progress': self.round_received_count,
                    'total': len(self.farm_ids)
                })

            # Aggregation complete - return weights in base64-encoded JSON
            try:
                # Serialize and encode to base64 for JSON compatibility
                serialized = serialize_weights(self.aggregated_weights)
                weights_b64 = base64.b64encode(serialized).decode('utf-8')

                print(f"  [OK] Returning aggregated weights ({len(serialized):,} bytes, base64 encoded)")
                print(f"  Resetting for next round...")

                # Reset state AFTER returning weights for next round
                self.round_weights = {}
                self.round_received_count = 0
                self.weights_aggregated = False

                return jsonify({
                    'status': 'success',
                    'weights': weights_b64,
                    'round': self.current_round,
                    'region': self.region_id
                })
            except Exception as e:
                print(f"  [ERROR] Failed to return weights: {e}")
                traceback.print_exc()
                return jsonify({'status': 'error', 'message': str(e)}), 400

        @self.app.route('/status', methods=['GET'])
        def status():
            """Return aggregator status."""
            return jsonify({
                'region': self.region_id,
                'current_round': self.current_round,
                'farms_in_region': len(self.farm_ids),
                'weights_received_this_round': self.round_received_count,
                'status': 'ready' if self.round_received_count == 0 else 'aggregating'
            })

    def _aggregate_weights(self):
        """
        Perform FedAvg aggregation.

        Weighted average: w_agg = sum(n_i / n_total * w_i)
        """
        if not self.round_weights:
            print(f"[Region {self.region_id}] No weights to aggregate")
            return

        # Calculate total samples
        total_samples = sum(num_samples for _, num_samples in self.round_weights.values())

        # Initialize aggregated weights
        aggregated_weights = {}

        # Get parameter names from first set of weights
        first_weights = list(self.round_weights.values())[0][0]

        for param_name in first_weights.keys():
            aggregated_weights[param_name] = torch.zeros_like(first_weights[param_name])

        # Weighted sum
        for farm_id, (weights_dict, num_samples) in self.round_weights.items():
            weight_ratio = num_samples / total_samples
            print(f"  Farm {farm_id}: weight_ratio={weight_ratio:.4f} ({num_samples}/{total_samples} samples)")

            for param_name, param in weights_dict.items():
                aggregated_weights[param_name] += weight_ratio * param

        self.aggregated_weights = aggregated_weights
        self.weights_aggregated = True  # Flag that weights are ready

        farms_aggregated = len(self.round_weights)

        # DO NOT reset yet - keep weights in memory until global server fetches them
        # Reset will happen when /get_aggregated_weights is called

        print(f"\n[Region {self.region_id}] [OK] FedAvg Aggregation Complete")
        print(f"  Total samples weighted: {total_samples}")
        print(f"  Farms aggregated: {farms_aggregated}")
        print(f"  Waiting for global server to fetch aggregated weights\n")

    def run(self):
        """Start Flask server."""
        print(f"\n[Region {self.region_id}] Starting Flask server on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False, threaded=True)

    def get_aggregated_weights(self):
        """Get current aggregated weights."""
        return self.aggregated_weights

    def get_status(self):
        """Get aggregator status."""
        return {
            'region': self.region_id,
            'current_round': self.current_round,
            'farms_in_region': len(self.farm_ids),
            'weights_received_this_round': self.round_received_count
        }


def run_aggregator(region_id, farm_ids, port):
    """
    Standalone function to run aggregator as a process.

    Args:
        region_id (str): Region ID
        farm_ids (list): Farm IDs in region
        port (int): Server port
    """
    aggregator = RegionalAggregator(region_id, farm_ids, port=port)
    aggregator.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regional Aggregator")
    parser.add_argument('--region', type=str, default='A', help='Region ID (A, B, C)')
    parser.add_argument('--farms', type=int, nargs='+', default=[0, 1], help='Farm IDs in region')
    parser.add_argument('--port', type=int, default=5001, help='Server port')

    args = parser.parse_args()

    run_aggregator(args.region, args.farms, args.port)
