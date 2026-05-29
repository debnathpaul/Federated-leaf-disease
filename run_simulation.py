"""
Complete federated learning simulation orchestrator.

Starts all components in correct order:
1. Regional aggregators (background subprocesses)
2. Farm nodes (training in parallel)
3. Global server (main process, coordinates everything)

Run with: python run_simulation.py --rounds 3 --epochs 2

This single command runs the entire federated learning simulation.
Windows-compatible using subprocess.Popen instead of multiprocessing.
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from global_server.server import GlobalServer


class SimulationOrchestrator:
    """Orchestrates the complete federated learning simulation."""

    def __init__(self, num_rounds=3, num_epochs=2):
        """
        Args:
            num_rounds (int): Number of federated learning rounds
            num_epochs (int): Number of local training epochs per farm per round
        """
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs

        # Regional aggregator configuration
        self.aggregators = {
            'A': {'port': 5001, 'farms': [0, 1]},
            'B': {'port': 5002, 'farms': [2, 3]},
            'C': {'port': 5003, 'farms': [4, 5]}
        }

        self.aggregator_processes = {}
        self.farm_processes = {}

        print("="*70)
        print("Federated Learning Simulation Orchestrator")
        print("="*70)
        print(f"Rounds: {num_rounds}")
        print(f"Epochs per round: {num_epochs}")
        print(f"Aggregators: {list(self.aggregators.keys())}")
        print(f"Total farms: 6")
        print(f"\nMode: SEQUENTIAL (farms train one at a time)")
        print(f"      Aggregators run in background")
        print(f"      Low memory usage on Windows")
        print("="*70)

    def start_aggregators(self):
        """Start all regional aggregators using subprocess."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting regional aggregators...")

        for region, config in self.aggregators.items():
            port = config['port']
            farms = config['farms']

            # Build command to run aggregator as subprocess
            cmd = [
                sys.executable,
                '-m',
                'regional_aggregator.aggregator',
                '--region', region,
                '--farms'] + [str(f) for f in farms] + ['--port', str(port)]

            try:
                # Start subprocess with stdout/stderr pipes
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=os.path.dirname(__file__)
                )
                self.aggregator_processes[region] = process
                print(f"  [START] Region {region} on port {port}")
                time.sleep(1)
            except Exception as e:
                print(f"  [ERROR] Failed to start Region {region}: {e}")
                raise

        print(f"\nAggregators started. No polling needed - using file-based flags.\n")

    def start_single_farm(self, farm_id, round_number):
        """
        Start a single farm for training (sequential execution).

        Args:
            farm_id (int): Farm identifier
            round_number (int): Current FL round number
        """
        # Determine aggregator URL based on farm ID
        farm_to_region = {
            0: ('A', 5001), 1: ('A', 5001),
            2: ('B', 5002), 3: ('B', 5002),
            4: ('C', 5003), 5: ('C', 5003)
        }
        region, port = farm_to_region[farm_id]
        aggregator_url = f"http://localhost:{port}"

        cmd = [
            sys.executable,
            'communication/edge_node/train_local.py',
            '--farm_id', str(farm_id),
            '--round', str(round_number),
            '--dataset_root', 'data/plantvillage/',
            '--epochs', str(self.num_epochs),
            '--batch_size', '32',
            '--lr', '0.001',
            '--device', 'cpu',
            '--aggregator_url', aggregator_url
        ]

        try:
            # Set PYTHONPATH so subprocess can find model module
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

            # Start subprocess with stdout/stderr pipes (IMPORTANT for capturing output)
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.dirname(__file__),
                env=env,
                text=True  # Auto-decode to string
            )
            self.farm_processes[farm_id] = process
            print(f"  [START] Farm {farm_id} training (Round {round_number})")
            print(f"         Aggregator: {aggregator_url}/upload_weights")
        except Exception as e:
            print(f"  [ERROR] Failed to start Farm {farm_id}: {e}")
            raise

    def wait_for_single_farm(self, farm_id, round_number, timeout=600):
        """
        Wait for a single farm to finish training.

        Args:
            farm_id (int): Farm identifier
            round_number (int): Current FL round
            timeout (int): Max wait time in seconds for this farm
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            process = self.farm_processes.get(farm_id)

            if process and process.poll() is not None:
                # Process has completed - capture and print output
                elapsed = int(time.time() - start_time)
                print(f"  [OK] Farm {farm_id} completed training in {elapsed}s")

                # Capture stdout and stderr
                stdout_output, stderr_output = process.communicate()

                # Print stdout
                if stdout_output:
                    print(f"\n  [FARM {farm_id} STDOUT]")
                    print("  " + "-" * 66)
                    for line in stdout_output.split('\n'):
                        if line.strip():
                            print(f"  {line}")
                    print("  " + "-" * 66)

                # Print stderr if any
                if stderr_output:
                    print(f"\n  [FARM {farm_id} STDERR]")
                    print("  " + "-" * 66)
                    for line in stderr_output.split('\n'):
                        if line.strip():
                            print(f"  {line}")
                    print("  " + "-" * 66)

                # Check exit code
                if process.returncode == 0:
                    print(f"  [OK] Farm {farm_id} exit code: 0 (SUCCESS)")
                    return True
                else:
                    print(f"  [ERROR] Farm {farm_id} exit code: {process.returncode} (FAILURE)")
                    return False

            elapsed = int(time.time() - start_time)
            print(f"  [WAIT] Farm {farm_id} training... ({elapsed}s)")
            time.sleep(15)

        print(f"  [ERROR] Timeout waiting for Farm {farm_id}")
        return False

    def train_farms_sequentially(self, round_number, timeout_per_farm=600):
        """
        Train all 6 farms SEQUENTIALLY (one at a time) for low-memory systems.

        After each farm completes, it sends weights to its regional aggregator.
        Aggregators automatically perform FedAvg when ALL farms send weights.

        Args:
            round_number (int): Current FL round
            timeout_per_farm (int): Max time per farm in seconds
        """
        print(f"[Round {round_number}] Starting farms training (SEQUENTIAL mode)...")
        print(f"[INFO] Farms will train one at a time to save memory\n")

        # Farm to region mapping
        farm_to_region = {
            0: 'A', 1: 'A',  # Region A: Farms 0, 1
            2: 'B', 3: 'B',  # Region B: Farms 2, 3
            4: 'C', 5: 'C'   # Region C: Farms 4, 5
        }

        # Train each farm sequentially
        for farm_id in range(6):
            region = farm_to_region[farm_id]
            print(f"\n{'='*60}")
            print(f"Farm {farm_id} (Region {region}) - Training")
            print(f"{'='*60}")

            # Start this farm
            self.start_single_farm(farm_id, round_number)

            # Wait for this farm to complete
            if not self.wait_for_single_farm(farm_id, round_number, timeout_per_farm):
                print(f"[ERROR] Farm {farm_id} failed to complete in time")
                return False

            # Farm automatically sends weights to its aggregator
            # (this happens in train_local.py - no action needed here)
            time.sleep(2)  # Small pause between farms

        print(f"\n[OK] All farms completed training for Round {round_number}\n")
        return True

    def wait_for_all_farms_done(self, round_number, timeout=600):
        """
        Wait for all farms to send weights by checking flag files.
        When all 6 flags exist, run FedAvg and clean up flags.

        Args:
            server: GlobalServer instance
            round_number (int): Current FL round
            timeout (int): Max wait time in seconds
        """
        print(f"\n[Flag Monitor] Waiting for all 6 farms to send weights (checking flags)...")

        start_time = time.time()
        flags_dir = os.path.join(os.path.dirname(__file__), 'flags')

        while time.time() - start_time < timeout:
            # Check if all 6 flag files exist
            all_farms_done = True
            farms_done_count = 0

            for farm_id in range(6):
                flag_file = os.path.join(flags_dir, f'round_{round_number}_farm_{farm_id}_done.txt')
                if os.path.exists(flag_file):
                    farms_done_count += 1
                else:
                    all_farms_done = False

            if all_farms_done:
                print(f"\n[Flag Monitor] [OK] All 6 farms completed! (6/6 flags found)")
                return True

            # Print progress every 10 seconds
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0:
                print(f"[Flag Monitor] Round {round_number} - Farms completed: {farms_done_count}/6 ({elapsed}s)")

            time.sleep(1)

        print(f"[Flag Monitor] [TIMEOUT] Not all farms completed in {timeout}s ({farms_done_count}/6 found)")
        return False

    def clean_flags(self, round_number):
        """Remove flag files for next round."""
        flags_dir = os.path.join(os.path.dirname(__file__), 'flags')

        for farm_id in range(6):
            flag_file = os.path.join(flags_dir, f'round_{round_number}_farm_{farm_id}_done.txt')
            if os.path.exists(flag_file):
                try:
                    os.remove(flag_file)
                except:
                    pass

    def cleanup(self):
        """Cleanup all processes."""
        print(f"\n[Cleanup] Terminating all processes...")

        # Terminate farms
        for farm_id, process in self.farm_processes.items():
            if process.poll() is None:  # If still running
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"  Terminated Farm {farm_id}")
                except:
                    process.kill()
                    print(f"  Killed Farm {farm_id}")

        # Terminate aggregators
        for region, process in self.aggregator_processes.items():
            if process.poll() is None:  # If still running
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"  Terminated Aggregator {region}")
                except:
                    process.kill()
                    print(f"  Killed Aggregator {region}")

        print(f"[OK] Cleanup complete\n")

    def run(self):
        """Run complete simulation."""
        try:
            # Create flags directory
            flags_dir = os.path.join(os.path.dirname(__file__), 'flags')
            os.makedirs(flags_dir, exist_ok=True)
            print(f"[SETUP] Flags directory: {flags_dir}")

            # Step 1: Start aggregators
            self.start_aggregators()

            print(f"\n{'#'*70}")
            print(f"STARTING FEDERATED LEARNING SIMULATION")
            print(f"Rounds: {self.num_rounds}, Epochs per round: {self.num_epochs}")
            print(f"{'#'*70}\n")

            # Create global server (but don't run it yet)
            server = GlobalServer(num_rounds=self.num_rounds, num_classes=6)

            # Manually run the training loop with integrated farm training
            for round_num in range(self.num_rounds):
                print(f"\n{'#'*70}")
                print(f"FL ROUND {round_num + 1}/{self.num_rounds}")
                print(f"{'#'*70}")
                print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                # Clean up flag files from previous round before starting new round
                if round_num > 0:
                    self.clean_flags(round_num)
                    print(f"[SETUP] Cleaned up flag files from Round {round_num}\n")

                # Step 1: Train farms SEQUENTIALLY (one at a time, low memory)
                # Farms automatically send weights to aggregators and create flag files
                if not self.train_farms_sequentially(round_num + 1):
                    print(f"[ERROR] Round {round_num + 1} failed during farm training")
                    break

                # Step 2: Wait for all farms to complete (by checking flag files)
                print(f"\n[Main] All farms training started. Waiting for weights via flags...")
                if not self.wait_for_all_farms_done(round_num + 1, timeout=600):
                    print(f"[ERROR] Timeout waiting for all farms to send weights")
                    break

                # Step 3: Run FedAvg on global server
                print(f"\n[Main] All farms done! Running FedAvg on global server...")
                try:
                    success = server.run_fedavg_round()
                    if success:
                        print(f"[OK] FedAvg complete for Round {round_num + 1}")
                    else:
                        print(f"[ERROR] FedAvg failed for Round {round_num + 1}")
                        break
                except Exception as e:
                    print(f"[ERROR] Error running FedAvg: {e}")
                    break

                # Step 4: Clean up flags for next round
                self.clean_flags(round_num + 1)

                print(f"[OK] Round {round_num + 1} complete\n")
                time.sleep(2)

            # Final summary
            print(f"\n{'#'*70}")
            print(f"FEDERATED LEARNING SIMULATION COMPLETE")
            print(f"Rounds Completed: {server.current_round}/{self.num_rounds}")
            print(f"{'#'*70}\n")

        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] Simulation interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Simulation failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


def main():
    """Main entry point for simulation."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Simulation Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py --rounds 3 --epochs 2
  python run_simulation.py --rounds 5 --epochs 3
        """
    )

    parser.add_argument(
        '--rounds',
        type=int,
        default=3,
        help='Number of federated learning rounds (default: 3)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of local training epochs per farm per round (default: 2)'
    )

    args = parser.parse_args()

    # Create and run orchestrator
    orchestrator = SimulationOrchestrator(
        num_rounds=args.rounds,
        num_epochs=args.epochs
    )

    orchestrator.run()


if __name__ == "__main__":
    main()
