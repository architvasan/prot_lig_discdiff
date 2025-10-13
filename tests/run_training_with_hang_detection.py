#!/usr/bin/env python3
"""
Wrapper script to run training with hang detection and automatic recovery.
This script monitors the training process and can detect/recover from hangs.
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path

class HangDetector:
    """Detects if a process has hung by monitoring its output."""
    
    def __init__(self, timeout_seconds=300):  # 5 minutes default
        self.timeout_seconds = timeout_seconds
        self.last_activity = time.time()
        self.is_monitoring = False
        self.process = None
        
    def reset_timer(self):
        """Reset the hang detection timer."""
        self.last_activity = time.time()
        
    def start_monitoring(self, process):
        """Start monitoring a process for hangs."""
        self.process = process
        self.is_monitoring = True
        self.reset_timer()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring and self.process and self.process.poll() is None:
            time.sleep(10)  # Check every 10 seconds
            
            if time.time() - self.last_activity > self.timeout_seconds:
                print(f"\n⚠️  HANG DETECTED: No activity for {self.timeout_seconds} seconds")
                print("🔄 Attempting to terminate process...")
                
                try:
                    # Try graceful termination first
                    self.process.terminate()
                    time.sleep(5)
                    
                    # Force kill if still running
                    if self.process.poll() is None:
                        self.process.kill()
                        print("💥 Process force-killed due to hang")
                    else:
                        print("✅ Process terminated gracefully")
                        
                except Exception as e:
                    print(f"❌ Failed to terminate process: {e}")
                
                break

def run_with_hang_detection(cmd, timeout_seconds=300):
    """Run a command with hang detection."""
    print(f"🚀 Running command with hang detection (timeout: {timeout_seconds}s)")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Setup hang detector
    detector = HangDetector(timeout_seconds)
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Start monitoring
        detector.start_monitoring(process)
        
        # Read output line by line
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                detector.reset_timer()  # Reset timer on any output
        
        # Wait for completion
        return_code = process.wait()
        detector.stop_monitoring()
        
        return return_code
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        detector.stop_monitoring()
        if process:
            process.terminate()
        return 130
        
    except Exception as e:
        print(f"❌ Error running command: {e}")
        detector.stop_monitoring()
        return 1

def setup_environment():
    """Setup environment to prevent hangs."""
    print("🔧 Setting up environment to prevent hangs...")
    
    # Set shorter temp directory
    temp_dirs = ["/tmp", "/dev/shm", "."]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir) and os.access(temp_dir, os.W_OK):
            import tempfile
            try:
                temp_path = tempfile.mkdtemp(prefix="training_", dir=temp_dir)
                os.environ['TMPDIR'] = temp_path
                os.environ['TEMP'] = temp_path
                os.environ['TMP'] = temp_path
                print(f"  Set TMPDIR to: {temp_path}")
                break
            except:
                continue
    
    # Set multiprocessing settings
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Disable some potential hang sources
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_CONSOLE'] = 'off'
    
    print("✅ Environment setup complete")

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python run_training_with_hang_detection.py <training_script> [args...]")
        print("Example: python run_training_with_hang_detection.py protlig_ddiff/train/run_train_clean.py --config config.yaml --datafile data.pt")
        return 1
    
    # Setup environment
    setup_environment()
    
    # Prepare command
    cmd = [sys.executable] + sys.argv[1:]
    
    # Parse timeout from environment or use default
    timeout = int(os.environ.get('HANG_TIMEOUT', '600'))  # 10 minutes default
    
    print("🔍 Training with Hang Detection")
    print("=" * 60)
    print(f"Timeout: {timeout} seconds")
    print(f"TMPDIR: {os.environ.get('TMPDIR', 'Not set')}")
    print()
    
    # Run with hang detection
    return_code = run_with_hang_detection(cmd, timeout)
    
    # Cleanup temp directory
    tmpdir = os.environ.get('TMPDIR')
    if tmpdir and tmpdir not in ['/tmp', '/dev/shm'] and os.path.exists(tmpdir):
        try:
            import shutil
            shutil.rmtree(tmpdir)
            print(f"🧹 Cleaned up temp directory: {tmpdir}")
        except Exception as e:
            print(f"⚠️  Failed to cleanup temp directory: {e}")
    
    if return_code == 0:
        print("\n🎉 Training completed successfully!")
    else:
        print(f"\n❌ Training failed with return code: {return_code}")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main())
