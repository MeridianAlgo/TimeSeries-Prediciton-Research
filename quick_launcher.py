#!/usr/bin/env python3
"""Quick launcher for separate programs."""

import subprocess
import sys
from pathlib import Path

def show_menu():
    """Show program menu."""
    print("\n🚀 Stock Prediction Programs")
    print("=" * 30)
    print("1. Simple Live Prediction")
    print("2. Advanced ML Predictor") 
    print("3. Dashboard Generator")
    print("4. Accuracy Monitor")
    print("5. Exit")
    print("=" * 30)

def run_program(script_name):
    """Run a program."""
    script_path = Path(script_name)
    if script_path.exists():
        print(f"🎯 Running {script_name}...")
        try:
            subprocess.run([sys.executable, script_name], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Error running {script_name}")
        except KeyboardInterrupt:
            print(f"\n⏹️ Stopped {script_name}")
    else:
        print(f"❌ File not found: {script_name}")

def main():
    """Main launcher."""
    while True:
        show_menu()
        try:
            choice = input("\nSelect (1-5): ").strip()
            
            if choice == "1":
                run_program("simple_live_prediction.py")
            elif choice == "2":
                run_program("advanced_ml_predictor.py")
            elif choice == "3":
                run_program("dashboard_generator.py")
            elif choice == "4":
                run_program("accuracy_monitor.py")
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()