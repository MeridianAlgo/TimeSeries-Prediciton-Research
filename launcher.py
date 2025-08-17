#!/usr/bin/env python3
"""Simple launcher for stock prediction system."""

import sys
import subprocess
from pathlib import Path

def show_menu():
    """Show main menu."""
    print("\n🚀 Stock Prediction System Launcher")
    print("=" * 40)
    print("1. Live Maximum Accuracy Simulation")
    print("2. View Performance Dashboards")
    print("3. Run Basic Simulation")
    print("4. Show Directory Structure")
    print("5. Exit")
    print("=" * 40)

def run_live_simulation():
    """Run live maximum accuracy simulation."""
    print("🎯 Starting Live Maximum Accuracy Simulation...")
    subprocess.run([sys.executable, "live_max_accuracy.py"])

def view_dashboards():
    """Show available dashboards."""
    dashboard_dir = Path("dashboards")
    if dashboard_dir.exists():
        html_files = list(dashboard_dir.glob("*.html"))
        if html_files:
            print("\n📊 Available Dashboards:")
            for i, file in enumerate(html_files[:10], 1):
                print(f"{i}. {file.name}")
            print(f"\n📁 Total: {len(html_files)} dashboard files")
            print("💡 Open any HTML file in your browser to view")
        else:
            print("❌ No dashboard files found")
    else:
        print("❌ Dashboards directory not found")

def run_basic_simulation():
    """Run basic simulation."""
    sim_dir = Path("simulations")
    if sim_dir.exists():
        sim_files = list(sim_dir.glob("*simulation*.py"))
        if sim_files:
            print(f"🎮 Running {sim_files[0].name}...")
            subprocess.run([sys.executable, str(sim_files[0])])
        else:
            print("❌ No simulation files found")
    else:
        print("❌ Simulations directory not found")

def show_structure():
    """Show directory structure."""
    print("\n📁 Directory Structure:")
    print("├── 📊 dashboards/     - Generated HTML dashboards")
    print("├── 🎮 simulations/    - Simulation scripts")
    print("├── 📋 examples/       - Example scripts")
    print("├── 🧪 tests_output/   - Test results")
    print("├── 🏗️ stock_predictor/ - Core framework")
    print("├── 🎯 live_max_accuracy.py - Live ML simulation")
    print("└── 📖 README.md       - Documentation")
    
    # Count files in each directory
    dirs = ["dashboards", "simulations", "examples", "tests_output"]
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"    {dir_name}: {file_count} files")

def main():
    """Main launcher function."""
    while True:
        show_menu()
        try:
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                run_live_simulation()
            elif choice == "2":
                view_dashboards()
            elif choice == "3":
                run_basic_simulation()
            elif choice == "4":
                show_structure()
            elif choice == "5":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()