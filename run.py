#!/usr/bin/env python3
"""
Simple script to run the Activation Function Visualizer
This script provides a simple way to launch the application
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from activation_visualizer import ActivationFunctionVisualizer
    from PyQt5.QtWidgets import QApplication
    
    def main():
        app = QApplication(sys.argv)
        window = ActivationFunctionVisualizer()
        window.show()
        sys.exit(app.exec_())
    
    if __name__ == '__main__':
        main()
        
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install required dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)
