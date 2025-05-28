# Code Documentation

## File Structure

```
SpectrumCompareOfActivationFumctions/
├── activation_visualizer.py    # Main application file
├── run.py                      # Simple execution script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
└── DOCS.md                     # This documentation file
```

## Main Components

### ActivationFunctionVisualizer Class

The main application class that inherits from `QMainWindow` and provides the GUI interface.

#### Key Methods:

- `__init__()`: Initializes the GUI components and layout
- `calculate_activation(x, function_name)`: Computes activation function values
- `create_plots()`: Generates both time-domain and frequency-domain plots

### Activation Functions Implemented

1. **ReLU**: `max(0, x)`
2. **GELU**: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
3. **Swish**: `x * sigmoid(x)`
4. **Mish**: `x * tanh(ln(1 + e^x))`
5. **ELU**: `x if x > 0 else α(e^x - 1)`
6. **Leaky ReLU**: `max(0.01x, x)`
7. **SELU**: `λ * (x if x > 0 else α(e^x - 1))`
8. **Sigmoid**: `1 / (1 + e^(-x))`
9. **Tanh**: `tanh(x)`
10. **Softplus**: `ln(1 + e^x)`

### FFT Analysis

The application computes the Fast Fourier Transform (FFT) of each activation function to analyze their frequency characteristics:

- Uses `numpy.fft.fft` for frequency domain transformation
- Applies `fftshift` for proper frequency centering
- Displays magnitude spectrum on logarithmic scale
- Helps understand the smoothness and frequency content of each function

### Numerical Stability Features

- **Clipping**: Prevents overflow in exponential functions
- **NaN handling**: Converts NaN and infinite values to finite numbers
- **Error handling**: Graceful error handling in plot generation

## Usage Examples

### Basic Usage
```python
from activation_visualizer import ActivationFunctionVisualizer
from PyQt5.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = ActivationFunctionVisualizer()
window.show()
sys.exit(app.exec_())
```

### Running with Script
```bash
python run.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Technical Notes

### GUI Framework
- Built with PyQt5 for cross-platform compatibility
- Uses matplotlib with Qt5Agg backend for plot integration
- Responsive layout using QVBoxLayout and QHBoxLayout

### Plot Configuration
- Left panel: Activation functions over range [-4, 4]
- Right panel: FFT magnitude spectrum (log scale)
- Color-coded lines with different styles for distinction
- Interactive legends and grid display

### Performance Considerations
- Efficient numpy operations for mathematical calculations
- Vectorized computations for 1000-point sampling
- Optimized matplotlib rendering with proper backends

## Future Enhancements

Potential improvements that could be added:

1. **Interactive Controls**: Sliders to adjust function parameters
2. **Export Features**: Save plots as images or data files
3. **Derivative Analysis**: Show first and second derivatives
4. **Comparison Metrics**: Quantitative comparison tools
5. **Custom Functions**: Allow user-defined activation functions
6. **3D Visualization**: Surface plots for parameter variations
7. **Animation**: Animated transitions between functions
8. **Statistical Analysis**: Mean, variance, and other statistics

## Dependencies Explanation

- **numpy**: Numerical computations and FFT analysis
- **matplotlib**: Plotting and visualization
- **PyQt5**: GUI framework for desktop application

## Troubleshooting

### Common Issues

1. **Import Error**: Install dependencies with `pip install -r requirements.txt`
2. **Display Issues**: Ensure proper Qt5 backend configuration
3. **Performance**: For slower systems, reduce sampling points in `create_plots()`

### Platform-Specific Notes

- **Windows**: May require Visual C++ redistributables for PyQt5
- **macOS**: Ensure proper Python framework installation
- **Linux**: May need additional Qt5 system packages
