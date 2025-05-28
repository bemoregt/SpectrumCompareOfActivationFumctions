import sys
import numpy as np
import matplotlib
# 백엔드를 명시적으로 설정 (가장 먼저 설정해야 함)
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt

class ActivationFunctionVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Neural Network Activation Functions Visualization')
        self.setGeometry(100, 100, 1400, 700)
        
        # 중앙 위젯 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(central_widget)
        
        # 제목 레이블
        title_label = QLabel('Top 10 Modern Activation Functions Comparison')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(title_label)
        
        # 그래프 레이아웃
        graph_layout = QHBoxLayout()
        
        # 왼쪽 그래프 (활성함수들)
        self.figure_function = plt.figure(figsize=(7, 5))
        self.canvas_function = FigureCanvas(self.figure_function)
        
        # 오른쪽 그래프 (FFT 스펙트럼들)
        self.figure_fft = plt.figure(figsize=(7, 5))
        self.canvas_fft = FigureCanvas(self.figure_fft)
        
        graph_layout.addWidget(self.canvas_function)
        graph_layout.addWidget(self.canvas_fft)
        
        main_layout.addLayout(graph_layout)
        
        # 주요 활성함수 10개 선정 (최신 트렌드 반영)
        self.activation_functions = [
            'ReLU', 'GELU', 'Swish', 'Mish', 'ELU',
            'Leaky ReLU', 'SELU', 'Sigmoid', 'Tanh', 'Softplus'
        ]
        
        # 더 구분되는 색상 팔레트 정의
        self.colors = [
            '#FF0000',  # Red - ReLU
            '#00FF00',  # Green - GELU
            '#0000FF',  # Blue - Swish
            '#FF00FF',  # Magenta - Mish
            '#FFA500',  # Orange - ELU
            '#800080',  # Purple - Leaky ReLU
            '#008080',  # Teal - SELU
            '#FFD700',  # Gold - Sigmoid
            '#8B4513',  # Brown - Tanh
            '#000000'   # Black - Softplus
        ]
        
        # 선 스타일 정의 (실선, 점선, 대시선 등)
        self.line_styles = [
            '-',     # ReLU - solid
            '--',    # GELU - dashed
            '-.',    # Swish - dash-dot
            ':',     # Mish - dotted
            '-',     # ELU - solid
            '--',    # Leaky ReLU - dashed
            '-.',    # SELU - dash-dot
            ':',     # Sigmoid - dotted
            '-',     # Tanh - solid
            '--'     # Softplus - dashed
        ]
        
        # 초기 플롯 생성
        self.create_plots()
        
    def calculate_activation(self, x, function_name):
        """선택된 활성함수에 따라 값을 계산"""
        if function_name == 'ReLU':
            return np.maximum(0, x)
        elif function_name == 'GELU':
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        elif function_name == 'Swish':
            return x * (1 / (1 + np.exp(-np.clip(x, -500, 500))))
        elif function_name == 'Mish':
            return x * np.tanh(np.log(1 + np.exp(np.clip(x, -20, 20))))
        elif function_name == 'ELU':
            alpha = 1.0
            return np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))
        elif function_name == 'Leaky ReLU':
            return np.where(x > 0, x, 0.01 * x)
        elif function_name == 'SELU':
            scale = 1.0507
            alpha = 1.67326
            return scale * np.where(x > 0, x, alpha * (np.exp(np.clip(x, -500, 500)) - 1))
        elif function_name == 'Sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif function_name == 'Tanh':
            return np.tanh(x)
        elif function_name == 'Softplus':
            return np.log(1 + np.exp(np.clip(x, -500, 500)))
        return x  # 기본값
    
    def create_plots(self):
        """모든 활성함수의 그래프를 겹쳐서 생성"""
        # x 범위 설정
        x = np.linspace(-4, 4, 1000)
        
        try:
            # 활성함수 그래프 생성
            self.figure_function.clear()
            ax_func = self.figure_function.add_subplot(111)
            
            # FFT 그래프 생성
            self.figure_fft.clear()
            ax_fft = self.figure_fft.add_subplot(111)
            
            for i, function_name in enumerate(self.activation_functions):
                # 활성함수 계산
                y = self.calculate_activation(x, function_name)
                
                # NaN이나 inf 값 처리
                y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 활성함수 플롯 (색상 + 선 스타일)
                ax_func.plot(x, y, color=self.colors[i], linestyle=self.line_styles[i],
                           linewidth=2.5, label=function_name, alpha=0.8)
                
                # FFT 계산 및 중앙 이동
                y_fft = np.fft.fft(y)
                y_fft_shifted = np.fft.fftshift(y_fft)
                # 진폭 스펙트럼 (로그 스케일)
                magnitude_spectrum = np.abs(y_fft_shifted)
                magnitude_spectrum = np.log10(magnitude_spectrum + 1e-10)  # 로그 스케일
                
                # 주파수 축 생성
                freq = np.fft.fftshift(np.fft.fftfreq(len(x), d=(x[1]-x[0])))
                
                # FFT 스펙트럼 플롯 (색상 + 선 스타일)
                ax_fft.plot(freq, magnitude_spectrum, color=self.colors[i], 
                          linestyle=self.line_styles[i], linewidth=2.0, 
                          label=function_name, alpha=0.8)
            
            # 활성함수 그래프 설정
            ax_func.set_title('Activation Functions Comparison', fontsize=14, fontweight='bold')
            ax_func.set_xlabel('Input (x)', fontsize=12)
            ax_func.set_ylabel('Output (y)', fontsize=12)
            ax_func.grid(True, alpha=0.3)
            ax_func.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax_func.set_xlim(-4, 4)
            
            # FFT 그래프 설정
            ax_fft.set_title('Fourier Transform Spectrum (Log Scale)', fontsize=14, fontweight='bold')
            ax_fft.set_xlabel('Frequency', fontsize=12)
            ax_fft.set_ylabel('Log Magnitude', fontsize=12)
            ax_fft.grid(True, alpha=0.3)
            ax_fft.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            # 레이아웃 조정
            self.figure_function.tight_layout()
            self.figure_fft.tight_layout()
            
        except Exception as e:
            print(f"Error creating plots: {e}")
        
        # 캔버스 업데이트
        self.canvas_function.draw()
        self.canvas_fft.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ActivationFunctionVisualizer()
    window.show()
    sys.exit(app.exec_())
