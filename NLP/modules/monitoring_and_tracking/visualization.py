import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import pandas as pd
from datetime import datetime
import os
import zipfile
import shutil
from pathlib import Path

class TrainingVisualizer:
    """Visualizes training metrics and optimizations."""
    
    def __init__(self, output_dir: str = "training_visualizations"):
        self.output_dir = output_dir
        self.experiment_dir = None
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._setup_directories()
        plt.style.use('seaborn')
        
    def _setup_directories(self):
        """Setup directories for experiment outputs."""
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create experiment-specific directory
        self.experiment_dir = os.path.join(self.output_dir, f"experiment_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create subdirectories for different types of visualizations
        self.graphs_dir = os.path.join(self.experiment_dir, "graphs")
        self.reports_dir = os.path.join(self.experiment_dir, "reports")
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def _save_figure(self, fig, filename: str, dpi: int = 300):
        """Save figure as high-resolution image."""
        filepath = os.path.join(self.graphs_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
    def plot_memory_usage(self, metrics_history: List[Dict], title: str = "Memory Usage Over Time"):
        """Plot GPU and CPU memory usage."""
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['gpu_memory_usage'], label='GPU Memory')
        plt.plot(df['timestamp'], df['cpu_memory_usage'], label='CPU Memory')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (%)')
        plt.legend()
        plt.grid(True)
        
        self._save_figure(fig, f"memory_usage_{self.timestamp}.png")
        
    def plot_training_speed(self, metrics_history: List[Dict], title: str = "Training Speed Over Time"):
        """Plot training speed metrics."""
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['training_speed'])
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Samples per Second')
        plt.grid(True)
        
        self._save_figure(fig, f"training_speed_{self.timestamp}.png")
        
    def plot_optimization_effectiveness(self, metrics_history: List[Dict], title: str = "Optimization Effectiveness"):
        """Plot optimization effectiveness metrics."""
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['optimization_effectiveness'])
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Effectiveness (%)')
        plt.grid(True)
        
        self._save_figure(fig, f"optimization_effectiveness_{self.timestamp}.png")
        
    def create_summary_dashboard(self, metrics_history: List[Dict]):
        """Create a comprehensive dashboard of all metrics."""
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Training Optimization Dashboard', fontsize=16)
        
        # Memory Usage
        axes[0, 0].plot(df['timestamp'], df['gpu_memory_usage'], label='GPU')
        axes[0, 0].plot(df['timestamp'], df['cpu_memory_usage'], label='CPU')
        axes[0, 0].set_title('Memory Usage')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Usage (%)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training Speed
        axes[0, 1].plot(df['timestamp'], df['training_speed'])
        axes[0, 1].set_title('Training Speed')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Samples/s')
        axes[0, 1].grid(True)
        
        # Optimization Effectiveness
        axes[1, 0].plot(df['timestamp'], df['optimization_effectiveness'])
        axes[1, 0].set_title('Optimization Effectiveness')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Effectiveness (%)')
        axes[1, 0].grid(True)
        
        # Batch Processing Time
        axes[1, 1].plot(df['timestamp'], df['batch_processing_time'])
        axes[1, 1].set_title('Batch Processing Time')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Time (s)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        self._save_figure(fig, f"training_dashboard_{self.timestamp}.png")
        
    def generate_html_report(self, metrics_history: List[Dict]):
        """Generate an HTML report with all visualizations."""
        html_content = """
        <html>
        <head>
            <title>Training Optimization Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .section { margin-bottom: 30px; }
                img { max-width: 100%; height: auto; }
                .timestamp { color: #666; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Training Optimization Report</h1>
                <p class="timestamp">Generated on: {timestamp}</p>
                
                <div class="section">
                    <h2>Memory Usage</h2>
                    <img src="graphs/memory_usage_{timestamp}.png" alt="Memory Usage">
                </div>
                
                <div class="section">
                    <h2>Training Speed</h2>
                    <img src="graphs/training_speed_{timestamp}.png" alt="Training Speed">
                </div>
                
                <div class="section">
                    <h2>Optimization Effectiveness</h2>
                    <img src="graphs/optimization_effectiveness_{timestamp}.png" alt="Optimization Effectiveness">
                </div>
                
                <div class="section">
                    <h2>Comprehensive Dashboard</h2>
                    <img src="graphs/training_dashboard_{timestamp}.png" alt="Training Dashboard">
                </div>
            </div>
        </body>
        </html>
        """
        
        html_content = html_content.format(timestamp=self.timestamp)
        
        # Save HTML report
        report_path = os.path.join(self.reports_dir, f"training_report_{self.timestamp}.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
            
    def create_experiment_archive(self):
        """Create a zip archive of the experiment results."""
        archive_name = f"experiment_results_{self.timestamp}.zip"
        archive_path = os.path.join(self.output_dir, archive_name)
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files from experiment directory
            for root, _, files in os.walk(self.experiment_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, self.experiment_dir)
                    zipf.write(file_path, arcname)
                    
        return archive_path
        
    def cleanup(self):
        """Clean up temporary files after creating archive."""
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)
            
    def visualize_experiment(self, metrics_history: List[Dict]):
        """Generate all visualizations and create experiment archive."""
        # Generate all visualizations
        self.plot_memory_usage(metrics_history)
        self.plot_training_speed(metrics_history)
        self.plot_optimization_effectiveness(metrics_history)
        self.create_summary_dashboard(metrics_history)
        
        # Generate HTML report
        self.generate_html_report(metrics_history)
        
        # Create and return archive path
        archive_path = self.create_experiment_archive()
        
        # Clean up temporary files
        self.cleanup()
        
        return archive_path 