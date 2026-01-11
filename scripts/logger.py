"""
TensorBoard Logger for FlowShield-UDRL Training.

Usage:
    from logger import TensorBoardLogger
    
    logger = TensorBoardLogger('lunarlander', 'policy')
    logger.log_scalar('loss', 0.5, step=1)
    logger.close()

View logs:
    tensorboard --logdir logs/tensorboard
"""

from pathlib import Path
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")


class TensorBoardLogger:
    """TensorBoard logger for training."""
    
    def __init__(self, env_name: str, model_name: str, log_dir: str = "logs/tensorboard"):
        """
        Initialize TensorBoard logger.
        
        Args:
            env_name: Environment name (e.g., 'lunarlander')
            model_name: Model name (e.g., 'policy', 'flow_shield')
            log_dir: Base directory for logs
        """
        self.env_name = env_name
        self.model_name = model_name
        
        if TENSORBOARD_AVAILABLE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{env_name}/{model_name}_{timestamp}"
            log_path = Path(log_dir) / run_name
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.writer = SummaryWriter(str(log_path))
            self.enabled = True
            print(f"TensorBoard logging to: {log_path}")
        else:
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img_tensor, step: int):
        """Log an image."""
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)
    
    def log_figure(self, tag: str, figure, step: int):
        """Log a matplotlib figure."""
        if self.enabled:
            self.writer.add_figure(tag, figure, step)
    
    def log_hparams(self, hparam_dict: dict, metric_dict: dict):
        """Log hyperparameters."""
        if self.enabled:
            self.writer.add_hparams(hparam_dict, metric_dict)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self.enabled:
            self.writer.add_text(tag, text, step)
    
    def flush(self):
        """Flush pending logs."""
        if self.enabled:
            self.writer.flush()
    
    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()


class DummyLogger:
    """Dummy logger when TensorBoard is not needed."""
    
    def log_scalar(self, *args, **kwargs): pass
    def log_scalars(self, *args, **kwargs): pass
    def log_histogram(self, *args, **kwargs): pass
    def log_image(self, *args, **kwargs): pass
    def log_figure(self, *args, **kwargs): pass
    def log_hparams(self, *args, **kwargs): pass
    def log_text(self, *args, **kwargs): pass
    def flush(self): pass
    def close(self): pass


def get_logger(env_name: str, model_name: str, use_tensorboard: bool = True):
    """Get logger instance."""
    if use_tensorboard and TENSORBOARD_AVAILABLE:
        return TensorBoardLogger(env_name, model_name)
    return DummyLogger()
