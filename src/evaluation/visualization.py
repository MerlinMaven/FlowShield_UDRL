"""
Visualization utilities for FlowShield-UDRL.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Optional imports for enhanced visualization
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def setup_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    
    if HAS_SEABORN:
        sns.set_style("whitegrid")


def plot_training_curves(
    metrics: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training curves for multiple metrics.
    
    Args:
        metrics: Dictionary of metric_name -> list of values
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display figure
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes)
    
    for idx, (name, values) in enumerate(metrics.items()):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        ax.plot(values, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.set_title(name.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_metrics, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_command_distribution(
    commands: torch.Tensor,
    states: Optional[torch.Tensor] = None,
    safety_module=None,
    title: str = "Command Distribution",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the distribution of commands in horizon-return space.
    
    Args:
        commands: (N, 2) tensor of [horizon, return] commands
        states: Optional states for conditional visualization
        safety_module: Optional safety module for coloring by safety score
        title: Plot title
        save_path: Optional save path
        show: Whether to show
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    commands = commands.cpu().numpy() if torch.is_tensor(commands) else commands
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1 = axes[0]
    
    if safety_module is not None and states is not None:
        # Color by safety score
        with torch.no_grad():
            if torch.is_tensor(states):
                states_t = states
            else:
                states_t = torch.FloatTensor(states)
            
            if torch.is_tensor(commands):
                commands_t = commands
            else:
                commands_t = torch.FloatTensor(commands)
            
            scores = safety_module.get_safety_score(states_t, commands_t)
            scores = scores.cpu().numpy()
        
        scatter = ax1.scatter(
            commands[:, 0], commands[:, 1],
            c=scores, cmap='RdYlGn', alpha=0.7, s=20
        )
        plt.colorbar(scatter, ax=ax1, label='Safety Score')
    else:
        ax1.scatter(commands[:, 0], commands[:, 1], alpha=0.5, s=20)
    
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel("Target Return")
    ax1.set_title("Commands in (H, R) Space")
    ax1.grid(True, alpha=0.3)
    
    # Marginal distributions
    ax2 = axes[1]
    
    ax2_h = ax2
    ax2_r = ax2.twinx()
    
    ax2_h.hist(commands[:, 0], bins=30, alpha=0.7, label='Horizon', color='blue')
    ax2_r.hist(commands[:, 1], bins=30, alpha=0.7, label='Return', color='orange')
    
    ax2_h.set_xlabel("Value")
    ax2_h.set_ylabel("Horizon Count", color='blue')
    ax2_r.set_ylabel("Return Count", color='orange')
    ax2.set_title("Marginal Distributions")
    
    # Combined legend
    lines1, labels1 = ax2_h.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_safety_boundaries(
    safety_module,
    state: torch.Tensor,
    horizon_range: Tuple[float, float] = (1, 100),
    return_range: Tuple[float, float] = (-100, 300),
    resolution: int = 100,
    threshold: Optional[float] = None,
    title: str = "Safety Boundaries",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize safety boundaries in command space for a given state.
    
    Args:
        safety_module: Safety module to visualize
        state: Conditioning state (single state)
        horizon_range: Range for horizon axis
        return_range: Range for return axis
        resolution: Grid resolution
        threshold: Safety threshold for boundary (uses module default if None)
        title: Plot title
        save_path: Optional save path
        show: Whether to show
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    if state.dim() == 1:
        state = state.unsqueeze(0)
    
    # Create grid
    horizons = torch.linspace(*horizon_range, resolution)
    returns = torch.linspace(*return_range, resolution)
    
    H, R = torch.meshgrid(horizons, returns, indexing='xy')
    grid_commands = torch.stack([H.flatten(), R.flatten()], dim=1)
    
    # Expand state to match grid
    state_expanded = state.expand(grid_commands.shape[0], -1)
    
    # Get safety scores
    with torch.no_grad():
        scores = safety_module.get_safety_score(state_expanded, grid_commands)
        is_safe = safety_module.is_safe(state_expanded, grid_commands)
    
    scores = scores.reshape(resolution, resolution).cpu().numpy()
    is_safe = is_safe.reshape(resolution, resolution).cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Safety score heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(
        scores.T, origin='lower', aspect='auto',
        extent=[*horizon_range, *return_range],
        cmap='RdYlGn'
    )
    plt.colorbar(im1, ax=ax1, label='Safety Score')
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel("Target Return")
    ax1.set_title("Safety Score Landscape")
    
    # Add threshold contour if available
    if threshold is not None:
        ax1.contour(
            horizons.numpy(), returns.numpy(), scores.T,
            levels=[threshold], colors='black', linewidths=2
        )
    
    # Binary safe/unsafe
    ax2 = axes[1]
    ax2.imshow(
        is_safe.T, origin='lower', aspect='auto',
        extent=[*horizon_range, *return_range],
        cmap='RdYlGn', vmin=0, vmax=1
    )
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("Target Return")
    ax2.set_title("Safe (green) vs Unsafe (red) Regions")
    
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_flow_trajectories(
    flow_model,
    state: torch.Tensor,
    n_trajectories: int = 20,
    n_steps: int = 100,
    title: str = "Flow Trajectories",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize flow trajectories from noise to data distribution.
    
    Args:
        flow_model: FlowMatchingModel or FlowMatchingSafetyModule
        state: Conditioning state
        n_trajectories: Number of trajectories to plot
        n_steps: Integration steps
        title: Plot title
        save_path: Optional save path
        show: Whether to show
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    if state.dim() == 1:
        state = state.unsqueeze(0)
    
    # Sample trajectories
    with torch.no_grad():
        if hasattr(flow_model, 'flow_model'):
            # It's a FlowMatchingSafetyModule
            samples, trajectories = flow_model.flow_model.sample(
                state, n_samples=n_trajectories,
                n_steps=n_steps, return_trajectory=True
            )
        else:
            samples, trajectories = flow_model.sample(
                state, n_samples=n_trajectories,
                n_steps=n_steps, return_trajectory=True
            )
    
    # trajectories: (1, n_traj, n_steps+1, 2)
    trajectories = trajectories[0].cpu().numpy()  # (n_traj, n_steps+1, 2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by trajectory index
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))
    
    for i in range(n_trajectories):
        traj = trajectories[i]  # (n_steps+1, 2)
        
        # Plot trajectory
        ax.plot(
            traj[:, 0], traj[:, 1],
            color=colors[i], alpha=0.5, linewidth=1
        )
        
        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=30, alpha=0.7)
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker='*', s=80, alpha=0.9)
    
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Target Return")
    ax.set_title(f"{title}\n(○: noise start, ★: generated sample)")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_obedient_suicide_analysis(
    commands: np.ndarray,
    outcomes: np.ndarray,
    terminations: List[str],
    title: str = "Obedient Suicide Analysis",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize the obedient suicide phenomenon.
    
    Args:
        commands: (N, 2) array of [horizon, return] commands
        outcomes: (N, 2) array of [steps, return] outcomes
        terminations: List of termination types
        title: Plot title
        save_path: Optional save path
        show: Whether to show
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Color by termination type
    colors = {
        'goal': 'green',
        'trap': 'red',
        'timeout': 'blue',
        'terminated': 'orange',
    }
    term_colors = [colors.get(t, 'gray') for t in terminations]
    
    # 1. Command space with outcomes
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(
        commands[:, 0], commands[:, 1],
        c=term_colors, alpha=0.6, s=30
    )
    ax1.set_xlabel("Commanded Horizon")
    ax1.set_ylabel("Commanded Return")
    ax1.set_title("Commands by Outcome")
    
    # Legend
    for term, color in colors.items():
        ax1.scatter([], [], c=color, label=term)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Achievement ratio vs command ambition
    ax2 = axes[0, 1]
    
    ambition = commands[:, 1] / (commands[:, 0] + 1)  # Return per step
    achievement = outcomes[:, 1] / (commands[:, 1] + 1e-8)
    
    ax2.scatter(ambition, achievement, c=term_colors, alpha=0.6, s=30)
    ax2.axhline(y=1.0, color='black', linestyle='--', label='Perfect achievement')
    ax2.axhline(y=0.0, color='gray', linestyle=':', label='Zero achievement')
    ax2.set_xlabel("Command Ambition (return/horizon)")
    ax2.set_ylabel("Achievement Ratio")
    ax2.set_title("Ambition vs Achievement")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Return histogram by termination
    ax3 = axes[1, 0]
    
    for term in ['goal', 'trap', 'timeout']:
        mask = np.array([t == term for t in terminations])
        if mask.sum() > 0:
            ax3.hist(
                outcomes[mask, 1], bins=30, alpha=0.5,
                label=f'{term} (n={mask.sum()})',
                color=colors.get(term, 'gray')
            )
    
    ax3.set_xlabel("Actual Return")
    ax3.set_ylabel("Count")
    ax3.set_title("Return Distribution by Outcome")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Trap rate by ambition percentile
    ax4 = axes[1, 1]
    
    percentiles = np.percentile(ambition, np.linspace(0, 100, 11))
    trap_rates = []
    
    for i in range(len(percentiles) - 1):
        mask = (ambition >= percentiles[i]) & (ambition < percentiles[i + 1])
        if mask.sum() > 0:
            trap_rate = np.mean([t == 'trap' for t, m in zip(terminations, mask) if m])
            trap_rates.append(trap_rate)
        else:
            trap_rates.append(0)
    
    ax4.bar(range(len(trap_rates)), trap_rates, color='red', alpha=0.7)
    ax4.set_xlabel("Ambition Percentile")
    ax4.set_ylabel("Trap Rate")
    ax4.set_title("Trap Rate by Command Ambition")
    ax4.set_xticks(range(len(trap_rates)))
    ax4.set_xticklabels([f'{i*10}-{(i+1)*10}%' for i in range(len(trap_rates))], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_method_comparison(
    results: Dict[str, Dict[str, float]],
    metrics_to_plot: Optional[List[str]] = None,
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Compare multiple methods across different metrics.
    
    Args:
        results: Dictionary of method_name -> metric_name -> value
        metrics_to_plot: Specific metrics to plot (None for all)
        title: Plot title
        save_path: Optional save path
        show: Whether to show
        
    Returns:
        Matplotlib figure
    """
    setup_style()
    
    methods = list(results.keys())
    
    if metrics_to_plot is None:
        # Get all unique metrics
        all_metrics = set()
        for method_results in results.values():
            all_metrics.update(method_results.keys())
        metrics_to_plot = sorted(list(all_metrics))
    
    n_metrics = len(metrics_to_plot)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        values = [results[m].get(metric, 0) for m in methods]
        
        bars = ax.bar(range(len(methods)), values, alpha=0.7)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide empty subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig
