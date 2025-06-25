"""
Comprehensive visualization tools for Optuna optimization analysis.

This module provides various plotting functions for analyzing optimization
results including parameter importance, optimization history, and trade-offs.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_edf,
    plot_contour,
    plot_intermediate_values
)

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class OptimizationPlotter:
    """
    Comprehensive plotter for optimization analysis and visualization.
    
    Provides both static (matplotlib) and interactive (plotly) visualizations
    for analyzing optimization results and model performance.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = "seaborn"):
        """
        Initialize the optimization plotter.
        
        Args:
            figsize: Default figure size for matplotlib plots
            style: Matplotlib style to use
        """
        self.figsize = figsize
        self.style = style
        
        # Set matplotlib style
        if style in plt.style.available:
            plt.style.use(style)
        
        logger.info("Optimization plotter initialized")
    
    def plot_optimization_history_custom(
        self,
        study: optuna.Study,
        target_name: str = "Objective Value",
        show_best: bool = True,
        interactive: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot optimization history with custom styling.
        
        Args:
            study: Optuna study object
            target_name: Name of the target variable
            show_best: Whether to show best value line
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if interactive:
            return self._plot_optimization_history_plotly(study, target_name, show_best)
        else:
            return self._plot_optimization_history_matplotlib(study, target_name, show_best)
    
    def _plot_optimization_history_matplotlib(
        self,
        study: optuna.Study,
        target_name: str,
        show_best: bool
    ) -> plt.Figure:
        """Create matplotlib optimization history plot."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get trial data
        trials = [t for t in study.trials if t.value is not None]
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]
        
        # Plot trial values
        ax.scatter(trial_numbers, values, alpha=0.6, s=50, label='Trial Values')
        
        if show_best and hasattr(study, 'best_value'):
            # Plot best value line
            best_values = []
            current_best = float('-inf') if study.direction.name == 'MAXIMIZE' else float('inf')
            
            for value in values:
                if study.direction.name == 'MAXIMIZE':
                    current_best = max(current_best, value)
                else:
                    current_best = min(current_best, value)
                best_values.append(current_best)
            
            ax.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best Value')
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel(target_name)
        ax.set_title('Optimization History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_optimization_history_plotly(
        self,
        study: optuna.Study,
        target_name: str,
        show_best: bool
    ) -> go.Figure:
        """Create plotly optimization history plot."""
        trials = [t for t in study.trials if t.value is not None]
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]
        
        fig = go.Figure()
        
        # Add trial values
        fig.add_trace(go.Scatter(
            x=trial_numbers,
            y=values,
            mode='markers',
            name='Trial Values',
            marker=dict(size=8, opacity=0.7),
            hovertemplate='Trial: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ))
        
        if show_best and hasattr(study, 'best_value'):
            # Add best value line
            best_values = []
            current_best = float('-inf') if study.direction.name == 'MAXIMIZE' else float('inf')
            
            for value in values:
                if study.direction.name == 'MAXIMIZE':
                    current_best = max(current_best, value)
                else:
                    current_best = min(current_best, value)
                best_values.append(current_best)
            
            fig.add_trace(go.Scatter(
                x=trial_numbers,
                y=best_values,
                mode='lines',
                name='Best Value',
                line=dict(color='red', width=3)
            ))
        
        fig.update_layout(
            title='Optimization History',
            xaxis_title='Trial Number',
            yaxis_title=target_name,
            hovermode='closest'
        )
        
        return fig
    
    def plot_parameter_importance_custom(
        self,
        study: optuna.Study,
        top_k: int = 10,
        interactive: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot parameter importance with custom styling.
        
        Args:
            study: Optuna study object
            top_k: Number of top parameters to show
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if interactive:
            return self._plot_parameter_importance_plotly(study, top_k)
        else:
            return self._plot_parameter_importance_matplotlib(study, top_k)
    
    def _plot_parameter_importance_matplotlib(
        self,
        study: optuna.Study,
        top_k: int
    ) -> plt.Figure:
        """Create matplotlib parameter importance plot."""
        try:
            importance = optuna.importance.get_param_importances(study)
            
            # Sort and select top k
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
            params, values = zip(*sorted_importance)
            
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create horizontal bar plot
            bars = ax.barh(range(len(params)), values)
            ax.set_yticks(range(len(params)))
            ax.set_yticklabels(params)
            ax.set_xlabel('Importance')
            ax.set_title('Parameter Importance')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + max(values) * 0.01, i, f'{value:.3f}', 
                       va='center', ha='left')
            
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating parameter importance plot: {str(e)}")
            return plt.figure()
    
    def _plot_parameter_importance_plotly(
        self,
        study: optuna.Study,
        top_k: int
    ) -> go.Figure:
        """Create plotly parameter importance plot."""
        try:
            importance = optuna.importance.get_param_importances(study)
            
            # Sort and select top k
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
            params, values = zip(*sorted_importance)
            
            fig = go.Figure(go.Bar(
                x=values,
                y=params,
                orientation='h',
                text=[f'{v:.3f}' for v in values],
                textposition='outside',
                hovertemplate='Parameter: %{y}<br>Importance: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Parameter Importance',
                xaxis_title='Importance',
                yaxis_title='Parameters',
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating parameter importance plot: {str(e)}")
            return go.Figure()
    
    def plot_parallel_coordinates_custom(
        self,
        study: optuna.Study,
        params: Optional[List[str]] = None,
        interactive: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot parallel coordinates with custom styling.
        
        Args:
            study: Optuna study object
            params: List of parameters to include
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if interactive:
            return plot_parallel_coordinate(study, params=params)
        else:
            return self._plot_parallel_coordinates_matplotlib(study, params)
    
    def _plot_parallel_coordinates_matplotlib(
        self,
        study: optuna.Study,
        params: Optional[List[str]]
    ) -> plt.Figure:
        """Create matplotlib parallel coordinates plot."""
        # Convert study to DataFrame
        df = study.trials_dataframe()
        
        if params is None:
            param_cols = [col for col in df.columns if col.startswith('params_')]
        else:
            param_cols = [f'params_{param}' for param in params if f'params_{param}' in df.columns]
        
        if not param_cols:
            logger.warning("No parameters found for parallel coordinates plot")
            return plt.figure()
        
        # Prepare data
        plot_data = df[param_cols + ['value']].dropna()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Normalize data for plotting
        normalized_data = (plot_data - plot_data.min()) / (plot_data.max() - plot_data.min())
        
        # Plot lines
        for i in range(len(normalized_data)):
            ax.plot(range(len(param_cols)), normalized_data.iloc[i, :-1], 
                   alpha=0.3, color=plt.cm.viridis(normalized_data.iloc[i, -1]))
        
        ax.set_xticks(range(len(param_cols)))
        ax.set_xticklabels([col.replace('params_', '') for col in param_cols], rotation=45)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Parallel Coordinates Plot')
        
        plt.tight_layout()
        return fig
    
    def plot_slice_custom(
        self,
        study: optuna.Study,
        params: Optional[List[str]] = None,
        interactive: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot slice plots for parameter analysis.
        
        Args:
            study: Optuna study object
            params: List of parameters to plot
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if interactive:
            return plot_slice(study, params=params)
        else:
            return self._plot_slice_matplotlib(study, params)
    
    def _plot_slice_matplotlib(
        self,
        study: optuna.Study,
        params: Optional[List[str]]
    ) -> plt.Figure:
        """Create matplotlib slice plots."""
        df = study.trials_dataframe()
        
        if params is None:
            param_cols = [col for col in df.columns if col.startswith('params_')][:6]  # Limit to 6
        else:
            param_cols = [f'params_{param}' for param in params if f'params_{param}' in df.columns]
        
        if not param_cols:
            logger.warning("No parameters found for slice plots")
            return plt.figure()
        
        n_params = len(param_cols)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, param_col in enumerate(param_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot parameter vs objective
            param_data = df[param_col].dropna()
            value_data = df.loc[param_data.index, 'value']
            
            ax.scatter(param_data, value_data, alpha=0.6)
            ax.set_xlabel(param_col.replace('params_', ''))
            ax.set_ylabel('Objective Value')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.suptitle('Parameter Slice Plots')
        plt.tight_layout()
        return fig
    
    def plot_edf_comparison(
        self,
        studies: Dict[str, optuna.Study],
        interactive: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot empirical distribution function comparison.
        
        Args:
            studies: Dictionary of study name to study object
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if interactive:
            return plot_edf(list(studies.values()), target_name="Objective Value")
        else:
            return self._plot_edf_matplotlib(studies)
    
    def _plot_edf_matplotlib(self, studies: Dict[str, optuna.Study]) -> plt.Figure:
        """Create matplotlib EDF comparison plot."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for name, study in studies.items():
            values = [t.value for t in study.trials if t.value is not None]
            if values:
                sorted_values = np.sort(values)
                y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                ax.plot(sorted_values, y, label=name, linewidth=2)
        
        ax.set_xlabel('Objective Value')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Empirical Distribution Function Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_multi_objective_pareto(
        self,
        study: optuna.Study,
        objective_names: List[str],
        interactive: bool = True
    ) -> Union[plt.Figure, go.Figure]:
        """
        Plot Pareto front for multi-objective optimization.
        
        Args:
            study: Multi-objective Optuna study
            objective_names: Names of the objectives
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib or Plotly figure
        """
        if not hasattr(study, 'best_trials'):
            raise ValueError("Study must be multi-objective")
        
        if len(objective_names) != 2:
            raise ValueError("Currently only supports 2-objective visualization")
        
        if interactive:
            return self._plot_pareto_plotly(study, objective_names)
        else:
            return self._plot_pareto_matplotlib(study, objective_names)
    
    def _plot_pareto_matplotlib(
        self,
        study: optuna.Study,
        objective_names: List[str]
    ) -> plt.Figure:
        """Create matplotlib Pareto front plot."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot all trials
        all_trials = [t for t in study.trials if t.values is not None]
        if all_trials:
            x_all = [t.values[0] for t in all_trials]
            y_all = [t.values[1] for t in all_trials]
            ax.scatter(x_all, y_all, alpha=0.5, color='lightblue', label='All Trials')
        
        # Plot Pareto front
        pareto_trials = study.best_trials
        if pareto_trials:
            x_pareto = [t.values[0] for t in pareto_trials]
            y_pareto = [t.values[1] for t in pareto_trials]
            ax.scatter(x_pareto, y_pareto, color='red', s=100, label='Pareto Front', zorder=5)
        
        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_title('Multi-Objective Optimization: Pareto Front')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_pareto_plotly(
        self,
        study: optuna.Study,
        objective_names: List[str]
    ) -> go.Figure:
        """Create plotly Pareto front plot."""
        fig = go.Figure()
        
        # Plot all trials
        all_trials = [t for t in study.trials if t.values is not None]
        if all_trials:
            x_all = [t.values[0] for t in all_trials]
            y_all = [t.values[1] for t in all_trials]
            
            fig.add_trace(go.Scatter(
                x=x_all,
                y=y_all,
                mode='markers',
                name='All Trials',
                marker=dict(color='lightblue', size=8, opacity=0.6),
                hovertemplate=f'{objective_names[0]}: %{{x:.4f}}<br>{objective_names[1]}: %{{y:.4f}}<extra></extra>'
            ))
        
        # Plot Pareto front
        pareto_trials = study.best_trials
        if pareto_trials:
            x_pareto = [t.values[0] for t in pareto_trials]
            y_pareto = [t.values[1] for t in pareto_trials]
            
            fig.add_trace(go.Scatter(
                x=x_pareto,
                y=y_pareto,
                mode='markers',
                name='Pareto Front',
                marker=dict(color='red', size=12),
                hovertemplate=f'{objective_names[0]}: %{{x:.4f}}<br>{objective_names[1]}: %{{y:.4f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Multi-Objective Optimization: Pareto Front',
            xaxis_title=objective_names[0],
            yaxis_title=objective_names[1],
            hovermode='closest'
        )
        
        return fig
    
    def create_optimization_dashboard(
        self,
        study: optuna.Study,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create comprehensive optimization dashboard.
        
        Args:
            study: Optuna study object
            save_path: Path to save the dashboard HTML
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Optimization History', 'Parameter Importance', 
                          'Parallel Coordinates', 'Slice Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"colspan": 2}, None]]
        )
        
        # Add optimization history
        trials = [t for t in study.trials if t.value is not None]
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]
        
        fig.add_trace(
            go.Scatter(x=trial_numbers, y=values, mode='markers', name='Trial Values'),
            row=1, col=1
        )
        
        # Add parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            params, imp_values = zip(*sorted_importance)
            
            fig.add_trace(
                go.Bar(x=imp_values, y=params, orientation='h', name='Importance'),
                row=1, col=2
            )
        except Exception:
            logger.warning("Could not add parameter importance to dashboard")
        
        # Update layout
        fig.update_layout(
            title_text="Optimization Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Dashboard saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Example usage
    import optuna
    
    # Create a simple study for demonstration
    def objective(trial):
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return -(x**2 + y**2)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    # Create plotter
    plotter = OptimizationPlotter()
    
    # Generate plots
    history_fig = plotter.plot_optimization_history_custom(study)
    importance_fig = plotter.plot_parameter_importance_custom(study)
    
    print("Visualization examples completed!")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
