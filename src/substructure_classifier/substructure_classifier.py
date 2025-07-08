
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

from config import TRAINED_CLASSIFIERS_DIR

from graphviz import Digraph
from IPython.display import display, HTML

#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
#import plotly.io as pio

class SubstructureClassifier:
    """
    A class to manage and visualize the stages of the SubstructureClassifier.
    It provides functionality to:
        - List all stages and their relationships
        - Visualize the stage hierarchy as a directed graph
        - Compare stage performance metrics
        - Set the active stage
        - Create new stages
        - Delete stages (with safety checks)
    """
    def __init__(self, classifier_name):
        self.classifier_name = classifier_name
        self.classifier_path = os.path.join(TRAINED_CLASSIFIERS_DIR, classifier_name)
        self.stages_path = os.path.join(self.classifier_path, "stages")
        
        # Load the classifier config
        self._load_classifier_config()
        
        # Load all stages
        self._load_all_stages()
    
    def _load_classifier_config(self):
        """Load the classifier configuration file."""
        config_path = os.path.join(self.classifier_path, "classifier_config.json")
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
            self.active_stage_id = self.config.get("active_stage")
        except FileNotFoundError:
            print(f"Classifier '{self.classifier_name}' not found.")
            self.config = None
            self.active_stage_id = None
    
    def _add_test_catalog(self, catalog_name):
        """Add a test catalog to the classifier configuration."""
        if catalog_name not in self.config["test_catalogs"]:
            self.config["test_catalogs"].append(catalog_name)
            self._save_classifier_config()
    
    def _load_all_stages(self):
        """Load metadata for all available stages."""
        self.stages = {}
        self.stage_tree = defaultdict(list)  # Parent stage -> list of child stages
        
        if not os.path.exists(self.stages_path):
            return
            
        for stage_id in os.listdir(self.stages_path):
            stage_config_path = os.path.join(self.stages_path, stage_id, "stage_config.json")
            if os.path.exists(stage_config_path):
                with open(stage_config_path, "r") as f:
                    stage_config = json.load(f)
                
                # Store basic stage info
                self.stages[stage_id] = {
                    "id": stage_id,
                    "parent_stage": stage_config.get("parent_stage"),
                    "train_catalog": stage_config.get("train_catalog"),
                    "perc_used": stage_config.get("perc_of_cat_used_for_train"),
                    "samples_used": stage_config.get("samples_used_for_train"),
                    "sample_for_test": stage_config.get("samples_used_for_test"),
                    "epochs": stage_config.get("epochs"),
                    "batch_size": stage_config.get("batch_size"),
                    "learning_rate": stage_config.get("learning_rate"),
                    "is_active": (stage_id == self.active_stage_id)
                }
                
                # Load loss data if available
                loss_path = os.path.join(self.stages_path, stage_id, "loss_data.npz")
                if os.path.exists(loss_path):
                    try:
                        loss_data = np.load(loss_path, allow_pickle=True)
                        self.stages[stage_id]["train_losses"] = loss_data['train_losses'].tolist()
                        self.stages[stage_id]["test_catalog_losses"] = loss_data.get('test_catalog_losses', {})
                        if not isinstance(self.stages[stage_id]["test_catalog_losses"], dict):
                            self.stages[stage_id]["test_catalog_losses"] = self.stages[stage_id]["test_catalog_losses"].item()
                    except Exception as e:
                        print(f"Error loading loss data for stage {stage_id}: {e}")
                        self.stages[stage_id]["train_losses"] = []
                        self.stages[stage_id]["test_catalog_losses"] = {}
                
                # Build the stage tree
                parent = stage_config.get("parent_stage")
                if parent:
                    self.stage_tree[parent].append(stage_id)
                else:
                    # Root node
                    self.stage_tree[None].append(stage_id)
    
    def _save_classifier_config(self):
        """Save the classifier configuration file."""
        config_path = os.path.join(self.classifier_path, "classifier_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=4)
    
    def set_active_stage(self, stage_id):
        """Set the active stage by ID."""
        if stage_id not in self.stages:
            print(f"Stage '{stage_id}' not found.")
            return
        
        # Deactivate the current active stage
        if self.active_stage_id:
            self.stages[self.active_stage_id]["is_active"] = False
        
        # Activate the new stage
        self.stages[stage_id]["is_active"] = True
        self.active_stage_id = stage_id
        
        # Update the classifier config
        self.config["active_stage"] = stage_id
        self._save_classifier_config()
    
    def delete_stage(self, stage_id):
        """Delete a stage by ID."""
        if stage_id not in self.stages:
            print(f"Stage '{stage_id}' not found.")
            return
        
        # Check if the stage has children
        if stage_id in self.stage_tree:
            children = self.stage_tree[stage_id]
            if children:
                print(f"Cannot delete stage '{stage_id}' because it has child stages: {', '.join(children)}")
                return
        
        # Delete the stage directory
        stage_path = os.path.join(self.stages_path, stage_id)
        if os.path.exists(stage_path):
            import shutil
            shutil.rmtree(stage_path)
        
        # Remove from the stages dictionary and tree
        del self.stages[stage_id]
        for parent, children in self.stage_tree.items():
            if stage_id in children:
                children.remove(stage_id)

        # Set the active stage to None if it was deleted
        if self.active_stage_id == stage_id:
            self.active_stage_id = None
            self.config["active_stage"] = None
            self._save_classifier_config()
        
        print(f"Stage '{stage_id}' deleted successfully.")



#=============================================================Visualizing===================================================================================
    def _generate_loss_plot(self, stage_id):
        """
        Generate loss plot for a stage and save it as a file.
        Returns True if plot was created, False otherwise.
        """
        stage_info = self.stages[stage_id]
        
        # Check if loss data exists
        train_losses = stage_info.get("train_losses", [])
        test_catalog_losses = stage_info.get("test_catalog_losses", {})
        
        if not train_losses and not test_catalog_losses:
            return False
        
        # Create a figure with a dark background
        plt.figure(figsize=(8, 4))
        ax = plt.gca()
        ax.set_facecolor('#333333')
        plt.gcf().set_facecolor('#333333')
        
        # Plot training loss
        if train_losses:
            epochs = range(1, len(train_losses) + 1)
            plt.plot(epochs, train_losses, 'o-', label='Training Loss', color='#ff79c6', linewidth=2, markersize=3)
        
        # Plot test catalog losses
        colors = ['#8be9fd', '#50fa7b', '#f1fa8c', '#ffb86c', '#bd93f9']
        color_idx = 0
        
        for catalog, losses in test_catalog_losses.items():
            if losses:
                epochs = range(1, len(losses) + 1)
                plt.plot(epochs, losses, 'o-', label=f'{catalog}', 
                        color=colors[color_idx % len(colors)], linewidth=2, markersize=3)
                color_idx += 1
        
        # Set labels and title with light colors
        plt.xlabel('Epoch', color='#f8f8f2', fontsize=10)
        plt.ylabel('Loss', color='#f8f8f2', fontsize=10)
        plt.title(f'Loss History for Stage: {stage_id}', color='#f8f8f2', fontsize=12)
        
        # Customize grid, legend, and ticks
        plt.grid(True, linestyle='--', alpha=0.3, color='#555555')
        plt.legend(loc='best', fontsize='small', facecolor='#333333', edgecolor='#555555', labelcolor='#f8f8f2')
        
        plt.tick_params(axis='both', colors='#f8f8f2', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#555555')
        
        # Make the plot tight
        plt.tight_layout()
        
        # Create directory for plots if it doesn't exist
        plots_dir = os.path.join(self.classifier_path, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save the plot to a file
        plot_file = os.path.join(plots_dir, f"{stage_id}_loss_plot.png")
        plt.savefig(plot_file, format='png', dpi=120, transparent=False)
        plt.close()
        
        print(f"Loss plot saved to {plot_file}")
        return True
    
    def visualize_stages(self, visualize_id=False):
        """
        Create a beautiful visualization of the stage hierarchy using Graphviz.
        Features:
        - Dark gray theme
        - Proper positioning of tables and their corresponding loss plots
        - Maintaining the hierarchical structure
        - Straight connecting lines
        """
        from graphviz import Digraph
        import os

        
        # Create a directed graph with dark theme
        dot = Digraph(comment='Stage Hierarchy', engine='dot')
        
        # Set graph attributes for dark theme and proper layout
        dot.attr(bgcolor='#333333')  # Dark gray background
        dot.attr('graph', fontcolor='#f8f8f2', fontname='Arial', ranksep='1.5')
        dot.attr('edge', color='#555555', fontcolor='#f8f8f2', fontname='Arial')
        
        # Important: Set format to svg for better image handling
        dot.format = 'svg'
        
        # Generate all plot files first
        plot_files = {}
        for stage_id in self.stages:
            has_loss_data = self._generate_loss_plot(stage_id)
            if has_loss_data:
                plot_file = os.path.abspath(os.path.join(self.classifier_path, "plots", f"{stage_id}_loss_plot.png"))
                plot_files[stage_id] = plot_file
        
        # Process stages in order of parent-child relationships
        processed_stages = set()
        
        # Helper function to process a stage and its children
        def process_stage(stage_id, parent_node=None, rank=0):
            if stage_id in processed_stages:
                return
            
            processed_stages.add(stage_id)
            stage_info = self.stages.get(stage_id, {})
            
            # Create the table HTML
            table_html = '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="1" BGCOLOR="#333333">'
            # Add the header row
            table_html += f'<TR><TD COLSPAN="2" BGCOLOR="#3C3C3C" BORDER="0">'
            table_html += f'<FONT COLOR="#bd93f9" FACE="Arial" POINT-SIZE="6"><B>{stage_id}</B></FONT>'
            table_html += '</TD></TR>'
            
            # Dynamically add any property that exists in stage_info
            for key, value in stage_info.items():
                # Skip non-display fields
                if (key not in ["is_active", "id", "parent_stage", "train_losses", "test_catalog_losses"] and 
                    value is not None):
                    
                    # Format the key to be more readable
                    display_key = key.replace("_", " ").title()
                    
                    # Special case abbreviations
                    if key == "learning_rate":
                        display_key = "LR"
                    elif key == "batch_size":
                        display_key = "Batch"
                    elif key == "epochs":
                        display_key = "Epochs"
                    elif key == "perc_used":
                        display_key = "Perc Used"
                    elif key == "train_catalog":
                        display_key = "Train Catalog"
                    
                    # Add a row to the table
                    table_html += f'<TR><TD ALIGN="LEFT">'
                    table_html += f'<FONT COLOR="#f8f8f2" FACE="Arial" POINT-SIZE="9">{display_key}:</FONT>'
                    table_html += '</TD><TD ALIGN="RIGHT">'
                    table_html += f'<FONT COLOR="#8be9fd" FACE="Arial" POINT-SIZE="9">{value}</FONT>'
                    table_html += '</TD></TR>'
            
            # Close the table
            table_html += '</TABLE>'
            
            # Create node color for active stages
            node_color = "#50fa7b" if stage_info.get("is_active") else "#333333"
            node_penwidth = "2" if stage_info.get("is_active") else "0"
            
            # Create a unique ID for the stage node
            node_id = f"stage_{stage_id}"
            
            # Create the stage node (this will be used for connections)
            dot.node(
                node_id, 
                label=f"<{table_html}>",
                shape="none",
                margin="0",
                color=node_color,
                penwidth=node_penwidth
            )
            
            # If there's a plot, add it as a node
            if stage_id in plot_files:
                plot_node_id = f"plot_{stage_id}"
                # Calculate new dimensions - make 5 times bigger than before
                # Initial reduction: 1.0 * 0.7 * 0.7 ≈ 0.49 width, 0.625 * 0.7 * 0.7 ≈ 0.31 height
                # New size: 0.49 * 5 ≈ 2.45 width, 0.31 * 5 ≈ 1.55 height
                new_width = 1.0 * 5  # Approximately 2.45
                new_height = 0.625 * 5  # Approximately 1.55
                
                dot.node(
                    plot_node_id,
                    shape="none",
                    image=plot_files[stage_id],
                    label="",
                    imagescale="false",  # Changed to false to respect absolute dimensions
                    width=str(new_width),  # Reduced by factor of 0.7 twice
                    height=str(new_height),  # Reduced by factor of 0.7 twice
                    margin="0",
                    fixedsize="true"  # Enforce the specified size
                )
                
                # Create invisible edge from stage to its plot
                dot.edge(node_id, plot_node_id, style="invis")
                
                # Use the plot node for connections to children
                source_for_children = plot_node_id
            else:
                # Use the stage node for connections to children
                source_for_children = node_id
            
            # Connect this stage to its parent if it has one
            if parent_node:
                dot.edge(parent_node, node_id)
            
            # Process all children of this stage
            next_rank = rank + 1
            children = self.stage_tree.get(stage_id, [])
            
            # First, ensure all children at the same rank level are in the same rank
            with dot.subgraph() as s:
                s.attr(rank='same')
                for child_id in children:
                    s.node(f"stage_{child_id}")
            
            # Then process each child
            for child_id in children:
                process_stage(child_id, source_for_children, next_rank)
        
        # Start processing from all root nodes
        for stage_id in self.stage_tree.get(None, []):
            process_stage(stage_id)
        
        # Set basic layout attributes
        dot.attr(rankdir="TB")  # Top to bottom layout
        dot.attr(ranksep="0.75")  # Vertical spacing
        dot.attr(nodesep="0.25")  # Horizontal spacing
        dot.attr(splines="polyline")  # Simpler lines that are less likely to cause crashes
        
        return dot


    def save_visualization(self, output_path="stage_hierarchy.svg"):
        """
        Create and save the stage hierarchy visualization.
        """
        dot = self.visualize_stages()
        
        # Use render instead of pipe to avoid memory issues
        try:
            dot.render(output_path, format="svg", cleanup=True)
            print(f"Visualization saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error rendering graph: {e}")
            
            # Fallback to simpler rendering options if the first attempt fails
            try:
                # Try with simpler settings
                dot.attr(splines="line")  # Simplest line type
                dot.attr('graph', pack="false")  # Disable packing
                dot.render(output_path, format="svg", cleanup=True)
                print(f"Visualization saved to {output_path} with simplified settings")
                return output_path
            except Exception as e2:
                print(f"Error with fallback rendering: {e2}")
                return None
            
    def evaluate_active_stage(self, device=None):
        """
        Evaluate the accuracy of the active stage model on all test catalogs.
        
        Args:
            save_results (bool, optional): Whether to save the accuracy results to a file.
                                        Defaults to True.
            device (torch.device, optional): The device to run evaluation on.
                                        If None, uses CUDA if available, else CPU.
        
        Returns:
            dict: Dictionary mapping catalog names to their accuracy percentages,
                or None if no active stage is set.
        """
        if self.active_stage_id is None:
            print("No active stage set. Please set an active stage first.")
            return None
        
        print(f"Evaluating active stage: {self.active_stage_id}")
        
        # Import necessary modules
        import torch
        from .training_stage import Stage
        
        # Set default device if not provided
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the active stage
        active_stage = Stage(
            classifier_name=self.classifier_name,
            stage_id=self.active_stage_id
        )
        
        # Run evaluation on all test catalogs
        catalog_accuracies = active_stage.evaluate_accuracy_on_test_catalogs(device=device)

       
        
        # Update stage information in the classifier's record
        if catalog_accuracies:
            self.stages[self.active_stage_id]["accuracies"] = catalog_accuracies
            
            # Calculate average accuracy across all catalogs
            accuracy_values = list(catalog_accuracies.values())
            avg_accuracy = sum(accuracy_values) / len(accuracy_values)
            self.stages[self.active_stage_id]["avg_accuracy"] = avg_accuracy
            
            print(f"Average accuracy across all test catalogs: {avg_accuracy:.2f}%")
        
        return catalog_accuracies

    def visualize_stage_accuracies(self, stage_ids=None, display_method='plotly'):
        """
        Visualize accuracy metrics for selected stages.
        
        Args:
            stage_ids (list, optional): List of stage IDs to visualize. 
                                    If None, uses all stages with accuracy data.
            display_method (str, optional): Visualization method to use.
                                        Options: 'plotly', 'matplotlib'
                                        Defaults to 'plotly'.
        
        Returns:
            object: The visualization object (Plotly figure or Matplotlib figure)
        """
        import numpy as np
        
        # Collect stages with accuracy data
        stages_with_accuracy = {}
        
        if stage_ids is None:
            # Find all stages with accuracy data
            for stage_id, stage_info in self.stages.items():
                if "accuracies" in stage_info:
                    stages_with_accuracy[stage_id] = stage_info
        else:
            # Use only the specified stages that have accuracy data
            for stage_id in stage_ids:
                if stage_id in self.stages and "accuracies" in self.stages[stage_id]:
                    stages_with_accuracy[stage_id] = self.stages[stage_id]
        
        if not stages_with_accuracy:
            print("No stages with accuracy data found.")
            return None
        
        # Get the set of all catalogs across all stages
        all_catalogs = set()
        for stage_info in stages_with_accuracy.values():
            all_catalogs.update(stage_info["accuracies"].keys())
        all_catalogs = sorted(all_catalogs)
        
        if display_method == 'plotly':
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplot with one row per catalog plus one for average
            fig = make_subplots(
                rows=len(all_catalogs) + 1,
                cols=1,
                subplot_titles=[f"Catalog: {catalog}" for catalog in all_catalogs] + ["Average Accuracy"],
                vertical_spacing=0.05
            )
            
            # Colors for stages
            colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
            
            # Add traces for each catalog
            for i, catalog in enumerate(all_catalogs, 1):
                for j, (stage_id, stage_info) in enumerate(stages_with_accuracy.items()):
                    accuracy = stage_info["accuracies"].get(catalog, 0)
                    
                    # Add bar for this stage's accuracy on this catalog
                    fig.add_trace(
                        go.Bar(
                            x=[stage_id],
                            y=[accuracy],
                            name=stage_id,
                            marker_color=colors[j % len(colors)],
                            showlegend=(i == 1)  # Only show legend for first catalog
                        ),
                        row=i, col=1
                    )
            
            # Add traces for average accuracy
            for j, (stage_id, stage_info) in enumerate(stages_with_accuracy.items()):
                avg_accuracy = stage_info.get("avg_accuracy", 
                                            sum(stage_info["accuracies"].values()) / len(stage_info["accuracies"]))
                
                fig.add_trace(
                    go.Bar(
                        x=[stage_id],
                        y=[avg_accuracy],
                        name=stage_id,
                        marker_color=colors[j % len(colors)],
                        showlegend=False
                    ),
                    row=len(all_catalogs) + 1, col=1
                )
            
            # Update layout
            fig.update_layout(
                title_text="Stage Accuracy Comparison",
                height=300 * (len(all_catalogs) + 1),
                width=800,
                barmode='group'
            )
            
            # Update y-axes to have the same range (0-100%)
            for i in range(1, len(all_catalogs) + 2):
                fig.update_yaxes(range=[0, 100], title_text="Accuracy (%)", row=i, col=1)
            
            return fig
        
        elif display_method == 'matplotlib':
            import matplotlib.pyplot as plt
            
            # Calculate figure size based on number of catalogs
            fig_height = 4 + 2 * len(all_catalogs)
            fig, axes = plt.subplots(len(all_catalogs) + 1, 1, figsize=(10, fig_height))
            
            # Flatten axes if it's a single subplot
            if len(all_catalogs) == 0:
                axes = [axes]
            
            # Plot each catalog
            for i, catalog in enumerate(all_catalogs):
                stage_ids = []
                accuracies = []
                
                for stage_id, stage_info in stages_with_accuracy.items():
                    stage_ids.append(stage_id)
                    accuracies.append(stage_info["accuracies"].get(catalog, 0))
                
                axes[i].bar(stage_ids, accuracies)
                axes[i].set_title(f"Catalog: {catalog}")
                axes[i].set_ylabel("Accuracy (%)")
                axes[i].set_ylim(0, 100)
                
                # Add value labels on bars
                for j, acc in enumerate(accuracies):
                    axes[i].text(j, acc + 1, f"{acc:.1f}%", ha='center')
            
            # Plot average accuracy
            avg_accuracies = []
            for stage_id, stage_info in stages_with_accuracy.items():
                avg_acc = stage_info.get("avg_accuracy", 
                                        sum(stage_info["accuracies"].values()) / len(stage_info["accuracies"]))
                avg_accuracies.append(avg_acc)
            
            axes[-1].bar(list(stages_with_accuracy.keys()), avg_accuracies)
            axes[-1].set_title("Average Accuracy")
            axes[-1].set_ylabel("Accuracy (%)")
            axes[-1].set_ylim(0, 100)
            
            # Add value labels on bars
            for j, acc in enumerate(avg_accuracies):
                axes[-1].text(j, acc + 1, f"{acc:.1f}%", ha='center')
            
            plt.tight_layout()
            return fig