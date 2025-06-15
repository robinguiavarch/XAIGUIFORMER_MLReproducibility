"""
Aggregate individual frequency-band graphs into unified PyG batches.
Combines multiple connectome graphs per subject for GNN processing.
"""

import os
import pickle
from collections import defaultdict
import torch
from torch_geometric.data import Data, Batch
from tqdm import tqdm


def create_unified_subject_graph(subject_graphs):
    """
    Combine multiple frequency-band graphs into a single unified graph.
    
    Args:
        subject_graphs: List of PyG Data objects for same subject
        
    Returns:
        Unified PyG Data object with frequency band ordering
    """
    if not subject_graphs:
        return None
    
    # Collect all nodes, edges, and metadata
    all_x = []
    all_edge_indices = []
    all_edge_attrs = []
    freq_bounds_list = []
    freqband_orders = []
    
    node_offset = 0
    
    # Get common metadata from first graph
    first_graph = subject_graphs[0]
    subject_id = first_graph.subject_id
    age = first_graph.age
    gender = first_graph.gender
    y = first_graph.y
    
    # Combine all frequency bands
    for band_idx, graph in enumerate(subject_graphs):
        # Node features
        all_x.append(graph.x)
        
        # Edge indices (adjust for node offset)
        edge_index_adjusted = graph.edge_index + node_offset
        all_edge_indices.append(edge_index_adjusted)
        
        # Edge attributes
        all_edge_attrs.append(graph.edge_attr)
        
        # Frequency bounds
        freq_bounds_list.append(graph.freq_bounds)
        
        # Track which nodes belong to which frequency band
        num_nodes = graph.x.shape[0]
        freqband_orders.extend([band_idx] * num_nodes)
        
        node_offset += num_nodes
    
    # Concatenate all components
    unified_x = torch.cat(all_x, dim=0)
    unified_edge_index = torch.cat(all_edge_indices, dim=1)
    unified_edge_attr = torch.cat(all_edge_attrs, dim=0)
    unified_freq_bounds = torch.stack(freq_bounds_list)
    freqband_order = torch.tensor(freqband_orders, dtype=torch.long)
    
    return Data(
        x=unified_x,
        edge_index=unified_edge_index,
        edge_attr=unified_edge_attr,
        freqband_order=freqband_order,
        freq_bounds=unified_freq_bounds,
        y=y,
        age=age,
        gender=gender,
        subject_id=subject_id
    )


def aggregate_connectome_graphs(input_graphs):
    """
    Group individual graphs by subject and create unified representations.
    
    Args:
        input_graphs: List of individual PyG Data objects
        
    Returns:
        List of unified PyG Data objects (one per subject)
    """
    # Group graphs by subject
    subject_groups = defaultdict(list)
    
    for graph in input_graphs:
        subject_key = (graph.subject_id, getattr(graph, 'session', 'ses-1'))
        subject_groups[subject_key].append(graph)
    
    unified_graphs = []
    
    for (subject_id, session), graphs in tqdm(subject_groups.items(), 
                                            desc="Aggregating subject graphs"):
        # Sort graphs by method and band for consistent ordering
        graphs.sort(key=lambda g: (g.method, g.band))
        
        unified_graph = create_unified_subject_graph(graphs)
        
        if unified_graph is not None:
            unified_graphs.append(unified_graph)
    
    return unified_graphs


def validate_graph_consistency(graphs):
    """Validate that aggregated graphs have consistent structure."""
    if not graphs:
        return False
    
    required_attrs = ['x', 'edge_index', 'edge_attr', 'freqband_order', 
                     'freq_bounds', 'y', 'age', 'gender']
    
    for graph in graphs[:5]:  # Check first 5 graphs
        for attr in required_attrs:
            if not hasattr(graph, attr):
                return False
    
    return True


def main():
    input_path = "data/TDBRAIN/tokens/connectomes_graphs.pkl"
    output_path = "data/TDBRAIN/tokens/unified_connectome_graphs.pkl"
    
    # Load individual graphs
    with open(input_path, "rb") as f:
        individual_graphs = pickle.load(f)
    
    # Aggregate by subject
    unified_graphs = aggregate_connectome_graphs(individual_graphs)
    
    # Validate results
    if not validate_graph_consistency(unified_graphs):
        raise ValueError("Graph aggregation produced inconsistent results")
    
    # Save unified graphs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(unified_graphs, f)


if __name__ == "__main__":
    main()