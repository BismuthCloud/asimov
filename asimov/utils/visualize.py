import graphviz

def create_agent_graph(agent, output_file='agent_graph'):
    """
    Creates a DOT graph visualization of an Agent's nodes and their relationships.
    
    Args:
        agent: The Agent object containing nodes and their relationships
        output_file: The name of the output file (without extension)
    
    Returns:
        The path to the generated DOT file
    """
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Agent Node Graph')
    dot.attr(rankdir='LR')  # Left to right layout
    
    # Add all nodes first
    for node in agent.nodes:
        # Create node with its name
        node_attrs = {
            'shape': 'box',
            'style': 'rounded',
            'label': node.name
        }
        dot.node(node.name, **node_attrs)
    
    # Add edges for dependencies
    for node in agent.nodes:
        # Add dependency edges
        for dep in node.dependencies:
            dot.edge(dep.name, node.name, color='blue', label='depends on')
            
        # Add flow control edges
        if hasattr(node, 'next_nodes'):
            for next_node in node.next_nodes:
                dot.edge(node.name, next_node.name, color='green', label='flow')
                
        # Add conditional edges if they exist
        if hasattr(node, 'conditional_nodes'):
            for condition, target in node.conditional_nodes.items():
                dot.edge(node.name, target.name, 
                        color='red', 
                        label=f'if {condition}')
    
    # Save the graph
    try:
        # Save as both .dot and rendered format
        dot.save(f'{output_file}.dot')
        # Also render as PDF for visualization
        dot.render(output_file, view=False)
        return f'{output_file}.dot'
    except Exception as e:
        print(f"Error saving graph: {e}")
        return None

def visualize_agent(agent, output_file='agent_graph'):
    """
    Convenience function to visualize an agent's structure.
    
    Args:
        agent: The Agent object to visualize
        output_file: The name of the output file (without extension)
    
    Returns:
        The path to the generated DOT file
    """
    return create_agent_graph(agent, output_file)

if __name__ == '__main__':
    # Example usage
    class MockNode:
        def __init__(self, name):
            self.name = name
            self.dependencies = []
            self.next_nodes = []
            self.conditional_nodes = {}
    
    class MockAgent:
        def __init__(self):
            self.nodes = []
    
    # Create a mock agent and nodes for testing
    agent = MockAgent()
    node1 = MockNode("Start")
    node2 = MockNode("Process")
    node3 = MockNode("End")
    
    # Set up relationships
    node1.next_nodes = [node2]
    node2.dependencies = [node1]
    node2.next_nodes = [node3]
    node3.dependencies = [node2]
    
    agent.nodes = [node1, node2, node3]
    
    # Generate visualization
    create_agent_graph(agent, "test_graph")

