import graphviz

from asimov.graph import Agent, FlowControlModule


def create_agent_graph(agent: Agent, output_file="agent_graph"):
    """
    Creates a DOT graph visualization of an Agent's nodes and their relationships.

    Args:
        agent: The Agent object containing nodes and their relationships
        output_file: The name of the output file (without extension)

    Returns:
        The path to the generated DOT file
    """
    # Create a new directed graph
    dot = graphviz.Digraph(comment="Agent Node Graph")
    dot.attr(rankdir="LR")  # Left to right layout

    # Add all nodes first
    for node in agent.nodes.values():
        if node.modules:
            with dot.subgraph(name=f"cluster_{node.name}") as sub:
                sub.attr(label=node.name)
                for mod in node.modules:
                    sub.node(f"{node.name}__{mod.name}", label=mod.name)
                for a, b in zip(node.modules, node.modules[1:]):
                    sub.edge(f"{node.name}__{a.name}", f"{node.name}__{b.name}")
        else:
            dot.node(node.name)

    # Add edges for dependencies
    for node in agent.nodes.values():
        # Add dependency edges
        for dep in node.dependencies:
            dot.edge(
                (
                    f"{dep}__{agent.nodes[dep].modules[-1].name}"
                    if agent.nodes[dep].modules
                    else dep
                ),
                f"{node.name}__{node.modules[0].name}" if node.modules else node.name,
            )

        # Add flow control edges
        for mod in node.modules:
            if isinstance(mod, FlowControlModule):
                for decision in mod.flow_config.decisions:
                    dot.edge(
                        f"{node.name}__{mod.name}",
                        (
                            f"{decision.next_node}__{agent.nodes[decision.next_node].modules[0].name}"
                            if agent.nodes[decision.next_node].modules
                            else decision.next_node
                        ),
                        label=decision.condition,
                    )
                if mod.flow_config.default:
                    dot.edge(
                        f"{node.name}__{mod.name}",
                        (
                            f"{mod.flow_config.default}__{agent.nodes[mod.flow_config.default].modules[0].name}"
                            if agent.nodes[mod.flow_config.default].modules
                            else mod.flow_config.default
                        ),
                        color="blue",
                        label="default",
                    )

    # Save the graph
    try:
        # Save as both .dot and rendered format
        dot.save(output_file)
        # Also render as PNG for visualization
        dot.render(output_file, format="png")
        return f"{output_file}.dot"
    except Exception as e:
        print(f"Error saving graph: {e}")
        return None
