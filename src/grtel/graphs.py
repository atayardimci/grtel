import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from beartype import beartype


@beartype
def plot_graph(
    weight_matrix: pd.DataFrame,
    ax: plt.Axes,
    min_correlation: float = 0.0,
    layout: str = "spring",
    adjust_size_with_degree_factor: int | None = None,
    node_size: int = 300,
    label_size: int = 8
) -> None:
	"""Create and display the graph built from the correlatin matrix"""
	G: nx.Graph = nx.from_pandas_adjacency(weight_matrix)

	# Remove weak edges
	for stock_one, stock_two, data in G.edges(data=True):
		if data["weight"] < min_correlation:
			G.remove_edge(stock_one, stock_two)

	edge_attributes: dict = nx.get_edge_attributes(G, "weight")
	edges, weights = zip(*edge_attributes.items())

	# nodes and their degrees
	node_and_degree_list: list[tuple[str, int]] = nx.degree(G)
	nodes, degrees = zip(*node_and_degree_list)

	node_sizes = (
		tuple([d**adjust_size_with_degree_factor for d in degrees])
		if adjust_size_with_degree_factor
		else node_size
	)

	# increase the value of weights for better visibility
	weights = tuple([ (1 + abs(x))**2 for x in weights] )
	# layout
	pos = nx.circular_layout(G) if layout == "circular" else nx.spring_layout(G)

	nx.draw_networkx_nodes(
		G,
		pos=pos,
		nodelist=nodes,
		node_size=node_sizes,
		node_color="#DA70D6",
		alpha=0.8,
		ax=ax,
	)
	nx.draw_networkx_labels(
		G,
		pos=pos,
		font_size=label_size,
		ax=ax,
	)
	nx.draw_networkx_edges(
		G,
		pos=pos,
		edgelist=edges,
		width=weights,
		edge_color=weights,
		style="solid",
		edge_cmap=plt.cm.GnBu,
		edge_vmin=min(weights), edge_vmax=max(weights),
		ax=ax,
	)
	ax.axis("off")
