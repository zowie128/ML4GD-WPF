import pandas as pd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


class WPFSpatialGraph:

    def __init__(self, kernel_size: int, coords: pd.DataFrame):
        self.kernel_size = kernel_size
        self.coords = coords
        self.G = self._create_graph()

    def _create_graph(self):
        # Initialize the graph
        G = nx.Graph()

        # Add nodes to the graph
        for _, row in self.coords.iterrows():
            G.add_node(row["TurbID"], pos=(row["x"], row["y"]))

        # Compute pairwise distances and add edges
        for i, row1 in self.coords.iterrows():
            for j, row2 in self.coords.iterrows():
                if i < j:  # to avoid duplicate calculations
                    distance = np.sqrt(
                        (row1["x"] - row2["x"]) ** 2 + (row1["y"] - row2["y"]) ** 2
                    )
                    if distance < self.kernel_size:
                        G.add_edge(row1["TurbID"], row2["TurbID"])

        return G

    def visualize(self):
        # Extract node positions
        pos = nx.get_node_attributes(self.G, "pos")

        # Create plotly scatter plot for nodes
        node_x = [pos[node][0] for node in self.G.nodes()]
        node_y = [pos[node][1] for node in self.G.nodes()]

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=list(self.G.nodes()),
            textposition="top center",
            marker=dict(size=10, color="skyblue", line=dict(width=2)),
        )

        # Create plotly scatter plot for edges
        edge_x = []
        edge_y = []

        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="gray"),
            hoverinfo="none",
            mode="lines",
        )

        # Create plotly figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Spatial Graph Visualization",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        fig.show()

    def sparsity(self):
        N = len(self.G.nodes())
        E = len(self.G.edges())

        max_edges = N * (N - 1) / 2

        density = E / max_edges
        sparsity = 1 - density

        return sparsity

    def save_graph_as_gml(self, file_path):
        nx.write_gml(self.G, file_path)

if __name__ == "__main__":
    data = pd.read_csv("raw_data/sdwpf_turb_location.csv")
    kernel_size = 2000

    G = WPFSpatialGraph(coords=data, kernel_size=kernel_size)
    G.visualize()

    #save the graph with kernel to use in later models
    G.save_graph_as_gml(f"data/spatial_graph_{kernel_size}.gml")

    # Plot different sparsities.
    kernel_sizes = [100, 500, 1000, 1500, 2000, 3000, 4000, 5000]
    sparsities = []

    for kernel_size in kernel_sizes:
        G = WPFSpatialGraph(coords=data, kernel_size=kernel_size)
        sparsities.append(G.sparsity())

    ax = sns.lineplot(x=kernel_sizes, y=sparsities, marker="o")
    ax.set(
        xlabel="Kernel Sizes", ylabel="Sparsities", title="Kernel Sizes vs Sparsities"
    )
    plt.grid(True)
    plt.show()

