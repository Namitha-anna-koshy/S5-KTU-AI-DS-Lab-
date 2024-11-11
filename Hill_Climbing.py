#Implementation of Hill Climbing Algorithm

import networkx as nx
import matplotlib.pyplot as plt

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'C': 2, 'D': 5},
    'C': {'D': 1},
    'D': {}
}

def hill_climb_path(graph, start, goal):
    current_node = start
    path = [current_node]
    
    while current_node != goal:
        neighbors = graph[current_node]
        
        if not neighbors:
            return path, float('inf')
            
        next_node = min(neighbors, key=neighbors.get)
        next_cost = neighbors[next_node]
        path.append(next_node)
        current_node = next_node
    
    total_cost = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
    return path, total_cost

start_node = 'A'
goal_node = 'D'
path, cost = hill_climb_path(graph, start_node, goal_node)

print(f"Final path: {path}, Total cost: {cost}")

G = nx.DiGraph()
for node, neighbors in graph.items():
    for neighbor, cost in neighbors.items():
        G.add_edge(node, neighbor, weight=cost)
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i + 1]) for i in range(len(path) - 1)], edge_color='red', width=2)
plt.show()
