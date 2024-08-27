import numpy as np 
import pandas as pd
from scipy.io import mmread
from igraph import Graph 
import igraph
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy.sparse import isspmatrix_csr
from igraph import Graph, plot
import networkx as nx



edges_csv = "C:/Users/Francesco L/Progetto_Grafi/edges4.csv"
nodes_csv = "C:/Users/Francesco L/Progetto_Grafi/nodes4.csv"
# Crea il DataFrame
edges = pd.read_csv(edges_csv)
nodes= pd.read_csv(nodes_csv)


print(edges)
print(nodes)



edge_list=edges[['source','target']].to_records(index=False).tolist()


#crea grafo
graph=igraph.Graph.TupleList(edge_list,directed=True)


print(graph.summary())

# Aggiungi gli attributi dei nodi (index, label, _pos)
graph.vs['index'] = nodes['index']
graph.vs['name'] = nodes['name']
graph.vs['_pos'] = nodes['_pos']



# Calcola il grado dei nodi
node_degrees = graph.degree()
print('il grafo ha grado: ', node_degrees)

print('la media del grafo e: ', np.mean(graph.degree()))
print( 'la densita del grafo e:', graph.density())

# Visualizza gli ID validi dei vertici nel tuo grafo
print('vertici validi con IDs:', graph.vs.indices)


print('\n il grado di distribuzione si ottiene con:\n')
deg_count = {}
for k in graph.degree():
    if k not in deg_count:
        deg_count[k] = 1
    else:
        deg_count[k] = deg_count[k]+1
deg_dist = {k:v/graph.vcount() for k,v in deg_count.items()}
print(deg_dist)
print('\n abbiamo contato quanti nodi hanno un determinato grado e poi diviso per il numero totale di nodi')



# estraggo il conteggio dei gradi per l'utente
deg_dist_igraph = graph.degree_distribution()

# un istogramma
for b1,b2,count in deg_dist_igraph.bins():
    print(f'ci sono {count} vertici aventi grado {b1} (the bin is {b1},{b2})')

print('il grado di ditribuzione del grafo e :')

# uso displot per vedere le distribuzioni a partire dal solo elenco di valori
sns.displot(graph.degree(), bins=20)

# Personalizzo gli assi x e y
plt.xlabel("Grado dei nodi")
plt.ylabel("Frequenza")

# Mostra il grafico
plt.show()


# Limita i gradi visualizzati
max_degree_to_plot = 10  # Imposta il massimo grado da visualizzare
degree_range = range(1, max_degree_to_plot + 1)

# Calcola la distribuzione dei gradi con un massimo limite superiore
graph_deg_count = Counter(graph.degree())
graph_deg_dist_array = np.array([graph_deg_count.get(k, 0) / graph.vcount() for k in degree_range])

# Calcola la CCDF
graph_CCDF = graph_deg_dist_array[::-1].cumsum()[::-1]

# Disabilita il layout automatico di Seaborn
sns.set(rc={'figure.autolayout': False})

# Crea un nuovo grafico
fig, ax = plt.subplots(figsize=(8, 6))

# Traccia la distribuzione dei gradi
plt.plot(degree_range, graph_deg_dist_array, label='$\Pr(k)$', color='b', linestyle='-')

# Traccia la CCDF Cumulative Complementary Distribution Function
#la probabilità di trovare un componente connesso di almeno una certa dimensione nella rete.

plt.plot(degree_range, graph_CCDF, label='$\Pr(K\geq k)$', color='r', linestyle='-')

# Imposta gli assi in scala logaritmica
plt.xscale("log")
plt.yscale("log")

# Aggiungi legenda
plt.legend()

# Aggiungi etichette agli assi
plt.xlabel("Grado dei nodi")
plt.ylabel("Probabilità")

# Mostra il grafico
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--')
plt.show()


from scipy.sparse import csr_matrix

def FFL_FBL(graph, sparse=True):
    # FFL: l'informazione si muove in avanti lungo un percorso senza essere influenzata da eventuali cambiamenti nell'output del sistema
    # FBL: è una struttura in cui una parte dell'output di un sistema viene "ritornata" (feedback) come input al sistema stesso

    if sparse:
        data = np.ones(graph.ecount())
        indptr = [0]
        indices = []
        for adjl in graph.get_adjlist():
            indices.extend(adjl)
            indptr.append(len(indices))
        A = csr_matrix((data, indices, indptr), shape=(graph.vcount(), graph.vcount()))
    else:
        A = np.asarray(graph.get_adjacency().data)

    A_sqr = A.dot(A)
    cycles_d = np.sum(A_sqr)
    cycles_n = np.sum((A_sqr.dot(A)).diagonal())
    if cycles_d != 0:
        tglob_cycles = cycles_n / cycles_d
    else:
        tglob_cycles = 0.0

    A_max = A.maximum(A.transpose())
    At_A = A.transpose().dot(A)
    any_d = np.sum(At_A)
    any_n = np.sum((At_A.dot(A_max)).diagonal())
    if any_d != 0:
        tglob_any = any_n / any_d
    else:
        tglob_any = 0.0

    return tglob_any, tglob_cycles

tglob_any, tglob_cycles = FFL_FBL(graph)
print(f"Transitività globale (qualsiasi): {tglob_any}")
print(f"Transitività globale (cicli): {tglob_cycles}")    

shortest_paths = graph.get_shortest_paths(0, to=range(10))
print('il percorso piu breve : ', shortest_paths)
print('alcuni nodi non sono raggiungibili')


# Crea il tuo grafo
graph = igraph.Graph.TupleList(edge_list, directed=True)

# Aggiungi gli attributi dei nodi
graph.vs['index'] = nodes['index']
graph.vs['name'] = nodes['name']
graph.vs['_pos'] = nodes['_pos']



# Calcola la matrice di adiacenza
# le righe e le colonne rappresentano i nodi del grafo, e gli elementi della matrice
# indicano se esiste o meno un'arco (connessione) tra i nodi corrispondenti.
data = np.ones(graph.ecount())
indptr = [0]
indices = []
for adjl in graph.get_adjlist():
    indices.extend(adjl)
    indptr.append(len(indices))
A = csr_matrix((data, indices, indptr), shape=(graph.vcount(), graph.vcount()))

# Debug: stampa la matrice di adiacenza
print("Adjacency matrix:")
print(A)

def is_simmetrica(matrice):
    return isspmatrix_csr(matrice) and (matrice - matrice.T).nnz == 0

# Verifica se la matrice di adiacenza è simmetrica
if is_simmetrica(A):
    print("Il grafo è indiretto.")
else:
    print("Il grafo è diretto.")


# Esegui il clustering
components = graph.components('WEAK')
print(f"Numero delle componenti connesse: {len(components)}")
print("Grandezza Clustering:", components.sizes())

# Trova l'indice della componente più grande
largest_component_index = components.sizes().index(max(components.sizes()))
print("Indice della piu arga componente:", largest_component_index)

# Estrai la componente più grande
largest_component = components.subgraph(largest_component_index)
print("Numero di vertici nella piu larga componente:", len(largest_component.vs))


components = graph.components()
print(f"Numero di componenti connesse: {len(components)}")

for idx, component in enumerate(components):
    print(f"Component {idx} has {len(component)} vertices.")



# Assicurati che la matrice di adiacenza sia corretta
print(A)

# Esegui il clustering
components = graph.components()
print(f"Numero delle componenti connesse : {len(components)}")
print("Simensione del Clustering:", components.sizes())

# Trova l'indice della componente più grande
largest_component_index = components.sizes().index(max(components.sizes()))
print("indice della piu larga component:", largest_component_index)

# Estrai la componente più grande
largest_component = components.subgraph(largest_component_index)
print("Il numero della componente maggiore:", len(largest_component.vs))

# Plot del grafo
igraph.plot(graph, bbox=(800, 800), margin=50)

# Mostra il grafico
plt.show()

# Calcola l'assortatività basata sul grado dei nodi
assortativity_degree = graph.assortativity_degree(directed=True)

# Stampa il risultato
print(f'Il grado di assortatività basato sul grado dei nodi è: {assortativity_degree}')

# Calcola le comunità utilizzando l'algoritmo Walktrap
walktrap = graph.community_walktrap()

# Trova la partizione ottimale
partition = walktrap.as_clustering()

# Stampa il numero di comunità
print(f'Numero di comunità individuate: {len(partition)}')

# Assegna le comunità ai nodi
graph.vs['community'] = partition.membership


# Trova la partizione ottimale
partition = walktrap.as_clustering()

# Assegna le comunità ai nodi
graph.vs['community'] = partition.membership

# Stampa le comunità del grafo basate sul metodo Walktrap
print('Le comunità del grafo basate sul metodo Walktrap sono:')
print(partition)

# Plot del grafo con colori delle comunità
vs = {"bbox": (300, 300), "margin": 20, "vertex_color": partition.membership}
igraph.plot(graph, **vs)
plt.show()

# Converti il grafo igraph in un grafo NetworkX
nx_graph = nx.Graph(graph.get_edgelist())

# Estrai la posizione dei nodi
pos = nx.spring_layout(nx_graph)



# Riduci la dimensione dei nodi
node_size = 30

# Personalizza i colori delle comunità
cmap = plt.cm.get_cmap('Set1', max(partition.membership) + 1)

# Disegna il grafo con nodi come punti e archi come linee
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(nx_graph, pos, node_color=partition.membership, cmap=cmap, node_size=node_size)
nx.draw_networkx_edges(nx_graph, pos, alpha=0.5)

# Mostra etichette solo per i nodi più grandi
labels = {}
for node in nx_graph.nodes():
    if graph.vs[node]['name'] in ['lista', 'dei', 'nodi', 'grandi']:
        labels[node] = graph.vs[node]['label']
nx.draw_networkx_labels(nx_graph, pos, labels, font_size=10, font_color='black')

plt.show()

