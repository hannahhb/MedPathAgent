# embedding_search.py
import networkx as nx
import numpy as np
from node2vec import Node2Vec           # pip install node2vec
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from typing import List, Tuple


def compute_node2vec_embeddings(G: nx.Graph, dims=128, walk_length=30, num_walks=200, workers=4):
    # G nodes must be strings
    n2v = Node2Vec(G, dimensions=dims, walk_length=walk_length, num_walks=num_walks, workers=workers, quiet=True)
    model = n2v.fit(window=10, min_count=1, batch_words=4)
    # produce a dict: node -> vector (numpy)
    emb = {str(n): model.wv[str(n)] for n in G.nodes()}
    return emb

def embedding_guided_beam_search(G: nx.Graph, embeddings: dict, src: str, dst: str,
                                 beam_width=10, max_depth=6, stop_when_found=5):
    """
    Beam search: at each depth expand top beam_width partial paths ranked by heuristic.
    Heuristic: cosine similarity of last_node emb to dst emb.
    Returns up to stop_when_found paths (list of node lists).
    """
    src = str(src).lower()
    dst = str(dst).lower()
    if src not in G or dst not in G:
        return []
    if dst not in embeddings or src not in embeddings:
        return []

    dst_emb = embeddings[dst].reshape(1, -1)

    # each item in beam: (score, path_list)
    # score is negative similarity (we use min-heap)
    initial_score = -cosine_similarity(embeddings[src].reshape(1,-1), dst_emb)[0,0]
    beam = [(initial_score, [src])]
    found = []
    visited_paths = set()

    for depth in range(max_depth):
        new_beam = []
        for score, path in beam:
            last = path[-1]
            for nbr in G.neighbors(last):
                if nbr in path:
                    continue  # avoid cycles
                new_path = path + [nbr]
                key = "->".join(new_path)
                if key in visited_paths:
                    continue
                visited_paths.add(key)
                # heuristic score: negative cosine similarity between nbr and dst
                if nbr in embeddings:
                    h = -cosine_similarity(embeddings[nbr].reshape(1,-1), dst_emb)[0,0]
                else:
                    h = 1.0
                # tie-breaker: shorter path preferred
                combined_score = h + 0.01 * len(new_path)
                heapq.heappush(new_beam, (combined_score, new_path))
        # keep best beam_width
        beam = heapq.nsmallest(beam_width, new_beam)
        # collect any paths that end at dst
        for s, p in beam:
            if p[-1] == dst:
                found.append(p)
                if len(found) >= stop_when_found:
                    return found
    return found

# Call once at startup
import numpy as np
import torch
import numpy as np
import torch

def build_node_embedding_index(knowledge_graph_df, nodeemb_dict, node_name_dict):
    """
    Returns:
      node_list: list of node names (strings) in same order as embeddings_mat
      embeddings_mat: numpy array shape (N, d)
      name_to_idx: dict mapping node name -> index in embeddings_mat
      node_type_map: dict node_name -> type (x_type)
    """
    node_list = []
    node_type_map = {}
    emb_rows = []

    # Iterate types in the same order as embeddings were generated
    for t, emb_for_type in nodeemb_dict.items():
        # Use the exact names list we stored at generation time
        names = node_name_dict.get(t)
        if names is None:
            # fallback: use KG df query
            names = knowledge_graph_df.query(f'x_type == "{t}"')['x_name'].unique().tolist()

        if len(names) != emb_for_type.shape[0]:
            print(f"Warning: mismatch for type {t}: {len(names)} names vs {emb_for_type.shape[0]} embeddings")
            # Optionally: intersect names/embeddings or raise error

        for i, name in enumerate(names):
            node_list.append(name)
            node_type_map[name] = t

            row = emb_for_type[i]
            if isinstance(row, torch.Tensor):
                row = row.detach().cpu().numpy()
            emb_rows.append(np.asarray(row, dtype=np.float32))

    embeddings_mat = np.stack(emb_rows, axis=0)
    name_to_idx = {n: i for i, n in enumerate(node_list)}
    return node_list, embeddings_mat, name_to_idx, node_type_map


# Example usage:
# # embeddings_mat shape (N, d)


# # pip install faiss-cpu  (or faiss-gpu if available)
import faiss


def faiss_knn_names(query_name, name_to_idx, emb_norm, index, node_list, k=200):
    idx = name_to_idx.get(query_name)
    if idx is None:
        return []
    q = emb_norm[idx].reshape(1, -1).astype(np.float32)
    D, I = index.search(q, k)   # I shape (1, k)
    indices = I[0].tolist()
    return [node_list[i] for i in indices if i < len(node_list)]

import numpy as np
def induce_and_shortest_paths(G, name_to_idx, embeddings_mat, src, dst, node_list,
                              k_each=1000, max_path_len=20, device=None):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Move numpy embeddings to torch tensor on device
    emb_tensor = torch.from_numpy(embeddings_mat).to(device)
    emb_norm   = torch.nn.functional.normalize(emb_tensor, p=2, dim=1)
    emb_norm_np = emb_norm.cpu().numpy().astype(np.float32)

    d = embeddings_mat.shape[1]
    import faiss  # ensure faiss supports numpy input
    index = faiss.IndexFlatIP(d)
    index.add(emb_norm_np)

    src_knn = faiss_knn_names(src.lower(), name_to_idx, emb_norm_np, index, node_list, k=k_each)
    dst_knn = faiss_knn_names(dst.lower(), name_to_idx, emb_norm_np, index, node_list, k=k_each)

    nodes = set(src_knn) | set(dst_knn) | {src.lower(), dst.lower()}
    subG = G.subgraph(nodes).copy()

    paths_with_relations = []
    try:
        for path in nx.all_shortest_paths(subG, src.lower(), dst.lower()):
            if len(path) <= max_path_len:
                relations = [G.get_edge_data(u, v, {}).get('relation') for u, v in zip(path[:-1], path[1:])]
                paths_with_relations.append((path, relations))
    except Exception:
        # fallback logic
        pass

    return paths_with_relations
