def serialize_paths_with_relations(path_tuples):
    serialized = []
    for nodes, relations in path_tuples:
        # ensure lengths align: len(relations) == len(nodes) - 1
        parts = []
        for i in range(len(relations)):
            parts.append(nodes[i])
            parts.append(relations[i])
        # append the final node
        parts.append(nodes[-1])
        serialized.append(" -> ".join(parts))
    serialized = "\n".join(serialized)
    return serialized

# Example usage:
example = [
    (['iga glomerulonephritis', 'glomerulonephritis', 'nephritis', 'kidney disease', 'proteinuria'],
     ['parent-child', 'parent-child', 'parent-child', 'parent-child']),
    (['iga glomerulonephritis', 'hereditary nephritis', 'nephritis', 'kidney disease', 'proteinuria'],
     ['parent-child', 'parent-child', 'parent-child', 'parent-child']),
    (['proteinuria', 'lysinuric protein intolerance', 'hepatic failure'],
     ['phenotype present', 'phenotype present'])
]

print(serialize_paths_with_relations(example))
