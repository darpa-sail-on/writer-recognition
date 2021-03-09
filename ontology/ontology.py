import os
import csv
import networkx


class Ontology:

    def __init__(self, d, file_name='ontology_labels.txt', ontology='ontology'):
        # id to (type name, name)
        self.gid_to_type_and_name = {}
        # type name to dictionary of name to  global id
        self.names_to_gid = {}
        # global id to type name
        self.gid_to_type_name = {}
         # global id to type id
        self.gid_to_type_id = {}
        # last gobal id used
        self.max_id = -1
        # last type id used for each type
        self.max_gid_to_type_id = {}
        self.graph = networkx.DiGraph()
        self.directory = d

        if os.path.exists(os.path.join(d, file_name)):
            with open(os.path.join(d, file_name), 'r') as fp:
                all_classInd = [i.strip().split(',') for i in fp.readlines()]
                for l in all_classInd:
                   type_name = l[2].strip()
                   type_id = int(l[3].strip())
                   gid = int(l[0].strip())
                   name = l[1].strip()
                   self.gid_to_type_and_name[gid] =  (type_name, name)
                   self.gid_to_type_name[gid] =  type_name
                   self.gid_to_type_id[gid]  = type_id
                   if type_name not in self.names_to_gid:
                        self.names_to_gid[type_name] = {}
                   self.names_to_gid[type_name][name] =  gid
                   self.graph.add_node(gid,name=name, item_type=type_name)
            self.max_id = max([int(i) for i in self.gid_to_type_and_name.keys()])
            for i, type_id in self.gid_to_type_id.items():
                type_name = self.gid_to_type_name[i]
                mx = self.max_gid_to_type_id.get(type_name, -1)
                if mx < int(type_id):
                    self.max_gid_to_type_id[type_name] = int(type_id)

            with open(os.path.join(d, ontology + '.csv'), 'r') as fp:
                all_classInd = fp.readlines()
                for line in all_classInd:
                    parts = line.strip().split(',')
                    child = int(parts[0])
                    parent = int(parts[1])
                    if not self.graph.has_node(child):
                        self.graph.add_node(int(child), name=self.gid_to_type_and_name[child])
                    if not self.graph.has_node(parent):
                        self.graph.add_node(int(parent), name=self.gid_to_type_and_name[parent])
                    self.graph.add_edge(child, parent)

    def save(self, file_name='ontology_labels.txt', ontology='ontology'):
        with open(os.path.join(self.directory, file_name), 'w') as fp:
            for i in range(self.max_id + 1):
                fp.write(f'{i},{self.gid_to_type_and_name[i][1]},{self.gid_to_type_name[i]},{self.gid_to_type_id[i]}\n')

        with open(os.path.join(self.directory, ontology + '.csv'), 'w') as fp:
            for edge in self.graph.edges:
                fp.write(f'{edge[0]},{edge[1]}\n')

    def add_check_label(self, name, type_name):
        if name in self.names_to_gid[type_name]:
            return self.names_to_gid[type_name][name]
        print(f'Add  {type_name}: {name}')
        gid_to_type_id = self.max_gid_to_type_id.get(type_name, -1) + 1
        self.max_id += 1
        if type_name not in self.names_to_gid:
            self.names_to_gid[type_name] = {}
        self.names_to_gid[type_name][name]= self.max_id
        self.gid_to_type_name[self.max_id] = type_name
        self.gid_to_type_id[self.max_id] = gid_to_type_id
        self.gid_to_type_and_name[self.max_id] = (type_name,  name)
        self.max_gid_to_type_id[type_name] = gid_to_type_id
        self.graph.add_node(int(self.max_id), name=name, item_type=type_name)
        return self.max_id

    def get_name_for_type_id(self, type_id, type_name):
        """
        @param type_id - the entity type
        @param type_name - the entity name
        @returns entity name
        """
        for gid, tid in self.gid_to_type_id.items():
            if self.gid_to_type_and_name[gid][0] == type_name and type_id == tid:
                return self.gid_to_type_and_name[gid][1]

    def get_type_id_for(self, type_name, entity_name):
        """
        @param type_name - the type name
        @param entity_name - the entity name
        @returns type id
        """
        return self.gid_to_type_id[self.names_to_gid[type_name][entity_name]]

    def add_edge(self, child_id, parent_id):
        self.graph.add_edge(child_id, parent_id)

    def add_sequence_of_parents(self, sequences, type_name):
        """
        Add a sequence of new entities to the graph.
        @param sequences: a sequence of entity names of the given type from ancestor to child
        """
        for sequence in sequences:
            child_id = self.add_check_label(sequence[0], type_name)
            for i in range(1, len(sequence)):
                parent_id = self.add_check_label(sequence[i], type_name)
                self.add_edge(child_id, parent_id)
                child_id = parent_id

    def get_leaf_labels_by_type(self, node_type):
        """
        @returns all nodes of given type that do not have children
        """
        def _has_children(node_id):
            return self.graph.in_degree(node_id) > 0

        return [self.graph.nodes[node_id]['name'] for node_id in self.graph.nodes()
                if self.graph.nodes[node_id]['item_type'] == node_type and not _has_children(node_id)]

    def get_labels_by_type(self, node_type):
        """
        @returns  list of nodes for a given type name
        """
        return [self.graph.nodes[node_id]['name'] for node_id in self.graph.nodes()
                if self.graph.nodes[node_id]['item_type'] == node_type]

def _distances(graph, node_1, node_2):
    preds_1 = ([p[0] for p in networkx.bfs_predecessors(graph, node_1)])
    preds_2 = ([p[0] for p in networkx.bfs_predecessors(graph, node_2)])
    def _to_count(path_gen):
        return min([len(p) - 1 for p in path_gen])
    distances_1 = {}
    distances_2 = {}
    distances = []
    for pred in preds_1:
        distances_1[pred] = _to_count(networkx.shortest_simple_paths(graph, node_1, pred))
    for pred in preds_2:
        distances_2[pred] = _to_count(networkx.shortest_simple_paths(graph, node_2, pred))
    for pred in preds_1:
        if pred in preds_2:
            distances.append(distances_1[pred] + distances_2[pred])
    # WORST CASE NOT CONNECTED
    distances.append(max(distances_1.values()) + max(distances_2.values()) + 1)
    return set(distances)

def node_distance(graph, node_1, node_2):
    distances = _distances(graph, node_1, node_2)
    #penalize by things in common
    return min(distances)*3.0/len(distances)

def clean_name(name:str):
    parts = name.split(' ')
    for i in range(len(parts)):
        parts[i] = parts[i].title()
    z = ''.join(parts).strip()
    if len(z) == 0:
        return '-'
    return z


def get_sub_graph_nodes(ontology, node_ids):
    all_nodes = []
    for node_id in node_ids:
        if node_id not in all_nodes:
            all_nodes.append(node_id)
            s = [x for x in ontology.graph.successors(node_id)]
            while len(s) > 0:
                if s[0] in all_nodes:
                    print(f'cycle {s[0]}')
                    s = s[1:]
                else:
                    all_nodes.append(s[0])
                    s = s[1:] + [x for x in ontology.graph.successors(s[0])]
            p = [x for x in ontology.graph.predecessors(node_id)]
            while len(p) > 0:
                if p[0] in all_nodes:
                    print(f'cycle {p[0]}')
                    p = p[1:]
                else:
                    all_nodes.append(p[0])
                    p = p[1:] + [x for x in ontology.graph.predecessors(p[0])]
    return all_nodes
