
import os
import json 
import numpy as np

class DirectedGraph:
    class DirectedEdge:
            def __init__(self, source, target):
                self.source = source
                self.target = target

            def get_source_id(self):
                return self.source
            def get_target_id(self):
                return self.target

    def __init__(self, stages_path):
        # get all the stages ids
        self.stages_path= stages_path
        stage_ids = [stage_id for stage_id in os.listdir(self.stages_path) if not stage_id.startswith(".")]

        self.unique_stage_ids= set(stage_ids)

        

        self.edges= []
        
        for stage_id in self.unique_stage_ids:
            stage_config_path= os.path.join(stages_path, stage_id, "stage_config.json")
            with open(stage_config_path, "r") as f:
                stage_config= json.load(f)
            # get the parent id
            parent_id= stage_config["parent_stage_id"]
            self.edges.append(self.DirectedEdge(parent_id, stage_id))

        
        self._compute_my_custom_matrix()
        self._check_matrix_my_application()

    def update_graph(self):
        self.__init__(self.stages_path)
                
    def _compute_my_custom_matrix(self):
        #my custom matrix:
        # A_ij : i -> j is 1 if there is an edge from i to j, and -1 if there is an edge from j to i
        # may not be the best way, but I am having fun.
        #a column and a row should be one the sign transposition of the other.

        nodes=set()
        for edge in self.edges:
            nodes.add(edge.get_target_id())
        #maybe could arrange the nodes in a time of creation order
        self.nodes_list= (list(nodes))
        self.N_nodes= len(nodes)

        # Initialize the custom matrix with zeros
        custom_matrix = np.zeros((self.N_nodes, self.N_nodes))
        #could be halfed to triangular matrix
        for edge in self.edges:
            if edge.get_source_id() is not None:
                pos_edge= self.nodes_list.index(edge.get_source_id())
                pos_target=self.nodes_list.index(edge.get_target_id())
        
                custom_matrix[pos_edge, pos_target]=1
                custom_matrix[pos_target, pos_edge]=-1

        self.custom_matrix=custom_matrix
        self._check_matrix_my_application()

    def get_nodes_list(self):
        return self.nodes_list

    def _check_matrix_my_application(self):
        # Every node can have only one parent.
        #along the rows, a -1 means parent, a +1 means child
        for i in range(self.N_nodes):
            found_parent=False
            for j in range(self.N_nodes):
                if self.custom_matrix[i, j]==-1:
                    if found_parent==True:
                        raise ValueError("More than one parent found for node {}".format(i))
                    found_parent=True
    def find_origin_node(self):
        for node in self.nodes_list:
            if self.find_parent(node) is None:
                return node

    def find_parent(self, node_id):
        # Find the parent of the node
        node_index= self.nodes_list.index(node_id)
        for j in range(self.N_nodes):
            if self.custom_matrix[node_index, j] == -1:
                return self.nodes_list[j]
        return None
    
    
    def find_route_to_origin(self, node_id):
        # Find the path from the node to the origin node
        route = []
        current_node = node_id
        origin_node = self.find_origin_node()
        
        if origin_node is None:
            print("No origin node found in the graph")
            return None
            
        # Set to track visited nodes for cycle detection
        visited = set()
        
        while current_node != origin_node:
            # Add current node to route and visited set
            route.append(current_node)
            visited.add(current_node)
            
            # Find parent of current node
            current_node = self.find_parent(current_node)
            
            # Check for missing parent
            if current_node is None:
                print(f"No path to origin - node has no parent")
                return None
                
            # Check for cycles
            if current_node in visited:
                print(f"Cycle detected in graph - no valid path")
                return None
        
        # Add the origin node
        route.append(origin_node)
        return route[::-1]
