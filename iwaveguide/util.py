import numpy as np
import math


class Edge:
    """Class representing an edge. Immutable (Values should only be read from, not written to)."""
    def __init__(self, node1, node2):
        """
        Constructor
        :param node1: The global node number of the first node of the edge
        :param node2: The global node number of the second node of the edge
        """
        self.node1 = node1
        self.node2 = node2
        # print(len(Element.all_nodes))
        node1_t, node2_t = Element.all_nodes[node1], Element.all_nodes[node2]
        # TODO: Evaluate whether the sign should be considered or not (currently does assign it a sign)
        sign_multiplier = 1 if node1 < node2 else -1
        self.length = sign_multiplier * math.sqrt((node2_t[0] - node1_t[0])**2 + (node2_t[1] - node1_t[1])**2)

    def flip(self):
        """Return a copy of this object with the node numbers switched"""
        return Edge(self.node2, self.node1)

    def line(self):
        """Return a tuple of matrices, the first containing a list of 2 x points, the second of y points"""
        return [Element.all_nodes[self.node1][0], Element.all_nodes[self.node2][0]], [Element.all_nodes[self.node1][1], Element.all_nodes[self.node2][1]]

    def __eq__(self, other):
        """Edges are considered equal if they have the same global node numbers"""
        return self.node1 == other.node1 and self.node2 == other.node2 or self.node1 == other.node2 and self.node2 == other.node1

    def __hash__(self):
        """
        Hash the edge object for fast performance on maps/sets.
        Two edges with flipped node numbers produce the same hash value (are considered equal when compared thru map/set)
        """
        if self.node1 < self.node2:
            hash_value = (self.node1, self.node2).__hash__()
        else:
            hash_value = (self.node2, self.node1).__hash__()
        # print(f"Hash value: {hash_value}")
        return hash_value


class Element:
    """A class representing an Element (containing 3 global node numbers and 3 global edge numbers, as well as a permittivity) """
    all_nodes = []
    all_edges = []

    def __init__(self, nodes, edges, permittivity):
        """
        :param nodes: Three global node numbers as a numpy array
        :param edges: Three global edge numbers as a numpy array
        :param permittivity: The permittivity associated with this element
        """
        self.nodes = nodes
        self.edges = edges
        self.permittivity = permittivity

    def area(self):
        """
        Compute the area of this element
        :return: The area of the triangle element
        """
        x1, y1 = Element.all_nodes[self.nodes[0]]
        x2, y2 = Element.all_nodes[self.nodes[1]]
        x3, y3 = Element.all_nodes[self.nodes[2]]
        return area(x1, y1, x2, y2, x3, y3)

    def is_inside(self, x, y):
        """
        Check if a point is inside this triangle element
        :param x: The x coordinate of the point
        :param y: The y coordinate of the point
        :return: True if the point lies in the triangle element, false otherwise
        """
        node1, node2, node3 = Element.all_nodes[self.nodes[0]], Element.all_nodes[self.nodes[1]], Element.all_nodes[self.nodes[2]]
        return is_inside(node1, node2, node3, x, y)

    def is_adjacent_to(self, element):
        """
        Check if this element is adjacent to the passed element
        :param element: The element to check if this is adjacent to
        :return: True if it is, false otherwise
        """
        node_count = 0
        if self.nodes[0] in element.nodes:
            node_count += 1
        if self.nodes[1] in element.nodes:
            node_count += 1
        if self.nodes[2] in element.nodes:
            node_count += 1
        return node_count == 2

    def nodal_interpolate(self, phi1, phi2, phi3, x, y):
        x1, y1 = Element.all_nodes[self.nodes[0]][0], Element.all_nodes[self.nodes[0]][1]
        x2, y2 = Element.all_nodes[self.nodes[1]][0], Element.all_nodes[self.nodes[1]][1]
        x3, y3 = Element.all_nodes[self.nodes[2]][0], Element.all_nodes[self.nodes[2]][1]
        a1 = x2 * y3 - x3 * y2
        a2 = x3 * y1 - x1 * y3
        a3 = x1 * y2 - x2 * y1
        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        # area = 0.5 * (b1 * c2 - b2 * c1)
        n1 = phi1 * (a1 + b1 * x + c1 * y)
        n2 = phi2 * (a2 + b2 * x + c2 * y)
        n3 = phi3 * (a3 + b3 * x + c3 * y)
        return (n1 + n2 + n3) / 2 / self.area()

    def edge_interpolate(self, phi1, phi2, phi3, x, y):
        area = self.area()
        x_component = 0
        y_component = 0
        phis = [phi1, phi2, phi3]
        count = 0
        # for l, k in ((0, 1), (1, 2), (2, 0)):
        #     # Generate the edge from 2 of the nodes (done in a CCW fashion by choice of tuples in for loop)
        #     edge = Edge(self.nodes[l], self.nodes[k])
        #     # TODO: Fix the below statement so that it doesn't print TRUE
        #     # if edge != Element.all_edges[self.edges[count]]:
        #     #     print("TRUE")
        #     # Index of the third node (the one not making up the edge) going in a CCW fashion
        #     m = (k + 1) % 3
        #     # Create the ccw node list started from the first node of the edge
        #     nodes_lk = (Element.all_nodes[self.nodes[l]], Element.all_nodes[self.nodes[k]], Element.all_nodes[self.nodes[m]])

        #     a_i_l, a_j_l = nodes_lk[1][0] * nodes_lk[2][1] - nodes_lk[2][0] * nodes_lk[1][1], nodes_lk[2][0] * nodes_lk[0][1] - nodes_lk[0][0] * nodes_lk[2][1]
        #     b_i_l, b_j_l = nodes_lk[1][1] - nodes_lk[2][1], nodes_lk[2][1] - nodes_lk[0][1]
        #     c_i_l, c_j_l = nodes_lk[2][0] - nodes_lk[1][0], nodes_lk[0][0] - nodes_lk[2][0]

        #     A_l = a_i_l * b_j_l - a_j_l * b_i_l
        #     B_l = c_i_l * b_j_l - c_j_l * b_i_l
        #     # Fix this like the other spots
        #     # C_l = a_i_l * c_j_l - b_j_l * c_i_l
        #     C_l = a_i_l * c_j_l - a_j_l * c_i_l
        #     D_l = b_i_l * c_j_l - b_j_l * c_i_l

        #     x_component += phis[count] * edge.length / 4 / area**2 * (A_l + B_l*y)
        #     y_component += phis[count] * edge.length / 4 / area**2 * (C_l + D_l*x)
        #     count += 1
        for edge_number in self.edges:
            # Generate the edge from 2 of the nodes (done in a CCW fashion by choice of tuples in for loop)
            edge = Element.all_edges[edge_number]
            # TODO: Fix the below statement so that it doesn't print TRUE
            # if edge != Element.all_edges[self.edges[count]]:
            #     print("TRUE")
            # Index of the third node (the one not making up the edge) going in a CCW fashion
            node1, node2 = edge.node1, edge.node2
            node3 = set(self.nodes).difference({node1, node2}).pop()
            n1_index = np.where(self.nodes == node1)[0][0]
            n2_index = np.where(self.nodes == node2)[0][0]
            # n2_index = self.nodes.index(node2)
            # print(str(n1_index[0][0]) + str(n2_index[0][0]))
            negate = 1
            match str(n1_index) + str(n2_index):
                case "01":
                    n3_index = 2
                case "02":
                    n1_index, n2_index, n3_index, negate = 2, 0, 1, -1
                case "10":
                    n1_index, n2_index, n3_index, negate = 0, 1, 2, -1
                case "12":
                    n1_index, n2_index, n3_index = 1, 2, 0
                case "20":
                    n1_index, n2_index, n3_index = 2, 0, 1
                case "21":
                    n1_index, n2_index, n3_index, negate = 1, 2, 0, -1

            # Create the ccw node list started from the first node of the edge
            # nodes_lk = (Element.all_nodes[self.nodes[l]], Element.all_nodes[self.nodes[k]], Element.all_nodes[self.nodes[m]])
            nodes_lk = (Element.all_nodes[self.nodes[n1_index]], Element.all_nodes[self.nodes[n2_index]], Element.all_nodes[self.nodes[n3_index]])

            a_i_l, a_j_l = nodes_lk[1][0] * nodes_lk[2][1] - nodes_lk[2][0] * nodes_lk[1][1], nodes_lk[2][0] * nodes_lk[0][1] - nodes_lk[0][0] * nodes_lk[2][1]
            b_i_l, b_j_l = nodes_lk[1][1] - nodes_lk[2][1], nodes_lk[2][1] - nodes_lk[0][1]
            c_i_l, c_j_l = nodes_lk[2][0] - nodes_lk[1][0], nodes_lk[0][0] - nodes_lk[2][0]

            A_l = a_i_l * b_j_l - a_j_l * b_i_l
            B_l = c_i_l * b_j_l - c_j_l * b_i_l
            # Fix this like the other spots
            # C_l = a_i_l * c_j_l - b_j_l * c_i_l
            C_l = a_i_l * c_j_l - a_j_l * c_i_l
            D_l = b_i_l * c_j_l - b_j_l * c_i_l

            x_component += phis[count] * negate * edge.length / 4 / area**2 * (A_l + B_l*y)
            y_component += phis[count] * negate * edge.length / 4 / area**2 * (C_l + D_l*x)
            count += 1
        return x_component, y_component

    def __eq__(self, other):
        if self.nodes[0] in other.nodes and self.nodes[1] in other.nodes and self.nodes[2] in other.nodes:
            return True
        return False


def load_mesh(filename, num_surfaces=1, permittivities=None):
    """
    Load a mesh from a file. Must have at least 2 blocks, one containing all surface element_to_node_conn, the other the edge ones.
    :param filename: The name of the mesh file (a .inp a.k.a. abaqus file) to load
    :param num_surfaces: The number of surfaces in the .inp file (for different permittivities)
    :param permittivities: The permittivities corresponding to each surface.
    :return: A number of items. See comments at bottom of this function, above the return statement.
    """
    # If we aren't provided permittivities, just assume they are all epsilon_r = 1
    if permittivities is None:
        permittivities = [1 for i in range(num_surfaces)]
    with open(filename, 'r') as file:
        lines = file.readlines()
        count = 0
        line = lines[count]
        # Go until we get to the nodes section of the file
        while "NODE" not in line:
            count += 1
            line = lines[count]
        # Skip over the line with "NODE" in it
        count += 1
        line = lines[count]
        skip = count
        while "*" not in line:
            count += 1
            line = lines[count]
        rows = count - skip
        # Load all the node information into a numpy array, slicing off the node numbers (these are implied by index)
        all_nodes = np.loadtxt(filename, skiprows=skip, max_rows=rows, delimiter=',')[:, 1:]
        # Provide a global copy of the nodes (this will mean that multiple calls to this function can be problematic)
        Element.all_nodes = all_nodes

        # CONSTRUCT ELEMENT NODE CONNECTIVITY LIST FOR EACH SURFACE
        surfaces_node_connectivity = []
        surfaces_edge_connectivity = []

        # All the edges (the index is the global edge number, i.e. all_edges[0] gets edge w/ global edge number 0)
        all_edges = []
        # A temporary map from an edge object to its global edge number (MUCH lower computational complexity)
        all_edges_map = {}
        # Keep track of what edge number we are on
        edge_count = 0
        # Iterate over all the surfaces in the .inp file
        for i in range(num_surfaces):
            # Go until we get to one of the surfaces (consisting of elements) in the .inp file
            while "ELEMENT" not in line:
                count += 1
                line = lines[count]
            count += 1
            line = lines[count]
            skip = count
            # Odd case where we have both a * and element on the same line as the only in-betweener of the surfaces
            if i >= 1:
                skip = skip - 1
            while "*" not in line:
                line = lines[count]
                count += 1
            rows = count - skip - 1

            # Load the elements' global nodes. This is our nodal connectivity list for this particular surface
            # Once again, slice off the element number, this is implied by the index into the array.
            # 1 is subtracted from each global node number so it aligns with starting at index 0 in the all_nodes array
            element_to_node_conn = np.subtract(np.loadtxt(filename, skiprows=skip, max_rows=rows, delimiter=',', dtype=np.int32)[:, 1:], 1)

            # EDGE SHENANIGANS
            # TODO: MAKE SURE NOTHING FUNNY GOING ON WITH CCW VS CW ROTATION
            # The element-to-edge connectivity list
            element_to_edge_conn = []
            # Iterate over each element in the nodal connectivity list
            for element in element_to_node_conn:
                # Construct 3 edges
                edge1 = Edge(element[0], element[1])
                edge2 = Edge(element[0], element[2])
                edge3 = Edge(element[1], element[2])
                # For each of the 3 edges in this element, check if we have created this edge before. If not, create it.
                for edge in (edge1, edge2, edge3):
                    if edge in all_edges_map:
                        # We do not want duplicate edges in our "all_edges" (global edge) list
                        pass
                    else:
                        # Otherwise, we have not come across this edge before, so add it
                        all_edges.append(edge)
                        all_edges_map[edge] = edge_count
                        edge_count += 1
                # Get the global edge numbers for these 3 edges
                edge1_number = all_edges_map[edge1]
                edge2_number = all_edges_map[edge2]
                edge3_number = all_edges_map[edge3]
                # Add the element_to_node_conn global edge numbers to the connectivity list
                element_to_edge_conn.append(np.array([edge1_number, edge2_number, edge3_number]))

            # Transform into numpy array
            element_to_edge_conn = np.array(element_to_edge_conn)
            # Add to edge connectivity list to complete list of edge connectivity lists
            surfaces_edge_connectivity.append(element_to_edge_conn)
            # Add nodal connectivity list to complete list of node connectivity lists
            surfaces_node_connectivity.append(element_to_node_conn)

        # Convert to numpy array
        all_edges = np.array(all_edges)
        Element.all_edges = all_edges

        # Go until we get to the boundary element block of the .inp file
        while "ELEMENT" not in line:
            line = lines[count]
            count += 1
        skip = count
        line = lines[count]
        while "*" not in line:
            line = lines[count]
            count += 1
        rows = count - skip - 1
        # Load the boundary elements
        boundary_elements = np.subtract(np.loadtxt(filename, skiprows=skip, max_rows=rows, delimiter=',', dtype=np.int32)[:, 1:], 1)

    # Get the set of all global node numbers that lie on the boundary of the geometry
    boundary_node_numbers = set(np.unique(boundary_elements))
    # Get the set of non-boundary global node numbers
    inner_node_numbers = set(np.arange(0, len(all_nodes))) - boundary_node_numbers
    # A map that takes one of the inner node numbers and maps it to a unique integer between [0, number of inner nodes]
    remap_inner_node_nums = {item: i for i, item in enumerate(inner_node_numbers)}
    # Old version of above list comprehension
    # for i, item in enumerate(inner_node_numbers):
        # remap_inner_node_nums[item] = i
    # Get a list of edges on the boundary (not a set, we will want the set of the global edge numbers, not the edges)
    boundary_edges = [Edge(element[0], element[1]) for element in boundary_elements]

    # DEBUG CODE TO MAKE SURE THINGS ARE WORKING PROPERLY. CAN REMOVE WHEN THOROUGHLY TESTED
    # Make sure that each edge in the boundary edges exists in the set of all edges
    for edge in boundary_edges:
        # Check against the all_edges_map for fast lookups. Do NOT check against all_edges, this will be slow.
        if edge in all_edges_map:
            # This is desired behavior, move on
            pass
        else:
            print("Error involving boundary edges. Boundary edge global node number not found")
            print(edge.node1)
            print(edge.node2)

    # Get the set of all global edge numbers that lie on the boundary of the geometry
    boundary_edge_numbers = set(all_edges_map[edge] for edge in boundary_edges)
    # Get the set of non-boundary global edge numbers
    inner_edge_numbers = set(np.arange(0, len(all_edges))) - boundary_edge_numbers
    # A map that takes one of the inner edge numbers and maps it to a unique integer between [0, number of inner edges]
    remap_inner_edge_nums = {item: i for i, item in enumerate(inner_edge_numbers)}

    # Verify the number of boundary nodes and boundary edges matches
    if len(boundary_node_numbers) != len(boundary_edge_numbers):
        print("Warning: Number of boundary edge numbers and node numbers differ. Make sure to merge the surfaces.")

    # Verify the number of elements involved with edges matches number of elements involved with nodes
    edge_element_sum = 0
    for surface in surfaces_edge_connectivity:
        edge_element_sum += len(surface)
    node_element_sum = 0
    for surface in surfaces_node_connectivity:
        node_element_sum += len(surface)
    if edge_element_sum != node_element_sum:
        print("Warning: Number of elements containing edges and number of elements containing nodes do not match.")

    # Construct a list of triangle Element objects, each holding the nodal, edge, and material information
    elements = []
    for i in range(len(surfaces_edge_connectivity)):
        for j in range(len(surfaces_edge_connectivity[i])):
            # permittivity = permittivities[i]
            elements.append(Element(surfaces_node_connectivity[i][j], surfaces_edge_connectivity[i][j], permittivities[i]))
    # Transform into numpy array
    elements = np.array(elements)

    # At last, all the necessary information has been parsed. Return it.
    # elements: The connectivity list for both the nodes and the edges (see "Element" class above)
    # all_nodes: A list of nodes. Index into this is the global node number, contents are x and y coords
    # all_edges: A list of edges. Index into this is the global edge number, contents are an Edge object (see above)
    # boundary_node_numbers: A set of global node numbers located on the boundary of the surface
    # boundary_edge_numbers: A set of global edge numbers located on the boundary of the surface
    # remap_inner_edge_nums: A map that takes one of the inner edge numbers and maps it to a unique integer between [0, number of inner edges]
    # remap_inner_node_nums: A map that takes one of the inner node numbers and maps it to a unique integer between [0, number of inner nodes]
    # all_edges_map: A map from an Edge object to its global edge number
    return elements, all_nodes, all_edges, boundary_node_numbers, boundary_edge_numbers, remap_inner_node_nums, remap_inner_edge_nums, all_edges_map


def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def is_inside(pt1, pt2, pt3, x, y):
    if sign((x, y), pt1, pt2) < 0.0:
        return False
    if sign((x, y), pt2, pt3) < 0.0:
        return False
    if sign((x, y), pt3, pt1) < 0.0:
        return False
    return True
