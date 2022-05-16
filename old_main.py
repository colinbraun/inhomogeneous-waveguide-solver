from util import load_mesh
from util import Edge
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

plt.ion()

mu0 = 4E-7*np.pi
epsilon0 = 8.8541878128E-12


def interpolate_edge_element(element, side=0, new_fig=True):
    nodes = element.nodes
    nodes = np.array((all_nodes[nodes[0]], all_nodes[nodes[1]], all_nodes[nodes[2]]))
    a_i = nodes[(1+side) % 3][0]*nodes[(2+side) % 3][1] - nodes[(2+side) % 3][0]*nodes[(1+side) % 3][1]
    b_i = nodes[(1+side) % 3][1] - nodes[(2+side) % 3][1]
    c_i = nodes[(2+side) % 3][0] - nodes[(1+side) % 3][0]
    a_j = nodes[(2+side) % 3][0]*nodes[(0+side) % 3][1] - nodes[(0+side) % 3][0]*nodes[(2+side) % 3][1]
    b_j = nodes[(2+side) % 3][1] - nodes[(0+side) % 3][1]
    c_j = nodes[(0+side) % 3][0] - nodes[(2+side) % 3][0]
    A_m = a_i*b_j - a_j*b_i
    B_m = c_i*b_j - c_j*b_i
    C_m = a_i*c_j - a_j*c_i
    D_m = b_i*c_j - b_j*c_i
    edge = Edge(element.nodes[(0+side) % 3], element.nodes[(1+side) % 3])
    x = -0.2
    y = 0.0001
    # x = -0.1888
    # y = 0.0615
    x_component = edge.length / 4 / element.area()**2 * (A_m + B_m*y)
    y_component = edge.length / 4 / element.area()**2 * (C_m + D_m*x)
    # print(nodes)
    # print(x_component)
    # print(y_component)
    x_min = np.min(nodes[:, 0])
    x_max = np.max(nodes[:, 0])
    y_min = np.min(nodes[:, 1])
    y_max = np.max(nodes[:, 1])
    num_x_points = 100
    num_y_points = 100
    x = np.linspace(x_min, x_max, num_x_points)
    y = np.linspace(y_min, y_max, num_y_points)
    x_components = np.zeros([num_x_points, num_y_points])
    y_components = np.zeros([num_x_points, num_y_points])
    for i in range(num_y_points):
        pt_y = y[i]
        for j in range(num_x_points):
            pt_x = x[j]
            x_component = edge.length / 4 / element.area() ** 2 * (A_m + B_m * pt_y)
            y_component = edge.length / 4 / element.area() ** 2 * (C_m + D_m * pt_x)
            if element.is_inside(pt_x, pt_y):
                x_components[i, j] = x_component
                y_components[i, j] = y_component

    X, Y = np.meshgrid(x, y)
    skip = (slice(None, None, 5), slice(None, None, 5))
    if new_fig:
        plt.figure()
    plt.quiver(X[skip], Y[skip], x_components[skip], y_components[skip], color="black", width=0.005)
    plt.scatter(nodes[:, 0], nodes[:, 1])


if __name__ == '__main__':
    # These need to be the same as the rectangular waveguide, we just happen to know the size ahead of time.
    a = 2
    b = 1
    # k0 = 1 is ~47.7 MHz
    # Try 50MHz-5GHz
    num_freqs = 10
    # k0s = np.linspace(5E3, 5E6, num_freqs) * 2 * np.pi * np.sqrt(mu0*epsilon0)
    k0s = np.linspace(2/a, 8/a, num_freqs)
    # Load the mesh from the file
    connectivity, all_nodes, all_edges, boundary_node_numbers, boundary_edge_numbers, remap_inner_node_nums, remap_inner_edge_nums, all_edges_map = load_mesh("rect_mesh_two_epsilons_coarse.inp", 2, [1, 4])
    # connectivity, all_nodes, all_edges, boundary_node_numbers, boundary_edge_numbers, remap_inner_node_nums, remap_inner_edge_nums, all_edges_map = load_mesh("rectangular_waveguide.inp", 1, [1])
    # Print out the lengths of some arrays of mesh data
    print(f"Number of Nodes: {len(all_nodes)}")
    print(f"Number of Edges: {len(all_edges)}")
    print(f"Number of Inner Nodes: {len(remap_inner_node_nums)}")
    print(f"Number of Inner Edges: {len(remap_inner_edge_nums)}")
    print()
    # Some code to verify that adjacent elements' edges are always in the opposite direction when cycling through nodes
    # element1 = connectivity[0]
    # interpolate_edge_element(element1)
    # element2 = None
    # for element in connectivity:
    #     if element.is_adjacent_to(element1):
    #         # print("Found an adjacent element")
    #         element2 = element
    #         interpolate_edge_element(element, side=2)
    #         break
    #         print(element1.nodes)
    #         print(element2.nodes)
    # # print(element1.nodes[2] in boundary_node_numbers)
    # # TODO: This is just for debugging to ensure that the edge basis functions are working properly
    # plt.show()

    # Create the figure for plotting
    plt.figure()
    plt.xlabel(r"$k_0 * a$")
    plt.ylabel(r"$\beta / k_0$")

    for k0 in k0s:
        print(f"Solving for k0 = {k0}")
        # Create empty matrices
        Att = np.zeros([len(remap_inner_edge_nums), len(remap_inner_edge_nums)])
        # These 3 will be left as zeros
        Azz = np.zeros([len(remap_inner_node_nums), len(remap_inner_node_nums)])
        Atz = np.zeros([len(remap_inner_edge_nums), len(remap_inner_node_nums)])
        Azt = np.zeros([len(remap_inner_node_nums), len(remap_inner_edge_nums)])

        Btt = np.zeros([len(remap_inner_edge_nums), len(remap_inner_edge_nums)])
        Bzz = np.zeros([len(remap_inner_node_nums), len(remap_inner_node_nums)])
        Btz = np.zeros([len(remap_inner_edge_nums), len(remap_inner_node_nums)])
        Bzt = np.zeros([len(remap_inner_node_nums), len(remap_inner_edge_nums)])

        # Iterate over all the elements in the mesh
        for n, element in enumerate(connectivity):
            # Get the nodes and edges that make up this element (avoids writing element.nodes[#], element.edges[#] later)
            nodes = element.nodes
            edges = element.edges

            # The area of the element
            area = element.area()
            # Average x and y values of the 3 points making up this element
            x_mean = (all_nodes[nodes[0]][0] + all_nodes[nodes[1]][0] + all_nodes[nodes[2]][0]) / 3
            y_mean = (all_nodes[nodes[0]][1] + all_nodes[nodes[1]][1] + all_nodes[nodes[2]][1]) / 3

            # ---------------------Put together the Att and Btt matrices using the edges------------------------
            # This is my new way of doing the construction of Att and Btt
            # Iterate over the 3 edges of the element
            for l, k in ((0, 1), (1, 2), (2, 0)):
                # The first edge whose basis vector will be integrated against another edge (edge2)
                edge1 = Edge(nodes[l], nodes[k])
                # Skip edges on the boundary
                if all_edges_map[edge1] in boundary_edge_numbers:
                    continue
                # Index of the third node (the one not making up the edge) going in a CCW fashion
                m = (k + 1) % 3
                # Create the ccw node list started from the first node of the edge
                nodes_lk = (all_nodes[nodes[l]], all_nodes[nodes[k]], all_nodes[nodes[m]])
                # Iterate over the 3 edges of the element (p stands for prime here, see Jin's book)
                for l_p, k_p in ((0, 1), (1, 2), (2, 0)):
                    # The second edge whose basis vector will be integrated against another edge (edge1)
                    edge2 = Edge(nodes[l_p], nodes[k_p])
                    # Skip edges on the boundary
                    if all_edges_map[edge2] in boundary_edge_numbers:
                        continue
                    # Index of the third node (the one not making up the edge) going in a CCW fashion
                    m_p = (k_p + 1) % 3
                    # Create the ccw node list started from the first node of the edge
                    nodes_lpkp = (all_nodes[nodes[l_p]], all_nodes[nodes[k_p]], all_nodes[nodes[m_p]])
                    # Constants needed to calculate the integral involving the 2 edges
                    # These come from the nodal basis function definitions (see Jin's book pg 441)
                    # These are for the first edge (call it edge l)
                    a_i_l, a_j_l = nodes_lk[1][0]*nodes_lk[2][1] - nodes_lk[2][0]*nodes_lk[1][1], nodes_lk[2][0]*nodes_lk[0][1] - nodes_lk[0][0]*nodes_lk[2][1]
                    b_i_l, b_j_l = nodes_lk[1][1] - nodes_lk[2][1], nodes_lk[2][1] - nodes_lk[0][1]
                    c_i_l, c_j_l = nodes_lk[2][0] - nodes_lk[1][0], nodes_lk[0][0] - nodes_lk[2][0]
                    # These are for the second edge (call it edge k)
                    a_i_k, a_j_k = nodes_lpkp[1][0]*nodes_lpkp[2][1] - nodes_lpkp[2][0]*nodes_lpkp[1][1], nodes_lpkp[2][0]*nodes_lpkp[0][1] - nodes_lpkp[0][0]*nodes_lpkp[2][1]
                    b_i_k, b_j_k = nodes_lpkp[1][1] - nodes_lpkp[2][1], nodes_lpkp[2][1] - nodes_lpkp[0][1]
                    c_i_k, c_j_k = nodes_lpkp[2][0] - nodes_lpkp[1][0], nodes_lpkp[0][0] - nodes_lpkp[2][0]
                    # These come from the edge basis function definitions (see NASA paper eqs. 51 and 56-60)
                    A_l = a_i_l * b_j_l - a_j_l * b_i_l
                    A_k = a_i_k * b_j_k - a_j_k * b_i_k
                    B_l = c_i_l * b_j_l - c_j_l * b_i_l
                    B_k = c_i_k * b_j_k - c_j_k * b_i_k
                    C_l = a_i_l * c_j_l - b_j_l * c_i_l
                    C_k = a_i_k * c_j_k - b_j_k * c_i_k
                    D_l = b_i_l * c_j_l - b_j_l * c_i_l
                    D_k = b_i_k * c_j_k - b_j_k * c_i_k
                    # See An Introduction to the finite element method 3rd edition page 426 and NASA paper eqs. 73-77
                    I1 = A_l*A_k + C_l*C_k
                    I2 = (C_l*D_k + C_k*D_l) * x_mean
                    I3 = (A_l*B_k + A_k*B_l) * y_mean
                    I4 = B_l*B_k * (sum([node[1]**2 for node in nodes_lk]) + 9*y_mean**2) / 12
                    I5 = D_l*D_k * (sum([node[0]**2 for node in nodes_lk]) + 9*x_mean**2) / 12
                    # curl(Ni_bar) dot curl(Nj_bar) - NASA paper pg 12
                    value1 = edge1.length * edge2.length * D_l * D_k / 4 / (area**3)
                    # Ni_bar dot Nj_bar - NASA paper pg 12
                    value2 = edge1.length * edge2.length * (I1 + I2 + I3 + I4 + I5) / 16 / (area**3)
                    Att[remap_inner_edge_nums[all_edges_map[edge1]], remap_inner_edge_nums[all_edges_map[edge2]]] += value1 - (k0**2 * element.permittivity * value2)
                    Btt[remap_inner_edge_nums[all_edges_map[edge1]], remap_inner_edge_nums[all_edges_map[edge2]]] += value2

            # This is my old way of doing the construction of Att and Btt
            # Iterate over these combinations of edges (see Jin page 455 and 456, but note 9.3.21 refers to local nodes)
            # for l, k in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
            #     # Skip over boundary edges
            #     if edges[l] in boundary_edge_numbers or edges[k] in boundary_edge_numbers:
            #         continue
            #     # Do the (very) ugly computation
            #     # Get the 2 edge nodes
            #     node1_l_num = all_edges[edges[l]].node1
            #     node2_l_num = all_edges[edges[l]].node2
            #     # 3rd node needed for some computations, determine this based on which of the three isn't part of the edge
            #     node3_l_num = nodes[0] if nodes[0] not in (node1_l_num, node2_l_num) else nodes[1] if nodes[1] not in (node1_l_num, node2_l_num) else nodes[2]
            #     # print(f"Three Nodes: {node1_l_num}, {node2_l_num}, {node3_l_num}")
            #     # print(edges)
            #     # Repeat identical process but instead for the k edge
            #     node1_k_num = all_edges[edges[k]].node1
            #     node2_k_num = all_edges[edges[k]].node2
            #     node3_k_num = nodes[0] if nodes[0] not in (node1_k_num, node2_k_num) else nodes[1] if nodes[1] not in (node1_k_num, node2_k_num) else nodes[2]
            #     # print(f"Three Nodes: {node1_l_num}, {node2_l_num}, {node3_l_num}")
            #     # Stitch the nodes together. One might wonder why bother doing what we just did? We already had the 3 nodes
            #     # of the element. We want to ensure That the node we start on is different for l and k, and also that
            #     # we traverse the element in a ccw fashion.
            #     nodes_l = (all_nodes[node1_l_num], all_nodes[node2_l_num], all_nodes[node3_l_num])
            #     nodes_k = (all_nodes[node1_k_num], all_nodes[node2_k_num], all_nodes[node3_k_num])

            #     # temp_l = [node1_l_num, node2_l_num, node3_l_num]
            #     # as_list = list(nodes)
            #     # if temp_l == as_list or temp_l[:1] + temp_l[1:] == as_list or temp_l[:2] + temp_l[2:] == as_list:
            #     #     print("True")
            #     # else:
            #     #     print("False")
            #     #     temp_l = list(reversed(temp_l))
            #     # print(nodes)
            #     # print(temp_l)
            #     # # print(all_edges[edges[l]].length)
            #     # print()

            #     # i and j are the nodes connecting our edge
            #     a_i_l, a_j_l = nodes_l[1][0]*nodes_l[2][1] - nodes_l[2][0]*nodes_l[1][1], nodes_l[2][0]*nodes_l[0][1] - nodes_l[0][0]*nodes_l[2][1]
            #     b_i_l, b_j_l = nodes_l[1][1] - nodes_l[2][1], nodes_l[2][1] - nodes_l[0][1]
            #     c_i_l, c_j_l = nodes_l[2][0] - nodes_l[1][0], nodes_l[0][0] - nodes_l[2][0]
            #     a_i_k, a_j_k = nodes_k[1][0]*nodes_k[2][1] - nodes_k[2][0]*nodes_k[1][1], nodes_k[2][0]*nodes_k[0][1] - nodes_k[0][0]*nodes_k[2][1]
            #     b_i_k, b_j_k = nodes_k[1][1] - nodes_k[2][1], nodes_k[2][1] - nodes_k[0][1]
            #     c_i_k, c_j_k = nodes_k[2][0] - nodes_k[1][0], nodes_k[0][0] - nodes_k[2][0]
            #     A_l = a_i_l * b_j_l - a_j_l * b_i_l
            #     A_k = a_i_k * b_j_k - a_j_k * b_i_k
            #     B_l = c_i_l * b_j_l - c_j_l * b_i_l
            #     B_k = c_i_k * b_j_k - c_j_k * b_i_k
            #     C_l = a_i_l * c_j_l - b_j_l * c_i_l
            #     C_k = a_i_k * c_j_k - b_j_k * c_i_k
            #     D_l = b_i_l * c_j_l - b_j_l * c_i_l
            #     D_k = b_i_k * c_j_k - b_j_k * c_i_k
            #     # See An Introduction to the finite element method 3rd edition page 426 and NASA paper
            #     I1 = A_l*A_k + C_l*C_k
            #     I2 = (C_l*D_k + C_k*D_l) * x_mean
            #     I3 = (A_l*B_k + A_k*B_l) * y_mean
            #     I4 = B_l*B_k * (sum([node[1]**2 for node in nodes_l]) + 9*y_mean**2) / 12
            #     I5 = D_l*D_k * (sum([node[0]**2 for node in nodes_l]) + 9*x_mean**2) / 12
            #     length_l, length_k = all_edges[edges[l]].length, all_edges[edges[k]].length
            #     # (curl(Ni_bar)) dot (curl(Nj_bar))
            #     value1 = length_l * length_k * D_l * D_k / 4 / (element.area()**3)
            #     # Ni_bar dot Nj_bar
            #     value2 = length_l * length_k * (I1 + I2 + I3 + I4 + I5) / 16 / (element.area()**3)
            #     Att[remap_inner_edge_nums[edges[l]], remap_inner_edge_nums[edges[k]]] += value1 - (element.permittivity * value2)
            #     Btt[remap_inner_edge_nums[edges[l]], remap_inner_edge_nums[edges[k]]] += value2

            # ------------------------Put together Bzz matrix using the nodes---------------------------
            # This is identical to what is done in the homogeneous waveguide.
            # Treat i as l from Jin's book page 469
            for i in range(len(nodes)):
                # Skip over edge nodes
                if nodes[i] in boundary_node_numbers:
                    continue
                # Treat j as k
                for j in range(len(nodes)):
                    # Skip over edge nodes
                    if nodes[j] in boundary_node_numbers:
                        continue
                    # Equations taken from Jin's book
                    b_l = all_nodes[nodes[(i + 1) % 3], 1] - all_nodes[nodes[(i + 2) % 3], 1]
                    b_k = all_nodes[nodes[(j + 1) % 3], 1] - all_nodes[nodes[(j + 2) % 3], 1]
                    c_l = all_nodes[nodes[(i + 2) % 3], 0] - all_nodes[nodes[(i + 1) % 3], 0]
                    c_k = all_nodes[nodes[(j + 2) % 3], 0] - all_nodes[nodes[(j + 1) % 3], 0]
                    b_1 = all_nodes[nodes[1], 1] - all_nodes[nodes[2], 1]
                    c_1 = all_nodes[nodes[2], 0] - all_nodes[nodes[1], 0]
                    b_2 = all_nodes[nodes[2], 1] - all_nodes[nodes[0], 1]
                    c_2 = all_nodes[nodes[0], 0] - all_nodes[nodes[2], 0]
                    # area = 0.5 * (b_1 * c_2 - b_2 * c_1)
                    l = 1
                    if i == j:
                        l = 2
                    # These two values are identical to what was computed for the homogeneous waveguide LHS and RHS matrices
                    value1 = (b_l * b_k + c_l * c_k) / 4 / area
                    value2 = l * 2 * area / 24
                    Bzz[remap_inner_node_nums[nodes[i]], remap_inner_node_nums[nodes[j]]] += value1 - k0**2 * element.permittivity * value2
                    # a[remap_inner_node_nums[nodes[i]], remap_inner_node_nums[nodes[j]]] += (b_l * b_k + c_l * c_k) / 4 / area
                    # b[remap_inner_node_nums[nodes[i]], remap_inner_node_nums[nodes[j]]] += l * 2 * area / 24

            # ---------------Put together Bzt and mtz matrices using both the nodes and edges------------------
            # This is my new way of doing the construction of Bzt and Btz matrices
            for i in range(len(nodes)):
                # Skip the boundary nodes
                if nodes[i] in boundary_node_numbers:
                    continue
                for l, k in ((0, 1), (1, 2), (2, 0)):
                    edge1 = Edge(nodes[l], nodes[k])
                    # Skip the boundary edges
                    if all_edges_map[edge1] in boundary_edge_numbers:
                        continue
                    m = (k + 1) % 3
                    # Create the ccw node list started from the first node of the edge
                    nodes_lk = (all_nodes[nodes[l]], all_nodes[nodes[k]], all_nodes[nodes[m]])
                    # Constants needed to calculate the integral involving the edge and the node
                    # These come from the nodal basis function definitions (see Jin's book pg 441)
                    # These are for the first edge (call it edge l)
                    a_i_l, a_j_l = nodes_lk[1][0]*nodes_lk[2][1] - nodes_lk[2][0]*nodes_lk[1][1], nodes_lk[2][0]*nodes_lk[0][1] - nodes_lk[0][0]*nodes_lk[2][1]
                    b_i_l, b_j_l = nodes_lk[1][1] - nodes_lk[2][1], nodes_lk[2][1] - nodes_lk[0][1]
                    c_i_l, c_j_l = nodes_lk[2][0] - nodes_lk[1][0], nodes_lk[0][0] - nodes_lk[2][0]
                    # These come from the edge basis function definitions (see NASA paper eqs. 51 and 56-60)
                    A_l = a_i_l * b_j_l - a_j_l * b_i_l
                    B_l = c_i_l * b_j_l - c_j_l * b_i_l
                    C_l = a_i_l * c_j_l - b_j_l * c_i_l
                    D_l = b_i_l * c_j_l - b_j_l * c_i_l
                    # Node constants
                    # a_k = all_nodes[nodes[(i+1) % 3], 0]*all_nodes[nodes[(i+2) % 3], 1] - all_nodes[nodes[(i+2) % 3], 0]*all_nodes[nodes[(i+1) % 3], 1]
                    b_k = all_nodes[nodes[(i + 1) % 3], 1] - all_nodes[nodes[(i + 2) % 3], 1]
                    c_k = all_nodes[nodes[(i + 2) % 3], 0] - all_nodes[nodes[(i + 1) % 3], 0]
                    part1 = area * b_k * A_l
                    part2 = area * b_k * B_l * y_mean
                    part3 = area * c_k * C_l
                    part4 = area * c_k * D_l * x_mean
                    total = edge1.length / 8 / area**3 * (part1 + part2 + part3 + part4)
                    Bzt[remap_inner_node_nums[nodes[i]], remap_inner_edge_nums[all_edges_map[edge1]]] += total
                    Btz[remap_inner_edge_nums[all_edges_map[edge1]], remap_inner_node_nums[nodes[i]]] += total

            # Iterate over the 3 nodes of this element
            # for i in range(len(nodes)):
            #     # Skip over boundary nodes
            #     if nodes[i] in boundary_node_numbers:
            #         continue
            #     # Iterate over the 3 edges of this element
            #     for j in range(len(edges)):
            #         # Skip over boundary edges
            #         if edges[j] in boundary_edge_numbers:
            #             continue
            #         # Do the ugly computation
            #         node1_num = nodes[0]
            #         node2_num = nodes[1]
            #         node3_num = nodes[2]
            #         # Get the 2 edge nodes
            #         node1_l_num = all_edges[edges[j]].node1
            #         node2_l_num = all_edges[edges[j]].node2
            #         # 3rd node needed for some computations, determine this based on which of the three isn't part of the edge
            #         node3_l_num = node1_num if node1_num not in (node1_l_num, node2_l_num) else node2_num if node2_num not in (node1_l_num, node2_l_num) else node3_num
            #         nodes_l = (all_nodes[node1_l_num], all_nodes[node2_l_num], all_nodes[node3_l_num])
            #         # Edge constants
            #         a_i_l, a_j_l = nodes_l[1][0]*nodes_l[2][1] - nodes_l[2][0]*nodes_l[1][1], nodes_l[2][0]*nodes_l[0][1] - nodes_l[0][0]*nodes_l[2][1]
            #         b_i_l, b_j_l = nodes_l[1][1] - nodes_l[2][1], nodes_l[2][1] - nodes_l[0][1]
            #         c_i_l, c_j_l = nodes_l[2][0] - nodes_l[1][0], nodes_l[0][0] - nodes_l[2][0]
            #         A_l = a_i_l * b_j_l - a_j_l * b_i_l
            #         B_l = c_i_l * b_j_l - c_j_l * b_i_l
            #         C_l = a_i_l * c_j_l - b_j_l * c_i_l
            #         D_l = b_i_l * c_j_l - b_j_l * c_i_l
            #         # Node constants
            #         # a_k = all_nodes[nodes[(i+1) % 3], 0]*all_nodes[nodes[(i+2) % 3], 1] - all_nodes[nodes[(i+2) % 3], 0]*all_nodes[nodes[(i+1) % 3], 1]
            #         b_k = all_nodes[nodes[(i + 1) % 3], 1] - all_nodes[nodes[(i + 2) % 3], 1]
            #         c_k = all_nodes[nodes[(i + 2) % 3], 0] - all_nodes[nodes[(i + 1) % 3], 0]
            #         area = element.area()
            #         part1 = area * b_k * A_l
            #         part2 = area * b_k * B_l * y_mean
            #         part3 = area * c_k * C_l
            #         part4 = area * c_k * D_l * x_mean
            #         Bzt[remap_inner_node_nums[nodes[i]], remap_inner_edge_nums[edges[j]]] = part1 + part2 + part3 + part4
            #         Btz[remap_inner_edge_nums[edges[j]], remap_inner_node_nums[nodes[i]]] = part1 + part2 + part3 + part4

        # print((Btz.transpose() == Bzt).all())
        # Negate Att so that our betas come out positive
        Att = -Att
        lhs = np.concatenate([np.concatenate([Att, Atz], axis=1), np.concatenate([Azt, Azz], axis=1)], axis=0)
        rhs = np.concatenate([np.concatenate([Btt, Btz], axis=1), np.concatenate([Bzt, Bzz], axis=1)], axis=0)

        # These eigenvalues give the -(\beta^2) values. They are quickly flipped in sign and only positive values taken.
        print("Computing eigenvalues")
        eigenvalues, eigenvectors = eig(lhs, rhs)
        # Take the transpose such that each row of the matrix now corresponds to an eigenvector (helpful for sorting)
        eigenvectors = eigenvectors.transpose()
        # Prepare to sort the eigenvalues and eigenvectors by propagation constant
        p = np.argsort(eigenvalues)
        # All the eigenvalues should have no imaginary component. If they do, something is wrong. Still need this.
        eigenvalues = np.real(eigenvalues[p])
        eigenvectors = np.real(eigenvectors[p])
        # Find the index of the first non-negative eigenvalue (propagation constant)
        first = np.argwhere(eigenvalues >= 0)[0, 0]
        positive_eigenvalues = eigenvalues[first:]
        positive_eigenvectors = eigenvectors[first:]
        print(eigenvalues)
        betas = np.sqrt(positive_eigenvalues)
        # Debug statement to observe how many eigenvalues are positive for a particular k0
        print(f"Number of propagating eigenvalues: {len(positive_eigenvalues)}, Total Eigenvalues: {len(eigenvalues)}")
        # Plot the beta values against k0
        plt.scatter(k0 * a * np.ones(len(betas)), betas/k0, color="blue", marker="o", facecolors='none')
        # break

    # ----------------------FIELD PLOTTING--------------------------
    # Find the minimum and maximum x and y values among the nodes:
    x_min = np.amin(all_nodes[:, 0])
    y_min = np.amin(all_nodes[:, 1])
    x_max = np.amax(all_nodes[:, 0])
    y_max = np.amax(all_nodes[:, 1])
    # Create a rectangular grid of points that the geometry is inscribed in
    num_x_points = 100
    x_points = np.linspace(x_min, x_max, num_x_points)
    num_y_points = 100
    y_points = np.linspace(y_min, y_max, num_y_points)
    Ez = np.zeros([num_x_points, num_y_points])
    Ex = np.zeros([num_x_points, num_y_points])
    Ey = np.zeros([num_x_points, num_y_points])

    # Just used to skip over the transverse edge coefficients, the Ez coefficients come right after them
    skip_tran = len(remap_inner_edge_nums)
    # Iterate over the 100x100 grid of points we want to plot
    for i in range(num_y_points):
        pt_y = y_points[i]
        for j in range(num_x_points):
            pt_x = x_points[j]
            element_num = -1
            for k, element in enumerate(connectivity):
                # Check if the point is inside this element
                # if is_inside(all_nodes[element[0]], all_nodes[element[1]], all_nodes[element[2]], pt_x, pt_y):
                if element.is_inside(pt_x, pt_y):
                    # The element this point lies in has been found, note it and stop searching
                    element_num = k
                    break
            # If we found an element that the point lies in, calculate the fields. If not found, Ex, Ey, Ez = 0
            if element_num != -1:
                # Get the phi values associated with each node
                phi1 = 0 if not element.nodes[0] in remap_inner_node_nums else eigenvectors[first+1, skip_tran + remap_inner_node_nums[element.nodes[0]]]
                phi2 = 0 if not element.nodes[1] in remap_inner_node_nums else eigenvectors[first+1, skip_tran + remap_inner_node_nums[element.nodes[1]]]
                phi3 = 0 if not element.nodes[2] in remap_inner_node_nums else eigenvectors[first+1, skip_tran + remap_inner_node_nums[element.nodes[2]]]
                # Ez[i, j] = interpolate(all_nodes[element[0]], all_nodes[element[1]], all_nodes[element[2]], phi1, phi2, phi3, pt_x, pt_y)
                Ez[i, j] = element.nodal_interpolate(phi1, phi2, phi3, pt_x, pt_y)

                phi1e = 0 if not element.edges[0] in remap_inner_edge_nums else eigenvectors[first+1, remap_inner_edge_nums[element.edges[0]]]
                phi2e = 0 if not element.edges[1] in remap_inner_edge_nums else eigenvectors[first+1, remap_inner_edge_nums[element.edges[1]]]
                phi3e = 0 if not element.edges[2] in remap_inner_edge_nums else eigenvectors[first+1, remap_inner_edge_nums[element.edges[2]]]
                Ex[i, j], Ey[i, j] = element.edge_interpolate(phi1e, phi2e, phi3e, pt_x, pt_y)

    plt.figure()
    color_image = plt.imshow(Ez, extent=[x_min, x_max, y_min, y_max], cmap="cividis")
    plt.colorbar(label="Ez")
    X, Y = np.meshgrid(x_points, y_points)
    skip = (slice(None, None, 5), slice(None, None, 5))
    # plt.quiver(X[skip], Y[skip], Ex[skip], Ey[skip], color="black", width=0.005)
    plt.quiver(X[skip], Y[skip], Ex[skip], Ey[skip], color="black")
    # plt.show()
