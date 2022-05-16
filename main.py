from util import load_mesh
from util import Edge
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

plt.ion()

mu0 = 4E-7*np.pi
epsilon0 = 8.8541878128E-12


def plot_rect_waveguide_mode(m, n, a=2, b=1, k0a_start=1, k0a_end=8, new_fig=False):
    """Plot the dispersion curve for the TE/TM mn mode (they look the same). a and b are the dimensions of waveguide."""
    if new_fig:
        plt.figure()
    min_k0 = k0a_start / a
    max_k0 = k0a_end / a
    num_freqs = 1000
    k0s = np.linspace(min_k0, max_k0, num_freqs)
    betas = np.sqrt(k0s**2 - (m*np.pi/a)**2 - (n*np.pi/b)**2)
    plt.plot(k0s*a, betas/k0s)


if __name__ == '__main__':
    # These need to be the same as the rectangular waveguide .inp file, we just happen to know the size ahead of time.
    a = 2
    b = 1
    # k0 = 1 is ~47.7 MHz
    num_freqs = 10
    # k0s = np.linspace(5E3, 5E6, num_freqs) * 2 * np.pi * np.sqrt(mu0*epsilon0)
    min_k0 = 1 / a
    max_k0 = 8 / a
    k0s = np.linspace(min_k0, max_k0, num_freqs)
    # Load the mesh from the file
    connectivity, all_nodes, all_edges, boundary_node_numbers, boundary_edge_numbers, remap_inner_node_nums, remap_inner_edge_nums, all_edges_map = load_mesh("rect_mesh_two_epsilons_coarse.inp", 2, [1, 1])
    # connectivity, all_nodes, all_edges, boundary_node_numbers, boundary_edge_numbers, remap_inner_node_nums, remap_inner_edge_nums, all_edges_map = load_mesh("rectangular_waveguide.inp", 1, [1])

    # Print out the lengths of some arrays of mesh data
    print(f"Number of Nodes: {len(all_nodes)}")
    print(f"Number of Edges: {len(all_edges)}")
    print(f"Number of Inner Nodes: {len(remap_inner_node_nums)}")
    print(f"Number of Inner Edges: {len(remap_inner_edge_nums)}")
    print()

    # Create the figure for plotting
    plt.figure()
    plt.xlabel(r"$k_0 * a$")
    plt.ylabel(r"$\beta / k_0$")
    # Analytical TE/TM Modes (the dispersion curves will look the same)
    for i in range(3):
        for j in range(3):
            if i == j == 0:
                continue
            plot_rect_waveguide_mode(i, j)

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
                # Iterate over the 3 edges of the element (p stands for prime here, see Jin's book pg 455)
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
                    # Just fixed the C_l and C_k (4/23/2022 5:01PM CT)
                    C_l = a_i_l * c_j_l - a_j_l * c_i_l
                    C_k = a_i_k * c_j_k - a_j_k * c_i_k
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
                    l = 1
                    if i == j:
                        l = 2
                    # These two values are identical to what was computed for the homogeneous waveguide LHS and RHS matrices
                    value1 = (b_l * b_k + c_l * c_k) / 4 / area
                    value2 = l * 2 * area / 24
                    Bzz[remap_inner_node_nums[nodes[i]], remap_inner_node_nums[nodes[j]]] += value1 - k0**2 * element.permittivity * value2

            # ---------------Put together Bzt and Btz matrices using both the nodes and edges------------------
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
                    C_l = a_i_l * c_j_l - a_j_l * c_i_l
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

        # Negate Att so that our betas come out positive
        Att = -Att
        lhs = np.concatenate([np.concatenate([Att, Atz], axis=1), np.concatenate([Azt, Azz], axis=1)], axis=0)
        rhs = np.concatenate([np.concatenate([Btt, Btz], axis=1), np.concatenate([Bzt, Bzz], axis=1)], axis=0)

        # These eigenvalues give the -(\beta^2) values. They are quickly flipped in sign and only positive values taken.
        print("Computing eigenvalues")
        eigenvalues, eigenvectors = eig(lhs, rhs, right=True)
        # Take the transpose such that each row of the matrix now corresponds to an eigenvector (helpful for sorting)
        eigenvectors = eigenvectors.transpose()
        # Prepare to sort the eigenvalues and eigenvectors by propagation constant
        p = np.argsort(eigenvalues)
        # All the eigenvalues should have no imaginary component. If they do, something is wrong. Still need this.
        eigenvalues = np.real(eigenvalues[p])
        eigenvectors = np.real(eigenvectors[p])
        first = np.argwhere(eigenvalues >= 0)[0, 0]
        positive_eigenvalues = eigenvalues[first:]
        positive_eigenvectors = eigenvectors[first:]
        print(eigenvalues)
        betas = np.sqrt(positive_eigenvalues)
        # Debug statement to observe how many eigenvalues are positive for a particular k0
        print(f"Number of propagating eigenvalues: {len(positive_eigenvalues)}, Total Eigenvalues: {len(eigenvalues)}")
        # Plot the beta values against k0
        plt.scatter(k0 * a * np.ones(len(betas)), betas/k0, color="blue", marker="o", facecolors='none')

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
    # The mode to show the fields for. This will tend to show the lowest possible mode first, but might not.
    # It really depends on the order of the beta values at that particular k0*a (frequency)
    mode = 0
    # Iterate over the 100x100 grid of points we want to plot
    for i in range(num_y_points):
        pt_y = y_points[i]
        for j in range(num_x_points):
            pt_x = x_points[j]
            element_num = -1
            for k, element in enumerate(connectivity):
                # Check if the point is inside this element
                if element.is_inside(pt_x, pt_y):
                    # The element this point lies in has been found, note it and stop searching
                    element_num = k
                    break
            # If we found an element that the point lies in, calculate the fields. If not found, Ex, Ey, Ez = 0
            if element_num != -1:
                # Get the phi values associated with each node
                phi1 = 0 if not element.nodes[0] in remap_inner_node_nums else eigenvectors[-1-mode, skip_tran + remap_inner_node_nums[element.nodes[0]]]
                phi2 = 0 if not element.nodes[1] in remap_inner_node_nums else eigenvectors[-1-mode, skip_tran + remap_inner_node_nums[element.nodes[1]]]
                phi3 = 0 if not element.nodes[2] in remap_inner_node_nums else eigenvectors[-1-mode, skip_tran + remap_inner_node_nums[element.nodes[2]]]
                # Interpolate to get Ez
                Ez[i, j] = element.nodal_interpolate(phi1, phi2, phi3, pt_x, pt_y)

                # Get the phi values associated with each edge
                phi1e = 0 if not element.edges[0] in remap_inner_edge_nums else eigenvectors[-1-mode, remap_inner_edge_nums[element.edges[0]]]
                phi2e = 0 if not element.edges[1] in remap_inner_edge_nums else eigenvectors[-1-mode, remap_inner_edge_nums[element.edges[1]]]
                phi3e = 0 if not element.edges[2] in remap_inner_edge_nums else eigenvectors[-1-mode, remap_inner_edge_nums[element.edges[2]]]
                # Interpolate to get Ex and Ey
                Ex[i, j], Ey[i, j] = element.edge_interpolate(phi1e, phi2e, phi3e, pt_x, pt_y)

    plt.figure()
    color_image = plt.imshow(Ez, extent=[x_min, x_max, y_min, y_max], cmap="cividis")
    plt.colorbar(label="Ez")
    X, Y = np.meshgrid(x_points, y_points)
    skip = (slice(None, None, 5), slice(None, None, 5))
    plt.quiver(X[skip], Y[skip], Ex[skip], Ey[skip], color="black")
    print(f"Plot shown for beta/k0 = {betas[-1-mode]/k0s[-1]}")
    # Do not need the below statement when plt.ion() is used
    # plt.show()
