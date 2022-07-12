from iwaveguide.util import load_mesh
from iwaveguide.util import Edge
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.integrate import trapz

mu0 = 4E-7*np.pi
epsilon0 = 8.8541878128E-12


class Waveguide:

    def __init__(self, filename, surface_names, boundary_name, permittivities=None, p1=None, p2=None):
        """
        Constructor for a waveguide. Can be inhomogeneous and any shape.
        :param filename: The name of the abaqus file (.inp) to be used for this Waveguide object.
        :param surface_names: A list containing the names of each surface of the mesh in the .inp file.
        :param boundary_name: A string of the name of the boundary nodes of the mesh in the .inp file.
        :param permittivities: The corresponding permittivities of each surface as a list. Passing None will assign 1.
        :param p1: If an integration line is desired, ``p1`` is the starting point of the line integral. Default = None.
        :param p2: If an integration line is desired, ``p2`` is the ending point of the line integral. Default = None.
        """
        self.connectivity, self.all_nodes, self.all_edges, self.boundary_node_numbers, self.boundary_edge_numbers, self.remap_inner_node_nums, self.remap_inner_edge_nums, self.all_edges_map = load_mesh(filename, surface_names, boundary_name, permittivities)
        # Compute the bounds of the waveguide
        # self.x_min = np.amin(self.all_nodes[:, 0])
        self.x_min = np.amin(self.all_nodes[self.boundary_node_numbers, 0])
        # self.y_min = np.amin(self.all_nodes[:, 1])
        self.y_min = np.amin(self.all_nodes[self.boundary_node_numbers, 1])
        # self.x_max = np.amax(self.all_nodes[:, 0])
        self.x_max = np.amax(self.all_nodes[self.boundary_node_numbers, 0])
        # self.y_max = np.amax(self.all_nodes[:, 1])
        self.y_max = np.amax(self.all_nodes[self.boundary_node_numbers, 1])
        # The mode to choose for the propagation constant and eigenvector. Used in getting and plotting field data
        # mode_index of -1 indicates the mode with the highest propagation constant. -2 is the next highest.
        self.mode_index = -1
        # Hold a copy of solutions for when the waveguide has been solved
        self.betas = None
        self.eigenvectors = None
        # Store the k0 value of the current solution
        self.k0 = -1
        # Hold on to the integration line points
        self.p1, self.p2 = p1, p2

    def set_mode_index(self, mode_index):
        """
        Set the mode index. Essentially chooses which mode to use for getting and plotting field data.
        Mode 0 indicates the mode with the highest propagation constant. Mode 1 is the next highest, and so on.
        :param mode_index: The index of the desired mode.
        :return: Nothing
        """
        self.mode_index = -1 - mode_index

    def get_selected_beta(self):
        """
        Get the beta value of the currently selected mode index (set using set_mode_index()).
        :return: The beta value of the currently selected mode.
        """
        return self.betas[self.mode_index]

    def solve_k0(self, k0):
        """
        Solve for the propagation constants and eigenvectors of a waveguide at a particular k0.
        Can be inhomogeneous and any shape.
        :param k0: The k0 (essentially a frequency) to solve for.
        :return: Two numpy arrays. The first holds the propagation constants for the k0. The second holds the eigenvectors
        for the k0. Each array is sorted by propagation constant. Eigenvectors are stored as rows in the returned numpy
        array, not columns.
        """
        # Store the k0 being used as the solution
        self.k0 = k0
        # Load the mesh from the file
        connectivity, all_nodes, all_edges, boundary_node_numbers, boundary_edge_numbers, remap_inner_node_nums, remap_inner_edge_nums, all_edges_map = self.connectivity, self.all_nodes, self.all_edges, self.boundary_node_numbers, self.boundary_edge_numbers, self.remap_inner_node_nums, self.remap_inner_edge_nums, self.all_edges_map

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
            # edges = element.edges

            # The area of the element
            area = element.area()
            # Average x and y values of the 3 points making up this element
            x_mean = (all_nodes[nodes[0]][0] + all_nodes[nodes[1]][0] + all_nodes[nodes[2]][0]) / 3
            y_mean = (all_nodes[nodes[0]][1] + all_nodes[nodes[1]][1] + all_nodes[nodes[2]][1]) / 3

            # ---------------------Put together the Att and Btt matrices using the edges------------------------
            # Iterate over the 3 edges of the element
            for l, k in ((0, 1), (1, 2), (2, 0)):
                # The first edge whose basis vector will be integrated against another edge (edge2)
                edge1 = Edge(nodes[l], nodes[k], self.all_nodes)
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
                    edge2 = Edge(nodes[l_p], nodes[k_p], self.all_nodes)
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
                    # b_1 = all_nodes[nodes[1], 1] - all_nodes[nodes[2], 1]
                    # c_1 = all_nodes[nodes[2], 0] - all_nodes[nodes[1], 0]
                    # b_2 = all_nodes[nodes[2], 1] - all_nodes[nodes[0], 1]
                    # c_2 = all_nodes[nodes[0], 0] - all_nodes[nodes[2], 0]
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
                    edge1 = Edge(nodes[l], nodes[k], self.all_nodes)
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
        eigenvalues, eigenvectors = eig(lhs, rhs, right=True)
        # Take the transpose such that each row of the matrix now corresponds to an eigenvector (helpful for sorting)
        eigenvectors = eigenvectors.transpose()
        # Prepare to sort the eigenvalues and eigenvectors by propagation constant
        p = np.argsort(eigenvalues)
        # All the eigenvalues should have no imaginary component. If they do, something is wrong. Still need this.
        eigenvalues = np.real(eigenvalues[p])
        eigenvectors = np.real(eigenvectors[p])
        # Find the first positive propagation constant
        first_pos = np.argwhere(eigenvalues >= 0)[0, 0]
        betas, eigenvectors = np.sqrt(eigenvalues[first_pos:]), eigenvectors[first_pos:]
        # Remove infinite eigenvalues (if any are found to exist)
        last_pos = np.argwhere(betas == float('inf'))
        last_pos = slice(0, last_pos[0, 0]) if len(last_pos) > 0 else slice(0, len(betas))
        betas, eigenvectors = betas[last_pos], eigenvectors[last_pos]
        # Return the \beta values and their corresponding eigenvectors
        self.betas = betas
        self.eigenvectors = eigenvectors
        # If an integration line is defined, determine if we should flip the direction of the fields or not.
        if self.p1 is not None and self.p2 is not None:
            self.eigenvectors *= 1 if self.integrate_line(self.p1, self.p2) >= 0 else -1
        return betas, eigenvectors

    def solve(self, start_k0=-1, end_k0=-1, num_k0s=10):
        """
        Solve for the propagation constants and eigenvectors of a waveguide.
        Can be inhomogeneous and any shape, but non-rectangular waveguides should specify the start and end k0 values.
        :param start_k0: The starting k0 (essentially a frequency) to solve for. Default is 2 / waveguide's x length.
        :param end_k0: The ending k0 (essentially a frequency) to solve for. Default is 8 / waveguide's x length.
        :param num_k0s: The number of k0 values to solve for (between start_k0 and end_k0).
        :return: Three numpy arrays. The first holds a numpy array of propagation constants for each k0. The second holds a
        numpy array of eigenvectors for each k0. Each numpy array that corresponds to a particular k0 is sorted by the
        propagation constants solved for that k0. Eigenvectors are stored as rows in the returned numpy arrays, not columns.
        The third numpy array holds the k0s that were solved for.
        """
        total_propagation_constants = []
        total_eigenvectors = []

        a = self.x_max - self.x_min
        # b = self.y_max - self.y_min
        # Set up the k0 values to solve for
        start_k0 = 2 / a if start_k0 == -1 else start_k0
        end_k0 = 8 / a if end_k0 == -1 else end_k0
        k0s = np.linspace(start_k0, end_k0, num_k0s)

        # Iterate over each k0 (frequency)
        for k0 in k0s:
            eigenvalues, eigenvectors = self.solve_k0(k0)
            total_propagation_constants.append(eigenvalues)
            total_eigenvectors.append(eigenvectors)

        return np.array(total_propagation_constants, dtype=object), np.array(total_eigenvectors, dtype=object), k0s

    def integrate_line(self, p1, p2, trapezoids=100):
        """
        Perform a line integral over the straight-line path from ``p1`` to ``p2``.
        :param p1: The starting point in the line integral.
        :param p2: The ending point in the line integral.
        :param trapezoids: The number of trapezoids to use in the integration. Default = 100.
        :return: The result of the integration.
        """
        # Convert to numpy arrays if not already
        p1, p2 = np.array(p1), np.array(p2)
        # The vector describing the direction of the line from ``p1`` to ``p2``.
        v = np.array((p2[0] - p1[0], p2[1] - p1[1]))
        # The normalized version of v
        norm_v = np.zeros([3])
        norm_v[0:2] = v / np.linalg.norm(v)
        # How far along the line segment we are (0 = starting point, 1 = ending point)
        t = np.linspace(0, 1, trapezoids)
        all_points = np.array([p1 + t_val * (p2 - p1) for t_val in t])
        y = np.array([np.dot(self.get_field_at(point[0], point[1]), norm_v) for point in all_points])
        return trapz(y, dx=np.linalg.norm(v)/trapezoids)

    def get_field_at(self, x, y):
        """
        Compute the electric field at the point (x, y).
        :param x: The x point to compute the field at.
        :param y: The y point to compute the field at.
        :return: A numpy array of length three containing (Ex, Ey, Ez) at the specified point.
        """
        # For skipping transverse fields and accessing the longitudinal fields
        skip_tran = len(self.remap_inner_edge_nums)

        # Keep track of whether we have found the element this point lies in.
        element_num = -1
        for k, element in enumerate(self.connectivity):
            # Check if the point is inside this element
            if element.is_inside(x, y):
                # The element this point lies in has been found, note it and stop searching
                element_num = k
                break
        # If we found an element that the point lies in, calculate the fields. If not found, Ex, Ey, Ez = 0
        if element_num != -1:
            # Get the phi values associated with each node
            phi1 = 0 if not element.nodes[0] in self.remap_inner_node_nums else self.eigenvectors[self.mode_index,
                skip_tran + self.remap_inner_node_nums[element.nodes[0]]]
            phi2 = 0 if not element.nodes[1] in self.remap_inner_node_nums else self.eigenvectors[self.mode_index,
                skip_tran + self.remap_inner_node_nums[element.nodes[1]]]
            phi3 = 0 if not element.nodes[2] in self.remap_inner_node_nums else self.eigenvectors[self.mode_index,
                skip_tran + self.remap_inner_node_nums[element.nodes[2]]]
            # Interpolate to get Ez
            ez = element.nodal_interpolate(phi1, phi2, phi3, x, y)

            # Get the phi values associated with each edge
            phi1e = 0 if not element.edges[0] in self.remap_inner_edge_nums else self.eigenvectors[self.mode_index,
                self.remap_inner_edge_nums[element.edges[0]]]
            phi2e = 0 if not element.edges[1] in self.remap_inner_edge_nums else self.eigenvectors[self.mode_index,
                self.remap_inner_edge_nums[element.edges[1]]]
            phi3e = 0 if not element.edges[2] in self.remap_inner_edge_nums else self.eigenvectors[self.mode_index,
                self.remap_inner_edge_nums[element.edges[2]]]
            # Interpolate to get Ex and Ey
            ex, ey = element.edge_interpolate(phi1e, phi2e, phi3e, x, y)
        else:
            # print(f"Electric field requested at point ({x}, {y}) lies outside the geometry")
            return np.array([0, 0, 0])
            # raise RuntimeError(f"Electric field requested at point ({x}, {y}) lies outside the geometry")

        return np.array([ex, ey, ez])

    def plot_dispersion(self, k0s, all_propagation_constants, rel_x=True):
        """
        Plot the Dispersion curves given the propagation constants. Creates a new figure in the process.
        Intended to integrate smoothly with solve() (i.e. pass the k0s and prop consts from solve() as args here).
        :param k0s: The k0 values that the propagation constants were solved at.
        :param all_propagation_constants: A numpy array of numpy arrays of propagation constants for a particular k0.
        :param rel_x: True if the x-axis of the plot (frequency) should be relative to the x-size (length) of the guide.
        :return: The generated figure (created using plt.figure()). Can usually do nothing with this.
        """
        if rel_x:
            x_label = r"$k_0 * a$"
            a = self.x_max - self.x_min
        else:
            x_label = r"$k_0$"
            # Set a = 1 (essentially do not factor in a for the x-axis)
            a = 1
        fig = plt.figure()
        plt.xlabel(x_label)
        plt.ylabel(r"$\beta / k_0$")
        for i, prop_const in enumerate(all_propagation_constants):
            plt.scatter(k0s[i] * a * np.ones(len(prop_const)), prop_const/k0s[i], color="blue", marker="o", facecolors='none')
        return fig

    def plot_fields(self, num_x_points=100, num_y_points=100):
        """
        Plot the electric fields. The Ez fields are plotted on a color plot, and the Ex and Ey fields are plotted on
        top of the color plot (on the same figure) as a vector field (using plt.quiver()).
        :param num_x_points: The number of x points to calculate the electric field at.
        :param num_y_points: The number of y points to calculate the electric field at.
        :return: The generated figure (created using plt.figure()). Can usually do nothing with this.
        """
        # ----------------------FIELD PLOTTING--------------------------
        # Find the minimum and maximum x and y values among the nodes:
        # Create a rectangular grid of points that the geometry is inscribed in
        x_points = np.linspace(self.x_min, self.x_max, num_x_points)
        y_points = np.linspace(self.y_min, self.y_max, num_y_points)
        Ez = np.zeros([num_y_points, num_x_points])
        Ex = np.zeros([num_y_points, num_x_points])
        Ey = np.zeros([num_y_points, num_x_points])

        # Iterate over the 100x100 grid of points we want to plot
        for i in range(num_y_points):
            pt_y = y_points[i]
            for j in range(num_x_points):
                pt_x = x_points[j]
                Ex[i, j], Ey[i, j], Ez[i, j] = self.get_field_at(pt_x, pt_y)

        # Fix Ez to be oriented correctly
        Ez = np.flipud(Ez)
        fig = plt.figure()
        plt.imshow(Ez, extent=[self.x_min, self.x_max, self.y_min, self.y_max], cmap="cividis")
        plt.colorbar(label="Ez")
        X, Y = np.meshgrid(x_points, y_points)
        skip = (slice(None, None, 5), slice(None, None, 5))
        plt.quiver(X[skip], Y[skip], Ex[skip], Ey[skip], color="black")
        return fig


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
