from iwaveguide.waveguide import Waveguide, plot_rect_waveguide_mode
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Turn on interactive mode for plotting. Causes plots to show without plt.show() and allows for debugging while
    # the plots are visible and usable.
    plt.ion()
    # k0 = 1 is ~47.7 MHz
    # connectivity, all_nodes, all_edges, boundary_node_numbers, boundary_edge_numbers, remap_inner_node_nums, remap_inner_edge_nums, all_edges_map = load_mesh("rectangular_waveguide.inp", 1, [1])

    # Load a homogeneous rectangular waveguide with epsilon_r = 1
    # waveguide = Waveguide("rect_mesh_two_epsilons_coarse.inp", ["EB1", "EB2"], "EB3")
    waveguide = Waveguide("rect_mesh_one_epsilon_finer_20220615.inp", ["EB1"], "EB2")
    # waveguide = Waveguide("z_plane_mesh.inp", ["EB1"], "EB2")
    # Solve for the propagation constants and eigenvectors.
    betas, all_eigenvectors, k0s = waveguide.solve()
    # Plot the dispersion curve using the results.
    waveguide.plot_dispersion(k0s, betas)
    # Create the figure for plotting. Not needed if figure already created by Waveguide#plot_dispersion().
    # plt.figure()
    # plt.xlabel(r"$k_0 * a$")
    # plt.ylabel(r"$\beta / k_0$")
    # Analytical TE/TM Modes (the dispersion curves will look the same)
    for i in range(3):
        for j in range(3):
            if i == j == 0:
                continue
            plot_rect_waveguide_mode(i, j)

    # Plot the fields of the waveguide. all_eigenvectors[-1] gives a set of eigenvectors for the end k0.
    # Indexing that set by [-1] gives the mode with the highest propagation constant for that k0.
    # This is usually but not always the first propagating mode. It is always the highest beta / k0 value at the
    # particular k0 value on the dispersion graph (see Waveguide#plot_dispersion()).
    waveguide.set_mode_index(0)
    # waveguide.plot_fields(all_eigenvectors[-1][-1])
    waveguide.plot_fields()
    print(f"Plot shown for beta/k0 = {betas[-1]/k0s[-1]}")
    # Do not need the below statement when plt.ion() is used
    # plt.show()
