import numpy as np
from numba import njit
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.spatial.distance import pdist, cdist, squareform
try:
    from .cutility import join_pairs as join_pairs_cpp
    CPP_ENABLED = True
except ImportError:
    CPP_ENABLED = False


def should_join(p1, p2): return 0 in pdist(np.concatenate((p1, p2))[:, None])


def join_pairs_py(pairs, copy=True):
    """
    Args:
        pairs (list): a list of tuples
        copy (bool): very difficult
    Example:
        >>> pairs = [(2, 3), (3, 5), (2, 6), (8, 9), (9, 10)]
        >>> join_pairs(pairs)
        [(2, 3, 5, 6), (8, 9, 10)]
    """
    pairs_joined = pairs[:] if copy else pairs
    for p1 in pairs_joined:
        for p2 in pairs_joined:
            if (p1 is not p2) and should_join(p1, p2):
                pairs_joined.append(tuple(set(list(p1) + list(p2))))
                pairs_joined.remove(p1)
                pairs_joined.remove(p2)
                pairs_joined = join_pairs(pairs_joined, copy=False)
                return pairs_joined  # join once per recursion
    return pairs_joined

if CPP_ENABLED:
    join_pairs = join_pairs_cpp
else:
    join_pairs = join_pairs_py


def is_inside(position, radius, boundary):
    result = True
    for dim in range(len(boundary)):
        result *= (position[dim] - np.ceil(radius) > 0)
        result *= (position[dim] + np.ceil(radius) < boundary[dim])
    return result


def get_sub_image_box(position, radius, image_shape=None):
    for dim in range((len(position))):
        p = int(position[dim])
        r = int(np.ceil(radius))
        lower_boundary = p - r
        upper_boundary = p + r + 1
        if not isinstance(image_shape, type(None)):
            lower_boundary = max(lower_boundary, 0)
            upper_boundary = min(upper_boundary, image_shape[dim])
        yield slice(lower_boundary, upper_boundary, None)


def get_sub_images_3d(image, centres, max_radius):
    int_maps = []
    for centre in centres:
        if not is_inside(centre, max_radius, image.shape):
            continue
        int_map = np.zeros((
            int(2 * np.ceil(max_radius) + 1),
            int(2 * np.ceil(max_radius) + 1),
            int(2 * np.ceil(max_radius) + 1)
        ))
        sub_image_box = list(get_sub_image_box(centre, max_radius, image.shape))
        int_map += image[tuple(sub_image_box)]
        int_maps.append(int_map)
    return int_maps


def see_slice(image, positions, s, radius, axis=2, sizes=(10, 8)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    to_show = np.moveaxis(image, axis, 0)
    plt.imshow(to_show[s].T)
    if axis == -1:
        axis = 2
    shown_axes = np.array([i for i in range(3) if i != axis])
    for p in positions:
        x, y = p[shown_axes]
        z = p[axis]
        dz = abs(z - s)
        if z > s:
            color='k'
        else:
            color='w'
        if dz < radius:
            r_slice = np.sqrt(radius**2 - dz**2)
            circle = plt.Circle([x, y], radius=r_slice, color=color, fill=None, lw=2)
            ax.add_patch(circle)
    fig.set_size_inches(sizes[0], sizes[1])
    plt.axis('off')
    plt.show()


def see_particle(position, image, radius):
    shape = np.array(image.shape)
    shape_canvas = shape + 2 * radius
    canvas = np.zeros(shape_canvas, dtype=image.dtype)
    canvas[
        radius : shape[0] + radius,
        radius : shape[1] + radius,
        radius : shape[2] + radius,
    ] = image
    ox, oy, oz = position.astype(int) + radius
    plt.subplot(121).imshow(
        canvas[ox-radius: ox+radius+1, oy-radius:oy+radius+1, oz].T
    )
    plt.scatter(radius, radius, marker='+', color='tomato', s=128)
    plt.ylim(0, 2*radius+1)
    plt.title("XY")
    plt.axis('off')
    plt.subplot(122).imshow(
        canvas[ox, oy-radius: oy+radius+1, oz-radius:oz+radius+1]
    )
    plt.scatter(radius, radius, marker='+', color='tomato', s=128)
    plt.ylim(0, 2*radius+1)
    plt.title("ZY")
    plt.axis('off')
    plt.show()


def get_gr(positions, cutoff, bins, minimum_gas_number=1e4):
    bins = np.linspace(0, cutoff, bins)
    drs = bins[1:] - bins[:-1]
    distances = pdist(positions).ravel()
    if positions.shape[0] < minimum_gas_number:
        rg_hists = []
        for i in range(int(minimum_gas_number) // positions.shape[0] + 2):
            random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
            rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]
            rg_hists.append(rg_hist)
        rg_hist = np.mean(rg_hists, 0)
    else:
        random_gas = np.random.random(positions.shape) * np.array([positions.max(axis=0)])
        rg_hist = np.histogram(pdist(random_gas), bins=bins)[0]
    hist = np.histogram(distances, bins=bins)[0]
    hist = hist / rg_hist
    hist[np.isnan(hist)] = 0
    centres = (bins[1:] + bins[:-1]) / 2
    return centres, hist


def cost(binary, centres, model, image, simulation, radii, threshold=0):
    simulation *= 0
    for c in centres[binary]:
        simulation[
            c[0] : c[0] + 2 * radii[0] + 1,
            c[1] : c[1] + 2 * radii[1] + 1,
            c[2] : c[2] + 2 * radii[2] + 1,
        ] += model * max(image[tuple(c)], threshold)
    sim = simulation[radii[0]:-radii[0], radii[1]:-radii[1], radii[2]:-radii[2]]
    return np.sum(np.abs(image - sim ))


def simulate(centres, model, image):
    r = ((np.array(model.shape) - 1) / 2).astype(int)
    canvas = np.zeros(np.array(image.shape) + 2 * r, dtype=model.dtype)
    radii = np.array((np.array(model.shape) - 1) / 2, dtype=int)
    box = tuple([slice(radii[d], - radii[d]) for d in range(3)])
    for c in centres.astype(int):
        canvas[
            c[0] : c[0] + 2 * radii[0] + 1,
            c[1] : c[1] + 2 * radii[1] + 1,
            c[2] : c[2] + 2 * radii[2] + 1,
        ] += model * image[tuple(c)]
    return canvas[box]


def gaussian_model_2d(mesh, intensity, sigma_1, sigma_2, offset):
    dim_1, dim_2 = mesh
    c0 = intensity
    c1 = -1 / 2
    dist_2d = offset + c0 * np.exp(c1 * (dim_1 ** 2 / sigma_1 ** 2 + dim_2 ** 2 / sigma_2 ** 2))
    return dist_2d.ravel()


def fit_shape(shape):
    mesh = np.indices(shape.shape) - np.array([[shape.shape]]).T / 2.0
    sigma = -mesh[0]**2 - mesh[1]**2
    initial_guess = float(shape.max()), 1.0, 1.0, float(shape.min())
    p_opt, p_cov = curve_fit(gaussian_model_2d, mesh, shape.ravel(), initial_guess, sigma=sigma.ravel())
    return p_opt


def fix_intensity(image):
    """
    Fix the intensity gradient in a confocal image, order: axis 0, 1, 2

    Args:
        image (numpy.ndarray): a 3D confocal image

    Return:
        (numpy.ndarray): a new image whose averaged intensity profile
            on any axis is uniform, the dtype is float, value is 0 ~ 255
    """
    fix_x = image / (np.reshape(image.mean(2).mean(1), (image.shape[0], 1, 1)) + 0)
    fix_y = fix_x / (np.reshape(fix_x.mean(2).mean(0), (1, image.shape[1], 1)) + 0)
    fix_z = fix_y / (np.reshape(fix_y.mean(0).mean(0), (1, 1, image.shape[2])) + 0)
    return (fix_z - fix_z.min()) / fix_z.max() * 255


def get_model(image, centres, radius, project_axis=0, want_measure=False):
    """
    Calculate a 2D gaussian model from positions

        1. measure the average image small inside a small box located
            at different centres
        2. get a 2D projection of the average image
        3. fit the 2D image with a 2D gaussian function with background
        4. Return a model shape just composed of a gaussian shape

    Args:
        image (numpy.ndarray): a 3D confocal image
        centres (numpy.ndarray): a collection of particle centres, shape (n, 3)
        radius (float): the radius of the particle / the size of the small box
        project_axis (int): the axis along wichi the 3D average shape will be projected
    """
    shape = np.mean(get_sub_images_3d(image, centres, radius), axis=0)
    res = np.abs(fit_shape(shape.mean(project_axis)))
    model = np.zeros(np.array(shape.shape))
    model[radius, radius, radius] = 1
    model = ndimage.gaussian_filter(model, (res[1], res[1], res[2]))
    model = model / model.max()
    if want_measure:
        return model, shape
    else:
        return model


def get_canvases(image, radii, dtype=np.float64):
    """
    Get two canvases where the image is padded with extra space.
        one is for the image, another is for the simulated image.
    """
    canvas = np.zeros(np.array(image.shape) + 2 * radii, dtype=dtype)
    canvas_img = canvas.copy()
    canvas_img[radii[0]:-radii[0], radii[1]:-radii[1], radii[2]:-radii[2]] += image
    return canvas_img, canvas


def get_model_derivitives(model):
    """
    Get the numerical spatial derivative of the model in different dimensions
    """
    model_der = []
    for d in range(model.ndim):
        shift = np.zeros(model.ndim)
        shift[d] = 0.5
        model_der.append(ndimage.shift(model, shift) - ndimage.shift(model, -shift))
    return model_der


def r_cost(r_1d, model, model_der, image, simulation, radii):
    centres = r_1d.reshape(int(len(r_1d)/3), 3).astype(int)
    simulation *= 0
    box = tuple([slice(radii[d], -radii[d]) for d in range(3)])
    for c in centres:
        simulation[
            c[0] : c[0] + 2 * radii[0] + 1,
            c[1] : c[1] + 2 * radii[1] + 1,
            c[2] : c[2] + 2 * radii[2] + 1,
        ] += model * image[tuple(c + radii)]
    return np.sum(np.power(image[box] - simulation[box], 2))


def r_jac(r_1d, model, model_der, image, simulation, radii):
    centres = r_1d.reshape(int(len(r_1d)/3), 3).astype(int)
    jac = np.empty(len(r_1d))
    for i, c in enumerate(centres):
        box = tuple([ slice(c[d], c[d] + 2*radii[d] + 1) for d in range(3) ])
        I = image[tuple(c + radii)]
        diff = (model - image[box])
        Idx = (image[tuple(c + radii + np.array((1, 0, 0)))] - image[tuple(c + radii - np.array((1, 0, 0)))]) / 2
        Idy = (image[tuple(c + radii + np.array((0, 1, 0)))] - image[tuple(c + radii - np.array((0, 1, 0)))]) / 2
        Idz = (image[tuple(c + radii + np.array((0, 0, 1)))] - image[tuple(c + radii - np.array((0, 0, 1)))]) / 2
        Ider = (Idx, Idy, Idz)
        for dim in range(3):
            jac[i*3 + dim] = np.sum(2 * diff * (model_der[dim] * I + Ider[dim] * model))
    return jac


def get_position_bounds(positions, image):
    """
    Get the boundary that maks all particles stay within the image
        for :obj:`scipy.optimize.minimize`

    Args:
        positions (np.ndarray): particle locations, shape (n, dim)
        image (np.ndarray): the image to be tracked as a ND numpy array
    Return:
        np.ndarray: ((0, xlim), (0, ylim), (0, zlim), ....)
    """
    n, dim = positions.shape
    image_size = np.array(image.shape)[np.newaxis, :]  # (1, ndim)
    lower_lim = np.zeros(n * dim)
    upper_lim = np.repeat(image_size - 1, n, axis=0).ravel()
    return np.array((lower_lim, upper_lim)).T


def remove_overlap(xyz, image, diameter):
    dist_mat = squareform(pdist(xyz))
    adj_mat = np.triu(dist_mat < diameter, k=1)
    pairs = np.array(np.nonzero(adj_mat)).T  # (n ,2)
    pairs = pairs.tolist()
    pairs = join_pairs(pairs)
    to_delete = []
    for pair in pairs:
        to_remain = np.argmax([image[tuple(xyz[p].astype(int))] for p in pair])
        to_delete += [p for i, p in enumerate(pair) if i != to_remain]
    return np.delete(xyz, to_delete, axis=0)


def refine(positions, image, r_model, diameter):
    """
    Refine the positions of particles by minimizing the difference between
        real image and simulated image

    Args:
        positions (np.array): the positions of particles, shape (n, dim)
        image (np.ndarray): the image to be tracked
        r_model (int): the radius of the model, size of the model
            is (2 * r_model + 1)
        diameter (float): the diameter between particles, the added particles
            who overlap with existing particles will be removed

    Return:
        np.array: the refined positions of particles
    """
    model = get_model(image, positions, r_model)
    radii = ((np.array(model.shape) - 1) / 2).astype(int)
    canvas_img, canvas = get_canvases(image, radii)
    model_der = get_model_derivitives(model)
    bounds = get_position_bounds(positions, image)
    res = minimize(
        fun=r_cost,
        jac=r_jac,
        x0=positions.ravel(),
        method='L-BFGS-B',
        args=(model, model_der, canvas_img, canvas, radii),
        bounds=bounds,
        tol=1,
        options={'maxiter': 30}
    )
    opt = np.reshape(res.x, positions.shape)
    opt = remove_overlap(opt, image, diameter)
    return opt


def add(positions, image, r_model, diameter, locate_func, threshold=0):
    """
    find extra particles from the difference between image and simulate dimage

    Args:
        positions (np.array): the positions of particles, shape (n, dim)
        image (np.ndarray): the image to be tracked
        r_model (int): the radius of the model, size of the model
            is (2 * r_model + 1)
        diameter (float): the diameter between particles, the added particles
            who overlap with existing particles will be removed
        locate_func (callable): the function that takes the image and
            output particle positions
        threshold:

    Return:
        np.array: the positions of particles where extra particle were added
    """
    model = get_model(image, positions, r_model)
    radii = ((np.array(model.shape) - 1) / 2).astype(int)
    sim = simulate(positions, model, image)
    diff = image - sim
    diff -= diff.min()
    new_pos = locate_func(diff)
    intensities = image[tuple(new_pos.astype(int).T)]
    mask = intensities > threshold
    if mask.sum() == 0:
        return positions
    else:
        new_pos = new_pos[mask]
    dists = cdist(new_pos, positions)
    not_overlap = np.min(dists, axis=1) > diameter
    positions = np.concatenate((positions, new_pos[not_overlap]), axis=0)
    return positions


def save_xyz(filename, frames):
    """
    Append many frames to an xyz file

    Args:
        filename (str): the name of the xyz file
        frames (list of np.ndarray): positiosn of particles in different frames
            shape of one frame: (particle_num, dim)
    """
    if '.xyz' == filename[-4:]:
        fname = filename
    else:
        fname = filename + '.xyz'
    f = open(fname, 'w')
    f.close()

    for i, frame in enumerate(frames):
        num, dim = frame.shape
        with open(fname, 'a') as f:
            np.savetxt(
                f, frame,
                delimiter='\t',
                fmt=['A\t%.8e'] + ['%.8e' for i in range(dim - 1)],
                comments='',
                header='%s\nframe %s' % (num, i)
            )


@njit
def spherical_ft(f, k, r, dr):
    ft = np.zeros(len(k))
    for i in range(len(k)):
        ft[i] = 4. * np.pi * np.sum(
            r * np.sin(k[i] * r) * f * dr
        ) / k[i]
    return ft


@njit
def inverse_spherical_ft(ff, k, r, dk):
    inft = np.zeros(len(r))
    for i in range(len(r)):
        inft[i] = np.sum(
            k * np.sin(k * r[i]) * ff * dk
        ) / r[i] / (2 * np.pi ** 2)
    return inft


def get_gr_pyhs(eta, r_max=5.0, points=1000):
    """
    Calculate the radial distribution function
        for Hard Spheres using the Percus-Yevick Approximation

    In the calculation the length is rescaled by the particle diameter 

    Args:
        eta (float): the volume fraction of the system
        r_max (float): the maximum value of the radius
        points (int): the number of points in the gr figure

    Example:
        >>> r, gr = get_gr_pyhs(0.5)
    """
    r = np.linspace(r_max / points, r_max, points)
    dk = 1 / r[-1]
    dr = r[1] - r[0]
    k = np.arange(1, points + 1) * dk
    rho = 6 / np.pi * eta

    # Percus-Yevick
    c0 = -(1 + 2 * eta)**2 / (1 - eta)**4
    c1 = 6 * eta * (1 + 0.5 * eta)**2 / (1 - eta) ** 4
    c2 = eta * c0 * 0.5
    c_direct = c0 + c1 * r + c2 * r**3
    c_direct[r > 1] = 0

    # Ornstein-Zernike Equation
    c_direct_ft = spherical_ft(c_direct, k, r, dr)
    h_ft = c_direct_ft / (1 - rho * c_direct_ft)
    h = inverse_spherical_ft(h_ft, k, r, dk)

    gr = h + 1
    gr[r < 1] = 0
    return r, gr

