import cython
cimport numpy as c_np
import numpy as np
import math
from libc.math cimport exp, erf, sqrt, pi, log, floor


@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef void platonic_3d(c_np.float_t[:, :, :] radius_map, float radius):
    cdef int x_max = radius_map.shape[0]
    cdef int y_max = radius_map.shape[1]
    cdef int z_max = radius_map.shape[2]
    cdef double term1, term2, term3
    cdef double alpha = 0.2765
    #cdef c_np.ndarray[c_np.float_t, ndim=3] platonic = np.zeros((x_max, y_max, z_max), dtype=np.float_)

    for x in range(x_max):
        for y in range(y_max):
            for z in range(z_max):
                term1 = erf((radius - radius_map[x, y, z]) / (alpha * sqrt(2))) + erf(
                    (radius + radius_map[x, y, z]) / (alpha * sqrt(2)))
                term2 = sqrt(0.5 / pi) * (alpha / (radius_map[x, y, z] + 1e-10))
                term3 = exp(-0.5 * (radius_map[x, y, z] - radius) ** 2 / alpha ** 2) - exp(
                    -0.5 * (radius_map[x, y, z] + radius) ** 2 / alpha ** 2)
                radius_map[x, y, z] = 0.5 * term1 - term2 * term3

def simulate_spheres(
    c_np.ndarray[c_np.float64_t, ndim=2] positions,
    c_np.ndarray[c_np.float64_t, ndim=1] intensities,
    c_np.ndarray[c_np.float64_t, ndim=1] radii,
    int x_size, int y_size, int z_size
):
    """
    Simulate a 3D volumetirc image of hard spheres, with different
        radii and intensities

    Args:
        positions (numpy.ndarray): particle locations, shape (n, 3)
        intensities (numpy.ndarray): the brightness of particle, shape (n, )
        radii (numpmy.ndarray): the radius of particle, shape (n, 1)
        x_size (int): the length of the image box in the x direction
        y_size (int): the length of the image box in the y direction
        z_size (int): the length of the image box in the z direction

    Return:
        numpy.ndarray: a 3D volumetric image, shape (x_size, y_size, z_size)
    """
    cdef c_np.float64_t[:, :] pos_view = positions
    cdef c_np.float64_t[:] rad_view = radii
    cdef c_np.float64_t[:] int_view = intensities

    cdef Py_ssize_t dimension = 3
    cdef Py_ssize_t particle_num = positions.shape[0]
    cdef int num, dim

    cdef c_np.float64_t pos_1d, radius
    cdef c_np.float64_t box_size[3]
    cdef int sub_image_box[3][2]
    cdef int sub_image_lengths[3]

    cdef int image_shape[3]
    image_shape[:] = [x_size, y_size, z_size]

    cdef Py_ssize_t lower_bound = 0
    cdef Py_ssize_t upper_bound = 0

    cdef c_np.ndarray[c_np.float_t, ndim=3] simulation =\
            np.zeros(image_shape, dtype=np.float_)
    cdef c_np.ndarray[c_np.float_t, ndim=2] sub_image_indices =\
            np.empty((3, np.max(image_shape)), dtype=np.float_)
    cdef c_np.ndarray[c_np.float64_t, ndim=4] mesh
    cdef c_np.ndarray[c_np.float64_t, ndim=3] radius_map

    for num in range(particle_num):  # for each particle
        radius = rad_view[num]
        box_size[:] = [radius+1, radius+1, radius+1]
        sub_image_indices[:, :] = 0

        for dim in range(dimension):
            pos_1d = pos_view[num, dim]
            lower_bound = max((
                0,
                int(math.floor(pos_1d - box_size[dim]))
            ))
            upper_bound = min((
                image_shape[dim] - 1,
                int(math.ceil(pos_1d + box_size[dim]))
            ))
            sub_image_box[dim][0] = lower_bound
            sub_image_box[dim][1] = upper_bound + 1
            sub_image_lengths[dim] = upper_bound - lower_bound + 1
            try:
                sub_image_indices[dim, :sub_image_lengths[dim]] = np.linspace(
                    lower_bound - pos_1d,
                    upper_bound - pos_1d,
                    upper_bound - lower_bound + 1,
                    endpoint=True
                )
            except ValueError:
                print(f'lower boundary: {lower_bound}, upper boundary: {upper_bound}')
                print(f'image shape: {image_shape}, dimension: {dim}, particle: {num}, box_size: {box_size}')
                print(f'radius: {radius}')
                print(f'particle_position: {[pos_view[num, d] for d in range(dimension)]}')
                raise ValueError()


        mesh = np.array(np.meshgrid(
            sub_image_indices[0, :sub_image_lengths[0]],
            sub_image_indices[1, :sub_image_lengths[1]],
            sub_image_indices[2, :sub_image_lengths[2]],
            indexing='ij'
        ))

        radius_map = np.sqrt(np.sum(mesh ** 2, axis=0))
        platonic_3d(radius_map, radius)

        simulation[
            sub_image_box[0][0]: sub_image_box[0][1],
            sub_image_box[1][0]: sub_image_box[1][1],
            sub_image_box[2][0]: sub_image_box[2][1]
        ] += radius_map * int_view[num]

    return simulation
