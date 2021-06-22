# -*- coding: utf-8 -*-
"""
Implementation of different models to track particles

Each model should be a self contained class
"""
import textwrap

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import correlate
from scipy.spatial.distance import cdist

from nplocate import csimulate
from nplocate.utility import remove_overlap, measure_shape


def get_intensities(image, positions):
    """
    Args:
        image (np.ndarray): a N-dimensional image
        positions (np.ndarray): the position of particles
            shape (n_particles, dimension)
        win_size (int): the intensity is taken as the average of central pixel
            and nieghbour pixels within a window size of win_size

    Return:
        np.ndarray: the intensity values on different positions
    """
    coord_int = positions.astype(int).T
    return image[tuple(coord_int)]


class GaussianSphere:
    """
    Model for the confocal image of particles, every particle is
        treated as a sperical core plus gaussian blur. A graphical
        illustration is below.

    ..code-block::

                lxy
               ◀───────▶
            ▲ ┌─────────┐
            │ │    ░    │
            │ │   ░░░   │
        lz  │ │  ░░░░░  │
            │ │ ░░███░░ │ ▲
            │ │░░░███░░░│ │ r
            │ │ ░░███░░ │ ▼
            │ │  ░░░░░  │
            │ │   ░░░   │
            ▼ │    ░    │
              └─────────┘

    Attributes:
        r (float): the radius of the solid core of the model particle.
        sxy (float): the sigma of the gaussian blur along x and y axes
        sz (float): the sigma of the gaussian blur along z axis
        __i0 (float): the intensity scale of one single particle
        __lxy (int): the minimum length of the x and y side of the box
            that contains a full single particle
        __lz (int): the minimum length of the z side of the box
            that contains a full single particle
        opt_eps (dict): the epsilon of r, sxy, and sz for the non-linear
            optimisation, serve as the initial step size.
        opt_ftol (float): the tolarance for the optimisatiion, see the
            scipy.optimize.minimize for its definition.
    """
    def __init__(self, r, sxy, sz):
        self.r = r
        self.sxy = sxy
        self.sz = sz
        self.opt_eps = {'r': 0.2, 'sxy': 1e-3, 'sz': 1e-3}
        self.opt_bounds = {
            'r': (1.0, 10.0), 'sxy': (0.5, 10.0), 'sz': (0.5, 20.0),
        }
        self.opt_ftol = 1
        self.__i0, self.__lxy, self.__lz = 1, 1, 1
        self.__update()  # gives better value for i0, lxy, and lz

    @property
    def __r_box(self):
        return np.array((
            (self.__lxy - 1) / 2,
            (self.__lxy - 1) / 2,
            (self.__lz - 1) / 2,
        ), dtype=int)

    def __str__(self):
        info = """\
        Model for mono dispersed spherical particles.
        The model is a solid sphere, blurred with anisotropic Gaussian PSF.

        The radius of the solid sphere      : {r:.4f}
        The sigma values of the Gaussian PSF: ({sxy:.4f}, {sxy:.4f}, {sz:.4f})
        """.format(r=self.r, sxy=self.sxy, sz=self.sz)
        return textwrap.dedent(info)

    def __repr__(self):
        info_id = """
        [nplocate.model.GaussianSphere at {}]
        """.format({hex(id(self))})
        return self.__str__() + textwrap.dedent(info_id)

    def __setattr__(self, name, val):
        """
        Automatically update the intensty scale and box size if image
            parameter were changed
        """
        if name in ['r', 'sxy', 'sz']:
            self.__dict__[name] = val
            if 'i0' in self.__dict__.keys():
                self.__update()
        else:
            self.__dict__[name] = val

    def __update_single_box(self, intensity_threshold=1e-3):
        """
        Get the size of the box that would contain a full single particle

        Args:
            intensity_threshold (float): the particle intensity is considered
                to be 0 when the actuall value is below the threshold.
        """
        should_increase_box = True
        r_init, dr = 5, 5
        while should_increase_box:
            single = self.simulate_single(r_window=r_init)  # ca 5 diamter
            single[single < intensity_threshold] = 0
            box = ndimage.find_objects(single > 0)[0]
            reach_low = [b.start == 0 for b in box]
            reach_high = [b.stop == r_init * 2 + 1 for b in box]
            if np.any(reach_low) or np.any(reach_high):
                r_init += dr
            else:
                self.__lxy, _, self.__lz = np.array(single[box].shape)
                should_increase_box = False

    def __update_intensity_scale(self):
        """
        update the intensity scale to match the measureed intensity and
            simulated intensity, after applying gaussian blur

        To simulate a particle in the image, with intensity [I], the
            platonic particle should have the intensity of [I * scale]
        """
        particle = self.simulate_single()
        cx, cy, cz = map(lambda x : (x - 1) //2, [self.__lxy, self.__lxy, self.__lz])
        self.__i0 = 1 / particle[cx, cy, cz]

    def __update(self):
        """
        If changing the shape parameters, this method should be called
        """
        self.__update_single_box()  # the order matters
        self.__update_intensity_scale()

    def __get_shift_for_mean(self, image, positions):
        n, dim = positions.shape
        shift = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n, n, 3)
        shift = np.reshape(shift, (n ** 2, dim), order='F')
        intensities = get_intensities(image, positions)
        intensities = intensities[np.newaxis, :] * np.ones((n, 1))
        return shift, intensities.flatten()

    def simulate(self, image, positions):
        """
        Generate a simulated image

        Args:
            image (numpy.ndarray): a 3D volumetric image
            positions (numpy.ndarray): locations of particles, shape (n, 3)

        Return:
            numpy.ndarray: a simulated volumetric image, which should resemble
                the actual image.
        """
        n = positions.shape[0]
        radii = np.ones(n, dtype=np.float64) * self.r
        intensities = get_intensities(image, positions).astype(np.float64)
        intensities *= self.__i0
        sim = csimulate.simulate_spheres(
            positions, intensities, radii, *image.shape
        )
        sim = ndimage.gaussian_filter(sim, sigma=(self.sxy, self.sxy, self.sz))
        return sim

    def simulate_single(self, r_window=None):
        """
        Generate a simulated single particle, the intensity is one

        Args:
            r_window (int): the radius of the window around the particle.

        Return:
            numpy.ndarray: simulated 3D image with one particle in the centre,
                shape of (L, L, L) and L = 2 * r_window + 1
        """
        if isinstance(r_window, type(None)):
            shape = np.array([self.__lxy, self.__lxy, self.__lz]).astype(int)
        else:
            shape = np.array([r_window * 2 + 1] * 3).astype(int)
        radii = np.ones(1, dtype=np.float64) * self.r
        position = shape / 2 - 0.5
        position = position[np.newaxis, :]  # (3, ) -> (1, 3)
        model = csimulate.simulate_spheres(
            position, np.ones(1) * self.__i0, radii, *shape
        )
        model = ndimage.gaussian_filter(model, (self.sxy, self.sxy, self.sz))
        return model

    def simulate_platonic(self, r_window):
        """
        Generate a simulated single particle without gaussian blur

        Args:
            r_window (int): the radius of the window around the particle.

        Return:
            numpy.ndarray: simulated 3D image with one particle in the centre,
                shape of (L, L, L) and L = 2 * r_window + 1
        """
        shape = [r_window * 2 + 1] * 3
        radii = np.ones(1, dtype=np.float64) * self.r
        position = np.array(shape) / 2 - 0.5
        position = position[np.newaxis, :]  # (3, ) -> (1, 3)
        model = csimulate.simulate_spheres(
            position, np.ones(1), radii, *shape
        )
        return model

    def simulate_mean_shape(self, image, positions):
        shift, shift_intensities = self.__get_shift_for_mean(image, positions)
        dx, dy, dz = np.abs(shift.T)
        mask  = dx < self.__lxy
        mask *= dy < self.__lxy
        mask *= dz < self.__lz
        r_box = np.min(self.__r_box)
        radii = self.r * np.ones(shift.shape[0], dtype=np.float64)
        shape = self.__r_box * 6 + 3  # also draw neighbours
        shift += self.__r_box * 3 + 1 # put particle centre to central box
        model = csimulate.simulate_spheres(
            shift[mask], shift_intensities[mask] * self.__i0, radii[mask], *shape
        ) / positions.shape[0]
        model = ndimage.gaussian_filter(model, (self.sxy, self.sxy, self.sz))
        c = (shape - 1) // 2
        model = model[
            c[0]-r_box:c[0]+r_box+1,
            c[1]-r_box:c[1]+r_box+1,
            c[2]-r_box:c[2]+r_box+1,
        ]
        return model

    def __shape_cost(self,
        parameters, mean_shape, weight,
        shift, shift_intensities, shape
    ):
        radius, sxy, sz = parameters
        dx, dy, dz = np.abs(shift.T)
        radii = radius * np.ones(shift.shape[0], dtype=np.float64)
        model = csimulate.simulate_spheres(
            shift, shift_intensities, radii, *shape
        )
        model = ndimage.gaussian_filter(model, (sxy, sxy, sz))
        r = min(self.__r_box)
        c = (shape - 1) // 2
        model = model[
            c[0]-r:c[0]+r+1,
            c[1]-r:c[1]+r+1,
            c[2]-r:c[2]+r+1,
        ]
        return np.sum(np.abs(mean_shape - model) * weight)

    def fit_shape(self, image, positions, maxiter=10):
        measurement = measure_shape(
            image, positions, win_size=min(self.__lz, self.__lxy)
        )
        shift, shift_intensities = self.__get_shift_for_mean(image, positions)
        # calculate mask
        dx, dy, dz = np.abs(shift.T)
        mask  = dx < self.__lxy
        mask *= dy < self.__lxy
        mask *= dz < self.__lz

        # apply mask
        shift_intensities = shift_intensities[mask]
        shift = shift[mask]

        shift_intensities *= self.__i0 / positions.shape[0]
        shift += self.__r_box * 3 + 1 # put particle centre to central box
        shape = self.__r_box * 6 + 3  # plot central box and neighbours

        # calculate weight
        r = min(self.__r_box)
        dr = np.arange(-r, r + 1)
        weight = np.array(np.meshgrid(dr, dr, dr)) + 1 / dr.max()
        weight = 1 / (np.sum(weight ** 2, axis=0))

        result = minimize(
            fun=self.__shape_cost,
            x0=[self.r, self.sxy, self.sz],
            args=(measurement, weight, shift, shift_intensities, shape),
            bounds=(
                self.opt_bounds['r'], self.opt_bounds['sxy'], self.opt_bounds['sz']
            ),
            options={
                'eps': np.array((
                    self.opt_eps['r'], self.opt_eps['sxy'], self.opt_eps['sz']
                )),
                'ftol': self.opt_ftol,
                'maxiter': maxiter
            }
        )
        if result.success:
            self.r, self.sxy, self.sz = result.x
        else:
            print("shape optimisation failed")
            print(result.message)

    def plot_model(self, r_window=None):
        """
        Plot a simulated single particle

        Args:
            r_window (int): the radius of the window around the particle.
                The final ouput image will have the shape of (L, L, L) and
                L = 2 * r_window + 1
        """
        sim = self.simulate_single(r_window=r_window)
        vmin, vmax = sim.min(), sim.max()
        if isinstance(r_window, type(None)):
            r_window_x = self.__r_box[0]
            r_window_z = self.__r_box[2]
        else:
            r_window_x = r_window
            r_window_z = r_window
        im = plt.subplot(121).imshow(sim[r_window_x], vmin=vmin, vmax=vmax)
        plt.colorbar(im)
        plt.xlabel('Z')
        plt.ylabel('Y')
        im = plt.subplot(122).imshow(sim[:, :, r_window_z], vmin=vmin, vmax=vmax)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.colorbar(im)
        plt.gcf().set_size_inches(8, 2.5)
        plt.show()

    def plot_mean_shape(self, image, positions):
        win_size = min(self.__lz, self.__lxy)
        r_window = (win_size - 1) // 2
        measure = measure_shape(image, positions, win_size)
        sim = self.simulate_mean_shape(image, positions)
        diff = measure - sim
        vmax = np.abs(diff).max()

        fig = plt.figure(figsize=(9, 6))
        ax = fig.subplots(2, 3).ravel()

        ax[0].set_title('simulation')
        ax[1].set_title('measurement')
        ax[2].set_title('difference')

        ax[0].imshow(sim[:, :, r_window])
        ax[1].imshow(measure[:, :, r_window])
        ax[2].imshow(diff[:, :, r_window], cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[3].imshow(sim[r_window])
        ax[4].imshow(measure[r_window])
        ax[5].imshow(diff[r_window], cmap='bwr', vmin=-vmax, vmax=vmax)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        plt.show()

    def plot_simulate(self, image, positions, s):
        """
        Generated a simulated image

        Args:
            image (numpy.ndarray): a 3D volumetric image
            positions (numpy.ndarray): locations of particles, shape (n, 3)
            s (int): index of a slice
        """
        fig = plt.figure(figsize=(9, 6))
        ax = fig.subplots(2, 3).ravel()

        sim = self.simulate(image, positions)
        diff = image - sim
        vmax = np.abs(diff).max()


        ax[0].set_title('simulation')
        ax[1].set_title('measurement')
        ax[2].set_title('difference')

        ax[0].imshow(sim[:, :, s])
        ax[1].imshow(image[:, :, s])
        ax[2].imshow(diff[:, :, s], cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[3].imshow(sim[s])
        ax[4].imshow(image[s])
        ax[5].imshow(diff[s], cmap='bwr', vmin=-vmax, vmax=vmax)

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        plt.show()

    def plot_compare(self, image, positions, figsize=(9, 6)):
        """
        Get the differnce between input image and simulation, then calculate
            the cross-correlation between the difference and the single particle
            simulation.

        Args:
            image (numpy.ndarray): a 3D volumetric image
            positions (numpy.ndarray): locations of particles, shape (n, 3)

        Return:
            numpy.ndarray: the cross-correlation between the difference and a
                single particle. The brighter voxels indicate higher chance to
                be a particle centre.
        """
        sim = self.simulate(image, positions)
        diff = image - sim
        sz = image.shape[2] // 2
        sx = image.shape[0] // 2
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots(2, 3).ravel()
        vmax = np.abs(diff).max()
        ax[0].imshow(sim[:, :, sz])
        ax[1].imshow(image[:, :, sz])
        ax[2].imshow(diff[:, :, sz], cmap='bwr', vmin=-vmax, vmax=vmax)
        ax[3].imshow(sim[sx])
        ax[4].imshow(image[sx])
        ax[5].imshow(diff[sx], cmap='bwr', vmin=-vmax, vmax=vmax)

        ax[0].set_title('simulation')
        ax[1].set_title('measurement')
        ax[2].set_title('difference')

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        plt.show()

    def __get_diff_cc(self, image, positions, plot=True):
        """
        Get the differnce between input image and simulation, then calculate
            the cross-correlation between the difference and the single particle
            simulation.

        Args:
            image (numpy.ndarray): a 3D volumetric image
            positions (numpy.ndarray): locations of particles, shape (n, 3)
            r_window (int): the window size to draw the one particle simulation,
                if r_window==0, the value of 2 * self.r will be taken
            plot (bool): if True, a figure will be generated as a report

        Return:
            numpy.ndarray: the cross-correlation between the difference and a
                single particle. The brighter voxels indicate higher chance to
                be a particle centre.
        """
        sim = self.simulate(image, positions)
        particle = self.simulate_single()
        diff = image - sim
        diff[diff < 0] = 0
        cc = correlate(
            diff-diff.mean(), particle-particle.mean(),
            method='fft', mode='same'
        )
        cc *= diff
        if plot:
            sz = image.shape[2] // 2
            sx = image.shape[0] // 2
            fig = plt.figure(figsize=(9, 6))
            ax = fig.subplots(2, 3).ravel()
            vmax = np.abs(cc).max()
            ax[0].imshow(sim[:, :, sz])
            ax[1].imshow(image[:, :, sz])
            ax[2].imshow(cc[:, :, sz], cmap='bwr', vmin=-vmax, vmax=vmax)
            ax[3].imshow(sim[sx])
            ax[4].imshow(image[sx])
            ax[5].imshow(cc[sx], cmap='bwr', vmin=-vmax, vmax=vmax)
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
            plt.show()
        return cc

    def __positions_cost(self, pos_1d, image, weight):
        pos_3d = pos_1d.reshape(int(len(pos_1d)/3), 3)
        intensities = get_intensities(image, pos_3d) * self.__i0
        radii = np.ones(pos_3d.shape[0]) * self.r
        simulation = csimulate.simulate_spheres(
            pos_3d, intensities, radii, *image.shape
        )
        simulation = ndimage.gaussian_filter(
            simulation, sigma=(self.sxy, self.sxy, self.sz)
        )
        cost = np.sum(np.power(simulation - image, 2) * weight)
        return cost

    def __positions_jacobian(self, pos_1d, image, weight):
        pos_3d = pos_1d.reshape(int(len(pos_1d)/3), 3)
        intensities = get_intensities(image, pos_3d).astype(np.float64)
        intensities *= self.__i0
        radii = np.ones(pos_3d.shape[0]) * self.r
        simulation = csimulate.simulate_spheres(
            pos_3d, intensities, radii, *image.shape
        )
        difference = simulation - image
        jac = np.empty(len(pos_1d))
        for i, p in enumerate(pos_3d):
            # get the central box with size lxy, lxy, lz
            #   image[box_im] = a image centered to this particle
            c = p.astype(int)
            box_im = tuple([
                slice(
                    c[d] - self.__r_box[d], c[d] + self.__r_box[d] + 1
                ) for d in range(3)
            ])

            # draw the simulation box that contains these particles
            model_ = csimulate.simulate_spheres(
                (p - c + self.__r_box)[np.newaxis, :],
                np.ones(1) * self.__i0, np.ones(1) * self.r,
                *self.__r_box * 2 + 1
            )
            model_dx =\
                csimulate.simulate_spheres(
                    (p - c + self.__r_box)[np.newaxis, :] + np.array((0.5, 0, 0)),
                    np.ones(1) * self.__i0, np.ones(1) * self.r,
                    *self.__r_box * 2 + 1
                ) - \
                csimulate.simulate_spheres(
                    (p - c + self.__r_box)[np.newaxis, :] - np.array((0.5, 0, 0)),
                    np.ones(1) * self.__i0, np.ones(1) * self.r,
                    *self.__r_box * 2 + 1
                )

            model_dy =\
                csimulate.simulate_spheres(
                    (p - c + self.__r_box)[np.newaxis, :] + np.array((0, 0.5, 0)),
                    np.ones(1) * self.__i0, np.ones(1) * self.r,
                    *self.__r_box * 2 + 1
                ) - \
                csimulate.simulate_spheres(
                    (p - c + self.__r_box)[np.newaxis, :] - np.array((0, 0.5, 0)),
                    np.ones(1) * self.__i0, np.ones(1) * self.r,
                    *self.__r_box * 2 + 1
                )

            model_dz =\
                csimulate.simulate_spheres(
                    (p - c + self.__r_box)[np.newaxis, :] + np.array((0, 0, 0.5)),
                    np.ones(1) * self.__i0, np.ones(1) * self.r,
                    *self.__r_box * 2 + 1
                ) - \
                csimulate.simulate_spheres(
                    (p - c + self.__r_box)[np.newaxis, :] - np.array((0, 0, 0.5)),
                    np.ones(1) * self.__i0, np.ones(1) * self.r,
                    *self.__r_box * 2 + 1
                )

            model_ = ndimage.gaussian_filter(
                model_, sigma=(self.sxy, self.sxy, self.sz)
            )
            model_dx = ndimage.gaussian_filter(
                model_dx, sigma=(self.sxy, self.sxy, self.sz)
            )
            model_dy = ndimage.gaussian_filter(
                model_dy, sigma=(self.sxy, self.sxy, self.sz)
            )
            model_dz = ndimage.gaussian_filter(
                model_dz, sigma=(self.sxy, self.sxy, self.sz)
            )

            # calculate the intensity and derivatives
            I = image[tuple(c)]
            Idx = (
                image[tuple(c + np.array((1, 0, 0)))] -\
                image[tuple(c - np.array((1, 0, 0)))]
            ) / 2
            Idy = (
                image[tuple(c + np.array((0, 1, 0)))] -\
                image[tuple(c - np.array((0, 1, 0)))]
            ) / 2
            Idz = (
                image[tuple(c + np.array((0, 0, 1)))] -\
                image[tuple(c - np.array((0, 0, 1)))]
            ) / 2

            # calculate the Jacobian
            D = difference[box_im]
            W = weight[box_im]
            A = 2 * D * W
            jac[i * 3 + 0] = np.sum(A * (Idx * model_ + I * model_dx))
            jac[i * 3 + 1] = np.sum(A * (Idy * model_ + I * model_dy))
            jac[i * 3 + 2] = np.sum(A * (Idz * model_ + I * model_dz))

        return jac

    def fit_positions(
        self, image, positions, max_iter=10, ftol=1, eps=1, method='BFGS',
        no_overlap=False
    ):
        """
        Refine the positions of particles by minimizing the difference between
            real image and simulated image. (All variables especially for
            optimisation have the format of [var_])

        Args:
            image (np.ndarray): the image to be tracked
            positions (np.array): the positions of particles, shape (n, dim)
            max_iter (int): the maximum iter number, see scipy doc for detail
            ftol (float): see scipy doc for detail
            eps (float): see scipy doc for detail
            method (str): see scipy doc for detail
            no_overlap (bool): if true, the overlapped particles would be
                removed

        Return:
            np.array: the refined positions of particles
        """
        # create padding around the image to contain all FULL particles
        pad_ = self.__r_box
        img_ = np.zeros(np.array(image.shape) + pad_ * 2, dtype=np.float64)
        img_[
            pad_[0] : -pad_[0],
            pad_[1] : -pad_[1],
            pad_[2] : -pad_[2],
        ] += image
        pos_ = positions + pad_[np.newaxis, :]

        opt_result_ = minimize(
            fun=self.__positions_cost,
            jac=self.__positions_jacobian,
            x0=pos_.ravel(),
            method=method,
            args=(img_, np.ones(img_.shape)),
            tol=ftol,
            options={'maxiter': max_iter, 'eps': eps, 'disp': True}
        )
        pos_opt_ = np.reshape(opt_result_.x, pos_.shape)
        pos_opt_ -= pad_
        if no_overlap:
            pos_opt_ = remove_overlap(pos_opt_, image, diameter=self.r * 2)
        return pos_opt_

    def gd_positions(self, image, positions, step):
        """
        Move partile positions according to the gradient

        Args:
            image (numpy.ndarray): a 3D volumetric image
            positions (numpy.ndarray): the 3D positions of particles, shape (n, 3)
            step (float): the particles were moved by step * gradient

        Return:
            numpy.ndarray: the positions that were moved
        """
        pad_ = self.__r_box
        img_ = np.zeros(np.array(image.shape) + pad_ * 2, dtype=np.float64)
        img_[
            pad_[0] : -pad_[0],
            pad_[1] : -pad_[1],
            pad_[2] : -pad_[2],
        ] += image
        pos_ = (positions + pad_[np.newaxis, :]).flatten()
        weight_ = np.ones(img_.shape)
        #cost = self.__positions_cost(pos_, img_, weight_)
        grad = self.__positions_jacobian(pos_, img_, weight_)
        pos_ -= grad * step
        new_pos = pos_.reshape(positions.shape)
        return new_pos - pad_


    def find_extra_particles(
            self, locate_func, image, positions, diameter='auto', max_iter=10,
            threshold=0, plot=True
    ):
        """
        Recursively find extra particles in the difference between the image
            and the simulation

        Args:
            locate_func (callable): the function to locate particles,
                example: `locate_func(image) -> positions`
            image (numpy.ndarray): a 3D volumetric image
            positions (numpy.ndarray): locations of particles, shape (n, 3)
            diameter (float or str)
            threshold (float): the intensity threshold, and the particles with
            plot (bool): if true, the cross-correlation of difference image
                and the one particle simulation will be plotted

        Return:
            numpy.ndarray: positions of all the extra particles, shape (n, 3)
        """
        extra_particles = np.empty((0, 3))
        if diameter == 'auto':
            diameter = self.r * 2
        confirmed_particles = positions
        for i in range(max_iter):
            # find new particles
            if plot:
                print("iteration: ", i)
            cc = self.__get_diff_cc(
                image, confirmed_particles, plot=plot
            )
            if plot:
                print()
            new_particles = locate_func(cc)

            # check if new particles are overlapping with confirmed particles
            new_intensities = get_intensities(image, new_particles)
            mask = new_intensities > threshold
            if mask.sum() == 0:
                return extra_particles

            # check if new particles are overlapping with confirmed particles
            new_particles = new_particles[mask]
            dists = cdist(new_particles, confirmed_particles)
            not_overlap = np.min(dists, axis=1) > diameter
            if not_overlap.sum() == 0:
                return extra_particles

            extra_particles = np.concatenate(
                (new_particles[not_overlap], extra_particles), axis=0
            )
            confirmed_particles = np.concatenate(
                (positions, extra_particles), axis=0
            )
        return extra_particles
