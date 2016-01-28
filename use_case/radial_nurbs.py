# mapping a 2d space into 3d via radial functions
# using the eigen.sparse_matrix

from __future__ import division
import eigen
import numpy as np


class control_point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.weight = 1

    def influence(self, p, x=2):
        r_q = (p[0] - self.x) ** 2 + (p[1] - self.y) ** 2
        if r_q > x ** 2:
            return False
        r = np.sqrt(r_q)
        return ((r - x)**4 * (5*r**2 + 4*r*x + x**2))/x**6

    @property
    def default_value(self):
        return [np.cos(self.x), np.sin(self.y), 0]

grid = np.array([[x, y] for x in np.linspace(0, 10, 50) for y in np.linspace(0, 10, 50)])
control_points = [control_point(x, y) for x in np.linspace(0, 10, 4) 
                                       for y in np.linspace(0, 10, 4)]

infl_triplets = []

for i, point in enumerate(grid):
    for j, cpc in enumerate(control_points):
        infl = cpc.influence(point)
        if infl:
            infl_triplets.append(eigen.triplet(i, j, infl))

infl_mat = eigen.sparse_matrix(len(grid), len(control_points))
infl_mat.set_from_triplets(infl_triplets)

sol = infl_mat * eigen.matrixx([cp.default_value for cp in control_points])