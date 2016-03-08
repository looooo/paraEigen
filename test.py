import sys

sys.path.insert(0, "build/")
from eigen import vector2, vector3, vector4, vectorx, matrixx, matrix44, sparse_matrix, triplet

import numpy as np
import unittest


class VectorTests(unittest.TestCase):
    def setUp(self):
        self.init_vector2()
        self.init_vector3()
        self.init_vector4()
        self.init_vectorx()

    def init_vector2(self):
        self.v2 = vector2()
        self.v2 = vector2(1)
        self.v2 = vector2([1, 3])
        self.v2 = vector2(self.v2)
        self.v2 = vector2(np.random.random([2]))

    def init_vector3(self):
        self.v3 = vector3()
        self.v3 = vector3(1)
        self.v3 = vector3([1, 2, 3])
        self.v3 = vector3(self.v3)
        self.v3 = vector3(np.random.random([3]))

    def init_vector4(self):
        self.v4 = vector4()
        self.v4 = vector4(1)
        self.v4 = vector4([1, 2, 3, 4])
        self.v4 = vector4(self.v3)
        self.v4 = vector4(np.random.random([4]))

    def init_vectorx(self):
        self.vx = vectorx(5)
        self.vx = vectorx(5, 1)
        self.vx = vectorx([1, 2, 3, 4, 5])
        self.vx = vectorx(self.vx)
        self.vx = vectorx(np.random.random([100]))

    def test_len(self):
        self.assertEqual(len(self.v2), 2)
        self.assertEqual(len(self.v3), 3)
        self.assertEqual(len(self.v4), 4)
        len(self.vx)

    def test_norm(self):
        self.assertAlmostEqual(self.v2.norm(), np.sqrt(self.v2.dot(self.v2)))
        self.assertAlmostEqual(self.v3.norm(), np.sqrt(self.v3.dot(self.v3)))
        self.assertAlmostEqual(self.v4.norm(), np.sqrt(self.v4.dot(self.v4)))
        self.assertAlmostEqual(self.vx.norm(), np.sqrt(self.vx.dot(self.vx)))

        self.assertAlmostEqual(abs(self.v2), np.sqrt(self.v2.dot(self.v2)))
        self.assertAlmostEqual(abs(self.v3), np.sqrt(self.v3.dot(self.v3)))
        self.assertAlmostEqual(abs(self.v4), np.sqrt(self.v4.dot(self.v4)))
        self.assertAlmostEqual(abs(self.vx), np.sqrt(self.vx.dot(self.vx)))
    
    def test_add(self):
        self.v2 + self.v2
        self.v3 + self.v3
        self.v4 + self.v4
        self.vx + self.vx
        
    def test_sub(self):
        self.v2 - self.v2
        self.v3 - self.v3
        self.v4 - self.v4
        self.vx - self.vx
        
    def test_mul(self):
        self.v2 * 2
        self.v3 * 2
        self.v4 * 2
        self.vx * 2
    
    def test_div(self):
        self.v2 / 2
        self.v3 / 2
        self.v4 / 2
        self.vx / 2
        
    def test_assign_add(self):
        self.v2 += self.v2
        self.v3 += self.v3
        self.v4 += self.v4
        self.vx += self.vx
        
    def test_assign_sub(self):
        self.v2 -= self.v2
        self.v3 -= self.v3
        self.v4 -= self.v4
        self.vx -= self.vx
        
    def test_assign_mul(self):
        self.v2 *= 2
        self.v3 *= 2
        self.v4 *= 2
        self.vx *= 2
    
    def test_assign_div(self):
        self.v2 /= 2
        self.v3 /= 2
        self.v4 /= 2
        self.vx /= 2


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.matrixx = matrixx(100, 100, 1)
        self.matrixx = matrixx([[0.,1.,1],[5.,6.,1]])
        self.matrix4 = matrix44([[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]])
        self.matrix4 = matrix44(np.array([[0,0,0,0],[1,1,1,1],[2,2,2,2],[3,3,3,3]]))
        self.matrix4 = matrix44(10)
    
    def test_init(self):
        self.matrixx.to_list()


class SparseTest(unittest.TestCase):
    def test_triplet(self):
        self.triplet = triplet(2, 2, 10.)
        self.sparse = sparse_matrix(3, 10)
        self.sparse.set_from_triplets([self.triplet])
        mat = matrixx(2, 3, 1.)

if __name__ == '__main__':
    unittest.main(verbosity=2)
