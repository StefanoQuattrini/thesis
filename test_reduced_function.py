from pyadjoint import *
from pyadjoint.reduced_functional_numpy import ReducedFunctionNumPy
import numpy as np
def test_reduced_function():
    x_1 = AdjFloat(3.0)
    x_2 = AdjFloat(5.0)
    x_3 = AdjFloat(7.0)
    y_1 = x_1*x_2
    y_2 = x_3*x_2
    y_3 = x_1*x_3
    target = (2, 6, 3)
    z_1 = target[0] - y_1
    z_2 = target[1] - y_2
    z_3 = target[2] - y_3
    rf = ReducedFunction([z_1, z_2, z_3], [Control(x_1), Control(x_2), Control(x_3)])
    correct = [-4.0, -6.0, -5.0]
    new_x = [2.0, 3.0, 4.0]
    assert rf(new_x) == correct
    rf_np = ReducedFunctionNumPy(rf)
    new_x_np = np.array(new_x)
    assert list(rf_np(new_x_np)) == correct
if __name__ == '__main__':
    test_reduced_function()