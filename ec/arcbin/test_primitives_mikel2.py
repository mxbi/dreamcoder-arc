import binutil

from dreamcoder.domains.arc import arcPrimitivesIC2
import numpy as np
import pytest

grid_test_case = arcPrimitivesIC2.Grid(np.array([[0, 0, 0], [0, 1, 0], [3, 0, 3], [4, 4, 5]]))
grid_test_case2 = arcPrimitivesIC2.Grid(np.array([[0, 0, 0], [0, 1, 8], [3, 0, 3], [4, 4, 5]]))
pos_test_case = (1, 5)
size_test_case = (5, 5)
count_test_case = 5
colour_test_case = 5

test_map = {
    arcPrimitivesIC2.tgrid: grid_test_case,
    arcPrimitivesIC2.tpos: pos_test_case,
    arcPrimitivesIC2.tsize: size_test_case,
    arcPrimitivesIC2.tcount: count_test_case,
    arcPrimitivesIC2.tcolour: colour_test_case,
    arcPrimitivesIC2.tlist: [grid_test_case, grid_test_case2], # assume tlist is always tlist(tgrid) 
}

@pytest.mark.parametrize("p", arcPrimitivesIC2.p.primitives.values())
def test_primitive_inputs(p):
    print(p.name)
    typesig = p.typesig
    func = p.value

    inputs = []
    print(typesig)
    for input_type in typesig[:-1]:
        # inputs.append(test_map[input_type])
        if hasattr(input_type, 'name') and input_type.name == 'list':
            input_type = arcPrimitivesIC2.tlist
        print(func, test_map[input_type])
        func = func(test_map[input_type]) # uncurry

    # out = func(*inputs)
    print(func)