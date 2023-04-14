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

    # Now we check the output type
    # Because we're mapping between type-systems, this is a bit hacky
    # But mainly just a heuristic

    output_type = type(func)
    # out = func(*inputs)
    if typesig[-1] in [arcPrimitivesIC2.tpos, arcPrimitivesIC2.tsize]:
        assert isinstance(func, tuple), f"Output type {output_type} does not match expected type {typesig[-1]}"
    elif typesig[-1] in [arcPrimitivesIC2.tcount, arcPrimitivesIC2.tcolour]:
        assert isinstance(func, (int, np.int64)), f"Output type {output_type} does not match expected type {typesig[-1]}"
    elif typesig[-1].name == 'list':
        assert isinstance(func, list), f"Output type {output_type} does not match expected type {typesig[-1]}"
        for item in func:
            assert isinstance(item, arcPrimitivesIC2.Grid), f"Output type {output_type} does not match expected type {typesig[-1]}"
    else:
        print(typesig[-1], type(typesig[-1]))
        assert arcPrimitivesIC2.typemap[output_type] == typesig[-1], f"Output type {output_type} does not match expected type {typesig[-1]}"