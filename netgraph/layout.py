import numpy as np

def layout_tensor(shape, xy_center=True, z_offset=0):

    assert shape[0] == 1, 'Expecting batch size to be 1'

    if len(shape) == 4:
        layout = np.array(list(np.ndindex(*shape[1:])))
    elif len(shape) == 2:
        layout = np.concatenate([
            np.arange(shape[1])[: ,np.newaxis], 
            np.zeros((shape[1],2))
        ], axis=1)
    else: 
        raise NotImplementedError()

    if xy_center:
        layout[:, 0] -= layout[:, 0].max()//2
        layout[:, 1] -= layout[:, 1].max()//2        

    layout[:, 2] += z_offset

    return layout

def layout_tensors(xs):
    layouts = []

    for x in xs:
        layout = layout_tensor(x)

        if layouts:
            layout[:, 2] += layouts[-1][:, 2].max() + 2

        layouts.append(layout)

    return np.concatenate(layouts)
