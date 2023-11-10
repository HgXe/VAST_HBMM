import csdl
import numpy as np

def local_cylindrical_transform(
        pre_transform_val, # value to transform
        local_origin, # local origin
        coordinates, # absolute coordinates
        dt, # timestep
        convert_back_to_cartesian=True # flag to convert back to cartesian
    ):
    local_radius = coordinates - local_origin
    return transformed_val