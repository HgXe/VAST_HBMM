import csdl
import numpy as np

from csdl import Model
# from csdl_om import Simulator
from scipy.sparse import coo_array
from scipy import sparse

def compute_spars(surface_shapes):
    '''
    This is a helper function to compute the sparse matrix
    for the wake panel to body panel mapping to get the reshaped M matrix
    '''

    num_total_bd_panel = 0
    # num_bd_panel = []
    num_bd_panel_array = np.array([])
    for i in range(len(surface_shapes)):
        surface_shape = surface_shapes[i]
        num_total_bd_panel += (surface_shape[1] - 1) * (surface_shape[2] - 1)

    num_total_bd_ind = np.arange(num_total_bd_panel)
    start = 0
    for i in range(len(surface_shapes)):
        surface_shape = surface_shapes[i]
        delta = (surface_shape[1] - 1) * (surface_shape[2] - 1)
        num_bd_panel_array = np.concatenate((
            num_bd_panel_array,
            num_total_bd_ind[start:start + delta][-(surface_shape[2] - 1):],
        ))
        start += delta
    '''this only works when there is only one row of wake panel streamwise
        can be generlized by given n_wake_pts_chord (num_wake_panel streamwise) as inputs'''
    num_wake_panel = num_bd_panel_array.size

    row = np.arange(num_wake_panel)
    col = num_bd_panel_array
    data = np.ones(num_wake_panel)
    sprs = coo_array(
        (data, (row, col)),
        shape=(num_wake_panel, num_total_bd_panel),
        dtype='float32',
    )
    return sprs



class Explicit(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_nodes')  #, types = tuple)
        self.parameters.declare('sprs')  #, types = tuple)
        self.parameters.declare('num_bd_panel')  #, types = tuple)
        self.parameters.declare('num_wake_panel')  #, types = tuple)

    def define(self):
        sprs = self.parameters['sprs']
        num_nodes = self.parameters['num_nodes']
        num_bd_panel = self.parameters['num_bd_panel']
        num_wake_panel = self.parameters['num_wake_panel']

        self.add_input('M_mat',
                       shape=(num_nodes, num_bd_panel, num_wake_panel))

        self.add_output('M_reshaped',
                        shape=(num_nodes, num_bd_panel, num_bd_panel))

        num_row_rep = sprs.shape[0]
        rows = np.outer(
            np.arange(num_nodes * num_bd_panel * num_bd_panel),
            np.ones(num_row_rep),
        )

        num_col_rep = sprs.shape[1]
        cols = np.hstack([
            np.arange(num_nodes * num_bd_panel * num_wake_panel).reshape(
                -1, num_wake_panel)
        ] * num_bd_panel)
        self.declare_derivatives('M_reshaped',
                                 'M_mat',
                                 rows=rows.flatten(),
                                 cols=cols.flatten())

    def compute(self, inputs, outputs):
        sprs = self.parameters['sprs']
        num_nodes = self.parameters['num_nodes']
        num_bd_panel = self.parameters['num_bd_panel']
        num_wake_panel = self.parameters['num_wake_panel']

        outputs['M_reshaped'] = np.einsum('ijk,kl->ijl', inputs['M_mat'],
                                          sprs.todense())

    def compute_derivatives(self, inputs, derivatives):
        sprs = self.parameters['sprs']
        num_nodes = self.parameters['num_nodes']
        num_bd_panel = self.parameters['num_bd_panel']
        num_wake_panel = self.parameters['num_wake_panel']

        derivatives['M_reshaped',
                    'M_mat'] = np.tile(sprs.T.todense().flatten(),
                                       num_nodes * num_bd_panel)
