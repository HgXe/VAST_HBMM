from csdl import Model
import csdl
import numpy as np


from VAST.utils.custom_explicit_mat_sprsmat import Explicit, compute_spars

from VAST.core.submodels.aerodynamic_submodels.kinematic_velocity_comp import KinematicVelocityComp
from VAST.core.submodels.aerodynamic_submodels.assemble_aic import AssembleAic
from VAST.core.submodels.aerodynamic_submodels.compute_normal_comp import ComputeNormal
from VAST.core.submodels.aerodynamic_submodels.projection_comp import Projection

class RHS(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.
    A \gamma_b + b + M \gamma_w = 0
    parameters
    ----------
    Returns
    -------
    rhs[num_nodes, system_size] : csdl array
    M[num_nodes, system_size, system_size] : csdl array

    """
    def initialize(self):
        self.parameters.declare('n_wake_pts_chord', default=5)
        self.parameters.declare('method',
                                values=['fw_euler', 'bk_euler'],
                                default='bk_euler')
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('bd_vortex_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('problem_type', default='fixed_wake')
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('sub',default=False)
        self.parameters.declare('sub_eval_list',default=None)
        self.parameters.declare('sub_induced_list',default=None)

    def define(self):
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        delta_t = self.parameters['delta_t']
        bd_vortex_shapes = self.parameters['bd_vortex_shapes']
        surface_names = self.parameters['surface_names']
        num_nodes = bd_vortex_shapes[0][0]

        problem_type = self.parameters['problem_type']

        sub = self.parameters['sub']
        sub_eval_list = self.parameters['sub_eval_list']
        sub_induced_list = self.parameters['sub_induced_list']

        bd_vtx_coords_names = [x + '_bd_vtx_coords' for x in surface_names]
        bd_vtx_normal_names = [x + '_bd_vtx_normals' for x in surface_names]
        coll_pts_coords_names = [x + '_coll_pts_coords' for x in surface_names]
        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        bd_coll_pts_shapes = [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in bd_vortex_shapes
        ]
        bd_normal_shape = bd_coll_pts_shapes
        wake_vortex_pts_shapes = [
            tuple((item[0], n_wake_pts_chord, item[2], 3))
            for item in bd_vortex_shapes
        ]


        '''1. project the kinematic velocity on to the bd_vertices'''
        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        m = KinematicVelocityComp(
            surface_names=surface_names,
            surface_shapes=bd_vortex_shapes,  # (2*3,3)
        )
        self.add(m, name='KinematicVelocityComp')

        # note changed here from bd_vtx_coords_names tp surface_names

        m = ComputeNormal(
            vortex_coords_names=bd_vtx_coords_names,
            normals_names=bd_vtx_normal_names,
            vortex_coords_shapes=bd_vortex_shapes,
        )
        self.add(m, name='ComputeNormal')  # shape=(2,3,3)

        kinematic_vel_names = [
            x + '_kinematic_vel' for x in self.parameters['surface_names']
        ]
        kinematic_vel_shapes = [
            tuple((item[0], item[1] * item[2], item[3]))
            for item in bd_coll_pts_shapes
        ]

        bd_vtx_normals = [x + '_bd_vtx_normals' for x in surface_names]

        m = Projection(
            input_vel_names=kinematic_vel_names,
            normal_names=bd_vtx_normal_names,
            output_vel_name='b',  # this is b
            input_vel_shapes=kinematic_vel_shapes,  # rotatonal_vel_shapes
            normal_shapes=bd_coll_pts_shapes,
        )
        self.add(m, name='Projection_k_vel')
        '''2. compute M (bk_euler) or M\gamma_w (fw_euler)'''
        if problem_type=='fixed_wake':
            m = AssembleAic(
                bd_coll_pts_names=coll_pts_coords_names,
                wake_vortex_pts_names=wake_coords_names,
                bd_coll_pts_shapes=bd_coll_pts_shapes,
                wake_vortex_pts_shapes=wake_vortex_pts_shapes,
                full_aic_name='aic_M',
                delta_t=delta_t,  # one line of wake vortex for fix wake,
                symmetry=self.parameters['symmetry'],
                sub = sub,
                sub_eval_list = sub_eval_list,
                sub_induced_list = sub_induced_list
            )

        elif problem_type=='prescribed_wake':
            TE_wake_coords_names = [x + '_TE_wake_coords' for x in surface_names]
            TE_wake_vortex_pts_shapes =  [
                tuple((item[0], n_wake_pts_chord+1, item[2], 3))
                for item in bd_vortex_shapes
            ]
            for i in range(len(surface_names)):
                bd_vortex_coords = self.declare_variable(bd_vtx_coords_names[i],shape=bd_vortex_shapes[i])
                wake_coords = self.declare_variable(wake_coords_names[i],shape=wake_vortex_pts_shapes[i])
                print(wake_coords_names)
                # self.print_var(wake_coords)
                TE_wake_coords=self.create_output(TE_wake_coords_names[i],shape=TE_wake_vortex_pts_shapes[i])
                TE_wake_coords[:,0,:,:] = bd_vortex_coords[:,bd_vortex_shapes[i][1]-1,:,:]
                TE_wake_coords[:,1:,:,:] = wake_coords
            m = AssembleAic(
                bd_coll_pts_names=coll_pts_coords_names,
                wake_vortex_pts_names=TE_wake_coords_names,
                bd_coll_pts_shapes=bd_coll_pts_shapes,
                wake_vortex_pts_shapes=TE_wake_vortex_pts_shapes,
                full_aic_name='aic_M',
                symmetry=self.parameters['symmetry'],
                sub = sub,
                sub_eval_list = sub_eval_list,
                sub_induced_list = sub_induced_list
                # delta_t=delta_t,  # one line of wake vortex for fix wake
            )
        self.add(m, name='AssembleAic')
        '''3. compute the size of the full AIC (coll_pts_coords_names, wake_coords_names) matrix'''

        aic_shape_row = aic_shape_col = 0

        for i in range(len(bd_coll_pts_shapes)):
            aic_shape_row += (bd_coll_pts_shapes[i][1] *
                              bd_coll_pts_shapes[i][2])
            if problem_type=='fixed_wake':
                aic_shape_col += ((wake_vortex_pts_shapes[i][1] - 1) *
                                (wake_vortex_pts_shapes[i][2] - 1))
            elif problem_type=='prescribed_wake':
                aic_shape_col += ((wake_vortex_pts_shapes[i][1] ) *
                                (wake_vortex_pts_shapes[i][2] - 1))
        '''3. project the aic on to the bd_vertices'''
        # print('shape----',num_nodes, aic_shape_row,
        #                                  aic_shape_col) 
       
        m = Projection(
            input_vel_names=['aic_M'],
            normal_names=bd_vtx_normal_names,
            output_vel_name='M_mat',  # this is b
            input_vel_shapes=[(num_nodes, aic_shape_row, aic_shape_col, 3)
                              ],  #rotatonal_vel_shapes
            normal_shapes=bd_coll_pts_shapes)  # NOTE: need to fix this later
        self.add(m, name='Projection_aic')

        M = self.declare_variable('M_mat',
                                  shape=(num_nodes, aic_shape_row,
                                         aic_shape_col))

        if problem_type=='fixed_wake':
        
            sprs = compute_spars(bd_vortex_shapes)
            M_reshaped = csdl.custom(M,
                                    op=Explicit(
                                        num_nodes=num_nodes,
                                        sprs=sprs,
                                        num_bd_panel=aic_shape_row,
                                        num_wake_panel=aic_shape_col,
                                    ))
        '''2. compute A_mtx'''
        m = AssembleAic(
            bd_coll_pts_names=coll_pts_coords_names,
            wake_vortex_pts_names=bd_vtx_coords_names,
            bd_coll_pts_shapes=bd_coll_pts_shapes,
            wake_vortex_pts_shapes=bd_vortex_shapes,
            full_aic_name='aic_bd',
            vc = False,
            symmetry=self.parameters['symmetry'],
            sub = sub,
            sub_eval_list = sub_eval_list,
            sub_induced_list = sub_induced_list
            # delta_t=delta_t,  # one line of wake vortex for fix wake
        )
        self.add(m, name='AssembleAic_bd')
        '''3. project the aic on to the bd_vertices'''
        aic_shape_row = aic_shape_col = 0
        for i in range(len(bd_coll_pts_shapes)):
            aic_shape_row += (bd_coll_pts_shapes[i][1] *
                              bd_coll_pts_shapes[i][2])
            aic_shape_col += ((bd_coll_pts_shapes[i][1]) *
                              (bd_coll_pts_shapes[i][2]))
        aic_bd_proj_name = 'aic_bd_proj'

        m = Projection(
            input_vel_names=['aic_bd'],
            normal_names=bd_vtx_normals,
            output_vel_name=aic_bd_proj_name,  # this is b
            input_vel_shapes=[(num_nodes, aic_shape_row, aic_shape_col, 3)
                              ],  #rotatonal_vel_shapes
            normal_shapes=bd_coll_pts_shapes,
        )
        self.add(m, name='Projection_aic_bd')
        sum_ny = sum((i[2] - 1) for i in bd_vortex_shapes)
        aic_bd_proj = self.declare_variable(aic_bd_proj_name,
                                            shape=(num_nodes, aic_shape_row,
                                                   aic_shape_col))
        if problem_type=='fixed_wake':
            self.register_output(
                'MTX', \
                    aic_bd_proj + csdl.reshape(
                    M_reshaped,
                    (num_nodes, aic_shape_row, aic_shape_row),
                ))


if __name__ == "__main__":

    def generate_simple_mesh(nx, ny, n_wake_pts_chord=None):
        if n_wake_pts_chord == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((n_wake_pts_chord, nx, ny, 3))
            for i in range(n_wake_pts_chord):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0.
        return mesh

    model_1 = Model()

    frame_vel_val = np.array([1, 0, 1])
    bd_vortex_coords_val = generate_simple_mesh(3, 4)
    wake_coords_val = np.array([
        [2., 0., 0.],
        [2., 1., 0.],
        [2., 2., 0.],
        [2., 3., 0.],
        [42., 0., 0.],
        [42., 1., 0.],
        [42., 2., 0.],
        [42., 3., 0.],
    ]).reshape(2, 4, 3)
    # coll_val = np.random.random((4, 5, 3))

    frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)
    bd_vortex_coords = model_1.create_input('bd_vortex_coords',
                                            val=bd_vortex_coords_val)
    wake_coords = model_1.create_input('wake_coords', val=wake_coords_val)
    nx = 3
    ny = 4
    coll_pts = 0.25 * (bd_vortex_coords[0:nx-1, 0:ny-1, :] +\
                                               bd_vortex_coords[0:nx-1, 1:ny, :] +\
                                               bd_vortex_coords[1:, 0:ny-1, :]+\
                                               bd_vortex_coords[1:, 1:, :])
    model_1.register_output('coll_coords', coll_pts)
    model_1.add(RHS())

    sim = Simulator(model_1)
    sim.run()
    sim.visualize_implementation()
    # print('aic is', sim['aic'])
    # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])
