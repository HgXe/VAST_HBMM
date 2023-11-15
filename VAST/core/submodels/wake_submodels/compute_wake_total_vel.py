# 1. compute wake kinematic vel
# -frame_vel + rot_vel
# TODO: figure out this rot_vel for the plunging wing case


# 2. compute wake induced vel
# BS (surface_wake_coords, (induced by) all bound and wake coords) 
# gamma_b and gamma_W (depending on the ordering in the last line)


# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix


from VAST.core.submodels.wake_submodels.compute_wake_kinematic_vel_temp import ComputeWakeKinematicVel
from VAST.core.submodels.wake_submodels.eval_pts_velocities_mls import EvalPtsVel
# from VAST.core.submodels.output_submodels.vlm_post_processing.eval_pts_velocities_mls import EvalPtsVel


class ComputeWakeTotalVel(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.
    A \gamma_b = b - M \gamma_w
    parameters
    ----------

    collocation_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the bd vertices collocation_pts     
    wake_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the wake panel collcation pts 
    wake_circulations[num_wake_panel] : csdl array
        a concatenate vector of the wake circulation strength
    Returns
    -------
    vel_col_w[num_evel_pts_x*num_vortex_panel_x* num_evel_pts_y*num_vortex_panel_y,3]
    csdl array
        the velocities computed using the aic_col_w from biot svart's law
        on bound vertices collcation pts induces by the wakes
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

        self.parameters.declare('n_wake_pts_chord') # we need this to combine bd and wake 
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('problem_type', default='prescribed_wake')
        self.parameters.declare('sub', default=False)
        self.parameters.declare('sub_eval_list', default=None)
        self.parameters.declare('sub_induced_list', default=None)
        self.parameters.declare('sym_struct_list', default=None)
        self.parameters.declare('core_size', default=5.e-2)

    def define(self):
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        problem_type = self.parameters['problem_type']

        sub = self.parameters['sub']
        sub_eval_list = self.parameters['sub_eval_list']
        sub_induced_list = self.parameters['sub_induced_list']
        sym_struct_list = self.parameters['sym_struct_list']

        submodel = ComputeWakeKinematicVel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            n_wake_pts_chord=n_wake_pts_chord
        )
        self.add(submodel, name='ComputeWakeKinematicVel')

        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        wake_vortex_pts_shapes = [tuple((item[0],n_wake_pts_chord, item[2], 3)) for item in surface_shapes]

        if problem_type == 'free_wake':
            submodel = EvalPtsVel(
                eval_pts_names=wake_coords_names,
                eval_pts_shapes=wake_vortex_pts_shapes,
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                n_wake_pts_chord=n_wake_pts_chord,
                problem_type='prescribed_wake',
                symmetry=self.parameters['symmetry'],
                sub=sub, 
                sub_eval_list=sub_eval_list,
                sub_induced_list=sub_induced_list,
                sym_struct_list=sym_struct_list,
                core_size=self.parameters['core_size']
                
            )
            self.add(submodel, name='EvalPtsVel')

        wake_kinematic_vel_names = [x + '_wake_kinematic_vel' for x in surface_names]
        wake_total_vel_names = [x + '_wake_total_vel' for x in surface_names]
        eval_induced_velocities_names = [x + '_wake_induced_vel' for x in surface_names]
        # print('eval_induced_velocities_names----------------------', eval_induced_velocities_names)

        for i in range(len(surface_names)):
            wake_kinematic_vel = self.declare_variable(wake_kinematic_vel_names[i],shape=wake_vortex_pts_shapes[i])
            if problem_type == 'free_wake':
                wake_induced_vel = self.declare_variable(eval_induced_velocities_names[i],shape=wake_vortex_pts_shapes[i], val=0.)
            else:
                wake_induced_vel = 0.
            # NOTE: wake_induced_vel can be non-zero only if problem_type == 'free_wake'

            # print('vars-------------')
            # self.print_var(wake_kinematic_vel)
            ''''TODO: fix this hardcoding'''
            wake_total_vel = wake_kinematic_vel + wake_induced_vel
            # wake_kinematic_vel * 0
            # wake_kinematic_vel+0#+wake_induced_vel
            
            self.register_output(wake_total_vel_names[i],wake_total_vel)

