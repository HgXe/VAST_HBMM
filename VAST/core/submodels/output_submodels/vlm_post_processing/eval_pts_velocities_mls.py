# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import axis, clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix
from VAST.core.submodels.aerodynamic_submodels.combine_bd_wake_comp import BdnWakeCombine
# from VAST.core.submodels.aerodynamic_submodels.biot_savart_vc_comp_org import BiotSavartComp
from VAST.core.submodels.aerodynamic_submodels.biot_savart_vc_comp import BiotSavartComp
# from VAST.core.submodels.aerodynamic_submodels.biot_savart_jax import BiotSavartComp
from VAST.core.submodels.aerodynamic_submodels.induced_velocity_comp import InducedVelocity

class EvalPtsVel(Model):
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
        self.parameters.declare('eval_pts_names', types=list)
        self.parameters.declare('eval_pts_shapes', default=None)

        self.parameters.declare('eval_pts_option',default='auto')
        self.parameters.declare('eval_pts_location',default=0.25)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        # stands for quarter-chord
        self.parameters.declare('n_wake_pts_chord')
        self.parameters.declare('delta_t',default=None)
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('problem_type',default='fixed_wake')
        self.parameters.declare('eps',default=1e-8)
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('sym_struct_list', default=None)
        self.parameters.declare('sub', default=False)
        self.parameters.declare('sub_eval_list', default=None)
        self.parameters.declare('sub_induced_list', default=None)

    def define(self):
        # eval_pts_names = self.parameters['eval_pts_names']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        eval_pts_location = self.parameters['eval_pts_location']
        mesh_unit = self.parameters['mesh_unit']
        eval_pts_option = self.parameters['eval_pts_option']

        sub = self.parameters['sub']
        sub_eval_list = self.parameters['sub_eval_list']
        sub_induced_list = self.parameters['sub_induced_list']
        symmetry = self.parameters['symmetry']
        sym_struct_list = self.parameters['sym_struct_list']

        num_nodes = surface_shapes[0][0]
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        delta_t = self.parameters['delta_t'] # NOT USED

        bdnwake_coords_names = [x + '_bdnwake_coords' for x in surface_names]
        wake_coords_reshaped_names = [x + '_wake_coords_reshaped' for x in surface_names]

        wake_vortex_pts_shapes = [
            tuple((num_nodes, n_wake_pts_chord, item[2], 3))
            for item in surface_shapes]

        circulation_names = [x + '_bdnwake_gamma' for x in surface_names]

        if self.parameters['problem_type'] == 'fixed_wake':
            bdnwake_shapes = [
                (num_nodes, x[1] + y[1] - 1, x[2], 3)
                for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
            ]
            circulations_shapes = [
                (num_nodes, (x[1] - 1) * (x[2] - 1) + (y[1] - 1) * (y[2] - 1))
                for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
            ]
            eval_induced_velocities_names = [
                x + '_eval_pts_induced_vel' for x in eval_pts_names
            ]
            v_total_eval_names = [x + '_eval_total_vel' for x in eval_pts_names]

        elif self.parameters['problem_type'] == 'prescribed_wake' or 'free_wake':
            bdnwake_shapes = [
                (num_nodes, x[1] + y[1], x[2], 3)
                for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
            ]
            circulations_shapes = [
                (num_nodes, (x[1] - 1) * (x[2] - 1) + (y[1] ) * (y[2] - 1))
                for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
            ]
            eval_induced_velocities_names = [
                x + '_eval_pts_induced_vel' for x in surface_names
            ]
            v_total_eval_names = [x + '_eval_total_vel' for x in surface_names]

        aic_shapes = [(num_nodes, x[1] * x[2] * (y[1] - 1) * (y[2] - 1), 3)
                      for x, y in zip(eval_pts_shapes, bdnwake_shapes)]

        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]

        else:
            eval_pts_names=self.parameters['eval_pts_names']

        if eval_pts_option=='auto':
            eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]

        eval_vel_shapes = [(num_nodes, x[1] * x[2], 3) for x in eval_pts_shapes]

        self.add(BdnWakeCombine(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            n_wake_pts_chord=n_wake_pts_chord,
            problem_type=self.parameters['problem_type'],),name='BdnWakeCombine')


        #!TODO:fix this for mls
        # !fixed!: this part is a temp fix-since we don't have +=in csdl, I just made a large velocity matrix contining
        # the induced velocity induced by each bdnwake_coords_names for mls, and sum this matrix by axis to get the
        # total induced vel
        eval_induced_velocities_col_names = [x + '_eval_pts_induced_vel_col' for x in eval_pts_names]

        for i in range(len(eval_pts_names)):
            eval_vel_shape = eval_vel_shapes[i]

            aic_shapes = [
                (num_nodes, x[1] * x[2] * (y[1] - 1) * (y[2] - 1), 3)
                for x, y in zip(([eval_pts_shapes[i]] *
                                 len(bdnwake_coords_names)), bdnwake_shapes)
            ]
            eval_pts_name_repeat = [eval_pts_names[i]
                                    ] * len(bdnwake_coords_names)

            output_names = [
                eval_pts_names[i] + x + '_out' for x in bdnwake_coords_names
            ]
            induced_vel_bdnwake_names = [
                eval_pts_names[i] + x + '_induced_vel'
                for x in bdnwake_coords_names
            ]

            # this is used for computing the force point induced velocity induced by the bdnwake_coords_names
            # the reason why we need vc (temp) for this is because the force points are on the leading edge
            # of each vortex ring, which cases r1_norm*r2_norm=-dot(r1,r2), making the denominator to zeros
            # this is fixed by finding the zeros and adding a small number to the denominator,
            # this is more acurate than adding a pertubation to every line vortex
            self.add(BiotSavartComp(
                eval_pt_names=eval_pts_name_repeat,
                vortex_coords_names=bdnwake_coords_names,
                eval_pt_shapes=[eval_pts_shapes[i]] *
                len(bdnwake_coords_names),
                vortex_coords_shapes=bdnwake_shapes,
                output_names=output_names,
                circulation_names=circulation_names,
                vc=True,
                eps=self.parameters['eps'],
                # symmetry=self.parameters['symmetry'],
                # aic_symmetry_dict=aic_symmetry_dict
            ),
            name='eval_pts_aics' + str(i))

            for j in range(len(bdnwake_coords_names)):
                aic = self.declare_variable(output_names[j],
                                            shape=(aic_shapes[j]))

            self.add(InducedVelocity(
                aic_names=output_names,
                circulation_names=circulation_names,
                aic_shapes=aic_shapes,
                circulations_shapes=circulations_shapes,
                v_induced_names=induced_vel_bdnwake_names),
                     name='eval_pts_ind_vel' + str(i))
                     
            # !!!!!!!!!!!TODO: need to check what is this April 18 2022
            # 08/03 this is used to add up the induced velocites
            surface_total_induced_col = self.create_output(
                eval_induced_velocities_col_names[i],
                shape=(num_nodes, len(bdnwake_coords_names),
                       eval_vel_shapes[i][1], 3))
            for j in range(len(bdnwake_coords_names)):
                induced_vel_bdnwake = self.declare_variable(
                    induced_vel_bdnwake_names[j],
                    shape=(num_nodes, eval_vel_shape[1], 3))
                surface_total_induced_col[:, j, :, :] = csdl.reshape(
                    induced_vel_bdnwake,
                    (num_nodes, 1, eval_vel_shapes[i][1], 3))
            eval_induced_velocity = self.register_output(
                eval_induced_velocities_names[i],
                csdl.sum(surface_total_induced_col, axes=(1, )))

            # print('eval_induced_velocity----------',eval_induced_velocities_names)

        # kinematic_vel_names = [
        #     x + '_kinematic_vel' for x in self.parameters['surface_names']
        # ]
        # TODO: check this part for the whole model
        model_wake_total_vel = Model()
        for i in range(len(eval_induced_velocities_names)):
            v_induced_wake_name = eval_induced_velocities_names[i]
            eval_vel_shape = eval_vel_shapes[i]

            wake_vortex_pts_shape = wake_vortex_pts_shapes[i]
            # kinematic_vel_name = kinematic_vel_names[i]

            v_induced_wake = model_wake_total_vel.declare_variable(
                v_induced_wake_name, shape=eval_vel_shape)
            
            v_kinematic = model_wake_total_vel.declare_variable(surface_names[i]+'_kinematic_vel', shape=eval_vel_shape)

            # !!TODO!! this needs to be fixed for more general cases to compute the undisturbed vel
            # Note - April 7 2022: the wake velocity seems to just
            # need to be the sum of free stream and the induced velocity - this part seems to be fine for now

            # kinematic_vel = model_wake_total_vel.declare_variable(
            #     kinematic_vel_name, shape=wake_vel_shape)
            frame_vel = model_wake_total_vel.declare_variable('frame_vel',
                                                              shape=(num_nodes,
                                                                     3))
            frame_vel_expand = csdl.expand(frame_vel,
                                           eval_vel_shape,
                                           indices='li->lji')

            # v_total_wake = csdl.reshape((v_induced_wake - frame_vel_expand),
            #                             new_shape=eval_vel_shape)


            v_total_wake = csdl.reshape((v_induced_wake + v_kinematic),
                                        new_shape=eval_vel_shape)

            model_wake_total_vel.register_output(v_total_eval_names[i],
                                                 v_total_wake)

            

        self.add(model_wake_total_vel, name='eval_pts_total_vel')