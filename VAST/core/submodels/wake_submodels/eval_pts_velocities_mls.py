# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import axis, clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix
from VAST.core.submodels.aerodynamic_submodels.combine_bd_wake_comp_ode import BdnWakeCombine
# from VAST.core.submodels.aerodynamic_submodels.biot_savart_vc_comp_org import BiotSavartComp
from VAST.core.submodels.aerodynamic_submodels.biot_savart_vc_comp import BiotSavartComp
# from VAST.core.submodels.aerodynamic_submodels.biot_savart_jax import BiotSavartComp
from VAST.core.submodels.aerodynamic_submodels.induced_velocity_comp import InducedVelocity

from VAST.utils.symmetry_sub_functions import generate_symmetry_groups, adjust_biot_savart_inputs_for_symmetry, modify_biot_savart_interactions

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
        self.parameters.declare('eval_pts_shapes', types=list)

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        # stands for quarter-chord
        self.parameters.declare('n_wake_pts_chord')
        self.parameters.declare('problem_type')
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('sub', default=False)
        self.parameters.declare('sub_eval_list', default=None)
        self.parameters.declare('sub_induced_list', default=None)
        self.parameters.declare('sym_struct_list', default=None)

    def define(self):
        eval_pts_names = self.parameters['eval_pts_names']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        
        sub = self.parameters['sub']
        sub_eval_list = self.parameters['sub_eval_list']
        sub_induced_list = self.parameters['sub_induced_list']
        sym_struct_list = self.parameters['sym_struct_list']

        num_nodes = surface_shapes[0][0]
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']

        bdnwake_coords_names = [x + '_bdnwake_coords' for x in surface_names]
        wake_coords_reshaped_names = [x + '_wake_coords_reshaped' for x in surface_names]

        wake_vortex_pts_shapes = [
            tuple((num_nodes, n_wake_pts_chord, item[2], 3))
            for item in surface_shapes]
        
        circulation_names = [x + '_bdnwake_gamma_wake' for x in surface_names]

        bdnwake_shapes = [
            (num_nodes, x[1] + y[1], x[2], 3)
            for x, y in zip(surface_shapes, wake_vortex_pts_shapes)]

        circulations_shapes = [(num_nodes, (x[1] - 1) * (x[2] - 1) + (y[1]) * (y[2] - 1))
                                for x, y in zip(surface_shapes, wake_vortex_pts_shapes)]
        
        eval_induced_velocities_names = [x + '_wake_induced_vel' for x in surface_names]
        
        aic_shapes = [(num_nodes, x[1] * x[2] * (y[1] - 1) * (y[2] - 1), 3)
                      for x, y in zip(eval_pts_shapes, bdnwake_shapes)]

        eval_vel_shapes = [(num_nodes, x[1] * x[2], 3) for x in eval_pts_shapes]

        #!TODO!: rewrite this comp for mls
        # !fixed!: defining the eval_pts
        for i in range(len(eval_pts_names)):
            mesh = self.declare_variable(surface_names[i], shape=surface_shapes[i])
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            eval_pts_coords = self.declare_variable(eval_pts_names[i], shape=(eval_pts_shapes[i]))

        self.add(BdnWakeCombine(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            n_wake_pts_chord=n_wake_pts_chord,
            problem_type=self.parameters['problem_type']), name='BdnWakeCombine')

        for i in range(len(surface_shapes)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            bdnwake_coords = self.declare_variable(bdnwake_coords_names[i], shape=(num_nodes, n_wake_pts_chord + nx, ny, 3))
            # print('bdnwake_coords_names[i]------------------', bdnwake_coords_names[i])
            # self.print_var(bdnwake_coords)
        #!TODO:fix this for mls
        # !fixed!: this part is a temp fix-since we don't have +=in csdl, I just made a large velocity matrix contining
        # the induced velocity induced by each bdnwake_coords_names for mls, and sum this matrix by axis to get the
        # total induced vel
        eval_induced_velocities_col_names = [x + '_eval_pts_induced_vel_col_wake' for x in surface_names]

        eval_pt_names_expanded = []
        eval_pt_shapes_expanded = []
        bdnwake_coords_names_expanded = []
        bdnwake_coords_shapes_expanded = []
        output_names_expanded = []
        full_aic_name = 'fw_eval_bdnwake' # free wake at eval points for bdn+wake points
        for i in range(len(eval_pts_names)):
            for j in range(len(wake_vortex_pts_shapes)):
                eval_pt_names_expanded.append(eval_pts_names[i])
                eval_pt_shapes_expanded.append(eval_pts_shapes[i])
                bdnwake_coords_names_expanded.append(bdnwake_coords_names[j])
                bdnwake_coords_shapes_expanded.append(bdnwake_shapes[j])
                # out_name = full_aic_name + eval_pts_names[i] + bdnwake_coords_names[j] + '_out'
                out_name = full_aic_name  +'_'+ str(i) +'_'+ str(j)
                output_names_expanded.append(out_name)
                # [eval_pts_names[i] + x + '_out' for x in bdnwake_coords_names]


        # region SETTING UP SYMMETRY
        symmetry_structure = False
        aic_symmetry_dict = False
        if self.parameters['symmetry'] and sym_struct_list is not None:
            symmetry_structure = True
            symmetry_outputs = generate_symmetry_groups(
                sym_struct_list,
                eval_pts_names,
                full_aic_name
            )
            interaction_groups_dict = symmetry_outputs[0]
            aic_names_dict = symmetry_outputs[1]
            aic_names_list = symmetry_outputs[2]
        # endregion

        if sub:
            eval_pt_names_expanded_sub, eval_pt_shapes_expanded_sub, bdnwake_coords_names_expanded_sub, bdnwake_coords_expanded_shapes_sub, output_names_expanded_sub = modify_biot_savart_interactions(
                sub_eval_list, sub_induced_list, eval_pts_names, eval_pts_shapes, bdnwake_coords_names, bdnwake_shapes, full_aic_name
            )

            if symmetry_structure == True: # NEED TO ADJUST INPUTS FOR THE ABOVE LISTS
                1
                sym_data = adjust_biot_savart_inputs_for_symmetry(
                    eval_pt_names_expanded_sub, 
                    eval_pt_shapes_expanded_sub, 
                    bdnwake_coords_names_expanded_sub, 
                    bdnwake_coords_expanded_shapes_sub,
                    output_names_expanded_sub, 
                    aic_names_dict, 
                )
                
                eval_pt_names_expanded_sub = sym_data[0] 
                eval_pt_shapes_expanded_sub = sym_data[1]
                bdnwake_coords_names_expanded_sub = sym_data[2]
                bdnwake_coords_expanded_shapes_sub = sym_data[3]
                output_names_expanded_sub = sym_data[4]

                aic_symmetry_dict = aic_names_dict
            # exit()
            self.add(BiotSavartComp(
                eval_pt_names=eval_pt_names_expanded_sub,
                eval_pt_shapes=eval_pt_shapes_expanded_sub,
                vortex_coords_names=bdnwake_coords_names_expanded_sub,
                vortex_coords_shapes=bdnwake_coords_expanded_shapes_sub,
                output_names=output_names_expanded_sub,
                circulation_names=circulation_names, # not used, doesn't matter here
                vc=True,
                eps=5e-2,
                symmetry=self.parameters['symmetry'],
                aic_symmetry_dict=aic_symmetry_dict
            ),
            name='eval_pts_aics')
            # print('==== eval_pts_names')
            # print(eval_pt_names_expanded_sub)
            # print('==== eval_pt_shape')
            # print(eval_pt_shapes_expanded_sub)
            # print('==== bdnwake_coords_names_expanded_sub')
            # print(bdnwake_coords_names_expanded_sub)
            # print('==== bdnwake_coords_expanded_shapes_sub')
            # print(bdnwake_coords_expanded_shapes_sub)
            # print('==== output_names_expanded_sub')
            # print(output_names_expanded_sub)
            # exit()
            
        else:
            if self.parameters['symmetry'] and sym_struct_list is not None:
                sym_data = adjust_biot_savart_inputs_for_symmetry(
                    eval_pt_names_expanded, 
                    eval_pt_shapes_expanded, 
                    bdnwake_coords_names_expanded, 
                    bdnwake_coords_shapes_expanded,
                    output_names_expanded, 
                    aic_names_dict, 
                )
                
                eval_pt_names_expanded_sub = sym_data[0] 
                eval_pt_shapes_expanded_sub = sym_data[1]
                bdnwake_coords_names_expanded_sub = sym_data[2]
                bdnwake_coords_shapes_expanded_sub = sym_data[3]
                output_names_expanded_sub = sym_data[4]

                aic_symmetry_dict = aic_names_dict

                m = BiotSavartComp(
                    eval_pt_names=eval_pt_names_expanded_sub,
                    eval_pt_shapes=eval_pt_shapes_expanded_sub,
                    vortex_coords_names=bdnwake_coords_names_expanded_sub,
                    vortex_coords_shapes=bdnwake_coords_shapes_expanded_sub,
                    output_names=output_names_expanded_sub,
                    vc=True,
                    symmetry=self.parameters['symmetry'],
                    aic_symmetry_dict=aic_symmetry_dict
                )
                self.add(m, name='eval_pts_aics')


            else:
                m = BiotSavartComp(
                    eval_pt_names=eval_pt_names_expanded,
                    eval_pt_shapes=eval_pt_shapes_expanded,
                    vortex_coords_names=bdnwake_coords_names_expanded,
                    vortex_coords_shapes=bdnwake_coords_shapes_expanded,
                    output_names=output_names_expanded,
                    vc=True,
                    symmetry=self.parameters['symmetry'],
                )
                self.add(m, name='eval_pts_aics')            

        # print('eval_pts_names: ', eval_pts_names)

        for i in range(len(eval_pts_names)):
            eval_vel_shape = eval_vel_shapes[i]

            aic_shapes = [
                (num_nodes, x[1] * x[2] * (y[1] - 1) * (y[2] - 1), 3)
                for x, y in zip(([eval_pts_shapes[i]] *
                                 len(bdnwake_coords_names)), bdnwake_shapes)]

            eval_pts_name_repeat = [eval_pts_names[i]
                                    ] * len(bdnwake_coords_names)

            # output_names = [eval_pts_names[i] + x + '_out' for x in bdnwake_coords_names]
            # out_name = full_aic_name  +'_'+ str(i) +'_'+ str(j)
            # output_names_expanded.append(out_name)
            output_names = [full_aic_name  +'_'+ str(i) +'_'+ str(j) for j in range(len(bdnwake_coords_names))]
            print(output_names)
            induced_vel_bdnwake_names = [
                eval_pts_names[i] + x + '_induced_vel'
                for x in bdnwake_coords_names
            ]
            # self.add(BiotSavartComp(
            #     eval_pt_names=eval_pts_name_repeat,
            #     vortex_coords_names=bdnwake_coords_names,
            #     eval_pt_shapes=[eval_pts_shapes[i]] *
            #     len(bdnwake_coords_names),
            #     vortex_coords_shapes=bdnwake_shapes,
            #     output_names=output_names,
            #     circulation_names=circulation_names, # not used, doesn't matter here
            #     vc=True,
            #     eps=5e-2,
            #     symmetry=self.parameters['symmetry']
            # ),
            # name='eval_pts_aics' + str(i))

            for j in range(len(bdnwake_coords_names)):
                aic = self.declare_variable(output_names[j],
                                            shape=(aic_shapes[j]),
                                            val=0.)
            # print('eval pts vel mls aic_shapes', aic_shapes)

            self.add(InducedVelocity(
                aic_names=output_names,
                circulation_names=circulation_names,
                aic_shapes=aic_shapes,
                circulations_shapes=circulations_shapes,
                v_induced_names=induced_vel_bdnwake_names),
                     name='eval_pts_ind_vel' + str(i))
            # !!!!!!!!!!!TODO: need to check what is this April 18 2022
            # surface_total_induced_col = self.create_output(
            #     eval_induced_velocities_col_names[i],
            #     shape=(num_nodes, len(bdnwake_coords_names),
            #            eval_vel_shapes[i][1], 3))
            induced_vel_list = []
            for j in range(len(bdnwake_coords_names)):
                induced_vel_bdnwake = self.declare_variable(
                    induced_vel_bdnwake_names[j],
                    shape=(num_nodes, eval_vel_shape[1], 3))
                # surface_total_induced_col[:, j, :, :] = csdl.reshape(
                #     var=induced_vel_bdnwake,
                #     new_shape=(num_nodes, 1, eval_vel_shapes[i][1], 3))
                induced_vel_list.append(induced_vel_bdnwake)

            eval_induced_velocity = sum(induced_vel_list)
            
            ny = surface_shapes[i][2]
            self.register_output(
                name=eval_induced_velocities_names[i],
                var=csdl.reshape(eval_induced_velocity,(num_nodes,n_wake_pts_chord,ny,3)))
        # exit()
                        
