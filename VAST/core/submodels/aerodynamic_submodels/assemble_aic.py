# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
# from VAST.core.submodels.aerodynamic_submodels.biot_savart_vc_comp_org import BiotSavartComp
from VAST.core.submodels.aerodynamic_submodels.biot_savart_vc_comp import BiotSavartComp
# from VAST.core.submodels.aerodynamic_submodels.biot_savart_jax import BiotSavartComp

from VAST.utils.symmetry_sub_functions import generate_symmetry_groups, adjust_biot_savart_inputs_for_symmetry, modify_biot_savart_interactions

class AssembleAic(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.

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
        # aic_col_w
        self.parameters.declare('bd_coll_pts_names', types=list)
        self.parameters.declare('wake_vortex_pts_names', types=list)

        self.parameters.declare('bd_coll_pts_shapes', types=list)
        self.parameters.declare('wake_vortex_pts_shapes', types=list)
        self.parameters.declare('full_aic_name', types=str)

        self.parameters.declare('delta_t',default=None)
        self.parameters.declare('vc',default=False)
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('sub',default=False)
        self.parameters.declare('sub_eval_list',default=None)
        self.parameters.declare('sub_induced_list',default=None)
        self.parameters.declare('sym_struct_list', default=None)

    def _generate_symmetry_groups(self, sym_struct_list, bd_coll_pts_names, full_aic_name):

        # Parsing surfaces and their plane of symmetry
        surf_reflection_dict = {}
        init_surfaces = []
        for i, sym_set in enumerate(sym_struct_list):
            init_surf_ind = sym_set[0]
            init_surfaces.append(init_surf_ind)
            len_sym_set = len(sym_set)
            
            if len_sym_set == 1: # single surface symmetric across y
                pass

            elif len_sym_set == 2:
                if 'image' in bd_coll_pts_names[sym_set[0]]:
                    # print('image in 1')
                    1

                elif 'image' in bd_coll_pts_names[sym_set[1]]:
                    # print('image in 2')
                    sub_dict = {'ref': [sym_set[1]], 'axis': ['z']}
            
            elif len_sym_set == 4:
                ref_list = []
                axis_list = []

                ref_surf = sym_set[1:]
                image_in_surf = ['image' in bd_coll_pts_names[i] for i in ref_surf]
                for j, bool_val in enumerate(image_in_surf):
                    if bool_val == False:
                        ref_list.append(ref_surf[j])
                        axis_list.append('y')
                    elif bool_val == True:
                        ref_list.append(ref_surf[j])
                        dummy_string = bd_coll_pts_names[ref_surf[j]].replace('image_','')
                        if dummy_string == bd_coll_pts_names[sym_set[0]]:
                            axis_list.append('z')
                        else:
                            axis_list.append('yz')

                sub_dict = {'ref': ref_list, 'axis': axis_list}
            
            surf_reflection_dict[init_surf_ind] = sub_dict
        
        # Setting up reflection sets and their respective planes of symmetry
        # SEARCH DOWN IN THE DICTIONARY
        # START WITH FIRST KEY AND SEARCH REST
        # THEN, START FROM SECOND AND MOVE DOWN
        interaction_groups_dict = {}
        for i, surf in enumerate(init_surfaces):
            ref_surfaces = surf_reflection_dict[surf]['ref']
            ref_axes = surf_reflection_dict[surf]['axis']
            outer_surf_list = []
            outer_surf_list.append(surf)
            outer_surf_list.extend(ref_surfaces)
            interactions = {}
            for j in range(len(init_surfaces)): # searching through dict entries
                init_surface = init_surfaces[j] # initial surface (the main one, also the dictionary key)
                inner_reflected_surfaces = surf_reflection_dict[init_surface]['ref'] # the surfaces symmetric to the line above
                inner_reflected_axes = surf_reflection_dict[init_surface]['axis'] # corresponding axes of reflected surfaces

                # Extracting surfaces within the inner loop
                inner_surfaces = []
                inner_surfaces.append(init_surface)
                inner_surfaces.extend(inner_reflected_surfaces)

                # Interaction group for main surface
                interaction_groups_main_surf = []
                interaction_groups_main_surf.append((surf, init_surface)) # Always appending interaction between outer surface and first surface defined in symmetry sets

                # looping through all of the reflected/symmetric surfaces
                if surf == init_surface: # looking at self interactions
                    for k, loop_surf in enumerate(inner_surfaces):
                        interaction_groups = []
                        interaction_axes = []
                        if surf == loop_surf:
                            interaction_groups.extend([(s,s) for s in inner_surfaces])
                            interaction_axes.extend(inner_reflected_axes)
                            ref_axis = 'self'
                        else:
                            if len(inner_surfaces) == 2:
                                interaction_groups.extend([(inner_surfaces[0], inner_surfaces[1]), (inner_surfaces[1], inner_surfaces[0])])
                                interaction_axes.append('z') # NOTE: assuming that any combination of 2 surfaces is symmetric across z ONLY
                                ref_axis = 'z'
                            else:
                                interaction_groups.extend([(surf, loop_surf), (loop_surf, surf)])
                                loop_surf_ind = inner_reflected_surfaces.index(loop_surf)
                                interaction_axes.append(inner_reflected_axes[loop_surf_ind])
                                rem_surf = [inner_reflected_surfaces[s] for s in range(len(inner_reflected_surfaces)) if s != loop_surf_ind]
                                rem_axes = [inner_reflected_axes[s] for s in range(len(inner_reflected_axes)) if s != loop_surf_ind]
                                interaction_groups.extend([(rem_surf[0], rem_surf[1]), (rem_surf[1], rem_surf[0])])
                                interaction_axes.extend([rem_axes[0], rem_axes[1]])
                                
                                ref_axis = inner_reflected_axes[loop_surf_ind]
                                
                        interaction_groups_dict[interaction_groups[0]] = {
                            'reflection' :interaction_groups[1:],
                            'axis': interaction_axes,
                            'ref_axis': ref_axis
                        }
                
                else:
                    if len(outer_surf_list) == 2:
                        main_surf_ref = 'z'
                        if len(inner_surfaces) == 4:
                            dict_connections = [inner_surfaces[0]]
                            dict_connections_axes = ['y']
                            for a, axis in enumerate(inner_reflected_axes):
                                if 'y' not in axis:
                                    dict_connections.append(inner_reflected_surfaces[a])
                                    dict_connections_axes.append('yz')

                            for k, loop_surf in enumerate(dict_connections):
                                interaction_groups = []
                                interaction_groups.append((surf, loop_surf))
                                interaction_axes = ['y', 'z', 'yz']
                                if dict_connections_axes[k] == 'y':
                                    y_ind = inner_reflected_axes.index('y')
                                    interaction_groups.append((surf, inner_reflected_surfaces[y_ind]))
                                    interaction_groups.extend([(outer_surf_list[1], inner_reflected_surfaces[i]) for i in range(len(inner_reflected_surfaces)) if i != y_ind])
                                    ref_axis = 'plane'
                                
                                elif dict_connections_axes[k] == 'yz':
                                    yz_ind = inner_reflected_axes.index('yz')
                                    interaction_groups.append((surf, inner_reflected_surfaces[yz_ind]))
                                    interaction_groups.append((outer_surf_list[1], inner_surfaces[0]))
                                    y_ind = inner_reflected_axes.index('y')
                                    interaction_groups.append((outer_surf_list[1], inner_reflected_surfaces[y_ind]))
                                    ref_axis = 'z'

                                interaction_groups_dict[interaction_groups[0]] = {
                                    'reflection' :interaction_groups[1:],
                                    'axis': interaction_axes,
                                    'ref_axis': ref_axis
                                }
                        elif len(inner_surfaces) == 2:
                            pass

                    elif len(outer_surf_list) == 4:
                        if len(inner_surfaces) == 2:
                            for k, loop_surf in enumerate(inner_surfaces):
                                loop_surf_ind = inner_surfaces.index(loop_surf)
                                rem_loop_ind = int(1-loop_surf_ind)
                                interaction_groups = [(surf, loop_surf)]
                                interaction_axes = ['y', 'z', 'yz']

                                inner_ref_ind = inner_reflected_axes[0]

                                y_ind = ref_axes.index('y')
                                z_ind = ref_axes.index('z')
                                yz_ind = ref_axes.index('yz')

                                interaction_groups.append((outer_surf_list[y_ind+1], loop_surf))
                                interaction_groups.append((outer_surf_list[z_ind+1], inner_surfaces[rem_loop_ind]))
                                interaction_groups.append((outer_surf_list[yz_ind+1], inner_surfaces[rem_loop_ind]))

                                if k == 0:
                                    ref_axis = 'plane'
                                else:
                                    ref_axis = 'z'

                                interaction_groups_dict[interaction_groups[0]] = {
                                    'reflection' :interaction_groups[1:],
                                    'axis': interaction_axes,
                                    'ref_axis': ref_axis
                                }

                        elif len(inner_surfaces) == 4:
                            y_ind = ref_axes.index('y')
                            z_ind = ref_axes.index('z')
                            yz_ind = ref_axes.index('yz')

                            y_ind_inner = inner_reflected_axes.index('y')
                            z_ind_inner = inner_reflected_axes.index('z')
                            yz_ind_inner = inner_reflected_axes.index('yz')

                            for k, loop_surf in enumerate(inner_surfaces):
                                loop_surf_ind = inner_surfaces.index(loop_surf)
                                interaction_groups = [(surf, loop_surf)]
                                interaction_axes = []
                                if k == 0:
                                    interaction_groups.append((ref_surfaces[y_ind], inner_reflected_surfaces[y_ind_inner]))
                                    interaction_groups.append((ref_surfaces[z_ind], inner_reflected_surfaces[z_ind_inner]))
                                    interaction_groups.append((ref_surfaces[yz_ind], inner_reflected_surfaces[yz_ind_inner]))

                                    interaction_axes.extend([inner_reflected_axes[y_ind_inner], inner_reflected_axes[z_ind_inner], inner_reflected_axes[yz_ind_inner]])
                                    
                                    interaction_groups_dict[interaction_groups[0]] = {
                                        'reflection' :interaction_groups[1:],
                                        'axis': interaction_axes,
                                        'ref_axis': 'plane'
                                    }
                                else:
                                    ind = inner_reflected_surfaces.index(loop_surf)
                                    surf_axis = inner_reflected_axes[ind]
                                    if surf_axis == 'y':
                                        interaction_groups.append((ref_surfaces[y_ind], inner_surfaces[0]))
                                        interaction_groups.append((ref_surfaces[z_ind], inner_reflected_surfaces[yz_ind_inner]))
                                        interaction_groups.append((ref_surfaces[yz_ind], inner_reflected_surfaces[z_ind_inner]))
                                        interaction_axes.extend([inner_reflected_axes[y_ind_inner], inner_reflected_axes[z_ind_inner], inner_reflected_axes[yz_ind_inner]])
                                    elif surf_axis == 'z':
                                        interaction_groups.append((ref_surfaces[y_ind], inner_reflected_surfaces[yz_ind_inner]))
                                        interaction_groups.append((ref_surfaces[z_ind], inner_surfaces[0]))
                                        interaction_groups.append((ref_surfaces[yz_ind], inner_reflected_surfaces[y_ind_inner]))
                                        interaction_axes.extend([inner_reflected_axes[y_ind_inner], inner_reflected_axes[z_ind_inner], inner_reflected_axes[yz_ind_inner]])
                                    elif surf_axis == 'yz':
                                        interaction_groups.append((ref_surfaces[y_ind], inner_reflected_surfaces[z_ind_inner]))
                                        interaction_groups.append((ref_surfaces[z_ind], inner_reflected_surfaces[y_ind_inner]))
                                        interaction_groups.append((ref_surfaces[yz_ind], inner_surfaces[0]))
                                        interaction_axes.extend([inner_reflected_axes[y_ind_inner], inner_reflected_axes[z_ind_inner], inner_reflected_axes[yz_ind_inner]])

                                    interaction_groups_dict[interaction_groups[0]] = {
                                        'reflection' :interaction_groups[1:],
                                        'axis': interaction_axes,
                                        'ref_axis': surf_axis
                                    }
        # SETTING UP AIC NAMES 
        aic_names_dict = {}
        aic_names_list = []
        for key in interaction_groups_dict.keys():
            # Assembling dictionary of AIC names
            interaction_group = interaction_groups_dict[key]['reflection']
            dict_key = full_aic_name + "_" + str(key[0]) + "_" + str(key[1])
            dict_list = []
            for group in interaction_group:
                dict_list.append(full_aic_name + "_" + str(group[0]) + "_" + str(group[1]))


            aic_names_dict[dict_key] = {
                'names': dict_list,
                'axis': interaction_groups_dict[key]['axis'],
                'ref_axis': interaction_groups_dict[key]['ref_axis'],
            }

            # Assembling list of AIC names
            aic_names_list.append(dict_key)
            aic_names_list.extend(dict_list)
        return interaction_groups_dict, aic_names_dict, aic_names_list
    
    def _adjust_biot_savart_inputs_for_symmetry(self, eval_pt_names, eval_pt_shapes, vortex_coords_names, vortex_coords_shapes, output_names, aic_names_dict):
        eval_pt_names_new = []
        eval_pt_shapes_new = []
        vortex_coords_names_new = []
        vortex_coords_shapes_new = []
        output_names_new = []
        for key in aic_names_dict:
            name_ind = output_names.index(key)
            eval_pt_names_new.append(eval_pt_names[name_ind])
            eval_pt_shapes_new.append(eval_pt_shapes[name_ind])
            vortex_coords_names_new.append(vortex_coords_names[name_ind])
            vortex_coords_shapes_new.append(vortex_coords_shapes[name_ind])
            output_names_new.append(output_names[name_ind])
        
        return eval_pt_names_new, eval_pt_shapes_new, vortex_coords_names_new, vortex_coords_shapes_new, output_names_new

    def define(self):
        # add_input
        bd_coll_pts_names = self.parameters['bd_coll_pts_names']
        wake_vortex_pts_names = self.parameters['wake_vortex_pts_names']
        # delta_t = self.parameters['delta_t']

        bd_coll_pts_shapes = self.parameters['bd_coll_pts_shapes']
        wake_vortex_pts_shapes = self.parameters['wake_vortex_pts_shapes']
        full_aic_name = self.parameters['full_aic_name']
        vc = self.parameters['vc']
        sub = self.parameters['sub']
        sub_eval_list = self.parameters['sub_eval_list']
        sub_induced_list = self.parameters['sub_induced_list']
        sym_struct_list = self.parameters['sym_struct_list']


        num_nodes = bd_coll_pts_shapes[0][0]
        row_ind = 0
        col_ind = 0

        eval_pt_names = []
        vortex_coords_names = []
        eval_pt_shapes = []
        vortex_coords_shapes = []
        output_names = []
        aic_shape_row = aic_shape_col = 0

        for i in range(len(bd_coll_pts_shapes)):

            bd_coll_pts = self.declare_variable(bd_coll_pts_names[i],
                                                shape=bd_coll_pts_shapes[i])
            wake_vortex_pts = self.declare_variable(
                wake_vortex_pts_names[i], shape=wake_vortex_pts_shapes[i])
            aic_shape_row += (bd_coll_pts_shapes[i][1] *
                              bd_coll_pts_shapes[i][2])
            aic_shape_col += ((wake_vortex_pts_shapes[i][1] - 1) *
                              (wake_vortex_pts_shapes[i][2] - 1))

            for j in range(len(wake_vortex_pts_shapes)):
                eval_pt_names.append(bd_coll_pts_names[i])
                vortex_coords_names.append(wake_vortex_pts_names[j])
                eval_pt_shapes.append(bd_coll_pts_shapes[i])
                vortex_coords_shapes.append(wake_vortex_pts_shapes[j])
                out_name = full_aic_name  +'_'+ str(i) +'_'+ str(j)
                output_names.append(out_name)

        # print('assemble_aic line 84 eval_pt_names', eval_pt_names)
        # print('assemble_aic line 85 vortex_coords_names', vortex_coords_names)
        # print('assemble_aic line 86 eval_pt_shapes', eval_pt_shapes)
        # print('assemble_aic l 87 vortex_coords_shapes', vortex_coords_shapes)
        # print('assemble_aic l 87 output_names', output_names)

        # this vc needs to be true because we are computing induced velocity
        # on the quarter chord of the mesh, which is just on the bound vortex lines

        # sub=False
        symmetry_structure = False
        aic_symmetry_dict = False
        if self.parameters['symmetry'] and sym_struct_list is not None:
            symmetry_structure = True
            # symmetry_outputs = self._generate_symmetry_groups(
            symmetry_outputs = generate_symmetry_groups(
                sym_struct_list,
                bd_coll_pts_names,
                full_aic_name
            )
            interaction_groups_dict = symmetry_outputs[0]
            aic_names_dict = symmetry_outputs[1]
            aic_names_list = symmetry_outputs[2]

        if sub:

            eval_pt_names_sub, eval_pt_shapes_sub, vortex_coords_names_sub, vortex_coords_shapes_sub, output_names_sub = modify_biot_savart_interactions(
                sub_eval_list, sub_induced_list, bd_coll_pts_names, bd_coll_pts_shapes, wake_vortex_pts_names, wake_vortex_pts_shapes, full_aic_name
            )

            if False:
                eval_pt_names_sub = []
                vortex_coords_names_sub = []
                eval_pt_shapes_sub = []
                vortex_coords_shapes_sub = []
                output_names_sub = []
                # sub_eval_list = [0, 1]
                # sub_induced_list = [0, 1]
                # aic_shape_row_sub = aic_shape_col = 0

                for i in range(len(sub_eval_list)):
                    eval_pt_names_sub.append(bd_coll_pts_names[sub_eval_list[i]])
                    eval_pt_shapes_sub.append(bd_coll_pts_shapes[sub_eval_list[i]])
                    vortex_coords_names_sub.append(wake_vortex_pts_names[sub_induced_list[i]])
                    vortex_coords_shapes_sub.append(wake_vortex_pts_shapes[sub_induced_list[i]])
                    output_name_sub = full_aic_name  +'_'+ str(sub_eval_list[i]) +'_'+ str(sub_induced_list[i])
                    output_names_sub.append(output_name_sub)
            
            if symmetry_structure == True: # NEED TO ADJUST INPUTS FOR THE ABOVE LISTS
                1
                # sym_data = self._adjust_biot_savart_inputs_for_symmetry(eval_pt_names_sub, eval_pt_shapes_sub, vortex_coords_names_sub, vortex_coords_shapes_sub,
                                                                        
                sym_data = adjust_biot_savart_inputs_for_symmetry(eval_pt_names_sub, eval_pt_shapes_sub, vortex_coords_names_sub, vortex_coords_shapes_sub,
                                                             output_names_sub, aic_names_dict)
                
                eval_pt_names_sub = sym_data[0] 
                eval_pt_shapes_sub = sym_data[1]
                vortex_coords_names_sub = sym_data[2]
                vortex_coords_shapes_sub = sym_data[3]
                output_names_sub = sym_data[4]

                aic_symmetry_dict = aic_names_dict
            '''
            THE GOAL WITH THE SYMMETRY:
                - reduce the length of names and shapes to the dictionary keys in the symmetry data
                - need to add the aic_names_dict to the BiotSavartComp to figure out symmetric outputs and their corresponding axes
            '''
    
            m = BiotSavartComp(
                eval_pt_names=eval_pt_names_sub,
                vortex_coords_names=vortex_coords_names_sub,
                eval_pt_shapes=eval_pt_shapes_sub,
                vortex_coords_shapes=vortex_coords_shapes_sub,
                output_names=output_names_sub,
                vc=True,
                symmetry=self.parameters['symmetry'],
                aic_symmetry_dict=aic_symmetry_dict
            )
            self.add(m, name='aic_bd_w_seperate')

            sub_array = np.array([sub_eval_list,sub_induced_list])

        else:
            if self.parameters['symmetry'] and sym_struct_list is not None:
                sym_data = adjust_biot_savart_inputs_for_symmetry(eval_pt_names, eval_pt_shapes, vortex_coords_names, vortex_coords_shapes,
                                                             output_names, aic_names_dict)
                
                eval_pt_names_sub = sym_data[0] 
                eval_pt_shapes_sub = sym_data[1]
                vortex_coords_names_sub = sym_data[2]
                vortex_coords_shapes_sub = sym_data[3]
                output_names_sub = sym_data[4]

                aic_symmetry_dict = aic_names_dict
                
                m = BiotSavartComp(
                    eval_pt_names=eval_pt_names_sub,
                    vortex_coords_names=vortex_coords_names_sub,
                    eval_pt_shapes=eval_pt_shapes_sub,
                    vortex_coords_shapes=vortex_coords_shapes_sub,
                    output_names=output_names_sub,
                    vc=True,
                    symmetry=self.parameters['symmetry'],
                    aic_symmetry_dict=aic_symmetry_dict
                )
                self.add(m, name='aic_bd_w_seperate')
            else:
                m = BiotSavartComp(
                    eval_pt_names=eval_pt_names,
                    vortex_coords_names=vortex_coords_names,
                    eval_pt_shapes=eval_pt_shapes,
                    vortex_coords_shapes=vortex_coords_shapes,
                    output_names=output_names,
                    vc=True,
                    symmetry=self.parameters['symmetry'],
                )
                self.add(m, name='aic_bd_w_seperate')

        aic_shape = (num_nodes, aic_shape_row, aic_shape_col, 3)

        # m1 = Model()
        aic_col_w = self.create_output(full_aic_name, shape=aic_shape, val=0.)
        row = 0
        col = 0
        
        for i in range(len(bd_coll_pts_shapes)):
            for j in range(len(wake_vortex_pts_shapes)):
                aic_i_shape = (
                    num_nodes,
                    bd_coll_pts_shapes[i][1] * bd_coll_pts_shapes[i][2] *
                    (wake_vortex_pts_shapes[j][1] - 1) *
                    (wake_vortex_pts_shapes[j][2] - 1),
                    3,
                )
                # aic_i = self.declare_variable(
                #     output_names[i * (len(wake_vortex_pts_shapes)) + j],
                #     shape=aic_i_shape)

                delta_row = bd_coll_pts_shapes[i][1] * bd_coll_pts_shapes[i][2]
                delta_col = (wake_vortex_pts_shapes[j][1] -
                             1) * (wake_vortex_pts_shapes[j][2] - 1)
                
                if (sub==True) and ((np.array([i,j]).reshape(2,1) == sub_array).all(axis=0).any()):
                    aic_i = self.declare_variable(
                        output_names[i * (len(wake_vortex_pts_shapes)) + j],
                        shape=aic_i_shape,val=0.)
                    aic_col_w[:, row:row + delta_row,
                            col:col + delta_col, :] = csdl.reshape(
                                aic_i,
                                new_shape=(num_nodes, delta_row, delta_col, 3))
                
                elif sub==False:
                    aic_i = self.declare_variable(
                        output_names[i * (len(wake_vortex_pts_shapes)) + j],
                        shape=aic_i_shape,val=0.)

                    aic_col_w[:, row:row + delta_row,
                            col:col + delta_col, :] = csdl.reshape(
                                aic_i,
                                new_shape=(num_nodes, delta_row, delta_col, 3))

                col = col + delta_col
            col = 0
            row = row + delta_row


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

    bd_coll_pts_names = ['bd_coll_pts_1', 'bd_coll_pts_2']
    wake_vortex_pts_names = ['wake_vortex_pts_1', 'wake_vortex_pts_2']
    bd_coll_pts_shapes = [(2, 3, 3), (3, 2, 3)]
    wake_vortex_pts_shapes = [(3, 3, 3), (3, 2, 3)]
    # bd_coll_pts_shapes = [(2, 3, 3), (2, 3, 3)]
    # wake_vortex_pts_shapes = [(3, 3, 3), (3, 3, 3)]

    model_1 = Model()
    bd_val = np.random.random((2, 3, 3))
    bd_val_1 = np.random.random((3, 2, 3))
    wake_vortex_val = np.random.random((3, 3, 3))
    wake_vortex_val_1 = np.random.random((3, 2, 3))

    # bd_val = np.random.random((2, 3, 3))
    # bd_val_1 = np.random.random((2, 3, 3))
    # wake_vortex_val = np.random.random((3, 3, 3))
    # wake_vortex_val_1 = np.random.random((3, 3, 3))

    bd = model_1.create_input('bd_coll_pts_1', val=bd_val)
    bd_1 = model_1.create_input('bd_coll_pts_2', val=bd_val_1)

    wake_vortex = model_1.create_input('wake_vortex_pts_1',
                                       val=wake_vortex_val)
    wake_vortex_1 = model_1.create_input('wake_vortex_pts_2',
                                         val=wake_vortex_val_1)
    model_1.add(AssembleAic(
        bd_coll_pts_names=bd_coll_pts_names,
        wake_vortex_pts_names=wake_vortex_pts_names,
        bd_coll_pts_shapes=bd_coll_pts_shapes,
        wake_vortex_pts_shapes=wake_vortex_pts_shapes,
    ),
                name='assemble_aic_comp')
    sim = Simulator(model_1)
    sim.run()
    # sim.visualize_implementation()
    # print('aic is', sim['aic'])
    # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])


#         num_nodes = bd_coll_pts_shapes[0][0]
#         row_ind = 0
#         col_ind = 0

#         eval_pt_names = []
#         vortex_coords_names = []
#         eval_pt_shapes = []
#         vortex_coords_shapes = []
#         output_names = []
#         aic_shape_row = aic_shape_col = 0

#         for i in range(len(bd_coll_pts_shapes)):

#             bd_coll_pts = self.declare_variable(bd_coll_pts_names[i],
#                                                 shape=bd_coll_pts_shapes[i])
#             wake_vortex_pts = self.declare_variable(
#                 wake_vortex_pts_names[i], shape=wake_vortex_pts_shapes[i])
#             aic_shape_row += (bd_coll_pts_shapes[i][1] *
#                               bd_coll_pts_shapes[i][2])
#             aic_shape_col += ((wake_vortex_pts_shapes[i][1] - 1) *
#                               (wake_vortex_pts_shapes[i][2] - 1))

#             for j in range(len(wake_vortex_pts_shapes)):
#                 eval_pt_names.append(bd_coll_pts_names[i])
#                 vortex_coords_names.append(wake_vortex_pts_names[j])
#                 eval_pt_shapes.append(bd_coll_pts_shapes[i])
#                 vortex_coords_shapes.append(wake_vortex_pts_shapes[j])
#                 out_name = full_aic_name  +'_'+ str(i) +'_'+ str(j)
#                 output_names.append(out_name)

#         # print('assemble_aic line 84 eval_pt_names', eval_pt_names)
#         # print('assemble_aic line 85 vortex_coords_names', vortex_coords_names)
#         # print('assemble_aic line 86 eval_pt_shapes', eval_pt_shapes)
#         # print('assemble_aic l 87 vortex_coords_shapes', vortex_coords_shapes)
#         # print('assemble_aic l 87 output_names', output_names)

#         # this vc needs to be true because we are computing induced velocity
#         # on the quarter chord of the mesh, which is just on the bound vortex lines

#         # sub=True

#         if sub:


#             eval_pt_names_sub = []
#             vortex_coords_names_sub = []
#             eval_pt_shapes_sub = []
#             vortex_coords_shapes_sub = []
#             output_names_sub = []
#             # sub_eval_list = [0,1]
#             # sub_induced_list = [0, 1]
#             # aic_shape_row_sub = aic_shape_col = 0

#             for i in range(len(sub_eval_list)):
#                 eval_pt_names_sub.append(bd_coll_pts_names[sub_eval_list[i]])
#                 eval_pt_shapes_sub.append(bd_coll_pts_shapes[sub_eval_list[i]])
#                 vortex_coords_names_sub.append(wake_vortex_pts_names[sub_induced_list[i]])
#                 vortex_coords_shapes_sub.append(wake_vortex_pts_shapes[sub_induced_list[i]])
#                 output_name_sub = full_aic_name  +'_'+ str(sub_eval_list[i]) +'_'+ str(sub_induced_list[i])
#                 output_names_sub.append(output_name_sub)

#             m = BiotSavartComp(
#                 eval_pt_names=eval_pt_names_sub,
#                 vortex_coords_names=vortex_coords_names_sub,
#                 eval_pt_shapes=eval_pt_shapes_sub,
#                 vortex_coords_shapes=vortex_coords_shapes_sub,
#                 output_names=output_names_sub,
#                 vc=True,
#                 symmetry=self.parameters['symmetry'],
#             )
#             self.add(m, name='aic_bd_w_seperate')

        


#         else:

#             m = BiotSavartComp(
#                 eval_pt_names=eval_pt_names,
#                 vortex_coords_names=vortex_coords_names,
#                 eval_pt_shapes=eval_pt_shapes,
#                 vortex_coords_shapes=vortex_coords_shapes,
#                 output_names=output_names,
#                 vc=True,
#                 symmetry=self.parameters['symmetry'],
#             )
#             self.add(m, name='aic_bd_w_seperate')

#         aic_shape = (num_nodes, aic_shape_row, aic_shape_col, 3)

#         # m1 = Model()
#         aic_col_w = self.create_output(full_aic_name, shape=aic_shape)
#         row = 0
#         col = 0
#         for i in range(len(bd_coll_pts_shapes)):
#             for j in range(len(wake_vortex_pts_shapes)):
#                 aic_i_shape = (
#                     num_nodes,
#                     bd_coll_pts_shapes[i][1] * bd_coll_pts_shapes[i][2] *
#                     (wake_vortex_pts_shapes[j][1] - 1) *
#                     (wake_vortex_pts_shapes[j][2] - 1),
#                     3,
#                 )
#                 # aic_i = self.declare_variable(
#                 #     output_names[i * (len(wake_vortex_pts_shapes)) + j],
#                 #     shape=aic_i_shape)

#                 delta_row = bd_coll_pts_shapes[i][1] * bd_coll_pts_shapes[i][2]
#                 delta_col = (wake_vortex_pts_shapes[j][1] -
#                              1) * (wake_vortex_pts_shapes[j][2] - 1)
                

#                 aic_i = self.declare_variable(
#                     output_names[i * (len(wake_vortex_pts_shapes)) + j],
#                     shape=aic_i_shape,val=0.)

#                 aic_col_w[:, row:row + delta_row,
#                           col:col + delta_col, :] = csdl.reshape(
#                               aic_i,
#                               new_shape=(num_nodes, delta_row, delta_col, 3))
#                 col = col + delta_col
#             col = 0
#             row = row + delta_row


# if __name__ == "__main__":

#     def generate_simple_mesh(nx, ny, n_wake_pts_chord=None):
#         if n_wake_pts_chord == None:
#             mesh = np.zeros((nx, ny, 3))
#             mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
#             mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
#             mesh[:, :, 2] = 0.
#         else:
#             mesh = np.zeros((n_wake_pts_chord, nx, ny, 3))
#             for i in range(n_wake_pts_chord):
#                 mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
#                 mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
#                 mesh[i, :, :, 2] = 0.
#         return mesh

#     bd_coll_pts_names = ['bd_coll_pts_1', 'bd_coll_pts_2']
#     wake_vortex_pts_names = ['wake_vortex_pts_1', 'wake_vortex_pts_2']
#     bd_coll_pts_shapes = [(2, 3, 3), (3, 2, 3)]
#     wake_vortex_pts_shapes = [(3, 3, 3), (3, 2, 3)]
#     # bd_coll_pts_shapes = [(2, 3, 3), (2, 3, 3)]
#     # wake_vortex_pts_shapes = [(3, 3, 3), (3, 3, 3)]

#     model_1 = Model()
#     bd_val = np.random.random((2, 3, 3))
#     bd_val_1 = np.random.random((3, 2, 3))
#     wake_vortex_val = np.random.random((3, 3, 3))
#     wake_vortex_val_1 = np.random.random((3, 2, 3))

#     # bd_val = np.random.random((2, 3, 3))
#     # bd_val_1 = np.random.random((2, 3, 3))
#     # wake_vortex_val = np.random.random((3, 3, 3))
#     # wake_vortex_val_1 = np.random.random((3, 3, 3))

#     bd = model_1.create_input('bd_coll_pts_1', val=bd_val)
#     bd_1 = model_1.create_input('bd_coll_pts_2', val=bd_val_1)

#     wake_vortex = model_1.create_input('wake_vortex_pts_1',
#                                        val=wake_vortex_val)
#     wake_vortex_1 = model_1.create_input('wake_vortex_pts_2',
#                                          val=wake_vortex_val_1)
#     model_1.add(AssembleAic(
#         bd_coll_pts_names=bd_coll_pts_names,
#         wake_vortex_pts_names=wake_vortex_pts_names,
#         bd_coll_pts_shapes=bd_coll_pts_shapes,
#         wake_vortex_pts_shapes=wake_vortex_pts_shapes,
#     ),
#                 name='assemble_aic_comp')
#     sim = Simulator(model_1)
#     sim.run()
#     # sim.visualize_implementation()
#     # print('aic is', sim['aic'])
#     # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])