def generate_symmetry_groups(sym_struct_list, bd_coll_pts_names, full_aic_name):

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
            if 'mirror' in bd_coll_pts_names[sym_set[0]]:
                # print('mirror in 1')
                1

            elif 'mirror' in bd_coll_pts_names[sym_set[1]]:
                # print('mirror in 2')
                sub_dict = {'ref': [sym_set[1]], 'axis': ['z']}
        
        elif len_sym_set == 4:
            ref_list = []
            axis_list = []

            ref_surf = sym_set[1:]
            mirror_in_surf = ['mirror' in bd_coll_pts_names[i] for i in ref_surf]
            for j, bool_val in enumerate(mirror_in_surf):
                if bool_val == False:
                    ref_list.append(ref_surf[j])
                    axis_list.append('y')
                elif bool_val == True:
                    ref_list.append(ref_surf[j])
                    ref_surface_name = bd_coll_pts_names[ref_surf[j]] 
                    del_ind = ref_surface_name.find('_mirror') # FINDING FIRST INDEX WITH _MIRROR
                    dummy_string = ref_surface_name.replace(ref_surface_name[del_ind:],'') # DELETING THAT PART OF THE STRING AND ON
                    # if dummy_string == bd_coll_pts_names[sym_set[0]]:
                    if dummy_string in bd_coll_pts_names[sym_set[0]]:
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


def adjust_biot_savart_inputs_for_symmetry(eval_pt_names, eval_pt_shapes, vortex_coords_names, vortex_coords_shapes, output_names, aic_names_dict):
    eval_pt_names_new = []
    eval_pt_shapes_new = []
    vortex_coords_names_new = []
    vortex_coords_shapes_new = []
    output_names_new = []
    # print(aic_names_dict)
    for key in aic_names_dict:
        if key in output_names:
            # print(key)
            name_ind = output_names.index(key)
            eval_pt_names_new.append(eval_pt_names[name_ind])
            eval_pt_shapes_new.append(eval_pt_shapes[name_ind])
            vortex_coords_names_new.append(vortex_coords_names[name_ind])
            vortex_coords_shapes_new.append(vortex_coords_shapes[name_ind])
            output_names_new.append(output_names[name_ind])
    
    return eval_pt_names_new, eval_pt_shapes_new, vortex_coords_names_new, vortex_coords_shapes_new, output_names_new

def modify_biot_savart_interactions(sub_eval_list, sub_induced_list, bd_coll_pts_names, bd_coll_pts_shapes, 
                                               wake_vortex_pts_names, wake_vortex_pts_shapes, full_aic_name):

    eval_pt_names_sub = []
    eval_pt_shapes_sub = []
    vortex_coords_names_sub = []
    vortex_coords_shapes_sub = []
    output_names_sub = []

    for i in range(len(sub_eval_list)):
        eval_pt_names_sub.append(bd_coll_pts_names[sub_eval_list[i]])
        eval_pt_shapes_sub.append(bd_coll_pts_shapes[sub_eval_list[i]])
        vortex_coords_names_sub.append(wake_vortex_pts_names[sub_induced_list[i]])
        vortex_coords_shapes_sub.append(wake_vortex_pts_shapes[sub_induced_list[i]])
        output_name_sub = full_aic_name  +'_'+ str(sub_eval_list[i]) +'_'+ str(sub_induced_list[i])
        output_names_sub.append(output_name_sub)

    return eval_pt_names_sub, eval_pt_shapes_sub, vortex_coords_names_sub, vortex_coords_shapes_sub, output_names_sub