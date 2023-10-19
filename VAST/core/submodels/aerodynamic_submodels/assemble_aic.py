# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
# from VAST.core.submodels.aerodynamic_submodels.biot_savart_vc_comp_org import BiotSavartComp
from VAST.core.submodels.aerodynamic_submodels.biot_savart_vc_comp import BiotSavartComp
# from VAST.core.submodels.aerodynamic_submodels.biot_savart_jax import BiotSavartComp

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
    
    def _adjust_inputs_for_symmetry(self, sym_struct_list, bd_coll_pts_names, wake_vortex_pts_names, eval_pt_names, vortex_coords_names, eval_pt_shapes, vortex_coords_shapes, output_names):
        eval_pts_names_new = []
        vortex_coords_names_new = []
        eval_pt_shapes_new = []
        vortex_coords_shapes_new = []
        output_names_new = []
        sym_type = []
        reflection_axes = []

        for i, sym_set in enumerate(sym_struct_list):
            len_sym_set = len(sym_set)
            init_surface = sym_set[0] # first surface in the symmetry combinations
            init_coll_surface = bd_coll_pts_names[init_surface]
            init_wake_vortex = wake_vortex_pts_names[init_surface]

            coll_surf_indices = [i for i in range(len(eval_pt_names)) if eval_pt_names[i] == init_coll_surface]
            print(init_coll_surface)
            # print(eval_pt_names)
            print(coll_surf_indices)
            # print([eval_pt_names[i] for i in coll_surf_indices])

            # adjusting the input names for the biot savart law computation 
            eval_pts_names_new.extend([eval_pt_names[i] for i in coll_surf_indices])
            vortex_coords_names_new.extend([vortex_coords_names[i] for i in coll_surf_indices])
            eval_pt_shapes_new.extend([eval_pt_shapes[i] for i in coll_surf_indices])
            vortex_coords_shapes_new.extend([vortex_coords_shapes[i] for i in coll_surf_indices])
            output_names_new.extend([output_names[i] for i in coll_surf_indices])

            print('====')
            for surf in sym_set:                
                print(bd_coll_pts_names[surf])
            print('====')

            if len_sym_set == 1: # single surface symmetric across y
                surface_sym = 'self'
                sym_type.extend([surface_sym] * len(coll_surf_indices))
                reflection_axes.extend(['y'] * len(coll_surf_indices))
            elif len_sym_set == 2:
                if 'image' in bd_coll_pts_names[sym_set[0]]:
                    print('image in 1')
                elif 'image' in bd_coll_pts_names[sym_set[1]]:
                    print('image in 2')

                else:
                    pass # case with no mirroring
                # need to check if image is in the name
                # if not, then it's just a y reflection
                # if image, then it is a yz reflection
            elif len_sym_set == 4:
                pass
            elif len_sym_set > 1: # multiple surfaces 
                surface_sym = 0
                # NEED TO KEEP TRACK OF SURFACE NAMES HERE
                add_surf = []
                for add_ind in sym_set[1:]:
                    add_surf.append(bd_coll_pts_names[add_ind])
                sym_type.extend([add_surf]*len(coll_surf_indices))
                if len_sym_set == 2:
                    pass
                elif len_sym_set == 4:
                    pass

                # need reflection axes here:
                # how to do it:
                #   - any name with "image" automatically has z
                #   - remaining has y reflection
                # parse name such that if "image" is removed, then 

        
        exit()

        return eval_pts_names_new, vortex_coords_names_new, eval_pt_shapes_new, vortex_coords_shapes_new, output_names_new, sym_type, reflection_axes

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
        # if self.parameters['symmetry'] and sym_struct_list is not None:


        if sub:
            print('====')
            print(bd_coll_pts_names)
            print(wake_vortex_pts_names)
            print('====')

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
            
            print(eval_pt_names_sub)
            print(vortex_coords_names_sub)
            print(output_names_sub)

            if self.parameters['symmetry'] and sym_struct_list is not None:
                new_sub = self._adjust_inputs_for_symmetry(
                    sym_struct_list,
                    bd_coll_pts_names,
                    wake_vortex_pts_names,
                    eval_pt_names_sub,
                    vortex_coords_names_sub,
                    eval_pt_shapes_sub,
                    vortex_coords_shapes_sub,
                    output_names_sub 
                )

                eval_pt_names_sub = new_sub[0]
                vortex_coords_names_sub = new_sub[1]
                eval_pt_shapes_sub = new_sub[2]
                vortex_coords_shapes_sub = new_sub[3]
                output_names_sub = new_sub[4]
                sym_type = new_sub[5]
                reflection_axes = new_sub[6]

            print('====')
            print(eval_pt_names_sub)
            print(vortex_coords_names_sub)
            print(output_names_sub)
            print(sym_type)
            print(reflection_axes)
            print("====")
            exit()
            m = BiotSavartComp(
                eval_pt_names=eval_pt_names_sub,
                vortex_coords_names=vortex_coords_names_sub,
                eval_pt_shapes=eval_pt_shapes_sub,
                vortex_coords_shapes=vortex_coords_shapes_sub,
                output_names=output_names_sub,
                vc=True,
                symmetry=self.parameters['symmetry'],
                sym_type=sym_type
            )
            self.add(m, name='aic_bd_w_seperate')

            sub_array = np.array([sub_eval_list,sub_induced_list])

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