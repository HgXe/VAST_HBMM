import csdl
import numpy as np
from VAST.utils.custom_array_indexing import MatrixIndexing
from VAST.utils.custom_find_zeros_replace_eps import ReplaceZeros
# from VAST.utils.custom_einsums import EinsumKijKijKi
# from VAST.utils.custom_expands import ExpandIjkIjlk
# from VAST.utils.custom_expands import ExpandIjkIljk
from scipy.sparse import csc_array
from scipy.sparse import eye as sparse_eye

class BiotSavartComp(csdl.Model):
    """
    Compute AIC.

    parameters
    ----------
    eval_pts[num_nodes,nc, ns, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface that the 
        AIC matrix is computed on.
    vortex_coords[num_nodes,nc_v, ns_v, 3] : numpy array
        Array defining the nodal coordinates of background mesh that induces
        the AIC.

    Returns
    -------
    AIC[nc*ns*(nc_v-1)*(ns_v-1), nc*ns*(nc_v-1)*(ns_v-1), 3] : numpy array
        Aerodynamic influence coeffients (can be interprete as induced
        velocities given circulations=1)
    2023-06-13: 
        need to check the node order and the bound vector against OAS
    """
    def initialize(self):
        # evaluation points names and shapes
        self.parameters.declare('eval_pt_names', types=list)
        self.parameters.declare('eval_pt_shapes', types=list)
        # induced background mesh names and shapes
        self.parameters.declare('vortex_coords_names', types=list)
        self.parameters.declare('vortex_coords_shapes', types=list)
        # output aic names
        self.parameters.declare('output_names', types=list)
        # whether to enable the fixed vortex core model
        self.parameters.declare('vc', default=False)
        self.parameters.declare('eps', default=5e-4)

        self.parameters.declare('circulation_names', default=None)
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('aic_symmetry_dict', default=False)

    def define(self):
        eval_pt_names = self.parameters['eval_pt_names']
        eval_pt_shapes = self.parameters['eval_pt_shapes']
        vortex_coords_names = self.parameters['vortex_coords_names']
        vortex_coords_shapes = self.parameters['vortex_coords_shapes']
        output_names = self.parameters['output_names']
        vc = self.parameters['vc']
        eps = self.parameters['eps']
        # circulation_names = self.parameters['circulation_names']
        symmetry = self.parameters['symmetry']
        aic_symmetry_dict = self.parameters['aic_symmetry_dict']
        # print('symmetry is---------------------------------------------', symmetry)
        # print(eval_pt_names)
        # print(vortex_coords_names)
        # exit()
        for i in range(len(eval_pt_names)):
            # input_names
            eval_pt_name = eval_pt_names[i]
            vortex_coords_name = vortex_coords_names[i]
            # output_name
            output_name = output_names[i]
            # print('output name: ', output_name)
            # input_shapes
            eval_pt_shape = eval_pt_shapes[i]
            vortex_coords_shape = vortex_coords_shapes[i]
            # declare_inputs
            eval_pts = self.declare_variable(eval_pt_name, shape=eval_pt_shape)
            vortex_coords = self.declare_variable(vortex_coords_name, shape=vortex_coords_shape)

            # define panel points (evaluation dependent on MESH ORDERING)
            #                  C -----> D
            # ---v_inf-(x)-->  ^        |
            #                  |        v
            #                  B <----- A
            A = vortex_coords[:,1:, :vortex_coords_shape[2] - 1, :]
            B = vortex_coords[:,:vortex_coords_shape[1] -
                              1, :vortex_coords_shape[2] - 1, :]
            C = vortex_coords[:,:vortex_coords_shape[1] - 1, 1:, :]
            D = vortex_coords[:,1:, 1:, :]


            if symmetry == False:
                self.r_A, self.r_A_norm = self.__compute_expand_vecs(eval_pts, A, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'A')
                self.r_B, self.r_B_norm = self.__compute_expand_vecs(eval_pts, B, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'B')
                self.r_C, self.r_C_norm = self.__compute_expand_vecs(eval_pts, C, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'C')
                self.r_D, self.r_D_norm = self.__compute_expand_vecs(eval_pts, D, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'D')
                v_ab = self._induced_vel_line(self.r_A, self.r_B, self.r_A_norm, self.r_B_norm,'AB')
                v_bc = self._induced_vel_line(self.r_B, self.r_C, self.r_B_norm, self.r_C_norm,'BC')
                v_cd = self._induced_vel_line(self.r_C, self.r_D, self.r_C_norm, self.r_D_norm,'CD')
                v_da = self._induced_vel_line(self.r_D, self.r_A, self.r_D_norm, self.r_A_norm,'DA')

                AIC = v_ab + v_bc + v_cd + v_da    
                # if 'aic_bd_' in output_name:
                #     self.print_var(AIC) 
            
            else:
                nx = eval_pt_shape[1]
                ny = eval_pt_shape[2]
                if aic_symmetry_dict == False: # original symmetry case
                    
                    self.r_A, self.r_A_norm = self.__compute_expand_vecs(eval_pts[:,:,:int(ny/2),:], A, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'A')
                    self.r_B, self.r_B_norm = self.__compute_expand_vecs(eval_pts[:,:,:int(ny/2),:], B, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'B')
                    self.r_C, self.r_C_norm = self.__compute_expand_vecs(eval_pts[:,:,:int(ny/2),:], C, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'C')
                    self.r_D, self.r_D_norm = self.__compute_expand_vecs(eval_pts[:,:,:int(ny/2),:], D, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'D')
                    
                    v_ab = self._induced_vel_line(self.r_A, self.r_B, self.r_A_norm, self.r_B_norm,'AB')
                    v_bc = self._induced_vel_line(self.r_B, self.r_C, self.r_B_norm, self.r_C_norm,'BC')
                    v_cd = self._induced_vel_line(self.r_C, self.r_D, self.r_C_norm, self.r_D_norm,'CD')
                    v_da = self._induced_vel_line(self.r_D, self.r_A, self.r_D_norm, self.r_A_norm,'DA')

                    AIC_half = v_ab + v_bc + v_cd + v_da      
                    AIC = csdl.custom(AIC_half, op = SymmetryFlip(in_name=AIC_half.name, eval_pt_shape=eval_pt_shape, vortex_coords_shape=vortex_coords_shape, out_name=output_name))
                
                else:
                    # FIRST COMPUTE STANDARD METHOD FOR APPROPRIATE SURFACE
                    self.r_A, self.r_A_norm = self.__compute_expand_vecs(eval_pts, A, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'A')
                    self.r_B, self.r_B_norm = self.__compute_expand_vecs(eval_pts, B, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'B')
                    self.r_C, self.r_C_norm = self.__compute_expand_vecs(eval_pts, C, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'C')
                    self.r_D, self.r_D_norm = self.__compute_expand_vecs(eval_pts, D, vortex_coords_shape,eval_pt_name,vortex_coords_name,output_name,'D')
                    v_ab = self._induced_vel_line(self.r_A, self.r_B, self.r_A_norm, self.r_B_norm,'AB')
                    v_bc = self._induced_vel_line(self.r_B, self.r_C, self.r_B_norm, self.r_C_norm,'BC')
                    v_cd = self._induced_vel_line(self.r_C, self.r_D, self.r_C_norm, self.r_D_norm,'CD')
                    v_da = self._induced_vel_line(self.r_D, self.r_A, self.r_D_norm, self.r_A_norm,'DA')

                    AIC = v_ab + v_bc + v_cd + v_da
                    # if 'aic_bd_' in output_name:
                    #     self.print_var(AIC) 

                    symmetry_names, symmetry_axes = aic_symmetry_dict[output_name]['names'], aic_symmetry_dict[output_name]['axis']
                    ref_axis = aic_symmetry_dict[output_name]['ref_axis']
                    AIC_shape = AIC.shape # num_nodes, total number of interactions, 3 coordinates (x,y,z)
                    AIC_inner_shape = np.prod(AIC_shape[1:]) # total size for each timestep
                    AIC_vectorized = self.create_output(f'{output_name}_vec', shape=(AIC_shape[0]*AIC_shape[1]*AIC_shape[2],))

                    for n in range(AIC_shape[0]):
                        for m in range(AIC.shape[2]): # always 3 b/c 3rd dimension
                            AIC_vectorized[n*AIC_inner_shape + m*AIC.shape[1]:n*AIC_inner_shape + (m+1)*AIC.shape[1]] = csdl.reshape(AIC[n,:,m], new_shape=(AIC.shape[1],))

                    # print(output_name)
                    for j, name in enumerate(symmetry_names):
                        axis = symmetry_axes[j]  
                        # print(name)
                        # AIC_reflected = csdl.custom(AIC, op=AICReflection(in_name=AIC.name, eval_pt_shape=eval_pt_shape, vortex_coords_shape=vortex_coords_shape, out_name=output_name, axis=axis, ref_axis=ref_axis))
                        plot=False
                        # if name == 'aic_bd_2_0':
                        #     plot=True
                        # if 'bd' in name:
                        #     print(name)
                        #     plot=True

                        AIC_reflected_vectorized = csdl.custom(AIC_vectorized, op=AICReflection(in_name=AIC_vectorized.name, AIC_shape=AIC.shape, out_name=name+'_vec', axis=axis, ref_axis=ref_axis, eval_pt_shape=eval_pt_shape, vortex_coords_shape=vortex_coords_shape,plot=plot))
                        AIC_reflected = self.create_output(name, shape=AIC.shape)
                        for k in range(AIC.shape[2]): # always 3 b/c 3rd dimension
                            AIC_reflected[0,:,k] = csdl.reshape(AIC_reflected_vectorized[k*AIC.shape[1]:(k+1)*AIC.shape[1]], new_shape=(1,AIC.shape[1],1))
                        # if 'aic_bd_' in name:
                        #     self.print_var(AIC_reflected) 

            self.register_output(output_name, AIC)

    def __compute_expand_vecs(self, eval_pts, p_1, vortex_coords_shape, eval_pt_name, vortex_coords_name, output_name, point_name):

        vc = self.parameters['vc']
        num_nodes = eval_pts.shape[0]
        name = eval_pt_name + vortex_coords_name + output_name + point_name

        # 1 -> 2 eval_pts(num_pts_x,num_pts_y, 3)
        # v_induced_line shape=(num_panel_x*num_panel_y, num_panel_x, num_panel_y, 3)
        num_repeat_eval = p_1.shape[1] * p_1.shape[2]
        num_repeat_p = eval_pts.shape[1] * eval_pts.shape[2]

        eval_pts_expand = csdl.reshape(
            csdl.expand(
                csdl.reshape(eval_pts,
                             new_shape=(num_nodes, (eval_pts.shape[1] *
                                                    eval_pts.shape[2]), 3)),
                (num_nodes, eval_pts.shape[1] * eval_pts.shape[2],
                 num_repeat_eval, 3),
                'lij->likj',
            ),
            new_shape=(num_nodes,
                       eval_pts.shape[1] * eval_pts.shape[2] * num_repeat_eval,
                       3))

        p_1_expand = csdl.reshape(\
            csdl.expand(
                csdl.reshape(p_1,
                         new_shape=(num_nodes, (p_1.shape[1] * p_1.shape[2]),
                                    3)),
            (num_nodes, num_repeat_p, p_1.shape[1] * p_1.shape[2], 3),
            'lij->lkij'),
                        new_shape=(num_nodes,
                                    p_1.shape[1] *p_1.shape[2] * num_repeat_p,
                                    3))

        r1 = eval_pts_expand - p_1_expand
        self.register_output(r1.name + '_test_output', r1)
        # r1_norm = csdl.pnorm(r1, axis=(2,))
        r1_norm = (csdl.sum(r1**2, axes=(2,)) + 1E-12)**0.5
        # r1_norm = csdl.sum(r1**2+1e-2, axes=(2,))**0.5 # TODO: make pnorm work
        # r1_sum = csdl.sum(r1**2, axes=(2,))
        # r1_norm = csdl.custom(r1_sum, op=PosSqrt(in_name=r1_sum.name, out_name=r1_sum.name + '_norm', shape=r1_sum.shape))
        return r1, r1_norm


    def _induced_vel_line(self, r_1, r_2, r_1_norm, r_2_norm,line_name):

        vc = self.parameters['vc']
        # print('vc is--------------------', vc)

        num_nodes = r_1.shape[0]

        # 1 -> 2 eval_pts(num_pts_x,num_pts_y, 3)

        # the denominator of the induced velocity equation
        # shape = (num_nodes,num_panel_x*num_panel_y, num_panel_x* num_panel_y, 3)
        one_over_den = 1 / (np.pi * 4) * csdl.cross(r_1, r_2, axis=2)

        if vc == False:
            dor_r1_r2 = csdl.sum(r_1*r_2,axes=(2,))
            num = (1/(r_1_norm * r_2_norm + dor_r1_r2)) * (1/r_1_norm + 1/r_2_norm)
            # print('the shape of num is', num.shape)
            num_expand = csdl.expand(num, (num_nodes, num.shape[1], 3), 'ij->ijl')
            # num_expand = jnp.einsum('ij,l->ijl', num, jnp.ones(3))
            v_induced_line = num_expand * one_over_den
        else:
            new_vc=True
            if new_vc:
                # core_size = 0.05
                core_size = 0.2
                dor_r1_r2 = csdl.sum(r_1*r_2,axes=(2,))
                r1s = r_1_norm**2
                r2s = r_2_norm**2
                eps_s = core_size**2
                # print('shapes in biot-savart law cross',csdl.cross(r_1, r_2, axis=2).shape)
                # print('shapes in biot-savart law r1s',r1s.shape)
                # print('shapes in biot-savart law dor_r1_r2',dor_r1_r2.shape)
                # print('shapes in biot-savart law r_1_norm',r_1_norm.shape)
                # f1 = ( (r1s - dor_r1_r2)/((r1s + eps_s + 1e-10)**0.5) + (r2s - dor_r1_r2)/((r2s + eps_s + 1e-10) **0.5) )/(r1s*r2s - dor_r1_r2**2 + eps_s*(r1s + r2s - 2*r_1_norm*r_2_norm) + 1e-10)
                f1 = ( (r1s - dor_r1_r2)/((r1s + eps_s)**0.5) + (r2s - dor_r1_r2)/((r2s + eps_s)**0.5) )/(r1s*r2s - dor_r1_r2**2 + eps_s*(r1s + r2s - 2*r_1_norm*r_2_norm) + 1e-10)
                f2 = one_over_den
                v_induced_line = csdl.expand(f1,(f2.shape),'ij->ijk') * f2 
                # self.print_var(v_induced_line)

                # finite core from BYU VortexLattice approach:
                # if finite_core
                #     rdot = dot(r1, r2)
                #     r1s, r2s, εs = nr1^2, nr2^2, core_size^2
                #     f1 = cross(r1, r2)/(r1s*r2s - rdot^2 + εs*(r1s + r2s - 2*nr1*nr2))
                #     f2 = (r1s - rdot)/sqrt(r1s + εs) + (r2s - rdot)/sqrt(r2s + εs)

                # Vhat = (f1*f2)/(4*pi)


            else:
                # this should be moved out instead of being in here, this is only used for dynamic case to compute the wake induced velocity indead
                dor_r1_r2 = csdl.sum(r_1*r_2,axes=(2,))
                dino = (r_1_norm * r_2_norm + dor_r1_r2)
                # deal with (r_1_norm * r_2_norm + dor_r1_r2) first
                # dino_non_singular = csdl.custom(dino, op=ReplaceZeros(in_name=dino.name,
                #                                                       in_shape=dino.shape,
                #                                                       out_name=dino.name + '_non_singular'))
                dino_non_singular = dino + 1e-4

                # num = (1/dino_non_singular) * (1/r_1_norm + 1/r_2_norm)
                num = (1/dino_non_singular) * (1/(r_1_norm+1e-3) + 1/(r_2_norm+1e-3))
                
                # print('the name of num is', num.name)
                self.register_output('num'+num.name, num)
                # print('the name of num is', num.name)
                # print('the shape of num is', num.shape)
                num_expand = csdl.expand(num, (num_nodes, num.shape[1], 3), 'ij->ijl')
                # num_expand = jnp.einsum('ij,l->ijl', num, jnp.ones(3))
                v_induced_line = num_expand * one_over_den
                # self.print_var(v_induced_line)
                # '''

        return v_induced_line

class PosSqrt(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('in_name')
        self.parameters.declare('out_name')
        self.parameters.declare('shape')
    def define(self):
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']
        shape = self.parameters['shape']
        self.add_input(in_name, shape=shape)
        self.add_output(out_name, shape=shape)
        self.declare_derivatives(out_name, in_name)
    def compute(self, inputs, outputs):
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']
        outputs[out_name] = inputs[in_name]**0.5
    def compute_derivatives(self, inputs, derivatives):
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']
        input = inputs[in_name]
        input[input < 1e-3] = 1e-3
        derivative = np.zeros((input.shape[0], input.shape[1], input.shape[0], input.shape[1]))
        for i in range(input.shape[0]):
            derivative[i,:,i,:] = np.diag(1/2 * input[i,:] ** (-1/2))
        derivatives[out_name, in_name] = derivative


class SymmetryFlip(csdl.CustomExplicitOperation):
    """
    Compute the whole AIC matrix given half of it

    parameters
    ----------
    <aic_half_names>[nc*ns*(nc_v-1)*(ns_v-1)* nc*ns*(nc_v-1)*(ns_v-1)/2, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface that the 
        AIC matrix is computed on.

    Returns
    -------
    <aic_names>[nc*ns*(nc_v-1)*(ns_v-1), nc*ns*(nc_v-1)*(ns_v-1), 3] : numpy array
        Aerodynamic influence coeffients (can be interprete as induced
        velocities given circulations=1)
    """    
    def initialize(self):
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('eval_pt_shape', types=tuple)
        self.parameters.declare('vortex_coords_shape', types=tuple)
        self.parameters.declare('out_name', types=str)
    def define(self):
        eval_pt_shape =self.eval_pt_shape= self.parameters['eval_pt_shape']
        vortex_coords_shape =self.vortex_coords_shape =  self.parameters['vortex_coords_shape']
        shape = eval_pt_shape[1]*eval_pt_shape[2]*(vortex_coords_shape[1]-1)*(vortex_coords_shape[2]-1)
        num_nodes = eval_pt_shape[0]
        self.add_input(self.parameters['in_name'],shape=(num_nodes,int(shape/2),3))
        self.add_output(self.parameters['out_name'],shape=(num_nodes,int(shape),3))

        self.full_aic_func = self.__get_full_aic_jax
        row_indices  = np.arange(int(num_nodes*shape*3))
        col_ind_temp = np.arange(int(num_nodes*shape/2*3)).reshape(num_nodes,eval_pt_shape[1],int(eval_pt_shape[2]/2),int(vortex_coords_shape[1]-1),int(vortex_coords_shape[2]-1),3)
        
        col_ind_flip = np.flip(col_ind_temp,axis=(2,4))
        col_indices = np.concatenate((col_ind_temp,col_ind_flip),axis=2).flatten()
        self.declare_derivatives(self.parameters['out_name'], self.parameters['in_name'],rows=row_indices,cols=col_indices,val=np.ones(row_indices.size, dtype=int))

    def compute(self, inputs, outputs):
        outputs[self.parameters['out_name']] = self.full_aic_func(inputs[self.parameters['in_name']]).reshape(1,-1,3)

    def __get_full_aic_jax(self,half_aic):
        nx_panel = self.eval_pt_shape[1]
        ny_panel = self.eval_pt_shape[2] 
        nx_panel_ind = self.vortex_coords_shape[1] - 1
        ny_panel_ind = self.vortex_coords_shape[2] - 1 
        num_nodes = self.eval_pt_shape[0]
        half_reshaped = half_aic.reshape(num_nodes,nx_panel,int(ny_panel/2),nx_panel_ind, ny_panel_ind, 3)
        # half_aic.reshape(nx_panel,int(ny_panel/2),nx_panel_ind, ny_panel_ind, 3)
        other_half_aic = np.flip(half_reshaped, (2,4))
        full_aic = np.concatenate((half_reshaped, other_half_aic), axis=2).reshape(num_nodes,-1,3)
        return full_aic

class AICReflection(csdl.CustomExplicitOperation):
    '''
    Compute the AIC matrix of the reflected component based on the symmetry axis.
    '''
    def initialize(self):
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('eval_pt_shape', types=tuple)
        self.parameters.declare('vortex_coords_shape', types=tuple)
        self.parameters.declare('AIC_shape', types=tuple)
        self.parameters.declare('out_name', types=str)
        self.parameters.declare('axis', types=str)
        self.parameters.declare('ref_axis', types=str)
        self.parameters.declare('plot', default=False)

    def define(self):
        self.AIC_shape = AIC_shape = self.parameters['AIC_shape']
        self.axis = axis = self.parameters['axis']
        self.ref_axis = ref_axis = self.parameters['ref_axis']
        num_row = AIC_shape[1]
        num_col = AIC_shape[2]
        eval_pt_shape = self.eval_pt_shape = self.parameters['eval_pt_shape']
        vortex_coords_shape = self.vortex_coords_shape = self.parameters['vortex_coords_shape']
        shape = eval_pt_shape[1]*eval_pt_shape[2]*(vortex_coords_shape[1]-1)*(vortex_coords_shape[2]-1)
        num_nodes = eval_pt_shape[0]

        vector_length = num_row*num_col
        self.add_input(self.parameters['in_name'], shape=(vector_length,))
        self.add_output(self.parameters['out_name'], shape=(vector_length,))

        # self.system_matrix = self.create_system_matrix(axis=axis, ref_axis=ref_axis, vector_length=vector_length)
        self.system_matrix = self.create_system_matrix_new(axis=axis, vector_length=vector_length)
        
        if self.parameters['plot'] == True:
            data_to_print = self.system_matrix.nonzero()
            data_rows, data_cols = data_to_print[0], data_to_print[1]
            for i in range(len(data_rows)):
                print(data_rows[i], data_cols[i], self.system_matrix[data_rows[i], data_cols[i]])
            import matplotlib.pyplot as plt
            plt.spy(self.system_matrix[:int(vector_length/3), :int(vector_length/3)])
            plt.show()
            # exit()

        # self.declare_derivatives(self.parameters['out_name'], self.parameters['in_name'],rows=row_indices,cols=col_indices,val=np.ones(row_indices.size))
        self.declare_derivatives(self.parameters['out_name'], self.parameters['in_name']) # actual derivative value given in compute_derivatives

    def compute(self, inputs, outputs):
        # outputs[self.parameters['out_name']] = np.matmul(self.system_matrix, inputs[self.parameters['in_name']])
        outputs[self.parameters['out_name']] = self.system_matrix.dot(inputs[self.parameters['in_name']])
    
    def compute_derivatives(self, inputs, derivatives):
        derivatives[self.parameters['out_name'], self.parameters['in_name']] = self.system_matrix # can be dense or sparse

    def generate_permutation_matrix_new(self, vector_length, axis):
        asdf = int(vector_length/3)
        # size of matrix for one dimension (x, y, or z) = a*b*c*d
        a = self.eval_pt_shape[1] # num chordwise collocation evaluation points
        b = self.eval_pt_shape[2] # num spanwise collocation evaluation points
        c = self.vortex_coords_shape[1]-1# num chordwise vortex coord points
        d = self.vortex_coords_shape[2]-1 # num spanwise vortex coord points
        # rows will not change
        # columns will be adjusted to maintain symmetry
        inner_cols_orig = np.arange(b*c*d,0,-1) - 1 # decreasing from (b*c*d-1) to 0
        inner_cols = np.zeros_like(inner_cols_orig).tolist() # adding elements to this one

        for i in range(b):
            vals = inner_cols_orig[(c*d)*i:(c*d)*(i+1)]
            for j in range(c):
                inner_cols[((c*d)*i + d*j):((c*d)*i + d*(j+1))] = vals[((d*(c-j-1))):(d*(c-j-1)+d)] # shifting first half back

        cols = inner_cols.copy()
        for i in range(a-1):
            shifted_cols = (np.array(inner_cols) + (i+1)*b*c*d).tolist()
            cols.extend(shifted_cols)

        num_per_axis = a*b*c*d
        for i in range(2):
            shifted_cols = (np.array(cols[:num_per_axis]) + (i+1)*num_per_axis).tolist()
            cols.extend(shifted_cols)

        cols = np.array(cols) # converting columns back to numpy array
        rows = np.arange(3*a*b*c*d)
        data = np.ones_like(cols)
        if axis == 'y':
            data[asdf:2*asdf] *= -1 # only on y
        else: # 'yz
            data[:asdf] *= -1 # only on x
        sparse_system_matrix = csc_array((data, (rows, cols)))
        return sparse_system_matrix

    def generate_permutation_matrix(self, vector_length, axis):
        asdf = int(vector_length/3)
        system_matrix = np.zeros((vector_length, vector_length))
        inner_inner_inner_sub_matrix = np.flipud(np.eye(self.vortex_coords_shape[2]-1)) # Identity-turned anti-diagonal matrix
        iiis_shape = inner_inner_inner_sub_matrix.shape

        inner_inner_sub_matrix = np.zeros(((self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1),(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1)))
        iis_shape =  inner_inner_sub_matrix.shape
        for i in range(self.vortex_coords_shape[1]-1):
            inner_inner_sub_matrix[i*(self.vortex_coords_shape[2]-1):(i+1)*(self.vortex_coords_shape[2]-1), i*(self.vortex_coords_shape[2]-1):(i+1)*(self.vortex_coords_shape[2]-1)] = inner_inner_inner_sub_matrix # inserting matrices along diagonals
        
        inner_sub_matrix = np.zeros((self.eval_pt_shape[2]*(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1), self.eval_pt_shape[2]*(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1)))
        is_shape = inner_sub_matrix.shape

        for i in range(self.eval_pt_shape[2]): # number of eval points in y-direction (self.eval_pt_shape[2])
            inner_sub_matrix[i*iis_shape[0]:(i+1)*iis_shape[0], (self.eval_pt_shape[2]-(i+1))*iis_shape[0]:(self.eval_pt_shape[2]-i)*iis_shape[0]] = inner_inner_sub_matrix
        # assembling along anti diagonal
        sub_matrix = np.zeros((asdf, asdf)) 
        for i in range(self.eval_pt_shape[1]): # number of eval points in x-direction (self.eval_pt_shape[1]):
            sub_matrix[i*is_shape[0]:(i+1)*is_shape[0], i*is_shape[0]:(i+1)*is_shape[0]] = inner_sub_matrix
        
        system_matrix[:asdf, :asdf] = sub_matrix
        system_matrix[asdf:2*asdf, asdf:2*asdf] = sub_matrix
        system_matrix[2*asdf:, 2*asdf:] = sub_matrix
        if axis == 'y':
            system_matrix[asdf:2*asdf, asdf:2*asdf] *= -1. # only on y
        else: # 'yz
            system_matrix[:asdf, :asdf] *= -1. # only on x
        return system_matrix

    def create_system_matrix_new(self, axis, vector_length):
        asdf = int(vector_length/3)
        rows, cols = np.arange(vector_length), np.arange(vector_length)
        data = np.ones((vector_length, ), dtype=int)

        if axis == 'y':
            data[:asdf] *= -1 # x changes sign
            data[2*asdf:] *= -1 # z changes sign
        elif axis == 'z':
            data[:2*asdf] *= -1 # x & y change sign
        elif axis == 'yz':
            data[asdf:] *= -1 # y & z change sign

        sparse_system_matrix = csc_array((data, (rows, cols)))
        return sparse_system_matrix

    def create_system_matrix(self, axis, ref_axis, vector_length):
        asdf = int(vector_length/3)
        if ref_axis == 'self':
            # sparse_system_matrix = csc_array(sparse_eye(vector_length))
            rows, cols = np.arange(vector_length), np.arange(vector_length)
            data = np.ones((vector_length, ), dtype=int)
            if 'z' in axis:
                data[:2*asdf]  *= -1
            sparse_system_matrix = csc_array((data, (rows, cols)))
        elif ref_axis == 'plane':
            if axis == 'z':
                rows, cols = np.arange(vector_length), np.arange(vector_length)
                data = np.ones((vector_length, ), dtype=int)
                if 'z' in axis:
                    data[:2*asdf]  *= -1
                sparse_system_matrix = csc_array((data, (rows, cols)))
            else:
                # system_matrix = self.generate_permutation_matrix(vector_length, axis)
                # sparse_system_matrix = csc_array(system_matrix)
                # del system_matrix
                sparse_system_matrix = self.generate_permutation_matrix_new(vector_length, axis)
                # system_matrix = np.zeros((vector_length, vector_length))
                    
                # inner_inner_inner_sub_matrix = np.flipud(np.eye(self.vortex_coords_shape[2]-1)) # Identity-turned anti-diagonal matrix
                # iiis_shape = inner_inner_inner_sub_matrix.shape

                # inner_inner_sub_matrix = np.zeros(((self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1),(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1)))
                # iis_shape =  inner_inner_sub_matrix.shape
                # for i in range(self.vortex_coords_shape[1]-1):
                #     inner_inner_sub_matrix[i*(self.vortex_coords_shape[2]-1):(i+1)*(self.vortex_coords_shape[2]-1), i*(self.vortex_coords_shape[2]-1):(i+1)*(self.vortex_coords_shape[2]-1)] = inner_inner_inner_sub_matrix # inserting matrices along diagonals
                
                # inner_sub_matrix = np.zeros((self.eval_pt_shape[2]*(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1), self.eval_pt_shape[2]*(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1)))
                # is_shape = inner_sub_matrix.shape

                # for i in range(self.eval_pt_shape[2]): # number of eval points in y-direction (self.eval_pt_shape[2])
                #     inner_sub_matrix[i*iis_shape[0]:(i+1)*iis_shape[0], (self.eval_pt_shape[2]-(i+1))*iis_shape[0]:(self.eval_pt_shape[2]-i)*iis_shape[0]] = inner_inner_sub_matrix
                # # assembling along anti diagonal
                # sub_matrix = np.zeros((asdf, asdf)) 
                # for i in range(self.eval_pt_shape[1]): # number of eval points in x-direction (self.eval_pt_shape[1]):
                #     sub_matrix[i*is_shape[0]:(i+1)*is_shape[0], i*is_shape[0]:(i+1)*is_shape[0]] = inner_sub_matrix
                
                # system_matrix[:asdf, :asdf] = sub_matrix
                # system_matrix[asdf:2*asdf, asdf:2*asdf] = sub_matrix
                # system_matrix[2*asdf:, 2*asdf:] = sub_matrix
                # if axis == 'y':
                #     system_matrix[asdf:2*asdf, asdf:2*asdf] *= -1. # only on y
                # else: # 'yz
                #     system_matrix[:asdf, :asdf] *= -1. # only on x
                    
        elif ref_axis == 'z':
            if axis == 'z':
                rows, cols = np.arange(vector_length), np.arange(vector_length)
                data = np.ones((vector_length, ), dtype=int)
                if 'z' in axis:
                    data[:2*asdf]  *= -1
                sparse_system_matrix = csc_array((data, (rows, cols)))
            else:
                # system_matrix = self.generate_permutation_matrix(vector_length, axis)
                # sparse_system_matrix = csc_array(system_matrix)
                # del system_matrix
                sparse_system_matrix = self.generate_permutation_matrix_new(vector_length, axis)
                # system_matrix = np.zeros((vector_length, vector_length))
                    
                # inner_inner_inner_sub_matrix = np.flipud(np.eye(self.vortex_coords_shape[2]-1)) # Identity-turned anti-diagonal matrix
                # iiis_shape = inner_inner_inner_sub_matrix.shape

                # inner_inner_sub_matrix = np.zeros(((self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1),(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1)))
                # iis_shape =  inner_inner_sub_matrix.shape
                # for i in range(self.vortex_coords_shape[1]-1):
                #     inner_inner_sub_matrix[i*(self.vortex_coords_shape[2]-1):(i+1)*(self.vortex_coords_shape[2]-1), i*(self.vortex_coords_shape[2]-1):(i+1)*(self.vortex_coords_shape[2]-1)] = inner_inner_inner_sub_matrix # inserting matrices along diagonals
                
                # inner_sub_matrix = np.zeros((self.eval_pt_shape[2]*(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1), self.eval_pt_shape[2]*(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1)))
                # is_shape = inner_sub_matrix.shape

                # for i in range(self.eval_pt_shape[2]): # number of eval points in y-direction (self.eval_pt_shape[2])
                #     inner_sub_matrix[i*iis_shape[0]:(i+1)*iis_shape[0], (self.eval_pt_shape[2]-(i+1))*iis_shape[0]:(self.eval_pt_shape[2]-i)*iis_shape[0]] = inner_inner_sub_matrix
                # # assembling along anti diagonal
                # sub_matrix = np.zeros((asdf, asdf)) 
                # for i in range(self.eval_pt_shape[1]): # number of eval points in x-direction (self.eval_pt_shape[1]):
                #     sub_matrix[i*is_shape[0]:(i+1)*is_shape[0], i*is_shape[0]:(i+1)*is_shape[0]] = inner_sub_matrix
                
                # system_matrix[:asdf, :asdf] = sub_matrix
                # system_matrix[asdf:2*asdf, asdf:2*asdf] = sub_matrix
                # system_matrix[2*asdf:, 2*asdf:] = sub_matrix
                # if axis == 'y':
                #     system_matrix[asdf:2*asdf, asdf:2*asdf] *= -1. # only on y
                # else: # 'yz
                #     system_matrix[:asdf, :asdf] *= -1. # only on x

        elif 'y' in ref_axis:
            if axis == 'z':
                rows, cols = np.arange(vector_length), np.arange(vector_length)
                data = np.ones((vector_length, ), dtype=int)
                if 'z' in axis:
                    data[:2*asdf]  *= -1
                sparse_system_matrix = csc_array((data, (rows, cols)))
            else:
                # system_matrix = self.generate_permutation_matrix(vector_length, axis)
                # sparse_system_matrix = csc_array(system_matrix)
                # del system_matrix
                sparse_system_matrix = self.generate_permutation_matrix_new(vector_length, axis)
                # system_matrix = np.zeros((vector_length, vector_length))
                
                # inner_inner_inner_sub_matrix = np.flipud(np.eye(self.vortex_coords_shape[2]-1)) # Identity-turned anti-diagonal matrix
                # iiis_shape = inner_inner_inner_sub_matrix.shape

                # inner_inner_sub_matrix = np.zeros(((self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1),(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1)))
                # iis_shape =  inner_inner_sub_matrix.shape
                # for i in range(self.vortex_coords_shape[1]-1):
                #     inner_inner_sub_matrix[i*(self.vortex_coords_shape[2]-1):(i+1)*(self.vortex_coords_shape[2]-1), i*(self.vortex_coords_shape[2]-1):(i+1)*(self.vortex_coords_shape[2]-1)] = inner_inner_inner_sub_matrix # inserting matrices along diagonals
                
                # inner_sub_matrix = np.zeros((self.eval_pt_shape[2]*(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1), self.eval_pt_shape[2]*(self.vortex_coords_shape[1]-1)*(self.vortex_coords_shape[2]-1)))
                # is_shape = inner_sub_matrix.shape

                # for i in range(self.eval_pt_shape[2]): # number of eval points in y-direction (self.eval_pt_shape[2])
                #     inner_sub_matrix[i*iis_shape[0]:(i+1)*iis_shape[0], (self.eval_pt_shape[2]-(i+1))*iis_shape[0]:(self.eval_pt_shape[2]-i)*iis_shape[0]] = inner_inner_sub_matrix
                # # assembling along anti diagonal
                # sub_matrix = np.zeros((asdf, asdf)) 
                # for i in range(self.eval_pt_shape[1]): # number of eval points in x-direction (self.eval_pt_shape[1]):
                #     sub_matrix[i*is_shape[0]:(i+1)*is_shape[0], i*is_shape[0]:(i+1)*is_shape[0]] = inner_sub_matrix
                
                # system_matrix[:asdf, :asdf] = sub_matrix
                # system_matrix[asdf:2*asdf, asdf:2*asdf] = sub_matrix
                # system_matrix[2*asdf:, 2*asdf:] = sub_matrix
                # if ref_axis == 'y':
                #     if axis == 'y':
                #         system_matrix[asdf:2*asdf, asdf:2*asdf] *= -1. # only on y
                #     else: # 'yz
                #         system_matrix[:asdf, :asdf] *= -1. # only on x

                # elif ref_axis == 'yz':
                #     if axis == 'y':
                #         system_matrix[asdf:2*asdf, asdf:2*asdf] *= -1. # only on y
                #     else: # 'yz
                #         system_matrix[:asdf, :asdf] *= -1. # only on x
        
        # sparse_system_matrix = csc_array(system_matrix)
        # del system_matrix
        # return system_matrix
        return sparse_system_matrix

if __name__ == "__main__":
    '''
    # import timeit
    # from python_csdl_backend import Simulator
    # import numpy as onp
    # AIC_half_val = onp.random.rand(1, 32, 3)
    # eval_pt_shape = (1,2,4)
    # vortex_coords_shape = (1,3,5)
    # m = csdl.Model()
    # AIC_half = m.declare_variable('AIC_half', val = AIC_half_val)
    # output_name = 'AIC'
    # AIC = csdl.custom(AIC_half, op = SymmetryFlip(in_name=AIC_half.name, eval_pt_shape=eval_pt_shape, vortex_coords_shape=vortex_coords_shape, out_name=output_name))
    # m.register_output(output_name, AIC)
    # sim = Simulator(m)
    # sim.run()
    '''

    '''
    import time
    import timeit
    ts = time.time()
    from python_csdl_backend import Simulator
    import numpy as onp
    def generate_simple_mesh(nx, ny):
        mesh = onp.zeros((nx, ny, 3))
        mesh[:, :, 0] = onp.outer(onp.arange(nx), onp.ones(ny))
        mesh[:, :, 1] = onp.outer(onp.arange(ny), onp.ones(nx)).T
        mesh[:, :, 2] = 0.
        return mesh

    nc = 10
    ns = 11

    eval_pt_names = ['coll_pts']
    vortex_coords_names = ['vtx_pts']
    # eval_pt_shapes = [(nx, ny, 3)]
    # vortex_coords_shapes = [(nx, ny, 3)]

    eval_pt_shapes = [(1, nc-1, ns-1, 3)]
    vortex_coords_shapes = [(1, nc, ns, 3)]

    output_names = ['aic']

    model_1 = csdl.Model()


    vor_val = generate_simple_mesh(nc, ns).reshape(1, nc, ns, 3)
    col_val = 0.25 * (vor_val[:,:-1, :-1, :] + vor_val[:,:-1, 1:, :] +
                      vor_val[:,1:, :-1, :] + vor_val[:,1:, 1:, :])
    # col_val = generate_simple_mesh(nx, ny)

    vor = model_1.create_input('vtx_pts', val=vor_val)
    col = model_1.create_input('coll_pts', val=col_val)

    # test if multiple ops work
    submodel=BiotSavartComp(eval_pt_names=eval_pt_names,
                               vortex_coords_names=vortex_coords_names,
                               eval_pt_shapes=eval_pt_shapes,
                               vortex_coords_shapes=vortex_coords_shapes,
                               output_names=output_names)

    model_1.add(submodel,'submodel')

    #####################
    # finshed adding model
    ####################
    sim = Simulator(model_1)
    print('time', time.time() - ts)
    sim.run()
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    sim.compute_totals(of='aic',wrt='vtx_pts')
    print('time', time.time() - ts)
    exit()
    '''