import m3l
import csdl
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_system import ODESystemModel
from VAST.core.submodels.kinematic_submodels.adapter_comp import AdapterComp
from VAST.core.submodels.aerodynamic_submodels.combine_gamma_w import CombineGammaW
from VAST.core.submodels.implicit_submodels.solve_group import SolveMatrix
from VAST.core.submodels.aerodynamic_submodels.seperate_gamma_b import SeperateGammab
from VAST.core.submodels.geometric_submodels.mesh_preprocessing_comp import MeshPreprocessingComp
from VAST.core.submodels.output_submodels.vlm_post_processing.horseshoe_circulations import HorseshoeCirculations
from VAST.core.submodels.output_submodels.vlm_post_processing.eval_pts_velocities_mls import EvalPtsVel
# from VAST.core.submodels.output_submodels.vlm_post_processing.compute_thrust_drag_dynamic_old import ThrustDrag
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_thrust_drag_dynamic import ThrustDrag, ThrustDragUndynamic
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_lift_drag import LiftDrag as LiftDrag_alt

from VAST.core.submodels.aerodynamic_submodels.combine_gamma_w import CombineGammaW
from VAST.core.submodels.geometric_submodels.mesh_preprocessing_comp import MeshPreprocessingComp
from VAST.core.submodels.kinematic_submodels.adapter_comp import AdapterComp

from VAST.core.submodels.implicit_submodels.solve_group import SolveMatrix
from VAST.core.submodels.aerodynamic_submodels.seperate_gamma_b import SeperateGammab
from VAST.core.submodels.wake_submodels.compute_wake_total_vel import ComputeWakeTotalVel
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_effective_aoa_cd_v import AOA_CD



import numpy as np


class ProfileOPModel3(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')
        self.parameters.declare('frame', default='wing_fixed')
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('compressible', default=False)
        self.parameters.declare('Ma',default=None)
        self.parameters.declare('free_wake', default=False)
        self.parameters.declare('sub',default=False)
        self.parameters.declare('sub_eval_list',default=None)
        self.parameters.declare('sub_induced_list',default=None)

    def define(self):
        # rename parameters
        n = self.parameters['num_nodes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']
        nt = self.parameters['nt']
        free_wake = self.parameters['free_wake']
        sub = self.parameters['sub']
        sub_eval_list = self.parameters['sub_eval_list']
        sub_induced_list = self.parameters['sub_induced_list']
        
        problem_type = 'prescribed_wake'
        if free_wake:
            problem_type = 'free_wake'

        # set conventional names
        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        # set shapes
        bd_vortex_shapes = surface_shapes
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)
        ode_surface_shapes = [(n, ) + item for item in surface_shapes]
        # wake_vortex_pts_shapes = [tuple((item[0],nt, item[2], 3)) for item in ode_surface_shapes]
        # wake_vel_shapes = [(n,x[1] * x[2], 3) for x in wake_vortex_pts_shapes]
        ode_bd_vortex_shapes = ode_surface_shapes
        gamma_w_shapes = [tuple((n,nt-1, item[2]-1)) for item in ode_surface_shapes]

        '''1. add a module here to compute surface_gamma_b, given mesh and ACstates'''
        # 1.1.1 declare the ode parameter surface for the current time step
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            surface = self.declare_variable(surface_name, shape=(n, nx, ny, 3))


        # 1.2.1 declare the ode parameter AcStates for the current time step
        u = self.declare_variable('u',  shape=(n,1))
        v = self.declare_variable('v',  shape=(n,1))
        w = self.declare_variable('w',  shape=(n,1))
        p = self.declare_variable('p',  shape=(n,1))
        q = self.declare_variable('q',  shape=(n,1))
        r = self.declare_variable('r',  shape=(n,1))
        theta = self.declare_variable('theta',  shape=(n,1))
        psi = self.declare_variable('psi',  shape=(n,1))
        x = self.declare_variable('x',  shape=(n,1))
        y = self.declare_variable('y',  shape=(n,1))
        z = self.declare_variable('z',  shape=(n,1))
        phiw = self.declare_variable('phiw',  shape=(n,1))
        gamma = self.declare_variable('gamma',  shape=(n,1))
        psiw = self.declare_variable('psiw',  shape=(n,1))

        #  1.2.2 from the AcStates, compute 5 preprocessing outputs
        # frame_vel, alpha, v_inf_sq, beta, rho
        m = AdapterComp(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            # frame=self.parameters['frame'],
            # frame='inertial',
        )
        self.add(m, name='adapter_comp')


        # 1.1.2 from the declared surface mesh, compute 6 preprocessing outputs
        # surface_bd_vtx_coords,coll_pts,l_span,l_chord,s_panel,bd_vec_all
        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=ode_surface_shapes,
                                       eval_pts_location=0.25,
                                       eval_pts_option='auto',
                                       delta_t=delta_t,
                                       problem_type='prescribed_wake',
                                       compressible=self.parameters['compressible'],
                                       Ma=self.parameters['Ma'],),
                 name='MeshPreprocessing_comp')


        self.add(CombineGammaW(surface_names=surface_names, surface_shapes=ode_surface_shapes, n_wake_pts_chord=nt-1),
            name='combine_gamma_w')

        self.add(SolveMatrix(n_wake_pts_chord=nt-1,
                                surface_names=surface_names,
                                bd_vortex_shapes=ode_surface_shapes,
                                delta_t=delta_t,
                                problem_type='prescribed_wake',
                                symmetry=self.parameters['symmetry'],
                                sub = sub,
                                sub_eval_list = sub_eval_list,
                                sub_induced_list = sub_induced_list),
                    name='solve_gamma_b_group')

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes),
                 name='seperate_gamma_b')

        # ODE system with surface gamma's
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            val = np.zeros((n, nt - 1, ny - 1))
            surface_name = surface_names[i]

            surface_gamma_w_name = surface_name + '_gamma_w'
            surface_dgammaw_dt_name = surface_name + '_dgammaw_dt'
            surface_gamma_b_name = surface_name +'_gamma_b'
            #######################################
            #states 1
            #######################################

            surface_gamma_w = self.declare_variable(surface_gamma_w_name,
                                                    shape=val.shape)
            #para for state 1

            surface_gamma_b = self.declare_variable(surface_gamma_b_name,
                                                    shape=(n, (nx - 1) *
                                                           (ny - 1), ))
            # self.print_var(surface_gamma_b)
            #outputs for state 1
            surface_dgammaw_dt = self.create_output(surface_dgammaw_dt_name,
                                                    shape=(n, nt - 1, ny - 1))

            gamma_b_last = csdl.reshape(surface_gamma_b[:,(nx - 2) * (ny - 1):],
                                        new_shape=(n, 1, ny - 1))

            surface_dgammaw_dt[:, 0, :] = (gamma_b_last -
                                           surface_gamma_w[:, 0, :]) / delta_t
            surface_dgammaw_dt[:, 1:, :] = (
                surface_gamma_w[:, :(surface_gamma_w.shape[1] - 1), :] -
                surface_gamma_w[:, 1:, :]) / delta_t

            # self.print_var(surface_gamma_w)
            # self.print_var(surface_gamma_b)


        #######################################
        #states 2
        #######################################
        # TODO: fix this comments to eliminate first row
        # t=0       [TE,              TE,                 TE,                TE]
        # t = 1,    [TE,              TE+v_ind(TE,w+bd),  TE,                TE] -> bracket 0-1
        # c11 = TE+v_ind(TE,w+bd)

        # t = 2,    [TE,              TE+v_ind(t=1, bracket 0),  c11+v_ind(t=1, bracket 1),   TE] ->  bracket 0-1-2
        # c21 =  TE+v_ind(t=1, bracket 0)
        # c22 =  c11+v_ind(t=1, bracket 1)

        # t = 3,    [TE,              TE+v_ind(t=2, bracket 0),  c21+vind(t=2, bracket 1), c22+vind(t=2, bracket 2)] -> bracket 0-1-2-3
        # Then, the shedding is
        '''2. add a module here to compute wake_total_vel, given mesh and ACstates'''
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            surface_wake_coords_name = surface_name + '_wake_coords'
            surface_dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            #states 2
            surface_wake_coords = self.declare_variable(surface_wake_coords_name,
                                                    shape=(n, nt - 1, ny, 3))
            '''2. add a module here to compute wake rollup'''

        self.add(ComputeWakeTotalVel(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes,
                                n_wake_pts_chord=nt-1,
                                problem_type=problem_type),
                 name='ComputeWakeTotalVel')            
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            surface_wake_coords_name = surface_name + '_wake_coords'
            surface_dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            surface_bd_vtx = self.declare_variable(surface_names[i]+'_bd_vtx_coords', shape=(n, nx, ny, 3))
            wake_total_vel = self.declare_variable(v_total_wake_names[i],val=np.zeros((n, nt - 1, ny, 3)))
            surface_wake_coords = self.declare_variable(surface_wake_coords_name, shape=(n, nt - 1, ny, 3))


            surface_dwake_coords_dt = self.create_output( surface_dwake_coords_dt_name, shape=((n, nt - 1, ny, 3)),val=0)
            # print(surface_dwake_coords_dt.name,surface_dwake_coords_dt.shape)

            TE = surface_bd_vtx[:, nx - 1, :, :]

            surface_dwake_coords_dt[:, 0, :, :] = (TE  + wake_total_vel[:, 0, :, :]*delta_t - surface_wake_coords[:, 0, :, :]) / delta_t
            surface_dwake_coords_dt[:, 1:, :, :] = (surface_wake_coords[:, :(surface_wake_coords.shape[1] - 1), :, :] - surface_wake_coords[:, 1:, :, :] + wake_total_vel[:, 1:, :, :] * delta_t) / delta_t

        if True:
            submodel = PPSubmodel(
                surface_names=surface_names,
                ode_surface_shapes=ode_surface_shapes,
                delta_t=delta_t,
                nt=nt,
                symmetry=self.parameters['symmetry']
            )

            kv_names= [x + '_kinematic_vel' for x in surface_names]
            gb_names= [x + '_gamma_b' for x in surface_names]
            op_gw_names = ['op_' + x + '_gamma_w' for x in surface_names]
            bvc_names = [x + '_bd_vtx_coords' for x in surface_names]
            op_wc_names = ['op_' + x + '_wake_coords' for x in surface_names]
            sp_names = [x + '_s_panel' for x in surface_names]
            eval_pts = [x + '_eval_pts_coords' for x in surface_names]
            output1 = [x + '_L' for x in surface_names]

            promotions = ['gamma_b','evaluation_pt','frame_vel','density','bd_vec','beta','alpha']
            promotions += surface_names
            promotions += kv_names
            promotions += gb_names
            promotions += op_gw_names 
            promotions += bvc_names 
            promotions += op_wc_names 
            promotions += sp_names 
            promotions += output1
            promotions += eval_pts

            self.add(submodel, name='pp_computations', promotes=promotions)

        else:
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
            eval_pts_shapes =        [
                tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
                for item in ode_surface_shapes
            ]

            # compute lift and drag
            submodel = HorseshoeCirculations(
                surface_names=surface_names,
                surface_shapes=ode_surface_shapes,
            )
            self.add(submodel, name='compute_horseshoe_circulation')

            submodel = EvalPtsVel(
                eval_pts_names=eval_pts_names,
                eval_pts_shapes=eval_pts_shapes,
                eval_pts_option='auto',
                eval_pts_location=0.25,
                surface_names=surface_names,
                surface_shapes=ode_surface_shapes,
                n_wake_pts_chord=nt-2,
                delta_t=delta_t,
                problem_type='prescribed_wake',
                eps=4e-5,
                symmetry=self.parameters['symmetry'],
            )
            self.add(submodel, name='EvalPtsVel')

            submodel = ThrustDragUndynamic(
                surface_names=surface_names,
                surface_shapes=ode_surface_shapes,
                eval_pts_option='auto',
                eval_pts_shapes=eval_pts_shapes,
                eval_pts_names=eval_pts_names,
                sprs=None,
                coeffs_aoa=None,
                coeffs_cd=None,
                delta_t=delta_t,
            )
            self.add(submodel, name='ThrustDrag')


class ProfileOPModel4(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')
        self.parameters.declare('frame', default='wing_fixed')
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('compressible', default=False)
        self.parameters.declare('Ma',default=None)
        self.parameters.declare('free_wake', default=False)
        self.parameters.declare('sub',default=False)
        self.parameters.declare('sub_eval_list',default=None)
        self.parameters.declare('sub_induced_list',default=None)

    def define(self):
        # rename parameters
        n = self.parameters['num_nodes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']
        nt = self.parameters['nt']
        free_wake = self.parameters['free_wake']
        sub = self.parameters['sub']
        sub_eval_list = self.parameters['sub_eval_list']
        sub_induced_list = self.parameters['sub_induced_list']
        
        problem_type = 'prescribed_wake'
        if free_wake:
            problem_type = 'free_wake'

        # set conventional names
        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        # set shapes
        bd_vortex_shapes = surface_shapes
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)
        ode_surface_shapes = [(n, ) + item for item in surface_shapes]
        # wake_vortex_pts_shapes = [tuple((item[0],nt, item[2], 3)) for item in ode_surface_shapes]
        # wake_vel_shapes = [(n,x[1] * x[2], 3) for x in wake_vortex_pts_shapes]
        ode_bd_vortex_shapes = ode_surface_shapes
        gamma_w_shapes = [tuple((n,nt-1, item[2]-1)) for item in ode_surface_shapes]

        '''1. add a module here to compute surface_gamma_b, given mesh and ACstates'''
        # 1.1.1 declare the ode parameter surface for the current time step
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            surface = self.declare_variable(surface_name, shape=(n, nx, ny, 3))


        # 1.2.1 declare the ode parameter AcStates for the current time step
        u = self.declare_variable('u',  shape=(n,1))
        v = self.declare_variable('v',  shape=(n,1))
        w = self.declare_variable('w',  shape=(n,1))
        p = self.declare_variable('p',  shape=(n,1))
        q = self.declare_variable('q',  shape=(n,1))
        r = self.declare_variable('r',  shape=(n,1))
        theta = self.declare_variable('theta',  shape=(n,1))
        psi = self.declare_variable('psi',  shape=(n,1))
        x = self.declare_variable('x',  shape=(n,1))
        y = self.declare_variable('y',  shape=(n,1))
        z = self.declare_variable('z',  shape=(n,1))
        phiw = self.declare_variable('phiw',  shape=(n,1))
        gamma = self.declare_variable('gamma',  shape=(n,1))
        psiw = self.declare_variable('psiw',  shape=(n,1))

        #  1.2.2 from the AcStates, compute 5 preprocessing outputs
        # frame_vel, alpha, v_inf_sq, beta, rho
        m = AdapterComp(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            # frame=self.parameters['frame'],
            # frame='inertial',
        )
        self.add(m, name='adapter_comp')


        # 1.1.2 from the declared surface mesh, compute 6 preprocessing outputs
        # surface_bd_vtx_coords,coll_pts,l_span,l_chord,s_panel,bd_vec_all
        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=ode_surface_shapes,
                                       eval_pts_location=0.25,
                                       eval_pts_option='auto',
                                       delta_t=delta_t,
                                       problem_type='prescribed_wake',
                                       compressible=self.parameters['compressible'],
                                       Ma=self.parameters['Ma'],),
                 name='MeshPreprocessing_comp')


        self.add(CombineGammaW(surface_names=surface_names, surface_shapes=ode_surface_shapes, n_wake_pts_chord=nt-1),
            name='combine_gamma_w')

        self.add(SolveMatrix(n_wake_pts_chord=nt-1,
                                surface_names=surface_names,
                                bd_vortex_shapes=ode_surface_shapes,
                                delta_t=delta_t,
                                problem_type='prescribed_wake',
                                symmetry=self.parameters['symmetry'],
                                sub = sub,
                                sub_eval_list = sub_eval_list,
                                sub_induced_list = sub_induced_list),
                    name='solve_gamma_b_group')

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes),
                 name='seperate_gamma_b')

        # ODE system with surface gamma's
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            val = np.zeros((n, nt - 1, ny - 1))
            surface_name = surface_names[i]

            surface_gamma_w_name = surface_name + '_gamma_w'
            surface_dgammaw_dt_name = surface_name + '_dgammaw_dt'
            surface_gamma_b_name = surface_name +'_gamma_b'
            #######################################
            #states 1
            #######################################

            surface_gamma_w = self.declare_variable(surface_gamma_w_name,
                                                    shape=val.shape)
            #para for state 1

            surface_gamma_b = self.declare_variable(surface_gamma_b_name,
                                                    shape=(n, (nx - 1) *
                                                           (ny - 1), ))
            # self.print_var(surface_gamma_b)
            #outputs for state 1
            surface_dgammaw_dt = self.create_output(surface_dgammaw_dt_name,
                                                    shape=(n, nt - 1, ny - 1))

            gamma_b_last = csdl.reshape(surface_gamma_b[:,(nx - 2) * (ny - 1):],
                                        new_shape=(n, 1, ny - 1))

            surface_dgammaw_dt[:, 0, :] = (gamma_b_last -
                                           surface_gamma_w[:, 0, :]) / delta_t
            surface_dgammaw_dt[:, 1:, :] = (
                surface_gamma_w[:, :(surface_gamma_w.shape[1] - 1), :] -
                surface_gamma_w[:, 1:, :]) / delta_t

            # self.print_var(surface_gamma_w)
            # self.print_var(surface_gamma_b)


        #######################################
        #states 2
        #######################################
        # TODO: fix this comments to eliminate first row
        # t=0       [TE,              TE,                 TE,                TE]
        # t = 1,    [TE,              TE+v_ind(TE,w+bd),  TE,                TE] -> bracket 0-1
        # c11 = TE+v_ind(TE,w+bd)

        # t = 2,    [TE,              TE+v_ind(t=1, bracket 0),  c11+v_ind(t=1, bracket 1),   TE] ->  bracket 0-1-2
        # c21 =  TE+v_ind(t=1, bracket 0)
        # c22 =  c11+v_ind(t=1, bracket 1)

        # t = 3,    [TE,              TE+v_ind(t=2, bracket 0),  c21+vind(t=2, bracket 1), c22+vind(t=2, bracket 2)] -> bracket 0-1-2-3
        # Then, the shedding is
        '''2. add a module here to compute wake_total_vel, given mesh and ACstates'''
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            surface_wake_coords_name = surface_name + '_wake_coords'
            surface_dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            #states 2
            surface_wake_coords = self.declare_variable(surface_wake_coords_name,
                                                    shape=(n, nt - 1, ny, 3))
            '''2. add a module here to compute wake rollup'''

        self.add(ComputeWakeTotalVel(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes,
                                n_wake_pts_chord=nt-1,
                                problem_type=problem_type),
                 name='ComputeWakeTotalVel')            
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            surface_name = surface_names[i]
            surface_wake_coords_name = surface_name + '_wake_coords'
            surface_dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            surface_bd_vtx = self.declare_variable(surface_names[i]+'_bd_vtx_coords', shape=(n, nx, ny, 3))
            wake_total_vel = self.declare_variable(v_total_wake_names[i],val=np.zeros((n, nt - 1, ny, 3)))
            surface_wake_coords = self.declare_variable(surface_wake_coords_name, shape=(n, nt - 1, ny, 3))


            surface_dwake_coords_dt = self.create_output( surface_dwake_coords_dt_name, shape=((n, nt - 1, ny, 3)),val=0)
            # print(surface_dwake_coords_dt.name,surface_dwake_coords_dt.shape)

            TE = surface_bd_vtx[:, nx - 1, :, :]

            surface_dwake_coords_dt[:, 0, :, :] = (TE  + wake_total_vel[:, 0, :, :]*delta_t - surface_wake_coords[:, 0, :, :]) / delta_t
            surface_dwake_coords_dt[:, 1:, :, :] = (surface_wake_coords[:, :(surface_wake_coords.shape[1] - 1), :, :] - surface_wake_coords[:, 1:, :, :] + wake_total_vel[:, 1:, :, :] * delta_t) / delta_t

        submodel = POSubmodel(
            surface_names=surface_names,
            ode_surface_shapes=ode_surface_shapes,
            delta_t=delta_t,
            nt=nt,
            symmetry=self.parameters['symmetry']
        )
        promotions = gen_promotions_list(surface_names, surface_shapes)
        self.add(submodel, name='po_submodel', promotes=promotions)

        












class POSubmodel(csdl.Model):
    def initialize(self):
        self.parameters.declare('surface_names')
        self.parameters.declare('ode_surface_shapes')
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')
        self.parameters.declare('symmetry')
    def define(self):
        surface_names = self.parameters['surface_names']
        ode_surface_shapes = self.parameters['ode_surface_shapes']
        delta_t = self.parameters['delta_t']
        nt = self.parameters['nt']

        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        eval_pts_shapes =        [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in ode_surface_shapes
        ]

        # compute lift and drag
        submodel = HorseshoeCirculations(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
        )
        self.add(submodel, name='compute_horseshoe_circulation')

        submodel = EvalPtsVel(
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option='auto',
            eval_pts_location=0.25,
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            n_wake_pts_chord=nt-2,
            delta_t=delta_t,
            problem_type='prescribed_wake',
            eps=4e-5,
            symmetry=self.parameters['symmetry'],
        )
        self.add(submodel, name='EvalPtsVel')








def gen_profile_output_list(surface_names, surface_shapes):
    outputs = []
    gamma_b_len = 0
    for surface_name, surface_shape in zip(surface_names, surface_shapes):
        bvn_name = surface_name + '_bd_vtx_normals'
        bvn_shape = (surface_shape[0]-1, surface_shape[1]-1, 3)
        outputs.append((bvn_name, bvn_shape))

        sp_name = surface_name + '_s_panel'
        sp_shape = (1,surface_shape[0]-1, surface_shape[1]-1)
        outputs.append((sp_name, sp_shape))

        eval_pts_name = surface_name + '_eval_pts_coords' 
        eval_pts_shape = (1,surface_shape[0]-1, surface_shape[1]-1, 3)
        outputs.append((eval_pts_name, eval_pts_shape))
        gamma_b_len += (surface_shape[1]-1)*(surface_shape[0]-1)

        etv_name = surface_name + '_eval_total_vel'
        etv_shape = (1,(surface_shape[1]-1)*(surface_shape[0]-1), 3)
        outputs.append((etv_name, etv_shape))

    outputs.append(('horseshoe_circulation', (1,gamma_b_len)))
    outputs.append(('gamma_b', (1,gamma_b_len)))
    outputs.append(('frame_vel', (1,3)))
    outputs.append(('density', (1,1)))
    outputs.append(('bd_vec', (1,gamma_b_len,3)))
    outputs.append(('beta', (1,1)))
    outputs.append(('alpha', (1,1)))
    return outputs


def gen_promotions_list(surface_names, surface_shapes):
    outputs = []
    gamma_b_len = 0
    for surface_name, surface_shape in zip(surface_names, surface_shapes):
        kv_name = surface_name + '_kinematic_vel'
        outputs.append(kv_name)

        gb_name = surface_name + '_gamma_b'
        outputs.append(gb_name)

        bvc_name = surface_name + '_bd_vtx_coords'
        outputs.append(bvc_name)

        bvn_name = surface_name + '_bd_vtx_normals' # newly needed
        outputs.append(bvn_name)

        sp_name = surface_name + '_s_panel'   # still needed
        outputs.append(sp_name)

        eval_pts_name = surface_name + '_eval_pts_coords' # still needed
        outputs.append(eval_pts_name)

        etv_name = surface_name + '_eval_total_vel' # newly needed
        outputs.append(etv_name)

    outputs.append('horseshoe_circulation')
    outputs.append('gamma_b')
    # outputs.append(('evaluation_pt', (3,)))
    outputs.append('frame_vel') # still needed
    outputs.append('density')  # still needed
    outputs.append('bd_vec') # still needed
    outputs.append('beta') # still needed
    outputs.append('alpha') # still needed
    return outputs







# input variables:
### gamma b


class PPSubmodel(csdl.Model):
    def initialize(self):
        self.parameters.declare('surface_names')
        self.parameters.declare('ode_surface_shapes')
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')
        self.parameters.declare('symmetry')
    def define(self):
        surface_names = self.parameters['surface_names']
        ode_surface_shapes = self.parameters['ode_surface_shapes']
        delta_t = self.parameters['delta_t']
        nt = self.parameters['nt']

        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        eval_pts_shapes =        [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in ode_surface_shapes
        ]

        submodel = ThrustDrag(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            eval_pts_option='auto',
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_names=eval_pts_names,
            sprs=None,
            coeffs_aoa=None,
            coeffs_cd=None,
            delta_t=delta_t,
        )
        self.add(submodel, name='ThrustDrag')


if __name__ == '__main__':
    from python_csdl_backend import Simulator
    surface_names = ['wing','wing2']
    ode_surface_shapes = [(5,5,5,3),(5,5,5,3)]
    eval_pts_shapes = ode_surface_shapes
    eval_pts_names = ['wing1']
    delta_t = 0.01
    nt = 10
    model = PPSubmodel(surface_names=surface_names, ode_surface_shapes=ode_surface_shapes,delta_t=delta_t,nt=nt,symmetry=False)
    sim = Simulator(model, analytics=True)
    

