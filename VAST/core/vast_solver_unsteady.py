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
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_thrust_drag_dynamic import ThrustDrag
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_lift_drag import LiftDrag as LiftDrag_alt

from VAST.core.submodels.aerodynamic_submodels.combine_gamma_w import CombineGammaW
from VAST.core.submodels.geometric_submodels.mesh_preprocessing_comp import MeshPreprocessingComp
from VAST.core.submodels.kinematic_submodels.adapter_comp import AdapterComp

from VAST.core.submodels.implicit_submodels.solve_group import SolveMatrix
from VAST.core.submodels.aerodynamic_submodels.seperate_gamma_b import SeperateGammab
from VAST.core.submodels.wake_submodels.compute_wake_total_vel import ComputeWakeTotalVel
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_effective_aoa_cd_v import AOA_CD



import numpy as np


class VASTSolverUnsteady(m3l.ImplicitOperation):
    def initialize(self, kwargs):
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')
        self.parameters.declare('frame', default='wing_fixed')
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('compressible', default=False)
        self.parameters.declare('Ma',default=None)
        self.parameters.declare('name', default='uvlm')
        self.parameters.declare('free_wake', default=False)
    def assign_atributes(self):
        self.num_nodes = self.parameters['num_nodes']
        self.surface_names = self.parameters['surface_names']
        self.surface_shapes = self.parameters['surface_shapes']
        self.delta_t = self.parameters['delta_t']
        self.nt = self.parameters['nt']
        self.frame = self.parameters['frame']
        self.compressible = self.parameters['compressible']
        self.Ma = self.parameters['Ma']
        self.name = self.parameters['name']
        self.symmetry = self.parameters['symmetry']
        self.free_wake = self.parameters['free_wake']
    def evaluate(self):
        self.assign_atributes()
        num_nodes = self.num_nodes
        self.residual_names = []
        self.ode_parameters = ['u',
                               'v',
                               'w',
                               'p',   # these are declared but not used
                               'q',
                               'r',
                               'theta',
                               'psi',
                            #    'x',
                            #    'y',
                            #    'z',
                            #    'phiw',
                               'gamma',
                               'psiw']     # TODO: add rho? - need to modify adapter_comp to not be hard coded
        surface_names = self.surface_names
        surface_shapes = self.surface_shapes
        for i in range(len(surface_names)):
            ####################################
            # ode parameter names
            ####################################
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]
            self.ode_parameters.append(surface_name)
            self.ode_parameters.append(surface_name + '_coll_vel')

            ####################################
            # ode states names
            ####################################
            gamma_w_name = surface_name + '_gamma_w'
            wing_wake_coords_name = surface_name + '_wake_coords'
            # gamma_w_name_list.append(gamma_w_name)
            # wing_wake_coords_name_list.append(wing_wake_coords_name)
            # Inputs names correspond to respective upstream CSDL variables
            ####################################
            # ode outputs names
            ####################################
            dgammaw_dt_name = surface_name + '_dgammaw_dt'
            dwake_coords_dt_name = surface_name + '_dwake_coords_dt'
            ####################################
            # IC names
            ####################################
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            ####################################
            # states and outputs names
            ####################################
            gamma_w_int_name = surface_name + '_gamma_w_int'
            wake_coords_int_name = surface_name + '_wake_coords_int'
            self.residual_names.append((gamma_w_name,dgammaw_dt_name,(num_nodes-1, ny-1)))
            self.residual_names.append((wing_wake_coords_name,dwake_coords_dt_name,(num_nodes-1,ny,3)))

        self.inputs = {}
        self.arguments = {}

        residual = m3l.Variable(self.residual_names[0][1], shape=(), operation=self)
        return residual #, frame_vel, bd_vec, horseshoe_circulation
    def compute_residual(self, num_nodes):
        model = ODESystemModel(num_nodes=num_nodes,
                               surface_names=self.surface_names,
                               surface_shapes=self.surface_shapes,
                               delta_t=self.delta_t,
                               nt=self.num_nodes,
                               symmetry=self.symmetry,
                               Ma=self.Ma,
                               compressible=self.compressible,
                               frame=self.frame,
                               free_wake=self.free_wake)
        return model


class PostProcessor(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('Ma',default=0.84)
        self.parameters.declare('frame', default='wing_fixed')

    def define(self):
        num_nodes = self.parameters['num_nodes']
        surface_shapes = self.parameters['surface_shapes']
        surface_names = self.parameters['surface_names']
        h_stepsize = self.parameters['delta_t']
        nt = self.parameters['nt']
        num_times = nt - 1
        symmetry = self.parameters['symmetry']
        frame = self.parameters['frame']


        ode_surface_shapes = [(nt-1, ) + item for item in surface_shapes]
        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        op_surface_names = ['op_' + x for x in surface_names]
        eval_pts_shapes =        [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in ode_surface_shapes
        ]

        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=ode_surface_shapes,
                                       eval_pts_location=0.25,
                                       eval_pts_option='auto',
                                       delta_t=h_stepsize,
                                       problem_type='prescribed_wake',
                                       Ma=self.parameters['Ma'],),
                 name='MeshPreprocessing_comp')

        m = AdapterComp(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            frame=frame,
        )
        self.add(m, name='adapter_comp')

        self.add(CombineGammaW(surface_names=op_surface_names, surface_shapes=ode_surface_shapes, n_wake_pts_chord=num_times-1),
            name='combine_gamma_w')

        self.add(SolveMatrix(n_wake_pts_chord=num_times-1,
                                surface_names=surface_names,
                                bd_vortex_shapes=ode_surface_shapes,
                                delta_t=h_stepsize,
                                problem_type='prescribed_wake',
                                end=True,
                                symmetry=self.parameters['symmetry'],),
                    name='solve_gamma_b_group')
        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes),
                 name='seperate_gamma_b')

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
            n_wake_pts_chord=num_times-1,
            delta_t=h_stepsize,
            problem_type='prescribed_wake',
            eps=4e-5,
            symmetry=self.parameters['symmetry'],
        )
        self.add(submodel, name='EvalPtsVel')

        submodel = ThrustDrag(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            eval_pts_option='auto',
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_names=eval_pts_names,
            sprs=None,
            coeffs_aoa=None,
            coeffs_cd=None,
            delta_t=h_stepsize,
        )
        self.add(submodel, name='ThrustDrag')


class ProfileOpModel(csdl.Model):
    '''
    contains
    1. MeshPreprocessing_comp
    2. SolveMatrix
    3. solve_gamma_b_group
    3. seperate_gamma_b_comp
    4. extract_gamma_w_comp
    '''
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t')
        self.parameters.declare('nt')
        self.parameters.declare('symmetry',default=False)

    def define(self):
        num_nodes = self.parameters['num_nodes']
        surface_shapes = self.parameters['surface_shapes']
        surface_names = self.parameters['surface_names']
        h_stepsize = self.parameters['delta_t']
        nt = self.parameters['nt']
        symmetry = self.parameters['symmetry']


        ode_surface_shapes = [(nt-1, ) + item for item in surface_shapes]
        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        op_surface_names = ['op_' + x for x in surface_names]
        eval_pts_shapes =        [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in ode_surface_shapes
        ]

        self.add(MeshPreprocessingComp(surface_names=surface_names,
                                       surface_shapes=ode_surface_shapes,
                                       eval_pts_location=0.25,
                                       eval_pts_option='auto',
                                       delta_t=h_stepsize,
                                       problem_type='prescribed_wake',
                                       Ma=None,),
                 name='MeshPreprocessing_comp')

        m = AdapterComp(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            # frame=frame,
        )
        self.add(m, name='adapter_comp')

        self.add(CombineGammaW(surface_names=op_surface_names, surface_shapes=ode_surface_shapes, n_wake_pts_chord=num_nodes),
            name='combine_gamma_w')

        self.add(SolveMatrix(n_wake_pts_chord=num_nodes,
                                surface_names=surface_names,
                                bd_vortex_shapes=ode_surface_shapes,
                                delta_t=h_stepsize,
                                problem_type='prescribed_wake',
                                end=True,
                                symmetry=symmetry,),
                    name='solve_gamma_b_group')
        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=ode_surface_shapes),
                 name='seperate_gamma_b')

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
            n_wake_pts_chord=num_nodes,
            delta_t=h_stepsize,
            problem_type='prescribed_wake',
            eps=4e-5,
            symmetry=False,
        )
        self.add(submodel, name='EvalPtsVel')

        submodel = LiftDrag(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            eval_pts_option='auto',
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_names=eval_pts_names,
            sprs=None,
            coeffs_aoa=None,
            coeffs_cd=None,
            # delta_t=h_stepsize,
        )
        self.add(submodel, name='LiftDrag')


class ProfileOpModel2(csdl.Model):
    '''
    contains
    1. MeshPreprocessing_comp
    2. SolveMatrix
    3. solve_gamma_b_group
    3. seperate_gamma_b_comp
    4. extract_gamma_w_comp
    '''
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
        
    def define(self):
        # rename parameters
        n = num_nodes = self.parameters['num_nodes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = h_stepsize = self.parameters['delta_t']
        nt = self.parameters['nt']

        # set conventional names
        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        # set shapes
        bd_vortex_shapes = surface_shapes
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)
        ode_surface_shapes = [(n, ) + item for item in surface_shapes]
        # ode_surface_shapes_2 = [(nt-1, ) + item for item in surface_shapes]
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
            frame=self.parameters['frame'],
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
                                symmetry=self.parameters['symmetry'],),
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
                                n_wake_pts_chord=nt-1),
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
            n_wake_pts_chord=num_nodes,
            delta_t=h_stepsize,
            problem_type='prescribed_wake',
            eps=4e-5,
            symmetry=False,
        )
        self.add(submodel, name='EvalPtsVel')

        submodel = ThrustDrag(
            surface_names=surface_names,
            surface_shapes=ode_surface_shapes,
            eval_pts_option='auto',
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_names=eval_pts_names,
            sprs=None,
            coeffs_aoa=None,
            coeffs_cd=None,
            delta_t=h_stepsize,
        )
        self.add(submodel, name='ThrustDrag')

class LiftDrag(csdl.Model):
    """
    L,D,cl,cd
    parameters
    ----------

    bd_vec : csdl array
        tangential vec    
    velocities: csdl array
        force_pts vel 
    gamma_b[num_bd_panel] : csdl array
        a concatenate vector of the bd circulation strength
    frame_vel[3,] : csdl array
        frame velocities
    Returns
    -------
    L,D,cl,cd
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

        self.parameters.declare('eval_pts_option')
        self.parameters.declare('eval_pts_shapes')
        self.parameters.declare('sprs')

        # self.parameters.declare('rho', default=0.9652)
        self.parameters.declare('eval_pts_names', types=None)

        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]
        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        cl_span_names = [x + '_cl_span' for x in surface_names]

        system_size = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            system_size += (nx - 1) * (ny - 1)

        rho = self.declare_variable('rho', shape=(num_nodes, 1))
        rho_expand = csdl.expand(csdl.reshape(rho, (num_nodes, )),
                                 (num_nodes, system_size, 3), 'k->kij')
        alpha = self.declare_variable('alpha', shape=(num_nodes, 1))
        beta = self.declare_variable('beta', shape=(num_nodes, 1))

        sprs = self.parameters['sprs']
        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_shapes = self.parameters['eval_pts_shapes']

        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']

        v_total_wake_names = [x + '_eval_total_vel' for x in surface_names]

        bd_vec = self.declare_variable('bd_vec',
                                       shape=((num_nodes, system_size, 3)))

        circulations = self.declare_variable('horseshoe_circulation',
                                             shape=(num_nodes, system_size))
        circulation_repeat = csdl.expand(circulations,
                                         (num_nodes, system_size, 3),
                                         'ki->kij')

        # print('beta shape', beta.shape)
        # print('sinbeta shape', sinbeta.shape)

        eval_pts_names = self.parameters['eval_pts_names']

        if eval_pts_option == 'auto':
            velocities = self.create_output('eval_total_vel',
                                            shape=(num_nodes, system_size, 3))
            s_panels_all = self.create_output('s_panels_all',
                                              shape=(num_nodes, system_size))
            eval_pts_all = self.create_output('eval_pts_all',
                                              shape=(num_nodes, system_size,
                                                     3))
            start = 0
            for i in range(len(v_total_wake_names)):

                nx = surface_shapes[i][1]
                ny = surface_shapes[i][2]
                delta = (nx - 1) * (ny - 1)
                vel_surface = self.declare_variable(v_total_wake_names[i],
                                                    shape=(num_nodes, delta,
                                                           3))
                s_panels = self.declare_variable(surface_names[i] + '_s_panel',
                                                 shape=(num_nodes, nx - 1,
                                                        ny - 1))

                spans = self.declare_variable(
                    surface_names[i] + '_span_length',
                    shape=(num_nodes, nx - 1, ny - 1))
                chords = self.declare_variable(
                    surface_names[i] + '_chord_length',
                    shape=(num_nodes, nx - 1, ny - 1))
                eval_pts = self.declare_variable(eval_pts_names[i],
                                                 shape=(num_nodes, nx - 1,
                                                        ny - 1, 3))
                # print('compute lift drag vel_surface shape', vel_surface.shape)
                # print('compute lift drag velocities shape', velocities.shape)
                velocities[:, start:start + delta, :] = vel_surface
                s_panels_all[:, start:start + delta] = csdl.reshape(
                    s_panels, (num_nodes, delta))
                eval_pts_all[:, start:start + delta, :] = csdl.reshape(
                    eval_pts, (num_nodes, delta, 3))
                # spans_all[:, start:start + delta] = csdl.reshape(
                #     spans, (num_nodes, delta))
                # chords_all[:, start:start + delta] = csdl.reshape(
                #     chords, (num_nodes, delta))
                start = start + delta

            # print('-----------------alpha', alpha.name, csdl.cos(alpha).name)
            # print('-----------------beta', beta.name, csdl.cos(beta).name)

            sina = csdl.expand(csdl.sin(alpha), (num_nodes, system_size, 1),
                               'ki->kji')
            cosa = csdl.expand(csdl.cos(alpha), (num_nodes, system_size, 1),
                               'ki->kji')
            sinb = csdl.expand(csdl.sin(beta), (num_nodes, system_size, 1),
                               'ki->kji')
            cosb = csdl.expand(csdl.cos(beta), (num_nodes, system_size, 1),
                               'ki->kji')
            # print('-----------------cosa', cosa.name)
            # print('-----------------sinb', sinb.name)
            # print('-----------------cosb', cosb.name)

            panel_forces = rho_expand * circulation_repeat * csdl.cross(
                velocities, bd_vec, axis=2)

            self.register_output('panel_forces', panel_forces)

            panel_forces_x = panel_forces[:, :, 0]
            panel_forces_y = panel_forces[:, :, 1]
            panel_forces_z = panel_forces[:, :, 2]
            # print('compute lift drag panel_forces', panel_forces.shape)
            b = frame_vel[:, 0]**2 + frame_vel[:, 1]**2 + frame_vel[:, 2]**2

            L_panel = -panel_forces_x * sina + panel_forces_z * cosa
            D_panel = panel_forces_x * cosa * cosb + panel_forces_z * sina * cosb - panel_forces_y * sinb
            traction_panel = panel_forces / csdl.expand(
                s_panels_all, panel_forces.shape, 'ij->ijk')

            s_panels_sum = csdl.reshape(csdl.sum(s_panels_all, axes=(1, )),
                                        (num_nodes, 1))

            start = 0
            for i in range(len(surface_names)):

                mesh = self.declare_variable(surface_names[i],
                                             shape=surface_shapes[i])
                nx = surface_shapes[i][1]
                ny = surface_shapes[i][2]

                # s_panels = self.declare_variable(surface_names[i] + '_s_panel',
                #                                  shape=(num_nodes, nx - 1,
                #                                         ny - 1))

                # nx = surface_shapes[i][1]
                # ny = surface_shapes[i][2]
                #!TODO: need to fix for uniformed mesh - should we take an average?
                # chord = csdl.reshape(mesh[:, nx - 1, 0, 0] - mesh[:, 0, 0, 0],
                #                      (num_nodes, 1))
                # span = csdl.reshape(mesh[:, 0, ny - 1, 1] - mesh[:, 0, 0, 1],
                #                     (num_nodes, 1))
                L_panel_name = surface_names[i] + '_L_panel'
                D_panel_name = surface_names[i] + '_D_panel'
                traction_surfaces_name = surface_names[i] + '_traction_surfaces'

                L_name = surface_names[i] + '_L'
                D_name = surface_names[i] + '_D'
                CL_name = surface_names[i] + '_C_L'
                CD_name = surface_names[i] + '_C_D_i'

                delta = (nx - 1) * (ny - 1)
                L_panel_surface = L_panel[:, start:start + delta, :]
                D_panel_surface = D_panel[:, start:start + delta, :]
                # cl_panel_surface = cl_panel[:, start:start + delta, :]
                # cdi_panel_surface = cd_i_panel[:, start:start + delta, :]
                traction_surfaces = traction_panel[:, start:start + delta, :]

                self.register_output(L_panel_name, L_panel_surface)
                self.register_output(D_panel_name, D_panel_surface)
                self.register_output(traction_surfaces_name, traction_surfaces)

                L = csdl.sum(L_panel_surface, axes=(1, ))
                D = csdl.sum(D_panel_surface, axes=(1, ))
                self.register_output(L_name, csdl.reshape(L, (num_nodes, 1)))
                self.register_output(D_name, csdl.reshape(D, (num_nodes, 1)))

                c_l = L / (0.5 * rho * s_panels_sum * b)
                self.register_output(CL_name,
                                     csdl.reshape(c_l, (num_nodes, 1)))

                c_d = D / (0.5 * rho * s_panels_sum * b)

                self.register_output(CD_name,
                                     csdl.reshape(c_d, (num_nodes, 1)))

                start += delta

            if self.parameters['coeffs_aoa'] != None:
                # print('coeffs_aoa is ', self.parameters['coeffs_aoa'])
                cl_span_names = [x + '_cl_span' for x in surface_names]
                cd_span_names = [x + '_cd_i_span' for x in surface_names]
                # nx = surface_shapes[i][1]
                # ny = surface_shapes[i][2]
                start = 0
                for i in range(len(surface_names)):
                    nx = surface_shapes[i][1]
                    ny = surface_shapes[i][2]
                    delta = (nx - 1) * (ny - 1)

                    s_panels = self.declare_variable(
                        surface_names[i] + '_s_panel',
                        shape=(num_nodes, nx - 1, ny - 1))
                    surface_span = csdl.reshape(csdl.sum(s_panels, axes=(1, )),
                                                (num_nodes, ny - 1, 1))
                    rho_b_exp = csdl.expand(rho * b, (num_nodes, ny - 1, 1),
                                            'ik->ijk')

                    cl_span = csdl.reshape(
                        csdl.sum(csdl.reshape(
                            L_panel[:, start:start + delta, :],
                            (num_nodes, nx - 1, ny - 1)),
                                 axes=(1, )),
                        (num_nodes, ny - 1, 1)) / (0.5 * rho_b_exp *
                                                   surface_span)
                    # print()
                    cd_span = csdl.reshape(
                        csdl.sum(csdl.reshape(
                            D_panel[:, start:start + delta, :],
                            (num_nodes, nx - 1, ny - 1)),
                                 axes=(1, )),
                        (num_nodes, ny - 1, 1)) / (0.5 * rho_b_exp *
                                                   surface_span)
                    self.register_output(cl_span_names[i], cl_span)
                    self.register_output(cd_span_names[i], cd_span)
                    start += delta

                sub = AOA_CD(
                    surface_names=surface_names,
                    surface_shapes=surface_shapes,
                    coeffs_aoa=coeffs_aoa,
                    coeffs_cd=coeffs_cd
                )
                self.add(sub, name='AOA_CD')

                #     cd_v_names = [x + '_cd_v' for x in surface_names]

                #     for i in range(len(surface_names)):
                D_total_name = surface_names[i] + '_D_total'

                #         cd_v = self.declare_variable(cd_v_names[i],
                #                                      shape=(num_nodes, 1))
                #         c_d_total = cd_v + c_d
                CD_total_names = [x + '_C_D' for x in surface_names]
                # for i in range(len(surface_names)):

                #     c_d_total = self.declare_variable(CD_total_names[i],
                #                                       shape=(num_nodes, 1))

                #     D_total = c_d_total * (0.5 * rho * s_panels_sum * b)
                # self.register_output(D_total_name, D_total)

            ##########################################################
            # temp fix total_forces = csdl.sum(panel_forces, axes=(1, ))
            ##########################################################
            # print('D shape', D.shape)
            # print('L shape', L.shape)
            # total_forces = self.create_output('F', shape=(num_nodes, 3))
            # total_forces[:, 0] = -D
            # total_forces[:, 2] = L

            # print('shapes total force', total_forces.shape)
            # print('shapes panel_forces', panel_forces.shape)
            # print('shapes eval_pts_all', eval_pts_all.shape)
            # print('shapes eval_pts_all',
            #       csdl.cross(panel_forces, eval_pts_all, axis=(2, )))

            #TODO: discuss about the drag computation
            # D_0 = self.declare_variable('Wing_D_0', shape=(num_nodes, 1))
            '''hardcode for testing'''

            # total_forces_temp = csdl.sum(panel_forces, axes=(1, ))
            # # F = self.create_output('F', shape=(num_nodes, 3))
            # # F[:, 0] = total_forces_temp[:, 0] - D_0 * csdl.cos(alpha)
            # # F[:, 1] = total_forces_temp[:, 1]
            # # F[:, 2] = total_forces_temp[:, 2] - D_0 * csdl.sin(alpha)

            # # F = self.create_output('F', shape=(num_nodes, 3))
            # # F[:, 0] = total_forces_temp[:, 0] - 0.
            # # F[:, 1] = total_forces_temp[:, 1] - 0.
            # # F[:, 2] = total_forces_temp[:, 2] - 0.
            # self.register_output('F', total_forces_temp)

            total_forces_temp = csdl.sum(panel_forces, axes=(1, ))
            F = self.create_output('F', shape=(num_nodes, 3))
            F[:, 0] = total_forces_temp[:, 0] 
            F[:, 1] = total_forces_temp[:, 1]
            F[:, 2] = -total_forces_temp[:, 2]

            # F = self.create_output('F', shape=(num_nodes, 3))
            # F[:, 0] = total_forces_temp[:, 0] - 0.
            # F[:, 1] = total_forces_temp[:, 1] - 0.
            # F[:, 2] = total_forces_temp[:, 2] - 0.
            # self.register_output('F', total_forces_temp)

            evaluation_pt = self.declare_variable('evaluation_pt',
                                                  val=np.zeros(3, ))
            evaluation_pt_exp = csdl.expand(
                evaluation_pt,
                (eval_pts_all.shape),
                'i->jki',
            )
            r_M = eval_pts_all - evaluation_pt_exp
            total_moment = csdl.sum(csdl.cross(r_M, panel_forces, axis=2),
                                    axes=(1, ))
            M = self.create_output('M', shape=total_moment.shape)
            # self.register_output('F', total_forces)
            # M[:, 0] = total_moment[:, 0] - 0.
            # M[:, 1] = -total_moment[:, 1] - 0.
            # M[:, 2] = total_moment[:, 2] - 0.

            M[:, 0] = total_moment[:, 0] - 0.
            M[:, 1] = -total_moment[:, 1] * 0.
            M[:, 2] = total_moment[:, 2] - 0.
            '''hardcode for testing'''

            # self.register_output('M', total_moment * 0)
            # else:
            #     for i in range(len(surface_names)):
            #         D_total_name = surface_names[i] + '_D_total'
            #     self.register_output(D_total_name, D + 0)

        # !TODO: need to fix eval_pts for main branch
        if eval_pts_option == 'user_defined':
            # sina = csdl.expand(csdl.sin(alpha), (system_size, 1), 'i->ji')
            # cosa = csdl.expand(csdl.cos(alpha), (system_size, 1), 'i->ji')

            # panel_forces = rho * circulation_repeat * csdl.cross(
            #     velocities, bd_vec, axis=1)

            # panel_forces_x = panel_forces[:, 0]
            # panel_forces_y = panel_forces[:, 1]
            # panel_forces_z = panel_forces[:, 2]
            # self.register_output('panel_forces_z', panel_forces_z)

            # L_panel = -panel_forces_x * sina + panel_forces_z * cosa
            # D_panel = panel_forces_x * cosa + panel_forces_z * sina
            start = 0
            for i in range(len(surface_names)):

                mesh = self.declare_variable(surface_names[i],
                                             shape=surface_shapes[i])
                nx = surface_shapes[i][1]
                ny = surface_shapes[i][2]

                delta = (nx - 1) * (ny - 1)

                nx_eval = eval_pts_shapes[i][1]
                ny_eval = eval_pts_shapes[i][2]
                delta_eval = nx_eval * ny_eval

                bd_vec_surface = bd_vec[:, start:start + delta, :]
                print('bd_vec shape', bd_vec.shape)
                print('bd_vec_surface shape', bd_vec_surface.shape)
                print('sprs shape', sprs[i].shape)

                bd_vec_eval = csdl.sparsematmat(bd_vec_surface, sprs[i])
                # sina = csdl.expand(csdl.sin(alpha), (num_nodes, delta_eval, 1),
                #                    'ki->kji')
                # cosa = csdl.expand(csdl.cos(alpha), (num_nodes, delta_eval, 1),
                #                    'ki->kji')
                # sinb = csdl.expand(csdl.sin(beta), (num_nodes, delta_eval, 1),
                #                    'ki->kji')
                # cosb = csdl.expand(csdl.cos(beta), (num_nodes, delta_eval, 1),
                #                    'ki->kji')
                # sina_eval = csdl.sparsematmat(sina, sprs[i])
                # cosa_eval = csdl.sparsematmat(cosa, sprs[i])

                circulation_repeat_surface = circulation_repeat[start:start +
                                                                delta, :]
                circulation_repeat_surface_eval = csdl.sparsematmat(
                    circulation_repeat_surface, sprs[i])

                vel_surface = self.declare_variable(v_total_wake_names[i],
                                                    shape=(num_nodes,
                                                           delta_eval, 3))
                velocities[start:start + delta, :] = vel_surface
                start = start + delta

                panel_forces = rho * circulation_repeat_surface_eval * csdl.cross(
                    vel_surface, bd_vec_eval, axis=2)

                self.register_output(surface_names[i] + 'panel_forces',
                                     panel_forces)

                # bd_vec_surface = bd_vec[start:start + delta, :]
                # circulation_repeat_surface = circulation_repeat[start:start +
                #                                                 delta, :]
                # bd_vec_eval = csdl.sparsematmat(bd_vec_surface, sprs[i])
                # sina_eval = csdl.sparsematmat(sina, sprs[i])
                # cosa_eval = csdl.sparsematmat(cosa, sprs[i])
                # # vel_surface_eval = csdl.sparsematmat(vel_surface, sprs[i])
                # circulation_repeat_surface_eval = csdl.sparsematmat(
                #     circulation_repeat_surface, sprs[i])

                # print('\nbd_vec_eval shape', bd_vec_eval.shape)
                # print('vel_surface shape', vel_surface.shape)
                # print('circulation_repeat_surface_eval shape',
                #       circulation_repeat_surface_eval.shape)

                panel_forces_surface = rho * circulation_repeat_surface_eval * csdl.cross(
                    vel_surface, bd_vec_eval, axis=1)

if __name__ == '__main__':
    import python_csdl_backend
    from VAST.utils.generate_mesh import *

    # from lsdo_uvlm.uvlm_outputs.compute_force.compute_lift_drag import LiftDrag

    ########################################
    # define mesh here
    ########################################
    nx = 29
    ny = 5 # actually 14 in the book


    AR = 8
    span = 12
    chord = span/AR
    # num_nodes = 9*16
    # num_nodes = 16 *2
    num_nodes = 5
    # num_nodes = 3
    nt = num_nodes+1

    alpha = np.deg2rad(15)

    # define the direction of the flapping motion (hardcoding for now)

    # u_val = np.concatenate((np.array([0.01, 0.5,1.]),np.ones(num_nodes-3))).reshape(num_nodes,1)
    # u_val = np.ones(num_nodes).reshape(num_nodes,1)
    u_vel = np.ones(num_nodes).reshape(num_nodes,1)*10
    w_vel = np.zeros((num_nodes,1)) *np.sin(alpha)*10
    # theta_val = np.linspace(0,alpha,num=num_nodes)
    theta_val = np.ones((num_nodes, 1))*alpha


    uvlm_parameters = [('u',True,u_vel),
                       ('v',True,np.zeros((num_nodes, 1))),
                       ('w',True,w_vel),
                       ('p',True,np.zeros((num_nodes, 1))),
                       ('q',True,np.zeros((num_nodes, 1))),
                       ('r',True,np.zeros((num_nodes, 1))),
                       ('theta',True,theta_val),
                       ('psi',True,np.zeros((num_nodes, 1))),
                    #    ('x',True,np.zeros((num_nodes, 1))),
                    #    ('y',True,np.zeros((num_nodes, 1))),
                    #    ('z',True,np.zeros((num_nodes, 1))),
                    #    ('phiw',True,np.zeros((num_nodes, 1))),
                       ('gamma',True,np.zeros((num_nodes, 1))),
                       ('psiw',True,np.zeros((num_nodes, 1)))]

    mesh_dict = {
        "num_y": ny,
        "num_x": nx,
        "wing_type": "rect",
        "symmetry": False,
        "span": span,
        "root_chord": chord,
        "span_cos_spacing": False,
        "chord_cos_spacing": False,
    }

    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict)

    # mesh_val = generate_simple_mesh(nx, ny, num_nodes)
    mesh_val = np.zeros((num_nodes, nx, ny, 3))

    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1]

    # mesh_val_2 = np.zeros((num_nodes, nx, ny, 3))

    # for i in range(num_nodes):
    #     mesh_val_2[i, :, :, 2] = mesh2.copy()[:, :, 2] + .25
    #     mesh_val_2[i, :, :, 0] = mesh2.copy()[:, :, 0] + 5
    #     mesh_val_2[i, :, :, 1] = mesh2.copy()[:, :, 1]

    uvlm_parameters.append(('wing', True, mesh_val))
    # uvlm_parameters.append(('uvlm_wing2',True,mesh_val_2))


    # surface_names=['wing','wing2']
    # surface_shapes=[(nx, ny, 3),(nx, ny, 3)]
    surface_names=['wing']
    surface_shapes=[(nx, ny, 3)]
    h_stepsize = delta_t = 1/16


    initial_conditions = []
    for i in range(len(surface_names)):
        surface_name = surface_names[i]
        gamma_w_0_name = surface_name + '_gamma_w_0'
        wake_coords_0_name = surface_name + '_wake_coords_0'
        surface_shape = surface_shapes[i]
        nx = surface_shape[0]
        ny = surface_shape[1]
        initial_conditions.append((gamma_w_0_name, np.zeros((num_nodes, ny - 1))))

        initial_conditions.append((wake_coords_0_name, np.zeros((num_nodes, ny, 3))))

    profile_outputs = []

    profile_outputs.append(('wing_L', (num_nodes,1)))
    profile_outputs.append(('wing_D', (num_nodes,1)))

    profile_params_dict = {
            'surface_names': ['wing'],
            'surface_shapes': surface_shapes,
            'delta_t': delta_t,
            'nt': nt
        }

    model = m3l.DynamicModel()
    uvlm = VASTSolverUnsteady(num_nodes=num_nodes, surface_names=surface_names, surface_shapes=surface_shapes, delta_t=delta_t, nt=num_nodes+1)
    uvlm_residual = uvlm.evaluate()
    model.register_output(uvlm_residual)
    model.set_dynamic_options(initial_conditions=initial_conditions,
                              num_times=num_nodes,
                              h_stepsize=delta_t,
                              parameters=uvlm_parameters,
                              integrator='ForwardEuler',
                              profile_outputs=None,
                              profile_system=None,
                              profile_parameters=None)
    model_csdl = model.assemble()

    submodel = ProfileOpModel(
        num_nodes = num_nodes,
        surface_names = surface_names,
        surface_shapes = surface_shapes,
        delta_t = h_stepsize,
        nt = num_nodes + 1
    )

    model_csdl.add(submodel, name='post_processing')



    sim = python_csdl_backend.Simulator(model_csdl, analytics=True)
    # Before code
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    sim.run()
    # After code
    profiler.disable()
    profiler.dump_stats('output')

    # print(sim['post_processing.ThrustDrag.wing_L'])
    # print(sim['post_processing.ThrustDrag.wing_D'])

    if True:
        from vedo import dataurl, Plotter, Mesh, Video, Points, Axes, show
        axs = Axes(
            xrange=(0, 35),
            yrange=(-10, 10),
            zrange=(-3, 4),
        )
        video = Video("uvlm_m3l_test.gif", duration=10, backend='ffmpeg')
        for i in range(nt - 1):
            vp = Plotter(
                bg='beige',
                bg2='lb',
                # axes=0,
                #  pos=(0, 0),
                offscreen=False,
                interactive=1)
            # Any rendering loop goes here, e.g.:
            for surface_name in surface_names:
                surface_name = 'prob.' + surface_name
                vps = Points(np.reshape(sim[surface_name][i, :, :, :], (-1, 3)),
                            r=8,
                            c='red')
                vp += vps
                vp += __doc__
                vps = Points(np.reshape(sim[surface_name+'_wake_coords_integrated'][i, 0:i, :, :],
                                        (-1, 3)),
                            r=8,
                            c='blue')
                vp += vps
                vp += __doc__
            # cam1 = dict(focalPoint=(3.133, 1.506, -3.132))
            # video.action(cameras=[cam1, cam1])
            vp.show(axs, elevation=-60, azimuth=-0,
                    axes=False, interactive=False)  # render the scene
            video.add_frame()  # add individual frame
            # time.sleep(0.1)
            # vp.interactive().close()
            vp.close_window()
        vp.close_window()
        video.close()  # merge all the recorded frames
