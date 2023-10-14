import time

import matplotlib.pyplot as plt

from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_system import ODESystemModel
from VAST.core.submodels.output_submodels.vlm_post_processing.compute_thrust_drag_dynamic_old import ThrustDrag

from ozone.api import ODEProblem
import csdl

import numpy as np

from VAST.core.submodels.kinematic_submodels.adapter_comp import AdapterComp
from VAST.core.submodels.aerodynamic_submodels.combine_gamma_w import CombineGammaW
from VAST.core.submodels.implicit_submodels.solve_group import SolveMatrix
from VAST.core.submodels.aerodynamic_submodels.seperate_gamma_b import SeperateGammab
from VAST.core.submodels.geometric_submodels.mesh_preprocessing_comp import MeshPreprocessingComp
from VAST.core.submodels.output_submodels.vlm_post_processing.horseshoe_circulations import HorseshoeCirculations
from VAST.core.submodels.output_submodels.vlm_post_processing.eval_pts_velocities_mls import EvalPtsVel

class ODEProblemTest(ODEProblem):
    def setup(self):

        # profile outputs are outputs from the ode integrator that are not states. 
        # instead they are outputs of a function of the solved states and parameters
        nt = self.num_times
        # surface_names = list(self.dictionary_inputs.keys())
        # surface_shapes = list(self.dictionary_inputs.values())
        # frame = self.dictionary_inputs.values()

        surface_shapes = self.dictionary_inputs['surface_shapes']
        surface_names = list(self.dictionary_inputs['surface_names'])
        frame = self.dictionary_inputs['frame']


        # self.add_profile_output('density')
        # self.add_profile_output('alpha')
        # self.add_profile_output('beta')
        # self.add_profile_output('frame_vel',shape=(3,))
        # self.add_profile_output('evaluation_pt')
        # self.add_profile_output('bd_vec', shape=((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),3))

        # self.add_profile_output('horseshoe_circulation', shape=((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),))

        ####################################
        # ode parameter names
        ####################################        
        self.add_parameter('u', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('v', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('w', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('p', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('q', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('r', dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('theta',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('x',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('y',dynamic=True, shape=(self.num_times, 1))
        # self.add_parameter('z',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('psi',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('gamma',dynamic=True, shape=(self.num_times, 1))
        self.add_parameter('psiw',dynamic=True, shape=(self.num_times, 1))

        gamma_w_name_list = []
        wing_wake_coords_name_list = []

        for i in range(len(surface_names)):
            ####################################
            # ode parameter names
            ####################################
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]
            # print('surface_name', surface_name)
            # print('surface_shape', nx, ny)
            self.add_parameter(surface_name,
                               dynamic=True,
                               shape=(self.num_times, nx, ny, 3))
            self.add_parameter(surface_name + '_coll_vel',
                                dynamic=True,
                                shape=(self.num_times, (nx-1), (ny-1), 3))

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
            gamma_w_int_name = 'op_'+ surface_name + '_gamma_w'
            wake_coords_int_name = 'op_' + surface_name + '_wake_coords'
            self.add_state(gamma_w_name,
                           dgammaw_dt_name,
                           shape=(nt - 1, ny - 1),
                           initial_condition_name=gamma_w_0_name,
                           output=gamma_w_int_name)
            self.add_state(wing_wake_coords_name,
                           dwake_coords_dt_name,
                           shape=(nt - 1, ny, 3),
                           initial_condition_name=wake_coords_0_name,
                           output=wake_coords_int_name)

            # self.add_profile_output(surface_name+'_gamma_b', shape=((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),))
            '''TODO: uncomment this'''
            # self.add_profile_output(surface_name+'_eval_pts_coords', shape=((surface_shapes[0][0]-1),(surface_shapes[0][1]-1),3))
            # self.add_profile_output(surface_name+'_s_panel', shape=((surface_shapes[0][0]-1),(surface_shapes[0][1]-1)))
            # self.add_profile_output(surface_name+'_eval_total_vel', shape=((surface_shapes[0][0]-1)*(surface_shapes[0][1]-1),3))

            ####################################
            # profile outputs
            ####################################
            # F_name = surface_name + '_F'
            # self.add_profile_output(F_name)

        self.add_times(step_vector='h')

        # Define ODE and Profile Output systems (Either CSDL Model or Native System)
        self.set_ode_system(ODESystemModel)
        # self.set_profile_system(ProfileOpModel)




class UVLMSolver(csdl.Model):
    '''This class generates the solver for the prescribed VLM.'''

    def initialize(self):
        self.parameters.declare('num_times')
        self.parameters.declare('h_stepsize')
        self.parameters.declare('states_dict')
        self.parameters.declare('surface_properties_dict')
        self.parameters.declare('mesh_val',default=None)
        self.parameters.declare('problem_type',default='fixed_wake')
        self.parameters.declare('symmetry',default=False)
        self.parameters.declare('compressible',default=False)
        self.parameters.declare('Ma',default=0.84)

    def define(self):
        num_times = self.parameters['num_times']

        h_stepsize = self.parameters['h_stepsize']
        mesh_val = self.parameters['mesh_val']

        AcStates_val_dict = self.parameters['states_dict']
        surface_properties_dict = self.parameters['surface_properties_dict']
        surface_names = list(surface_properties_dict['surface_names'])
        surface_shapes = list(surface_properties_dict['surface_shapes'])
        frame = surface_properties_dict['frame']
        ode_surface_shapes = [(num_times, ) + item for item in surface_shapes]


        ####################################
        # Create parameters
        ####################################
        for data in AcStates_val_dict:
            string_name = data
            val = AcStates_val_dict[data]            
            # print('{:15} = {},shape{}'.format(string_name, val, val.shape))
            variable = self.create_input(string_name,
                                         val=val)
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            gamma_w_0_name = surface_name + '_gamma_w_0'
            wake_coords_0_name = surface_name + '_wake_coords_0'
            surface_shape = surface_shapes[i]
            nx = surface_shape[0]
            ny = surface_shape[1]
            ####################################
            # Create parameters
            ####################################
            '''1. wing'''
            # if mesh_val!=None:
            #     wing_val = mesh_val
            #     wing = self.create_input(surface_name, wing_val)

            ########################################
            # Initial condition for states
            ########################################
            '''1. wing_gamma_w_0'''
            wing_gamma_w_0 = self.create_input(gamma_w_0_name, np.zeros((num_times - 1, ny - 1)))

            '''2. wing_wake_coords_0'''
            wing_wake_coords_0_val = np.zeros((num_times - 1, ny, 3))
            wing_wake_coords_0 = self.create_input(wake_coords_0_name, wing_wake_coords_0_val)

        ########################################
        # Timestep vector
        ########################################
        h_vec = np.ones(num_times - 1) * h_stepsize
        h = self.create_input('h', h_vec)
        ########################################
        # params_dict to the init of ODESystem
        ########################################
        params_dict = {
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': h_stepsize,
            'nt': num_times,
            'frame': frame,
             'symmetry': self.parameters['symmetry'],   
        }

        profile_params_dict = {
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': h_stepsize,
            'nt': num_times,
        }

        # add an actuation model on the upstream
        # self.add(ActuationModel(surface_names=surface_names, surface_shapes=surface_shapes, num_nodes=nt-1),'actuation_temp')

        # Create Model containing integrator
        ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching checkpointing', num_times, display='default', visualization='None',dictionary_inputs=surface_properties_dict)
        # ODEProblem = ODEProblemTest('ForwardEuler', 'time-marching', num_times, display='default', visualization='None',dictionary_inputs=surface_properties_dict)

        self.add(ODEProblem.create_solver_model(ODE_parameters=params_dict,
                                                profile_parameters=profile_params_dict), 'subgroup')
        # self.add(ProfileSystemModel(**profile_params_dict),'profile_outputs')
        # self.add_design_variable('u',lower=1e-3, upper=10)
        # self.add_objective('res')
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

        # self.add_design_variable('theta',upper=np.deg2rad(5),lower=-np.deg2rad(5))
        # cl = self.declare_variable('wing_C_L',shape=(num_times,1))
        # wing_C_L_sum = -csdl.sum(cl,axes=(0,))
        # self.register_output('wing_C_L_sum', wing_C_L_sum)
        # self.add_objective('wing_C_L_sum')
