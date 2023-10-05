import m3l
from VAST.core.vlm_llt.vlm_dynamic_old.VLM_prescribed_wake_system import ODESystemModel

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
            self.residual_names.append((gamma_w_name,dgammaw_dt_name,(num_nodes, ny-1)))
            self.residual_names.append((wing_wake_coords_name,dwake_coords_dt_name,(num_nodes,ny,3)))

        self.inputs = {}
        self.arguments = {}

        residual = m3l.Variable(self.residual_names[0][1], shape=(), operation=self)
        return residual #, frame_vel, bd_vec, horseshoe_circulation
    def compute_residual(self, num_nodes):
        model = ODESystemModel(num_nodes=num_nodes,
                               surface_names=self.surface_names,
                               surface_shapes=self.surface_shapes,
                               delta_t=self.delta_t,
                               nt=self.nt)
        return model
    


if __name__ == '__main__':
    pass