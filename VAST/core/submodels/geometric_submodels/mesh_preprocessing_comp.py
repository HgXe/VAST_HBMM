import csdl
import numpy as np
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL

class MeshPreprocessingComp(ModuleCSDL):
    """
    Compute various geometric properties for VLM analysis.

    parameters
    ----------
    def_mesh[num_nodes,num_pts_chord, num_pts_span, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.
    w

    Returns
    -------
    1. bd_vtx_coords[num_nodes,num_pts_chord, num_pts_span, 3] : csdl array
    bound vortices points
    2. coll_pts[num_nodes, num_pts_chord-1, num_pts_span-1, 3] : csdl array
        collocation points for the horseshoe vortices, found along the 3/4 chord.
    w
    3. l_span[num_nodes, (num_pts_chord-1), (num_pts_span-1)] : csdl array
        The spanwise widths of each individual panel.
    4. l_chord[num_nodes, (num_pts_chord-1), (num_pts_span-1)] : csdl array
        The chordwise length of of each individual panel.
    5. s_panel [num_nodes, (num_pts_chord-1), (num_pts_span-1)]: csdl array
        The panel areas.
    6. bd_vec_all[num_nodes,system_size,3]: bd_vec of all lifting surfaces
    w
    7. eval_pts_coords
    w
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('eval_pts_option', default='auto')
        self.parameters.declare('eval_pts_location',default=0.25)
        self.parameters.declare('delta_t',default=0)
        self.parameters.declare('problem_type',default='fixed_wake')
        self.parameters.declare('compressible',default=False)
        self.parameters.declare('Ma',default=None)

    def define(self):
        # load options
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        mesh_unit = self.parameters['mesh_unit']
        num_nodes = surface_shapes[0][0]
        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_location = self.parameters['eval_pts_location']
        delta_t = self.parameters['delta_t']
        problem_type = self.parameters['problem_type']
        compressible = self.parameters['compressible']
        Ma = self.parameters['Ma']

        system_size = sum((i[1] - 1) * (i[2] - 1) for i in surface_shapes)
        def_mesh_list = []

        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]

        else:
            eval_pts_names=self.parameters['eval_pts_names']

        # loop through lifting surfaces to compute outputs
        start = 0
        chord_lengths = []
        for i in range(len(surface_names)):
            # load name of the geometry mesh, number of points in chord and spanwise direction
            surface_name = surface_names[i]
            num_pts_chord = surface_shapes[i][1]
            num_pts_span = surface_shapes[i][2]

            delta = (num_pts_chord - 1) * (num_pts_span - 1)

            # get names of the outputs:
            # 1. bd_vtx_coords_name, 2. coll_pts_coords_name, 3. chord_name, 4. span_name, 5. s_panel_name
            bd_vtx_coords_name = surface_name + '_bd_vtx_coords'
            coll_pts_coords_name = surface_name + '_coll_pts_coords'
            chord_name = surface_name + '_chord_length'
            span_name = surface_name + '_span_length'
            s_panel_name = surface_name + '_s_panel'

            # add_input

            # declare the input variable lifting surface (deformed) mesh,
            # this should come from CADDEE geometry if connected,
            # or up to the user to create an input if using the solver alone.
            if mesh_unit == 'm':
                def_mesh = self.register_module_input(surface_name, shape=surface_shapes[i], promotes=True)
            elif mesh_unit == 'ft':
                def_mesh_ft = self.register_module_input(surface_name, shape=surface_shapes[i], promotes=True)

                def_mesh = def_mesh_ft * 0.
            mesh_org = def_mesh+0
            
            if compressible:
                beta = (1-Ma**2)**0.5
                def_mesh = self.create_output(surface_name+'_compressible',shape=surface_shapes[i])
                def_mesh[:, :, :, 0] = mesh_org[:, :, :, 0] 
                def_mesh[:, :, :, 1] = mesh_org[:, :, :, 1] * beta
                def_mesh[:, :, :, 2] = mesh_org[:, :, :, 2] * beta

            ################################################################################
            # create the output: 1. bd_vtx_coords
            ################################################################################
            bd_vtx_coords = self.create_output(bd_vtx_coords_name, shape=(def_mesh.shape))
            # the 0th until the second last one chordwise is (0.75*left +0.25*right)
            bd_vtx_coords[:, 0:num_pts_chord - 1, :, :] = def_mesh[:, 0:num_pts_chord - 1, :, :] * .75 +\
                                                             def_mesh[:, 1:num_pts_chord, :, :] * 0.25
            # the last one chordwise is 1/4 chord offset from the last chordwise def_mesh panel
            if problem_type == 'fixed_wake':
                bd_vtx_coords[:, num_pts_chord -
                            1, :, :] = def_mesh[:, num_pts_chord -
                                                1, :, :] + 0.25 * (
                                                    def_mesh[:, num_pts_chord -
                                                            1, :, :] -
                                                    def_mesh[:, num_pts_chord -
                                                            2, :, :])

                # in BYU lab VortexLattice.ji, the last one chordwise bd_vtx is the same as the TE panel


            elif problem_type == 'prescribed_wake':
                frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))
                w = self.declare_variable('w', shape=(num_nodes, 1))
                fs  = self.create_output(surface_name + '_fs', shape=(num_nodes, 3))
                fs[:,0] = -frame_vel[:,0]
                fs[:,1] = -frame_vel[:,1]
                # fs[:,2] = -frame_vel[:,2]
                fs[:,2] = w
                # self.print_var(fs)
                # self.print_var(w)
                eta = 0.25
                add_starting_wake = csdl.expand(fs*eta*delta_t,(num_nodes,1,num_pts_span,3),'il->ijkl')

                bd_vtx_coords[:, num_pts_chord -1, :, :] = def_mesh[:, num_pts_chord - 1, :, :] + add_starting_wake

                # bd_vtx_coords[:, num_pts_chord -
                #             1, :, :] = def_mesh[:, num_pts_chord -
                #                                 1, :, :] + 0.25 * (
                #                                     def_mesh[:, num_pts_chord -
                #                                             1, :, :] -
                #                                     def_mesh[:, num_pts_chord -
                #                                             2, :, :])

            ################################################################################
            # compute the output: 2. coll_pts_coords (center point of the bd_vtx panels,
            # approx 3/4 chord middle span of def_mesh)
            ################################################################################
            # coll_pts_coords = def_mesh[:, 0:num_pts_chord -1, :, :] * .25 + def_mesh[:, 1:num_pts_chord, :, :] * 0.75

            # TODO: this might not work very well for the panels that are not rectangular
            coll_pts_coords = 0.25/2 * (def_mesh[:,0:num_pts_chord-1, 0:num_pts_span-1, :] +def_mesh[:,0:num_pts_chord-1, 1:num_pts_span, :]) +\
                                         0.75/2 * (def_mesh[:,1:, 0:num_pts_span-1, :]+def_mesh[:,1:, 1:, :])

            # coll_pts_coords = 0.25 * (bd_vtx_coords[:,0:num_pts_chord-1, 0:num_pts_span-1, :] +\
            #                                 bd_vtx_coords[:,0:num_pts_chord-1, 1:num_pts_span, :] +\
            #                                 bd_vtx_coords[:,1:, 0:num_pts_span-1, :]+\
            #                                 bd_vtx_coords[:,1:, 1:, :])

            self.register_output(coll_pts_coords_name, coll_pts_coords)
            # self.print_var(coll_pts_coords)

            ################################################################################
            # compute the output: 3. chords- I don't think VLM deal w/ unstrunctured mesh
            # (a vector that contains the chord lengths of each panel)
            # TODO: implement chords, and spans as an average of top and bottom???
            ################################################################################
            chords_vec = def_mesh[:, 0:num_pts_chord - 1, :, :] - def_mesh[:, 1:, :, :]
            chords_temp = csdl.pnorm(chords_vec, axis=(3))
            chords = 0.5*(chords_temp[:, :, 0:num_pts_span-1] + chords_temp[:, :, 1:])
            self.register_output(chord_name, chords)
            chord_lengths.append(chords)

            ################################################################################
            # compute the output: 4. spans
            # (a vector that contains the span lengths of each panel)
            ################################################################################
            # compute the spans:
            span_vec = def_mesh[:, :, 0:num_pts_span -1, :] - def_mesh[:, :, 1:, :]
            spans_temp = csdl.pnorm(span_vec, axis=(3))
            spans = 0.5*(spans_temp[:, 0:num_pts_chord-1, :] + spans_temp[:, 1:, :])
            self.register_output(span_name, spans)

            ################################################################################
            # compute the output: 5. s_panels: panel area of each panel
            # TODO: implement projected area if needed
            ################################################################################
            i = mesh_org[:, :-1, 1:, :] - mesh_org[:, 1:, :-1, :]
            j = mesh_org[:, :-1, :-1, :] - mesh_org[:, 1:, 1:, :]
            # compute the wetted area:
            normals = csdl.cross(i, j, axis=3)
            s_panels = (csdl.sum(normals**2, axes=(3, )))**0.5 * 0.5
            self.register_output(s_panel_name, s_panels)

            def_mesh_list.append(def_mesh)

        ################################################################################
        # create the output: 6. bd_vec_all: bd_vec of all lifting surfaces
        ################################################################################
        bd_vec_all = self.create_output('bd_vec',
                                        shape=(num_nodes, system_size, 3))
        start = 0
        for i in range(len(surface_names)):
            # load name of the geometry mesh, number of points in chord and spanwise direction
            surface_name = surface_names[i]
            num_pts_chord = surface_shapes[i][1]
            num_pts_span = surface_shapes[i][2]

            delta = (num_pts_chord - 1) * (num_pts_span - 1)
            bound_vecs = -csdl.reshape(
                (0.75 * def_mesh_list[i][:, 0:-1, 0:-1, :] +
                 0.25 * def_mesh_list[i][:, 1:, 0:-1, :] +
                 -0.75 * def_mesh_list[i][:, 0:-1, 1:, :] +
                 -0.25 * def_mesh_list[i][:, 1:, 1:, :]),
                new_shape=(num_nodes, (num_pts_chord - 1) * (num_pts_span - 1),
                           3))

            bd_vec_all[:, start:start + delta, :] = bound_vecs
            start += delta
        ################################################################################
        if self.parameters['eval_pts_option'] == 'auto':
            for i in range(len(surface_shapes)):
                surface_name = surface_names[i]
                # if mesh_unit == 'm':
                #     mesh = self.register_module_input(surface_name, shape=surface_shapes[i], promotes=True)
                # elif mesh_unit == 'ft':
                #     mesh_ft = self.register_module_input(surface_name, shape=surface_shapes[i], promotes=True)
                #     mesh = mesh_ft * 0.3048

                # if compressible:
                #     beta = (1-Ma**2)**0.5
                #     mesh_org = mesh+0
                #     mesh = self.create_output(surface_name+'_compressible',shape=surface_shapes[i])
                #     mesh[:, :, :, 0] = mesh_org[:, :, :, 0] 
                #     mesh[:, :, :, 1] = mesh_org[:, :, :, 1] * beta
                #     mesh[:, :, :, 2] = mesh_org[:, :, :, 2] * beta

                eval_pts_coords = (
                    (1 - eval_pts_location) * 0.5 * def_mesh_list[i][:, 0:-1, 0:-1, :] +
                    (1 - eval_pts_location) * 0.5 * def_mesh_list[i][:, 0:-1, 1:, :] +
                    eval_pts_location * 0.5 * def_mesh_list[i][:, 1:, 0:-1, :] +
                    eval_pts_location * 0.5 * def_mesh_list[i][:, 1:, 1:, :])

                self.register_output(eval_pts_names[i], eval_pts_coords)
        del def_mesh_list

        # elif self.parameters['eval_pts_option'] == 'user_defined':
        #     for i in range(len(eval_pts_shapes)):
        #         eval_pts_coords = self.declare_variable(
        #             eval_pts_names[i], shape=(eval_pts_shapes[i]))


        ################################################################################
        # create the output: 7. re_span: Reynolds # of each strip
        ################################################################################
        rho = self.register_module_input('density', shape=(num_nodes, 1), val=1.1)
        v_inf_sq = self.register_module_input('v_inf_sq', shape=(num_nodes, 1), val=1.1)

        mu = 1.8e-5

        for i in range(len(surface_names)):
            chord_length = chord_lengths[i]
            ny = chord_length.shape[2]
            chord_length_span = csdl.reshape(csdl.sum(chord_length, axes=(1,)), new_shape=(num_nodes, ny, 1))

            rho_v_exp = csdl.expand(rho * (v_inf_sq**0.5), (num_nodes,chord_length_span.shape[1],1),'ik->ijk')

            re_span = rho_v_exp * chord_length_span / mu
            self.register_module_output(surface_names[i] + '_re_span', re_span)


if __name__ == "__main__":
    import python_csdl_backend
    simulator_name = 'python_csdl_backend'

    num_nodes = 2
    num_pts_chord = 3
    num_pts_span = 4

    def generate_simple_mesh(num_nodes, num_pts_chord, num_pts_span):
        mesh = np.zeros((num_nodes, num_pts_chord, num_pts_span, 3))
        for i in range(num_nodes):
            mesh[i, :, :, 0] = np.outer(np.arange(num_pts_chord),
                                        np.ones(num_pts_span))
            mesh[i, :, :, 1] = np.outer(np.arange(num_pts_span),
                                        np.ones(num_pts_chord)).T
            mesh[i, :, :, 2] = 0.
        return mesh

    surface_names = ['wing_1', 'wing_2']
    surface_shapes = [(num_nodes, num_pts_chord, num_pts_span, 3),
                      (num_nodes, num_pts_chord + 1, num_pts_span + 1, 3)]
    model_1 = csdl.Model()

    wing_1_mesh = generate_simple_mesh(num_nodes, num_pts_chord, num_pts_span)
    wing_2_mesh = generate_simple_mesh(num_nodes, num_pts_chord + 1,
                                       num_pts_span + 1)

    wing_1_inputs = model_1.create_input('wing_1', val=wing_1_mesh)
    wing_2_inputs = model_1.create_input('wing_2', val=wing_2_mesh)

    model_1.add(MeshPreprocessingComp(surface_names=surface_names,
                                      surface_shapes=surface_shapes),
                name='preprocessing_group')

    if simulator_name == 'csdl_om':

        sim = Simulator(model_1)

        sim.run()
        # sim.prob.check_partials(compact_print=True)
        partials = sim.prob.check_partials(compact_print=True, out_stream=None)
        sim.assert_check_partials(partials, 1e-5, 1e-7)
        sim.visualize_implementation()
        sim.prob.check_config(checks=['unconnected_inputs'], out_file=None)

    elif simulator_name == 'python_csdl_backend':
        sim = python_csdl_backend.Simulator(model_1)

        sim.run()
        sim.check_partials(compact_print=True)

    import matplotlib.pyplot as plt

    plt.plot(sim[surface_names[0] + '_bd_vtx_coords'][0, :, :, 0].flatten(),
             sim[surface_names[0] + '_bd_vtx_coords'][0, :, :,
                                                      1].flatten(), 'ro')
    plt.plot(sim[surface_names[0] + '_coll_pts_coords'][0, :, :, 0].flatten(),
             sim[surface_names[0] + '_coll_pts_coords'][0, :, :,
                                                        1].flatten(), 'bo')
    plt.legend(['bs', 'coll'])

    plt.axis('equal')

    plt.show()