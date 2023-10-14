from turtle import shape
# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import axis
import numpy as np

from VAST.core.submodels.output_submodels.vlm_post_processing.compute_effective_aoa_cd_v import AOA_CD


class ThrustDrag(Model):
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
        self.parameters.declare('delta_t',default=0.5)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']
        # print('delta_t-------------------------------',delta_t)


        num_nodes = surface_shapes[0][0]
        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        cl_span_names = [x + '_cl_span' for x in surface_names]

        system_size = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            system_size += (nx - 1) * (ny - 1)

        rho = self.declare_variable('density', shape=(num_nodes, 1))
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
        
        # b = self.declare_variable('b',shape=(num_nodes,system_size))


        # gamma_b_repeat = csdl.expand(gamma_b/csdl.reshape(((b**2)**0.5),(num_nodes,system_size)), (num_nodes, system_size, 3),'ki->kij')
        # gamma_b_repeat = circulation_repeat

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
                # print('v_total_wake_names-----------------------', v_total_wake_names[i])

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

            panel_forces = rho_expand * circulation_repeat * csdl.cross(
                velocities, bd_vec, axis=2)

            self.register_output('panel_forces', panel_forces)
            wing_inital = self.declare_variable(surface_names[0],shape=(num_nodes,nx,ny,3))
            # TODO: delete this hardcoding


            gamma_b = self.declare_variable('gamma_b',shape=(num_nodes, system_size))
            gamma_b_repeat = csdl.expand(gamma_b* s_panels_all,(num_nodes, system_size, 3),'ki->kij')   



            # gamma_b_repeat = csdl.expand(gamma_b,(num_nodes, system_size, 3),'ki->kij')  
          
            c_bar = wing_inital[0,nx-1,0,0] - wing_inital[0,nx-2,0,0]
            # self.print_var(c_bar)
            c_bar_exp = csdl.reshape(csdl.expand(csdl.reshape(c_bar,(1,)), (num_nodes*system_size*3,1),'i->ji'),(num_nodes,system_size,3))
            dcirculation_repeat_dt = self.create_output('dcirculation_repeat_dt',shape=(num_nodes,system_size,3))
            # print('dcirculation_repeat_dt shape is:\n',dcirculation_repeat_dt.shape)
            # dcirculation_repeat_dt[0,:,:] = (gamma_b_repeat[1,:,:]-gamma_b_repeat[0,:,:])/delta_t
            dcirculation_repeat_dt[0,:,:] = (gamma_b_repeat[0,:,:])/delta_t
            dcirculation_repeat_dt[1:num_nodes-1,:,:] = (gamma_b_repeat[2:num_nodes,:,:]-gamma_b_repeat[0:num_nodes-2,:,:])/(delta_t*2)
            dcirculation_repeat_dt[num_nodes-1,:,:] = (gamma_b_repeat[num_nodes-1,:,:]-gamma_b_repeat[num_nodes-2,:,:])/delta_t

            # panel_forces_dynamic = rho_expand * dcirculation_repeat_dt* c_bar_exp * csdl.cross(
                # velocities, bd_vec, axis=2)

            normals = self.declare_variable(surface_names[0] + '_bd_vtx_normals',shape=(num_nodes,system_size,3))
            # NOTE: this direction needs some verification
            panel_forces_dynamic = rho_expand * dcirculation_repeat_dt * normals

            panel_forces_x = panel_forces[:, :, 0] + panel_forces_dynamic[:, :, 0]
            panel_forces_y = panel_forces[:, :, 1] + panel_forces_dynamic[:, :, 1]
            panel_forces_z = panel_forces[:, :, 2] + panel_forces_dynamic[:, :, 2]
            self.register_output('panel_forces_x', panel_forces_x)
            self.register_output('panel_forces_y', panel_forces_y)
            self.register_output('panel_forces_z', panel_forces_z)
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
                # self.print_var(rho)
                # self.print_var(s_panels_sum)
                # self.print_var(b)

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
                    coeffs_cd=coeffs_cd,
                )
                self.add(sub, name='AOA_CD')

                #     cd_v_names = [x + '_cd_v' for x in surface_names]

                #     for i in range(len(surface_names)):
                D_total_name = surface_names[i] + '_D_total'

                #         cd_v = self.declare_variable(cd_v_names[i],
                #                                      shape=(num_nodes, 1))
                #         c_d_total = cd_v + c_d
                CD_total_names = [x + '_C_D' for x in surface_names]


            total_forces_temp = csdl.sum(panel_forces, axes=(1, ))

            '''this is hardcodes with one surface'''
            initial_mesh_names = [
                x + '_initial_mesh' for x in self.parameters['surface_names']
            ]
            # panel_forces_dynamic = rho_expand * dcirculation_repeat_dt* c_bar_exp * csdl.cross(
            #     velocities, bd_vec, axis=2)
            total_forces_temp_dynamic = csdl.sum(panel_forces_dynamic, axes=(1, ))
            self.register_output('rho_expand',rho_expand)
            # self.register_output('dcirculation_repeat_dt',dcirculation_repeat_dt)
            self.register_output('c_bar_exp',c_bar_exp)
            self.register_output('total_forces_temp_dynamic',total_forces_temp_dynamic)
            self.register_output('crosspd',csdl.cross(
                velocities, bd_vec, axis=2))

            panel_forces_all = panel_forces + panel_forces_dynamic
            self.register_output('panel_forces_all', panel_forces_all)
            self.register_output('panel_forces_dynamic', panel_forces_dynamic)
            panel_forces_all_mag = csdl.sum(panel_forces_all**2,axes=(2,))**0.5
            velocities_mag = csdl.sum(velocities**2,axes=(2,))**0.5
            panel_power = csdl.sum(panel_forces_all_mag*velocities_mag,axes=(1,))

            # panel_power = csdl.sum(csdl.dot(panel_forces_all,velocities,axis=2),axes=(1,))
            # panel_power = csdl.dot(panel_forces_all,velocities,axis=2)
            self.register_output('panel_power',panel_power)

            F = self.create_output('F', shape=(num_nodes, 3))
            F[:, 0] = total_forces_temp[:, 0] + total_forces_temp_dynamic[:, 0]
            F[:, 1] = total_forces_temp[:, 1] + total_forces_temp_dynamic[:, 1]
            F[:, 2] = -total_forces_temp[:, 2] - total_forces_temp_dynamic[:, 2]

            F_s = self.create_output('F_s', shape=(num_nodes, 3))
            F_s[:, 0] = total_forces_temp[:, 0] 
            F_s[:, 1] = total_forces_temp[:, 1] 
            F_s[:, 2] = -total_forces_temp[:, 2] 
            self.register_output('thrust',F[:,0]) # thurst is negative x force

            CD_0 = 0.1936
            CD_1 = 0.1412
            alpha_deg = 0
            alpha = alpha_deg / 180 * np.pi
            CD_v = CD_0 + CD_1 * alpha**2
            s_panels_sum = csdl.reshape(csdl.sum(s_panels_all, axes=(1, )),
                                        (num_nodes, 1))
            

            Drag = 0.5*rho*b*s_panels_sum*CD_0
            self.register_output('Drag', Drag)

            # print(csdl.sum(F[:,0]).shape)
            # print(Drag.shape)

            res = (csdl.sum(F[:,0])/num_nodes-csdl.sum(Drag)/num_nodes)**2
            self.register_output('res', res)

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


if __name__ == "__main__":

    nx = 3
    ny = 4
    model_1 = Model()
    surface_names = ['wing']
    surface_shapes = [(nx, ny, 3)]

    frame_vel_val = np.array([-1, 0, -1])
    f_val = np.einsum(
        'i,j->ij',
        np.ones(6),
        np.array([-1, 0, -1]) + 1e-3,
    )

    # coll_val = np.random.random((4, 5, 3))

    frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)
    gamma_b = model_1.create_input('gamma_b',
                                   val=np.random.random(((nx - 1) * (ny - 1))))
    force_pt_vel = model_1.create_input('force_pt_vel', val=f_val)

    model_1.add(
        LiftDrag(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
        ))

    frame_vel = model_1.declare_variable('L', shape=(1, ))
    frame_vel = model_1.declare_variable('D', shape=(1, ))

    sim = Simulator(model_1)
    sim.run()
