import csdl
import numpy as np

class KinematicVelocityComp(csdl.Model):
    """
    Compute the kinematic velocity of the surface
    based on the frame velocity, the angular velocity,
    the rotation reference point
    , and the coll pts velocity (e.g., the fish body deformation).

    parameters
    ----------
    frame_vel
    p
    q
    r
    rotation reference point (surface_name + '_rot_ref') (num_nodes, 3)
    coll_vel (surface_name+'_coll_vel')

    Returns
    -------
    ang_vel[num_nodes,3]: p, q, r
    kinematic_vel[num_nodes, (num_evel_pts_x* num_evel_pts_y), 3] : csdl array
        (surface_name + '_kinematic_vel')
        undisturbed fluid velocity (x is positive)
    """
    def initialize(self):

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        # get num_nodes from surface shape
        num_nodes = surface_shapes[0][0]

        # add_output name and shapes
        kinematic_vel_names = [
            x + '_kinematic_vel' for x in self.parameters['surface_names']
        ]

        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        p = self.declare_variable('p', shape=(num_nodes, 1), val=0)
        q = self.declare_variable('q', shape=(num_nodes, 1), val=0)
        r = self.declare_variable('r', shape=(num_nodes, 1), val=0)

        ang_vel = self.create_output('ang_vel', shape=(num_nodes, 3))
        ang_vel[:, 0] = p
        ang_vel[:, 1] = q
        ang_vel[:, 2] = r

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            num_pts_chord = surface_shapes[i][1]
            num_pts_span = surface_shapes[i][2]
            kinematic_vel_name = kinematic_vel_names[i]
            out_shape = (num_nodes, (num_pts_chord - 1) * (num_pts_span - 1),
                         3)

            coll_pts_coords_name = surface_name + '_coll_pts_coords'
            rot_ref_name = surface_name + '_rot_ref'

            coll_pts = self.declare_variable(coll_pts_coords_name,
                                             shape=(num_nodes,
                                                    num_pts_chord - 1,
                                                    num_pts_span - 1, 3))
            rot_ref = self.declare_variable(rot_ref_name,
                                             np.zeros((num_nodes, 3)))
            rot_ref_exp = csdl.expand(rot_ref,shape=(coll_pts.shape),indices='il->ijkl')
            r_vec = coll_pts - rot_ref_exp
            ang_vel_exp = csdl.expand(
                ang_vel, (num_nodes, num_pts_chord - 1, num_pts_span - 1, 3),
                indices='il->ijkl')
            rot_vel = csdl.reshape(csdl.cross(ang_vel_exp, r_vec, axis=3),
                                   out_shape)
            # self.print_var(rot_vel)
            frame_vel_expand = csdl.expand(frame_vel,
                                           out_shape,
                                           indices='ij->ikj')
            # velocity of the colloction points due to the fish body deformation
            # need a nagtive sign here because the coll pts velocity is the velocity of the fish body, we convert to air velocity
            coll_vel = self.declare_variable(surface_name+'_coll_vel',val=np.zeros((num_nodes,num_pts_chord-1,num_pts_span-1,3)))
            # self.print_var(coll_vel)
            # self.print_var(frame_vel_expand)

            kinematic_vel = -(rot_vel + frame_vel_expand + csdl.reshape(coll_vel,new_shape=(num_nodes,(num_pts_chord-1)*(num_pts_span-1),3)))
            self.register_output(kinematic_vel_name, kinematic_vel)

