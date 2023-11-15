# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix
# from VAST.utils.custom_einsums import EinsumIjKjKi

class HorseshoeCirculationsOld(Model):
    """ 
    Compute horseshoe circulation for all the panels for all the surfaces
    horseshoe_circulation = csdl.dot(mtx, gamma_b)
    parameters
    ----------

    gamma_b : csdl array
        all the circulations   

    Returns
    -------
    horseshoe_circulation
    csdl array
        horseshoe circulations for force computation
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]

        system_size = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            system_size += (nx - 1) * (ny - 1)

        data = [np.ones(system_size, dtype=int)]
        rows = [np.arange(system_size)]
        cols = [np.arange(system_size)]

        ind_1 = 0
        ind_2 = 0

        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            arange = np.arange(num).reshape((nx - 1), (ny - 1))

            data_ = -np.ones((nx - 2) * (ny - 1), dtype=int)
            rows_ = ind_1 + arange[1:, :].flatten()
            cols_ = ind_1 + arange[:-1, :].flatten()

            data.append(data_)
            rows.append(rows_)
            cols.append(cols_)
            ind_1 += num

        data = np.concatenate(data).astype(int)
        rows = np.concatenate(rows).astype(int)
        cols = np.concatenate(cols).astype(int)


        #mtx_val = csc_matrix((data, (rows, cols)),
        #                     shape=(system_size, system_size)).astype(int).toarray()
        

        # Create a dense NumPy array without using csc_matrix
        mtx_val = np.zeros((system_size, system_size), dtype=int)

        # Populate the dense array with values from 'data' at positions specified by 'rows' and 'cols'
        mtx_val[rows, cols] = data


        A = mtx = self.create_input('mtx', val=mtx_val.astype(int))

        # gamma_b = self.declare_variable(
        #     'gamma_b', shape_by_conn=True)  # shape_by_conn not working

        #!TODO:fix this for mls!
        # surface_gamma_b_name = surface_names[0] + '_gamma_b'
        surface_gamma_b_name = 'gamma_b'
        B = surface_gamma_b = self.declare_variable(surface_gamma_b_name,
                                                shape=(num_nodes, system_size))
        # gamma_b = self.declare_variable('gamma_b', shape=(system_size, ))

        # print(gamma_b.shape)
        # print(mtx.shape)
        # horseshoe_circulation = csdl.custom(mtx,
        #                                     surface_gamma_b,
        #                                     op=EinsumIjKjKi(in_name_1='mtx',
        #                                                         in_name_2='gamma_b',
        #                                                         ijk=(system_size, system_size, num_nodes),
        #                                                         out_name='horseshoe_circulation'))

        # manual horseshoe_circulation calc:
        #B = self.create_output('B', shape=(num_nodes, system_size), val=0)

        C = csdl.transpose(csdl.matmat(A, csdl.transpose(B)))
        
        # i = 2
        # j = 3
        # k = 4
        # A = np.ones((i,j))*0.1
        # B = np.ones((k,j))

        # C = np.einsum('ij,kj->ki',A,B)
        # D = (A @ B.T).T

        # mtx (rows, cols)
        # surface_gamma_b (num_nodes, system_size)
        
        # csdl.einsum(mtx,
        #                                     surface_gamma_b,
        #                                     subscripts='ij,kj->ki')
        # print('horseshoe_circulation horseshoe_circulation shape',
        #       horseshoe_circulation.shape)

        #self.register_output('horseshoe_circulation', horseshoe_circulation)
        self.register_output('horseshoe_circulation', C)

class HorseshoeCirculations(Model):
    """ 
    Compute horseshoe circulation for all the panels for all the surfaces
    horseshoe_circulation = csdl.dot(mtx, gamma_b)
    parameters
    ----------

    gamma_b : csdl array
        all the circulations   

    Returns
    -------
    horseshoe_circulation
    csdl array
        horseshoe circulations for force computation
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]

        system_size = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            system_size += (nx - 1) * (ny - 1)

        gamma_b_name = 'gamma_b'
        gamma_b = self.declare_variable(gamma_b_name,
                                                shape=(num_nodes, system_size))
        
        gamma_b_T = csdl.transpose(gamma_b)
        gamma_b_T_vectorized = csdl.reshape(gamma_b_T, (num_nodes*system_size,))

        custom=False
        if custom:
            horseshoe_circ_T_vectorized = csdl.custom(
                gamma_b_T_vectorized,
                op=SparseMatVecGamma(
                    in_name=gamma_b_T_vectorized.name,
                    out_name='asdf',
                    system_size=system_size,
                    surface_names=surface_names,
                    num_nodes=num_nodes
                )
            )
        else:
            data = [np.ones(system_size, dtype=int)]
            rows = [np.arange(system_size)]
            cols = [np.arange(system_size)]

            ind_1 = 0
            ind_2 = 0

            for i in range(len(surface_names)):
                nx = surface_shapes[i][1]
                ny = surface_shapes[i][2]
                num = (nx - 1) * (ny - 1)

                ind_2 += num

                arange = np.arange(num).reshape((nx - 1), (ny - 1))

                data_ = -np.ones((nx - 2) * (ny - 1), dtype=int)
                rows_ = ind_1 + arange[1:, :].flatten()
                cols_ = ind_1 + arange[:-1, :].flatten()

                data.append(data_)
                rows.append(rows_)
                cols.append(cols_)
                ind_1 += num

            data = np.concatenate(data).astype(int).tolist()
            rows = np.concatenate(rows).astype(int)
            cols = np.concatenate(cols).astype(int)
            # mtx_val = csc_matrix((data, (rows, cols)), shape=(system_size, system_size)).astype(int)

            num_data = len(data)

            data_exp = np.array(data*num_nodes)
            rows_exp = np.zeros((num_data*num_nodes,))
            cols_exp = np.zeros_like(rows_exp)
            for i in range(num_nodes):
                rows_exp[num_data*i:num_data*(i+1)] = rows + int(num_data*i)
                cols_exp[num_data*i:num_data*(i+1)] = cols + int(num_data*i)

            mtx_val = csc_matrix((data_exp, (rows_exp, cols_exp)),
                                shape=(system_size*num_nodes, system_size*num_nodes)).astype(int)
            
            horseshoe_circ_T_vectorized = csdl.matvec(mtx_val, gamma_b_T_vectorized)

        horseshoe_circ_T = csdl.reshape(horseshoe_circ_T_vectorized, (system_size, num_nodes))
        horseshoe_circ = csdl.transpose(horseshoe_circ_T)

        self.register_output('horseshoe_circulation', horseshoe_circ)



class SparseMatVecGamma(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('out_name', types=str)
        self.parameters.declare('system_size', types=int)
        self.parameters.declare('surface_names')
        self.parameters.declare('num_nodes')

    def define(self):
        self.in_name = self.parameters['in_name']
        self.out_name = self.parameters['out_name']

        system_size = self.parameters['system_size']
        surface_names = self.parameters['surface_names']
        num_nodes = self.parameters['num_nodes']

        self.sparse_mtx = self.assemble_sparse_matrix(system_size, surface_names, num_nodes)

    def compute(self, inputs, outputs):
        outputs[self.out_name] = self.sparse_mtx.dot(inputs[self.in_name])

    def compute_derivatives(self, inputs, derivatives):
        derivatives[self.out_name] = self.sparse_mtx

    def assemble_sparse_matrix(self, system_size, surface_names, num_nodes):
        data = [np.ones(system_size, dtype=int)]
        rows = [np.arange(system_size)]
        cols = [np.arange(system_size)]

        ind_1 = 0
        ind_2 = 0

        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            arange = np.arange(num).reshape((nx - 1), (ny - 1))

            data_ = -np.ones((nx - 2) * (ny - 1), dtype=int)
            rows_ = ind_1 + arange[1:, :].flatten()
            cols_ = ind_1 + arange[:-1, :].flatten()

            data.append(data_)
            rows.append(rows_)
            cols.append(cols_)
            ind_1 += num

        data = np.concatenate(data).astype(int).tolist()
        rows = np.concatenate(rows).astype(int)
        cols = np.concatenate(cols).astype(int)

        # mtx_val = csc_matrix((data, (rows, cols)),
        #                     shape=(system_size, system_size)).astype(int)

        data_exp = np.array(data*num_nodes)
        rows_exp = np.zeros((system_size*num_nodes,))
        cols_exp = np.zeros_like(rows_exp)
        for i in range(num_nodes):
            rows_exp[system_size*i:system_size*(i+1)] = rows + int(system_size*i)
            cols_exp[system_size*i:system_size*(i+1)] = cols + int(system_size*i)

        mtx_val = csc_matrix((data_exp, (rows_exp, cols_exp)),
                            shape=(system_size*num_nodes, num_nodes)).astype(int)
        
        return mtx_val


if __name__ == "__main__":

    model_1 = Model()
    nx = 3
    ny = 4
    surface_names = ['wing']
    surface_shapes = [(nx, ny, 3)]

    gamma_b_val = np.random.random(((nx - 1) * (ny - 1)))

    gamma_b = model_1.create_input('wing_gamma_b', val=gamma_b_val)

    model_1.add(
        HorseshoeCirculations(surface_names=surface_names,
                              surface_shapes=surface_shapes))

    sim = Simulator(model_1)
    sim.run()
    sim.visualize_implementation()
