
import numpy as np 
import tensorflow as tf 
import os

from .utils import *
from .energy_iceflow import *
from .neural_network import *

from igm import emulators
import importlib_resources
from IPython import embed
  
def initialize_iceflow_emulator(params,state):

    if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
        state.opti_retrain = getattr(tf.keras.optimizers,params.iflo_optimizer_emulator)(
            learning_rate=params.iflo_retrain_emulator_lr
        )
    else:
        state.opti_retrain = getattr(tf.keras.optimizers.legacy,params.iflo_optimizer_emulator)( 
            learning_rate=params.iflo_retrain_emulator_lr
        )

    direct_name = (
        "pinnbp"
        + "_"
        + str(params.iflo_Nz)
        + "_"
        + str(int(params.iflo_vert_spacing))
        + "_"
    )
    direct_name += (
        params.iflo_network
        + "_"
        + str(params.iflo_nb_layers)
        + "_"
        + str(params.iflo_nb_out_filter)
        + "_"
    )
    direct_name += (
        str(params.iflo_dim_arrhenius)
        + "_"
        + str(int(params.iflo_new_friction_param))
    )

    if params.iflo_pretrained_emulator:
        if params.iflo_emulator == "":
            if os.path.exists(
                importlib_resources.files(emulators).joinpath(direct_name)
            ):
                dirpath = importlib_resources.files(emulators).joinpath(direct_name)
                print(
                    "Found pretrained emulator in the igm package: " + direct_name
                )
            else:
                print("No pretrained emulator found in the igm package")
        else:
            if os.path.exists(params.iflo_emulator):
                dirpath = params.iflo_emulator
                print("----------------------------------> Found pretrained emulator: " + params.iflo_emulator)
            else:
                print("----------------------------------> No pretrained emulator found ")

        fieldin = []
        fid = open(os.path.join(dirpath, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            fieldin.append(part[0])
        fid.close()
        assert params.iflo_fieldin == fieldin
        state.iceflow_model = tf.keras.models.load_model(
            os.path.join(dirpath, "model.h5"), compile=False
        )
        state.iceflow_model.compile() 
    else:
        print("----------------------------------> No pretrained emulator, start from scratch.") 
        nb_inputs = len(params.iflo_fieldin) + (params.iflo_dim_arrhenius == 3) * (
            params.iflo_Nz - 1
        )
        nb_outputs = 2 * params.iflo_Nz
        # state.iceflow_model = getattr(igm, params.iflo_network)(
        #     params, nb_inputs, nb_outputs
        # )
        if params.iflo_network=='cnn':
            state.iceflow_model = cnn(params, nb_inputs, nb_outputs)
        elif params.iflo_network=='unet':
            state.iceflow_model = unet(params, nb_inputs, nb_outputs)



def update_iceflow_emulated(params, state):
    # Define the input of the NN, include scaling

    Ny, Nx = state.thk.shape
    N = params.iflo_Nz

    fieldin = [vars(state)[f] for f in params.iflo_fieldin]

    X = fieldin_to_X(params, fieldin)

    if params.iflo_exclude_borders>0:
        iz = params.iflo_exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")
        
    if params.iflo_multiple_window_size==0:
        Y = state.iceflow_model(X)
    else:
        Y = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

    if params.iflo_exclude_borders>0:
        iz = params.iflo_exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    U, V = Y_to_UV(params, Y)
    U = U[0]
    V = V[0]

    #    U = tf.where(state.thk > 0, U, 0)

    state.U.assign(U)
    state.V.assign(V)

    # If requested, the speeds are artifically upper-bounded
    if params.iflo_force_max_velbar > 0:
        velbar_mag = getmag3d(state.U, state.V)
        state.U.assign(
            tf.where(
                velbar_mag >= params.iflo_force_max_velbar,
                params.iflo_force_max_velbar * (state.U / velbar_mag),
                state.U,
            )
        )
        state.V.assign(
            tf.where(
                velbar_mag >= params.iflo_force_max_velbar,
                params.iflo_force_max_velbar * (state.V / velbar_mag),
                state.V,
            )
        )

    update_2d_iceflow_variables(params, state)


def update_iceflow_emulator(params, state):
    if (state.it < 0) | (state.it % params.iflo_retrain_emulator_freq == 0):
        fieldin = [vars(state)[f] for f in params.iflo_fieldin]
        thk = state.thk

        XX = fieldin_to_X(params, fieldin)

        X = _split_into_patches(XX, params.iflo_retrain_emulator_framesizemax)
        
        Ny = X.shape[1]
        Nx = X.shape[2]
        
        PAD = compute_PAD(params,Nx,Ny)

        state.COST_EMULATOR = []

        nbit = (state.it >= 0) * params.iflo_retrain_emulator_nbit + (
            state.it < 0
        ) * params.iflo_retrain_emulator_nbit_init

        if (not params.iflo_optimizer_lbfgs):

            iz = params.iflo_exclude_borders 

            for epoch in range(nbit):
                cost_emulator = tf.Variable(0.0)

                for i in range(X.shape[0]):
                    with tf.GradientTape() as t:
 
                        Y = state.iceflow_model(tf.pad(X[i:i+1, :, :, :], PAD, "CONSTANT"))[:,:Ny,:Nx,:]
                    
                        if iz>0:
                            C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(params, X[i : i + 1, iz:-iz, iz:-iz, :], Y[:, iz:-iz, iz:-iz, :])
                        else:
                            C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(params, X[i : i + 1, :, :, :], Y[:, :, :, :])
 
                        COST = tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) \
                             + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)
                   

                        if (epoch + 1) % 100 == 0:
                            Csh = tf.reduce_mean(C_shear).numpy()
                            Cgr = tf.reduce_mean(C_grav).numpy()
                            Csl = tf.reduce_mean(C_slid).numpy()
                            Cfl = tf.reduce_mean(C_float).numpy()
                            print("---------- > ",Csh,Csl,Cgr,Cfl)

                        cost_emulator = cost_emulator + COST

                        if (epoch + 1) % 100 == 0:
                            U, V = Y_to_UV(params, Y)
                            U = U[0]
                            V = V[0]
                            velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)
                            velsurf_mag = tf.where(thk==0,0,velsurf_mag)
                            print("train : ", epoch, COST.numpy(), np.max(velsurf_mag))

                    grads = t.gradient(COST, state.iceflow_model.trainable_variables)

                    state.opti_retrain.apply_gradients(
                        zip(grads, state.iceflow_model.trainable_variables)
                    )

                    state.opti_retrain.lr = params.iflo_retrain_emulator_lr * (
                        0.95 ** (epoch / 1000)
                    )

                state.COST_EMULATOR.append(cost_emulator)
        else:

            _update_iceflow_emulator_lbfgs(params, state, X, nbit)
            
    
    if len(params.save_cost_emulator)>0:
        np.savetxt(params.save_cost_emulator+'-'+str(state.it)+'.dat', np.array(state.COST_EMULATOR), fmt="%5.10f")



def _update_iceflow_emulator_lbfgs(params, state, X, nbit):

    import tensorflow_probability as tfp

    # based on guidance here: https://gist.github.com/piyueh/712ec7d4540489aad2dcfb80f9a54993
    print ('beginning lbfgs, max iterations ' + str(nbit))

    Ny = X.shape[1]
    Nx = X.shape[2]
    PAD = compute_PAD(params,Nx,Ny)
    iz = params.iflo_exclude_borders
    thk = state.thk

    # according to what i've read, tfp.optimizer.lbfgs_minimize requires the controls to be a 1D tensor.
    # below the trainable variables, which exist as a list of tensors of various shapes, need to be converted
    # to and from a 1D tensor via dynamic_partition() and dynamic_stitch(), which uses idx and part.

    shapes = tf.shape_n(state.iceflow_model.trainable_variables)
    n_tensors = len(shapes)

    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    ######

    @tf.function
    def assign_new_model_parameters(model,params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    ######

# this cannot have the tf.function decorator because energy_iceflow.iceflow_energy()
# uses the symbol table. this can be avoided by fixing iflo_fieldin rather than 
# having it as a settable parameter. (Or at least fixing it to several hardcoded choices)
# same is true of the next function (below)

#    @tf.function
    def COST(oneD_tensor):

        cost_emulator = tf.Variable(0.0)
    
        assign_new_model_parameters(state.iceflow_model,oneD_tensor)

        epoch = len(state.COST_EMULATOR)

        for i in range(X.shape[0]):

            Y = state.iceflow_model(tf.pad(X[i:i+1, :, :, :], PAD, "CONSTANT"))[:,:Ny,:Nx,:]
    
            if iz>0:
                C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(params, X[i : i + 1, iz:-iz, iz:-iz, :], Y[:, iz:-iz, iz:-iz, :])
            else:
                C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(params, X[i : i + 1, :, :, :], Y[:, :, :, :])

            COST = tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) \
                 + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)

            cost_emulator = cost_emulator + COST
           
        state.COST_EMULATOR.append(cost_emulator)

        if (epoch + 1) % 100 == 0:
            print ('epoch ' + str(epoch))
            Csh = tf.reduce_mean(C_shear).numpy()
            Cgr = tf.reduce_mean(C_grav).numpy()
            Csl = tf.reduce_mean(C_slid).numpy()
            Cfl = tf.reduce_mean(C_float).numpy()
            print("---------- > ",Csh,Csl,Cgr,Cfl)

        if (epoch + 1) % 100 == 0:
            U, V = Y_to_UV(params, Y)
            U = U[0]
            V = V[0]
            velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)
            velsurf_mag = tf.where(thk==0,0,velsurf_mag)
            print("train : ", epoch, COST.numpy(), np.max(velsurf_mag))

        return cost_emulator

    ######

#    @tf.function
    def loss_and_gradients_function(oneD_tensor):
        with tf.GradientTape() as tape:
            loss = COST(oneD_tensor)
        gradients = tape.gradient(loss, state.iceflow_model.trainable_variables)
        gradients = tf.dynamic_stitch(idx, gradients)
        return loss, gradients

    ######

    # max_iterations << number of gradient evals.

    init_params = tf.dynamic_stitch(idx, state.iceflow_model.trainable_variables)
    results = tfp.optimizer.lbfgs_minimize(
                value_and_gradients_function=loss_and_gradients_function,
                initial_position=init_params,
                max_iterations=nbit,
                f_relative_tolerance=1e-3)

    assign_new_model_parameters(state.iceflow_model,results.position)



def _split_into_patches(X, nbmax):
    XX = []
    ny = X.shape[1]
    nx = X.shape[2]
    sy = ny // nbmax + 1
    sx = nx // nbmax + 1
    ly = int(ny / sy)
    lx = int(nx / sx)

    for i in range(sx):
        for j in range(sy):
            XX.append(X[0, j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :])

    return tf.stack(XX, axis=0)


def save_iceflow_model(params, state):
    directory = "iceflow-model"
    
    import shutil

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)

    state.iceflow_model.save(os.path.join(directory, "model.h5"))

    #    fieldin_dim=[0,0,1*(params.iflo_dim_arrhenius==3),0,0]

    fid = open(os.path.join(directory, "fieldin.dat"), "w")
    #    for key,gg in zip(params.iflo_fieldin,fieldin_dim):
    #        fid.write("%s %.1f \n" % (key, gg))
    for key in params.iflo_fieldin:
        print(key)
        fid.write("%s \n" % (key))
    fid.close()

    fid = open(os.path.join(directory, "vert_grid.dat"), "w")
    fid.write("%4.0f  %s \n" % (params.iflo_Nz, "# number of vertical grid point (Nz)"))
    fid.write(
        "%2.2f  %s \n"
        % (params.iflo_vert_spacing, "# param for vertical spacing (vert_spacing)")
    )
    fid.close()
