################################################
################## IMPORT ######################
################################################

from posixpath import split
import sys
import os
from datetime import datetime
from functools import partial, wraps

import fire
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.experimental import ode
from shadow.plot import *
from shadow.plot import panel

from psystems.npendulum import (PEF, edge_order, get_init, hconstraints,
                                pendulum_connections)


MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import jraph
import src
from jax.config import config
from src import lnn
from src.graph import *
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import nve
from src.utils import *
from src.hamiltonian import *


config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def pprint(*args, namespace=globals()):
    for arg in args:
        print(f"{namestr(arg, namespace)[0]}: {arg}")



# N = 2
# epochs = 10000
# seed = 42
# rname = True
# dt = 1.0e-5
# ifdrag = 0
# stride=1000
# trainm = 1
# lr = 0.001
# withdata = None
# datapoints = None
# batch_size = 1000

def main(N = 5, epochs = 10000, seed = 42, rname = True,
        dt = 1.0e-5, ifdrag = 0, trainm = 1, stride=1000, lr = 0.001, withdata = None, datapoints = None, batch_size = 100):    
    print("Configs: ")
    pprint(N, epochs, seed, rname, dt, lr, ifdrag, batch_size, namespace=locals())
    
    randfilename = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + f"_{datapoints}"
    
    PSYS = f"{N}-Pendulum"
    TAG = f"3HGNN"
    out_dir = f"../results"
    
    
    def _filename(name, tag=TAG):
        rstring = randfilename if (rname and (tag != "data-ham")) else (
            "0" if (tag == "data-ham") or (withdata == None) else f"0_{withdata}")
        filename_prefix = f"{out_dir}/{PSYS}-{tag}/{rstring}/"
        file = f"{filename_prefix}/{name}"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        filename = f"{filename_prefix}/{name}".replace("//", "/")
        print("===", filename, "===")
        return filename
    
    
    def OUT(f):
        @wraps(f)
        def func(file, *args, tag=TAG, **kwargs):
            return f(_filename(file, tag=tag), *args, **kwargs)
        return func
    
    
    loadfile = OUT(src.io.loadfile)
    savefile = OUT(src.io.savefile)
    
    
    ################################################
    ################## CONFIG ######################
    ################################################
    
    np.random.seed(seed)
    key = random.PRNGKey(seed)
    
    
    try:
        dataset_states = loadfile(f"model_states_{ifdrag}.pkl", tag="data-ham")[0]
    except:
        raise Exception("Generate dataset first.")
    
    if datapoints is not None:
        dataset_states = dataset_states[:datapoints]
    
    
    model_states = dataset_states[0]
    z_out, zdot_out = model_states
    
    
    print(
        f"Total number of data points: {len(dataset_states)}x{z_out.shape[0]}")
    
    N2, dim = z_out.shape[-2:]
    N = N2//2
    species = jnp.zeros(N, dtype=int)
    masses = jnp.ones(N)
    
    array = jnp.array([jnp.array(i) for i in dataset_states])
    
    Zs = array[:, 0, :, :, :]
    Zs_dot = array[:, 1, :, :, :]
    
    Zs = Zs.reshape(-1, N2, dim)
    Zs_dot = Zs_dot.reshape(-1, N2, dim)
    
    mask = np.random.choice(len(Zs), len(Zs), replace=False)
    allZs = Zs[mask]
    allZs_dot = Zs_dot[mask]
    
    Ntr = int(0.75*len(Zs))
    Nts = len(Zs) - Ntr
    
    Zs = allZs[:Ntr]
    Zs_dot = allZs_dot[:Ntr]
    
    Zst = allZs[Ntr:]
    Zst_dot = allZs_dot[Ntr:]
    
    ################################################
    ################## SYSTEM ######################
    ################################################


    def phi(x):
        X = jnp.vstack([x[:1, :]*0, x])
        return jnp.square(X[:-1, :] - X[1:, :]).sum(axis=1) - 1.0


    constraints = get_constraints(N, dim, phi)


    ################################################
    ################### ML Model ###################
    ################################################

    senders, receivers = pendulum_connections(N)
    eorder = edge_order(N)
    
    Ef = 1  # eij dim
    Nf = dim
    Oh = 1

    Eei = 5
    Nei = 5

    hidden = 5
    nhidden = 2


    def get_layers(in_, out_):
        return [in_] + [hidden]*nhidden + [out_]


    def mlp(in_, out_, key, **kwargs):
        return initialize_mlp(get_layers(in_, out_), key, **kwargs)


    # # fne_params = mlp(Oh, Nei, key)
    fneke_params = initialize_mlp([Oh, Nei], key)
    fne_params = initialize_mlp([Oh, Nei], key)

    fb_params = mlp(Ef, Eei, key)
    fv_params = mlp(Nei+Eei, Nei, key)
    fe_params = mlp(Nei, Eei, key)

    ff1_params = mlp(Eei, 1, key)
    ff2_params = mlp(Nei, 1, key)
    ff3_params = mlp(dim+Nei, 1, key)
    ke_params = initialize_mlp([1+Nei, 10, 10, 1], key, affine=[True])

    Hparams = dict(fb=fb_params,
                fv=fv_params,
                fe=fe_params,
                ff1=ff1_params,
                ff2=ff2_params,
                ff3=ff3_params,
                fne=fne_params,
                fneke=fneke_params,
                ke=ke_params)
    

    def H_energy_fn(params, graph):
        g, V, T = cal_graph(params, graph, eorder=eorder,
                            useT=True)
        return T + V


    R, V = jnp.split(Zs[0], 2, axis=0)
    
    
    def energy_fn(species):
        senders, receivers = [np.array(i)
                            for i in pendulum_connections(R.shape[0])]
        state_graph = jraph.GraphsTuple(nodes={
            "position": R,
            "velocity": V,
            "type": species
        },
            edges={},
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([R.shape[0]]),
            n_edge=jnp.array([senders.shape[0]]),
            globals={})

        def apply(R, V, params):
            state_graph.nodes.update(position=R)
            state_graph.nodes.update(velocity=V)
            return H_energy_fn(params, state_graph)
        return apply


    apply_fn = energy_fn(species)
    v_apply_fn = vmap(apply_fn, in_axes=(None, 0))


    def Hmodel(x, v, params):
        return apply_fn(x, v, params["H"])


    params = {"H": Hparams}


    def nndrag(v, params):
        return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v


    if ifdrag == 0:
        print("Drag: 0.0")

        def drag(x, p, params):
            return 0.0
    elif ifdrag == 1:
        print("Drag: -0.1*v")

        def drag(x, p, params):
            return vmap(nndrag, in_axes=(0, None))(p.reshape(-1), params["drag"]).reshape(-1, 1)

    params["drag"] = initialize_mlp([1, 5, 5, 1], key)

    # def external_force(x, p, params):
    #         F = 0*p
    #         F = jax.ops.index_update(F, (1, 1), -1.0)
    #         return F.reshape(-1, 1)
        
        

    zdot_model, lamda_force_model = get_zdot_lambda(
        N, dim, hamiltonian=Hmodel, drag=None, constraints=constraints,external_force=None)


    v_zdot_model = vmap(zdot_model, in_axes=(0, 0, None))

    ################################################
    ################## ML Training #################
    ################################################


    @jit
    def loss_fn(params, Rs, Vs, Zs_dot):
        pred = v_zdot_model(Rs, Vs, params)
        return MSE(pred, Zs_dot)


    def gloss(*args):
        return value_and_grad(loss_fn)(*args)


    def update(i, opt_state, params, loss__, *data):
        """ Compute the gradient for a batch and update the parameters """
        value, grads_ = gloss(params, *data)
        opt_state = opt_update(i, grads_, opt_state)
        return opt_state, get_params(opt_state), value


    @ jit
    def step(i, ps, *args):
        return update(i, *ps, *args)


    opt_init, opt_update_, get_params = optimizers.adam(lr)


    @ jit
    def opt_update(i, grads_, opt_state):
        grads_ = jax.tree_map(jnp.nan_to_num, grads_)
        # grads_ = jax.tree_map(partial(jnp.clip, a_min=-1000.0, a_max=1000.0), grads_)
        return opt_update_(i, grads_, opt_state)


    def batching(*args, size=None):
        L = len(args[0])
        if size != None:
            nbatches1 = int((L - 0.5) // size) + 1
            nbatches2 = max(1, nbatches1 - 1)
            size1 = int(L/nbatches1)
            size2 = int(L/nbatches2)
            if size1*nbatches1 > size2*nbatches2:
                size = size1
                nbatches = nbatches1
            else:
                size = size2
                nbatches = nbatches2
        else:
            nbatches = 1
            size = L

        newargs = []
        for arg in args:
            newargs += [jnp.array([arg[i*size:(i+1)*size]
                                for i in range(nbatches)])]
        return newargs


    Rs, Vs = jnp.split(Zs, 2, axis=1)
    Rst, Vst = jnp.split(Zst, 2, axis=1)

    bRs, bVs, bZs_dot = batching(Rs, Vs, Zs_dot,
                                size=min(len(Rs), batch_size))

    print(f"training ...")

    opt_state = opt_init(params)
    epoch = 0
    optimizer_step = -1
    larray = []
    ltarray = []
    last_loss = 1000
    for epoch in range(epochs):
        for data in zip(bRs, bVs, bZs_dot):
            optimizer_step += 1
            opt_state, params, l_ = step(
                optimizer_step, (opt_state, params, 0), *data)
        
        if epoch % 1 == 0:
            larray += [loss_fn(params, Rs, Vs, Zs_dot)]
            ltarray += [loss_fn(params, Rst, Vst, Zst_dot)]
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch}/{epochs} Loss (MSE):  train={larray[-1]}, test={ltarray[-1]}")
            savefile(f"hgnn_trained_model_{ifdrag}_{trainm}.dil",
                    params, metadata={"savedat": epoch})
            savefile(f"loss_array_{ifdrag}_{trainm}.dil",
                    (larray, ltarray), metadata={"savedat": epoch})
            if last_loss > larray[-1]:
                last_loss = larray[-1]
                savefile(f"hgnn_trained_model_{ifdrag}_{trainm}_low.dil",
                        params, metadata={"savedat": epoch})
    
            fig, axs = panel(1, 1)
            plt.semilogy(larray, label="Training")
            plt.semilogy(ltarray, label="Test")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))
    
    fig, axs = panel(1, 1)
    plt.semilogy(larray, label="Training")
    plt.semilogy(ltarray, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(_filename(f"training_loss_{ifdrag}_{trainm}.png"))
    
    params = get_params(opt_state)
    savefile(f"hgnn_trained_model_{ifdrag}_{trainm}.dil",
            params, metadata={"savedat": epoch})
    savefile(f"loss_array_{ifdrag}_{trainm}.dil",
            (larray, ltarray), metadata={"savedat": epoch})


fire.Fire(main)