################################################
################## IMPORT ######################
################################################

import json
import re
import sys
import unicodedata
from datetime import datetime
from functools import partial, wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random, value_and_grad, vmap
from jax.experimental import optimizers
from jax_md import space
from shadow.font import set_font_size
from shadow.plot import *
from sklearn.metrics import r2_score

from psystems.npendulum import (PEF, edge_order, get_init, hconstraints,
                                pendulum_connections)
from psystems.nsprings import (chain, edge_order, get_connections,
                               get_fully_connected_senders_and_receivers,
                               get_fully_edge_order)

MAINPATH = ".."  # nopep8
sys.path.append(MAINPATH)  # nopep8

import fire
import jraph
import src
from jax.config import config
from src.graph import *
from src.md import *
from src.models import MSE, initialize_mlp
from src.nve import NVEState, NVEStates, nve
from src.utils import *
from src.hamiltonian import *
import time

set_font_size(size=30)

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
# jax.config.update('jax_platform_name', 'gpu')


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


useN1 = 5
useN2 = 5
dim = 2
trainm = 0
ifdrag = 0
runs = 1000
rname = 0
semilog = 1
maxtraj = 100
grid = 0
mpass1 = 1
mpass2 = 1
plotonly = 0
dt = 1.0e-2
atol=1.0e-8
rtol=1.0e-4
mxstep=jnp.inf
seed = 42
saveovito=0
plotthings=0

# time_step = 1
# def main(useN1 = 3,useN2 = 2,dim = 2,trainm = 0,time_step = 1,ifdrag = 0,runs = 100,rname = 0,semilog = 1,maxtraj = 5,grid = 0,mpass1 = 1,mpass2 = 1,plotonly = 0,dt = 1.0e-5,stride = 1000):

PSYS = f"ham-4-8-hybrid"
TAG = f"3HGNN"
out_dir = f"../results"


def _filename(name, tag=TAG, trained=None):
    lt = name.split(".")
    if len(lt) > 1:
        a = lt[:-1]
        b = lt[-1]
        name = slugify(".".join(a)) + "." + b
    if tag == "data":
        part = f"_{ifdrag}."
    else:
        part = f"_{ifdrag}_{0}."
    if trained is not None:
        psys = f"{trained}"
    else:
        psys = PSYS
    name = ".".join(name.split(".")[:-1]) + \
        part + name.split(".")[-1]
    rstring = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") if rname else "0"
    filename_prefix = f"{out_dir}/{psys}-{tag}/{rstring}/"
    file = f"{filename_prefix}/{name}"
    os.makedirs(os.path.dirname(file), exist_ok=True)
    filename = f"{filename_prefix}/{name}".replace("//", "/")
    print("===", filename, "===")
    return filename


def OUT(f):
    @wraps(f)
    def func(file, *args, tag=TAG, trained=None, **kwargs):
        return f(_filename(file, tag=tag, trained=trained), *args, **kwargs)
    return func


loadmodel = OUT(src.models.loadmodel)
savemodel = OUT(src.models.savemodel)

loadfile = OUT(src.io.loadfile)
savefile = OUT(src.io.savefile)
save_ovito = OUT(src.io.save_ovito)


def putpin(state):
    def f(x): return jnp.vstack([0*x[:1], x])
    return NVEState(f(state.position), f(state.velocity), f(state.force), f(state.mass))

################################################
################## CONFIG ######################
################################################


np.random.seed(seed)
key = random.PRNGKey(seed)


def get_RV():
    θ1 = np.pi/5*(np.random.rand()-0.5)
    θ2 = -np.pi/5*(np.random.rand()-0.5)
    θ3 = np.pi/5*(np.random.rand()-0.5)
    θ4 = -np.pi/5*(np.random.rand()-0.5)
    θ5 = np.pi/5*(np.random.rand()-0.5)
    θ6 = -np.pi/5*(np.random.rand()-0.5)
    θ7 = np.pi/5*(np.random.rand()-0.5)
    θ8 = -np.pi/5*(np.random.rand()-0.5)
    sin1 = np.sin(θ1)
    cos1 = np.cos(θ1)
    sin2 = np.sin(θ2)
    cos2 = np.cos(θ2)
    sin3 = np.sin(θ3)
    cos3 = np.cos(θ3)
    sin4 = np.sin(θ4)
    cos4 = np.cos(θ4)
    sin5 = np.sin(θ5)
    cos5 = np.cos(θ5)
    sin6 = np.sin(θ6)
    cos6 = np.cos(θ6)
    sin7 = np.sin(θ7)
    cos7 = np.cos(θ7)
    sin8 = np.sin(θ8)
    cos8 = np.cos(θ8)
    l1 = 1
    l2 = 1
    R = jnp.array([[0, 0],
                [l1*sin1, -l1*cos1],
                [l1*sin1+l2*sin2, -l1*cos1-l2*cos2],
                [1, 0],
                [1+l1*sin3, -l1*cos3],
                [1+l1*sin3+l2*sin4, -l1*cos3-l2*cos4],
                [2, 0],
                [2+l1*sin5, -l1*cos5],
                [2+l1*sin5+l2*sin6, -l1*cos5-l2*cos6],
                [3, 0],
                [3+l1*sin7, -l1*cos7],
                [3+l1*sin7+l2*sin8, -l1*cos7-l2*cos8]
                ])
    # t4 = np.pi/5*(np.random.rand()-0.5)
    # rot = jnp.array([
    #     [np.cos(t4), -np.sin(t4)],
    #     [np.sin(t4), np.cos(t4)]
    # ])
    # R = rot.dot(R.T).T
    # R = R.at[5,0].set(1)
    # R = R.at[5,1].set(0)
    V = 0*R
    # V = jnp.array([[5.0, 0],
    #                [1.0, 0],
    #                [-5.0, 0],
    #                [5.0, 0]])
    return R, V


R, V = get_RV()
# plt.plot(R[:, 0], R[:, 1],"*")
# plt.clf()


print("Saving init configs...")
savefile(f"initial-configs.pkl", [(R, V)])

N = R.shape[0]

species = jnp.zeros(N, dtype=int)
masses = jnp.ones(N)

################################################
################## SYSTEM ######################
################################################

#################
###  Spring  ####
#################



# _, _, senders1, receivers1 = chain(8)
senders1, receivers1 = jnp.array(
    [4, 1, 2, 5, 8, 11, 10, 7]+ [1, 2, 5, 8, 11, 10, 7, 4], dtype=int), jnp.array([ 1, 2, 5, 8, 11, 10, 7, 4]+[4, 1, 2, 5, 8, 11, 10, 7], dtype=int)
eorder1 = edge_order(len(senders1))

print("kinetic energy: 0.5mv^2")
kin_energy = partial(src.hamiltonian._T, mass=masses)


def H_energy_fn(params, graph):
    g, V, T = cal_graph(params, graph, mpass=mpass1, eorder=eorder1,
                        useT=True, useonlyedge=True)
    return + V


def energy_fn(species):
    state_graph1 = jraph.GraphsTuple(nodes={
        "position": R,
        "velocity": V,
        "type": species
    },
        edges={},
        senders=senders1,
        receivers=receivers1,
        n_node=jnp.array([R.shape[0]]),
        n_edge=jnp.array([senders1.shape[0]]),
        globals={})
    def apply(R, V, params):
        state_graph1.nodes.update(position=R)
        state_graph1.nodes.update(velocity=V)
        return H_energy_fn(params, state_graph1)
    return apply


apply_fn = energy_fn(species)

# apply_fn(R[1:5], V[1:5], params_spring['H'])


def Hmodel_spring(x, v, params): return apply_fn(x, v, params["H"])


params_spring = loadfile(f"hgnn_trained_model.dil",
                        trained=f"a-{useN1}-Spring")[0]

#######################################################

#################
### Pendulum ####
#################

# senders2, receivers2 = pendulum_connections(2)
senders2, receivers2 = jnp.array(
    [0, 3 ,6, 9, 4, 7]+[1, 4, 7, 10, 5, 8], dtype=int), jnp.array([1, 4, 7, 10, 5, 8]+[0, 3 ,6, 9, 4, 7], dtype=int)

eorder2 = edge_order(len(senders2))


print("kinetic energy: 0.5mv^2")
kin_energy = partial(src.hamiltonian._T, mass=masses)


def H_energy_fn2(params, graph):
    g, V, T = cal_graph(params, graph, eorder=eorder2,
                        useT=True)
    return + V


def energy_fn2(species):
    state_graph2 = jraph.GraphsTuple(nodes={
        "position": R,
        "velocity": V,
        "type": species
    },
        edges={},
        senders=senders2,
        receivers=receivers2,
        n_node=jnp.array([R.shape[0]]),
        n_edge=jnp.array([senders2.shape[0]]),
        globals={})
    def apply(R, V, params):
        state_graph2.nodes.update(position=R)
        state_graph2.nodes.update(velocity=V)
        return H_energy_fn2(params, state_graph2)
    return apply


apply_fn2 = energy_fn2(species)

# apply_fn2(R, V, params_pen["H"])


def Hmodel_pen(x, v, params): return apply_fn2(x, v, params["H"])


params_pen = loadfile(f"hgnn_trained_model.dil",
                    trained=f"{useN2}-Pendulum")[0]

# jax.grad(Hmodel_pen)(R, V, params["pen"])

def Hmodel(x, v, params): return (kin_energy(v) +
                                Hmodel_pen(x, v, params["pen"]) +
                                Hmodel_spring(x, v, params["spring"]) +
                                0.0)


# def Hmodel(x, v, params): return (kin_energy(v) +
#                                   Hmodel_pen(x[(0, 1), :], v[(0, 1), :], params["pen"]) +
#                                   Hmodel_pen(x[(5, 4), :], v[(5, 4), :], params["pen"]) +
#                                   Hmodel_spring(x[1:5], v[1:5], params["spring"]) +
#                                 #   1*10*x[(2, 3), 1].sum() +
#                                   Hmodel_pen(x[(2, 3), :], v[(2, 3), :], params["pen"]) +
#                                   0.0)


# Hmodel(R, V, params)


def nndrag(v, params):
    return - jnp.abs(models.forward_pass(params, v.reshape(-1), activation_fn=models.SquarePlus)) * v


if ifdrag == 0:
    print("Drag: 0.0")
    
    def drag(x, v, params):
        return 0.0
elif ifdrag == 1:
    print("Drag: nn")
    
    def drag(x, v, params):
        return vmap(nndrag, in_axes=(0, None))(v.reshape(-1), params["drag"]).reshape(-1, 1)


def external_force(x, v, params):
    # F = 0*R
    # F = jax.ops.index_update(F, (1, 0), 10.0)
    # return F.reshape(-1, 1)
    return 0.0


# _a = [0,3,6,9]
# _b = [1,4,7,10,5,8]
# _c = [0,3,6,9,4,7]
# _origin = jnp.array([[0, 0],[1,0],[2,0],[3,0]])

# @jit
# def phi(x):
#     # X = jnp.vstack([r[:1, :]*0, r])
#     # jnp.square(X[:-1, :] - X[1:, :]).sum(axis=1) - 1.0
#     ss = jnp.hstack([
#         jnp.square(x[_a, :] - _origin).sum(0) - 0.0,
#         jnp.square(x[_b, :] - x[_c, :]).sum(0) - 1.0,
#         ])
#     return ss.flatten()

def phi(x):
    ss = jnp.vstack([
        jnp.square(x[0, :] - jnp.array([0, 0])).sum() - 0.0,
        jnp.square(x[1, :] - x[0, :]).sum() - 1.0,
        jnp.square(x[3, :] - jnp.array([1, 0])).sum() - 0.0,
        jnp.square(x[4, :] - x[3, :]).sum() - 1.0,
        jnp.square(x[5, :] - x[4, :]).sum() - 1.0,
        jnp.square(x[6, :] - jnp.array([2, 0])).sum() - 0.0,
        jnp.square(x[7, :] - x[6, :]).sum() - 1.0,
        jnp.square(x[8, :] - x[7, :]).sum() - 1.0,
        jnp.square(x[9, :] - jnp.array([3, 0])).sum() - 0.0,
        jnp.square(x[10, :] - x[9, :]).sum() - 1.0
        ])
    return ss.flatten()


# phi(R)

constraints = jit(get_constraints(N, dim, phi))


zdot_model, lamda_force_model = get_zdot_lambda(
    N, dim, hamiltonian=Hmodel, drag=None, constraints=constraints)

zdot_model = jit(zdot_model)


def zdot_model_func(z, t, params):
    x, p = jnp.split(z, 2)
    return zdot_model(x, p, params)


def z0(x, p):
    return jnp.vstack([x, p])


def get_forward_sim(params=None, zdot_func=None, runs=10):
    def fn(R, V):
        t = jnp.linspace(0.0, runs*dt, runs)
        _z_model_out = (ode.odeint(zdot_func, z0(R, V), t, params,rtol=rtol, atol=atol, mxstep=mxstep))
        return _z_model_out
    return fn


params = {
    "spring": params_spring,
    "pen": params_pen,
}

sim_model = get_forward_sim(
    params=params, zdot_func=zdot_model_func, runs=runs)

# zdot_model(R, V, params)
# sim_model(R, V)

################################################
################## ACTUAL ######################
################################################

def PEactual(x):
    PE = 0
    PE += 1*10*x[:, 1].sum()
    dr = jnp.square(x[senders1] - x[receivers1]).sum(axis=1)
    PE += vmap(partial(lnn.SPRING, stiffness=1.0, length=1.0))(dr).sum()
    return PE


def Hactual(x, v, params):
    KE = kin_energy(v)
    PE = PEactual(x)
    H = KE + PE
    return H


zdot, lamda_force = get_zdot_lambda(
    N, dim, hamiltonian=Hactual, drag=None, constraints=constraints)

zdot = jit(zdot)


def zdot_func(z, t, params):
    x, p = jnp.split(z, 2)
    return zdot(x, p, params)

sim_origin = get_forward_sim(
    params=None, zdot_func=zdot_func, runs=runs)

# sim_origin(R, V)
################################################
###############   FORWARD SIM   ################
################################################

def norm(a):
    a2 = jnp.square(a)
    n = len(a2)
    a3 = a2.reshape(n, -1)
    return jnp.sqrt(a3.sum(axis=1))

def RelErr(ya, yp):
    return norm(ya-yp) / (norm(ya) + norm(yp))

def Err(ya, yp):
    return ya-yp

def AbsErr(*args):
    return jnp.abs(Err(*args))

def caH_energy_fn(lag=None, params=None):
    def fn(states):
        KE = vmap(kin_energy)(states.velocity)
        H = vmap(lag, in_axes=(0, 0, None)
                    )(states.position, states.velocity, params)
        PE = (H - KE)
        # return jnp.array([H]).T
        return jnp.array([PE, KE,KE-PE, H]).T
    return fn

Es_fn = caH_energy_fn(lag=Hactual, params=None)
Es_pred_fn = caH_energy_fn(lag=Hmodel, params=params)
# Es_pred_fn(pred_traj)

def net_force_fn(force=None, params=None):
    def fn(states):
        zdot_out = vmap(force, in_axes=(0, 0, None))(
            states.position, states.velocity, params)
        _, force_out = jnp.split(zdot_out, 2, axis=1)
        return force_out
    return fn

net_force_orig_fn = net_force_fn(force=zdot)
net_force_model_fn = net_force_fn(force=zdot_model, params=params)

nexp = {
    "z_pred": [],
    "z_actual": [],
    "v_pred": [],
    "v_actual": [],
    "Zerr": [],
    "Herr": [],
    "Es": [],
    "Eshat":[],
    "Perr": [],
    "simulation_time": [],
    "constraintsF_pred":[],
    "constraintsF_actual":[]
}

trajectories = []

sim_orig2 = get_forward_sim(params=None, zdot_func=zdot_func, runs=runs)

for ind in range(maxtraj):
    print(f"Simulating trajectory {ind}/{maxtraj}")
    
    R, V = get_init(N, dim=dim, angles=(-90, 90))
    
    z_actual_out = sim_orig2(R, V)
    x_act_out, p_act_out = jnp.split(z_actual_out, 2, axis=1)
    zdot_act_out = jax.vmap(zdot, in_axes=(0, 0, None))(
        x_act_out, p_act_out, None)
    _, force_act_out = jnp.split(zdot_act_out, 2, axis=1)
    c_force_actual = jax.vmap(lamda_force, in_axes=(0, 0, None))(x_act_out, p_act_out, None)
    
    my_state = States()
    my_state.position = x_act_out
    my_state.velocity = p_act_out
    my_state.constraint_force = c_force_actual
    my_state.force = force_act_out
    my_state.mass = jnp.ones(x_act_out.shape[0])
    actual_traj = my_state
    
    start = time.time()
    z_pred_out = sim_model(R, V)
    x_pred_out, p_pred_out = jnp.split(z_pred_out, 2, axis=1)
    zdot_pred_out = jax.vmap(zdot_model, in_axes=(
        0, 0, None))(x_pred_out, p_pred_out, params)
    _, force_pred_out = jnp.split(zdot_pred_out, 2, axis=1)
    c_force_model = jax.vmap(lamda_force_model, in_axes=(0, 0, None))(x_pred_out, p_pred_out, params)
    
    my_state_pred = States()
    my_state_pred.position = x_pred_out
    my_state_pred.velocity = p_pred_out
    my_state_pred.constraint_force = c_force_model
    my_state_pred.force = force_pred_out
    my_state_pred.mass = jnp.ones(x_pred_out.shape[0])
    pred_traj = my_state_pred
    
    end = time.time()
    nexp["simulation_time"] += [end-start]
    
    if saveovito:
        if ind < 1:
            save_ovito(f"pred_{ind}.data", [
                state for state in NVEStates(pred_traj)], lattice="")
            save_ovito(f"actual_{ind}.data", [
                state for state in NVEStates(actual_traj)], lattice="")
        else:
            pass
    
    if plotthings:
        if ind < 1:
            for key, traj in {"actual": actual_traj, "pred": pred_traj}.items():
                
                print(f"plotting energy ({key})...")
                
                Es = Es_fn(traj)
                Es_pred = Es_pred_fn(traj)
                Es_pred = Es_pred - Es_pred[0] + Es[0]
                
                net_force_orig = net_force_orig_fn(traj)
                net_force_model = net_force_model_fn(traj)
                
                fig, axs = panel(1+R.shape[0], 1, figsize=(20,
                                                            R.shape[0]*5), hshift=0.1, vs=0.35)
                for i, ax in zip(range(R.shape[0]+1), axs):
                    if i == 0:
                        ax.text(0.6, 0.8, "Averaged over all particles",
                                transform=ax.transAxes, color="k")
                        ax.plot(net_force_orig.sum(axis=1), lw=6, label=[
                                r"$F_x$", r"$F_y$", r"$F_z$"][:R.shape[1]], alpha=0.5)
                        ax.plot(net_force_model.sum(
                            axis=1), "--", color="k")
                        ax.plot([], "--", c="k", label="Predicted")
                    else:
                        ax.text(0.6, 0.8, f"For particle {i}",
                                transform=ax.transAxes, color="k")
                        ax.plot(net_force_orig[:, i-1, :], lw=6, label=[r"$F_x$",
                                r"$F_y$", r"$F_z$"][:R.shape[1]], alpha=0.5)
                        ax.plot(
                            net_force_model[:, i-1, :], "--", color="k")
                        ax.plot([], "--", c="k", label="Predicted")
                    
                    ax.legend(loc=2, bbox_to_anchor=(1, 1),
                                labelcolor="markerfacecolor")
                    ax.set_ylabel("Net force")
                    ax.set_xlabel("Time step")
                    ax.set_title(f"{N}-Pendulum Exp {ind}")
                plt.savefig(_filename(f"net_force_Exp_{ind}_{key}.png"))
            
            Es = Es_fn(actual_traj)
            Eshat = Es_fn(pred_traj)
            H = Es[:, -1]
            Hhat = Eshat[:, -1]
            
            fig, axs = panel(1, 2, figsize=(20, 5))
            axs[0].plot(Es, label=["PE", "KE", "L", "TE"], lw=6, alpha=0.5)
            axs[1].plot(Eshat, "--", label=["PE", "KE", "L", "TE"])
            plt.legend(bbox_to_anchor=(1, 1), loc=2)
            axs[0].set_facecolor("w")
            
            xlabel("Time step", ax=axs[0])
            xlabel("Time step", ax=axs[1])
            ylabel("Energy", ax=axs[0])
            ylabel("Energy", ax=axs[1])
            
            title = f"HGNN {N}-Pendulum Exp {ind} Hmodel"
            axs[1].set_title(title)
            title = f"HGNN {N}-Pendulum Exp {ind} Hactual"
            axs[0].set_title(title)
            
            plt.savefig(_filename(title.replace(" ", "-")+f".png"))
        else:
            pass
    
    Es = Es_fn(actual_traj)
    Eshat = Es_fn(pred_traj)
    H = Es[:, -1]
    Hhat = Eshat[:, -1]
    
    herrrr = RelErr(H, Hhat)
    herrrr = herrrr.at[0].set(herrrr[1])
    nexp["Herr"] += [herrrr]
    
    nexp["Es"] += [Es]
    nexp["Eshat"] += [Eshat]
    
    
    nexp["z_pred"] += [pred_traj.position]
    nexp["z_actual"] += [actual_traj.position]
    
    nexp["v_pred"] += [pred_traj.velocity]
    nexp["v_actual"] += [actual_traj.velocity]
    
    nexp["constraintsF_pred"] += [pred_traj.constraint_force]
    nexp["constraintsF_actual"] += [actual_traj.constraint_force]
    
    zerrrr = RelErr(actual_traj.position, pred_traj.position)
    zerrrr = zerrrr.at[0].set(zerrrr[1])
    nexp["Zerr"] += [zerrrr]    
    
    ac_mom = jnp.square(actual_traj.velocity.sum(1)).sum(1)
    pr_mom = jnp.square(pred_traj.velocity.sum(1)).sum(1)
    nexp["Perr"] += [ac_mom - pr_mom]
    
    trajectories += [(actual_traj, pred_traj)]
    if ind%10==0:
        savefile(f"error_parameter.pkl", nexp)    
        savefile("trajectories.pkl", trajectories)

def make_plots(nexp, key, yl="Err", xl="Time", key2=None):
    print(f"Plotting err for {key}")
    fig, axs = panel(1, 1)
    filepart = f"{key}"
    for i in range(len(nexp[key])):
        y = nexp[key][i].flatten()
        if key2 is None:
            x = range(len(y))
        else:
            x = nexp[key2][i].flatten()
            filepart = f"{filepart}_{key2}"
        if semilog:
            plt.semilogy(x, y)
        else:
            plt.plot(x, y)
    
    plt.ylabel(yl)
    plt.xlabel(xl)
    
    plt.savefig(_filename(f"RelError_{filepart}.png"))
    
    fig, axs = panel(1, 1)
    
    mean_ = jnp.log(jnp.array(nexp[key])).mean(axis=0)
    std_ = jnp.log(jnp.array(nexp[key])).std(axis=0)
    
    up_b = jnp.exp(mean_ + 2*std_)
    low_b = jnp.exp(mean_ - 2*std_)
    y = jnp.exp(mean_)
    
    x = range(len(mean_))
    if semilog:
        plt.semilogy(x, y)
    else:
        plt.plot(x, y)
    plt.fill_between(x, low_b, up_b, alpha=0.5)
    plt.ylabel(yl)
    plt.xlabel("Time")
    plt.savefig(_filename(f"RelError_std_{key}.png"))

make_plots(
    nexp, "Zerr", yl=r"$\frac{||\hat{z}-z||_2}{||\hat{z}||_2+||z||_2}$")
make_plots(
    nexp, "Herr", yl=r"$\frac{||H(\hat{z})-H(z)||_2}{||H(\hat{z})||_2+||H(z)||_2}$")


# fire.Fire(main)


# def norm(a):
#     a2 = jnp.square(a)
#     n = len(a2)
#     a3 = a2.reshape(n, -1)
#     return jnp.sqrt(a3.sum(axis=1))


# def RelErr(ya, yp):
#     return norm(ya-yp) / (norm(ya) + norm(yp))


# Zerr = []
# Herr = []


# for ii in range(maxtraj):
#     if plotonly:
#         pred_traj, _ = loadfile(f"model_states_pred_ii_{ii}.pkl")
#         actual_traj, _ = loadfile(f"model_states_actual_ii_{ii}.pkl")
#     else:
#         print("FORWARD SIM ...")
#         R, V = get_RV()
#         z_actual_out = sim_origin(R, V)
#         x_act_out, p_act_out = jnp.split(z_actual_out, 2, axis=1)
#         zdot_act_out = jax.vmap(zdot_orig, in_axes=(0, 0, None))(
#             x_act_out, p_act_out, None)
#         _, force_act_out = jnp.split(zdot_act_out, 2, axis=1)

#         my_state = States()
#         my_state.position = x_act_out
#         my_state.velocity = p_act_out
#         my_state.force = force_act_out
#         my_state.mass = jnp.ones(x_act_out.shape[0])
#         actual_traj = my_state

#         z_pred_out = sim_model(R, V)
#         x_pred_out, p_pred_out = jnp.split(z_pred_out, 2, axis=1)
#         zdot_pred_out = jax.vmap(zdot_model, in_axes=(
#             0, 0, None))(x_pred_out, p_pred_out, params)
#         _, force_pred_out = jnp.split(zdot_pred_out, 2, axis=1)

#         my_state_pred = States()
#         my_state_pred.position = x_pred_out
#         my_state_pred.velocity = p_pred_out
#         my_state_pred.force = force_pred_out
#         my_state_pred.mass = jnp.ones(x_pred_out.shape[0])
#         pred_traj = my_state_pred

#         print("Saving datafile...")
#         savefile(f"model_states_pred_ii_{ii}.pkl", pred_traj)
#         savefile(f"model_states_actual_ii_{ii}.pkl", actual_traj)
#         save_ovito(f"pred_ii_{ii}.data", [
#             state for state in NVEStates(pred_traj)], lattice="")
#         save_ovito(f"actual_ii_{ii}.data", [
#             state for state in NVEStates(actual_traj)], lattice="")

#     def cal_energy(states):
#         KE = vmap(kin_energy)(states.velocity)
#         H = vmap(Hmodel, in_axes=(0, 0, None))(
#             states.position, states.velocity, params)
#         H = (H - H[0]) + 1
#         PE = (H - KE)
#         return jnp.array([PE, KE, PE-KE, H]).T
    
#     def cal_energy_actual(states):
#         KE = vmap(kin_energy)(states.velocity)
#         PE = vmap(PEactual)(states.position)
#         H = KE + PE
#         H = H - H[0] + 1
#         PE = (H - KE)
#         return jnp.array([PE, KE, PE-KE, H]).T
    
#     print("plotting energy...")
#     Es = cal_energy(actual_traj)
#     Es_actual = cal_energy_actual(actual_traj)
    
#     fig, axs = panel(1, 1, figsize=(20, 5))
#     colors = ["b", "r", "g", "m"]
#     labels = ["PE", "KE", "H", "TE"]
#     for it1, it2, c, lab in zip(Es_actual.T, Es.T, colors, labels):
#         plt.plot(it1, color=c, label=lab, lw=6, alpha=0.5)
#         plt.plot(it2, "--", color=c, label=lab+"_pred", lw=2)
#     plt.legend(bbox_to_anchor=(1, 1))
#     plt.ylabel("Energy")
#     plt.xlabel("Time step")
    
#     title = f"ham-2-4-hybrid actual"
#     plt.title(title)
#     plt.savefig(
#         _filename(title.replace(" ", "_")+f"_ii_{ii}.png"), dpi=300)
    
#     print("plotting energy... (pred)")
#     Esh = cal_energy(pred_traj)
#     Es_actualh = cal_energy_actual(pred_traj)
    
#     fig, axs = panel(1, 1, figsize=(20, 5))
#     colors = ["b", "r", "g", "m"]
#     labels = ["PE", "KE", "H", "TE"]
#     for it1, it2, c, lab in zip(Es_actualh.T, Esh.T, colors, labels):
#         plt.plot(it1, color=c, label=lab, lw=6, alpha=0.5)
#         plt.plot(it2, "--", color=c, label=lab+"_pred", lw=2)
#     plt.legend(bbox_to_anchor=(1, 1))
#     plt.ylabel("Energy")
#     plt.xlabel("Time step")
    
#     title = f"ham-2-4-hybrid pred"
#     plt.title(title)
#     plt.savefig(
#         _filename(title.replace(" ", "_")+f"_ii_{ii}.png"), dpi=300)
    
#     H = Es_actual[:, -1]
#     Hhat = Es_actualh[:, -1]
    
#     Herr += [RelErr(H, Hhat)+1e-30]
#     Zerr += [RelErr(actual_traj.position,
#                     pred_traj.position)+1e-30]


# def make_plots(y, x=None, yl="Err", xl="Time", ax=None, time_step=1):
    
#     print(f"Plotting err for {yl}")
#     fig, axs = panel(1, 2)
    
#     mean_ = jnp.log(jnp.array(y)).mean(axis=0)
#     std_ = jnp.log(jnp.array(y)).std(axis=0)

#     up_b = jnp.exp(mean_ + 2*std_)
#     low_b = jnp.exp(mean_ - 2*std_)
#     y = jnp.exp(mean_)

#     if ax == None:
#         pass
#     else:
#         plt.sca(ax)

#     x = np.array(range(len(mean_)))*time_step
#     if semilog:
#         plt.semilogy(x, y)
#     else:
#         plt.plot(x, y)
#     plt.fill_between(x, low_b, up_b, alpha=0.5)
#     plt.ylabel(yl)
#     plt.xlabel(xl)
#     plt.savefig(_filename(f"RelError_{yl}_{xl}.png"))


# fig, axs = panel(1, 2, label=["", ""], brackets=False)

# make_plots(Zerr, yl="Rollout error", time_step=time_step, ax=axs[0])
# make_plots(
#     Herr, yl="Energy violation", time_step=time_step, ax=axs[1])


# # fire.Fire(main)
