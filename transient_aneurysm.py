import torch
import numpy as np
from sympy import Symbol, sqrt, Max
from stl import mesh


import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.domain.inferencer  import VoxelInferencer

from modulus.domain.validator import PointwiseValidator
from modulus.domain.monitor import PointwiseMonitor
from modulus.key import Key
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.basic import NormalDotVec
from modulus.utils.io import csv_to_dict
from modulus.geometry.tessellation import Tessellation

from modulus.geometry.tessellation import Tessellation
from modulus.domain.inferencer import PointwiseInferencer
from modulus.utils.io.vtk import var_to_polyvtk

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from sympy import Symbol, Eq, Abs, sin, cos

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import SequentialSolver
from modulus.domain import Domain

from modulus.geometry.primitives_3d import Box

from modulus.models.fully_connected import FullyConnectedArch
from modulus.models.moving_time_window import MovingTimeWindowArch
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.domain.inferencer import PointVTKInferencer
from modulus.utils.io import (
    VTKUniformGrid,
)
from modulus.key import Key
from modulus.node import Node
from modulus.eq.pdes.navier_stokes import NavierStokes

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:

    # read stl files to make geometry
    point_path = to_absolute_path("examples/aneurysm/stl_files")
    inlet_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_inlet.stl", airtight=False
    )
    outlet_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_outlet.stl", airtight=False
    )
    noslip_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_noslip.stl", airtight=False
    )
    integral_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_integral.stl", airtight=False
    )
    interior_mesh = Tessellation.from_stl(
        point_path + "/aneurysm_closed.stl", airtight=True
    )

    # params
    nu = 0.025
    inlet_vel = 1.5
    
    # time window parameters
    time_window_size = 1.0
    t_symbol = Symbol("t")
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 10

    
    
    # inlet velocity profile
    def circular_parabola(x, y, z, center, normal, radius, max_vel):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]
        distance = sqrt(centered_x ** 2 + centered_y ** 2 + centered_z ** 2)
        parabola = max_vel * Max((1 - (distance / radius) ** 2), 0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola

    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh

    # normalize invars
    def normalize_invar(invar, center, scale, dims=2):
        invar["x"] -= center[0]
        invar["y"] -= center[1]
        invar["z"] -= center[2]
        invar["x"] *= scale
        invar["y"] *= scale
        invar["z"] *= scale
        if "area" in invar.keys():
            invar["area"] *= scale ** dims
        return invar

    # scale and normalize mesh and openfoam data
    center = (-18.40381048596882, -50.285383353981196, 12.848136936899031)
    scale = 0.4
    inlet_mesh = normalize_mesh(inlet_mesh, center, scale)
    outlet_mesh = normalize_mesh(outlet_mesh, center, scale)
    noslip_mesh = normalize_mesh(noslip_mesh, center, scale)
    integral_mesh = normalize_mesh(integral_mesh, center, scale)
    interior_mesh = normalize_mesh(interior_mesh, center, scale)

    # geom params
    inlet_normal = (0.8526, -0.428, 0.299)
    inlet_area = 21.1284 * (scale ** 2)
    inlet_center = (-4.24298030045776, 4.082857101816247, -4.637790193399717)
    inlet_radius = np.sqrt(inlet_area / np.pi)
    outlet_normal = (0.33179, 0.43424, 0.83747)
    outlet_area = 12.0773 * (scale ** 2)
    outlet_radius = np.sqrt(outlet_area / np.pi)

   

    # make list of nodes to unroll graph on
    ns = NavierStokes(nu=nu * scale, rho=1.0, dim=3, time=True)
    normal_dot_vel = NormalDotVec(["u", "v", "w", "p"])
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p")],
        layer_size=256,
    )
    time_window_net = MovingTimeWindowArch(flow_net, time_window_size)

    nodes = (
        ns.make_nodes()  
        + normal_dot_vel.make_nodes()
        +[time_window_net.make_node(name="time_window_network")]
    )
    
       # make initial condition domain
    ic_domain = Domain("initial_conditions")

    # make moving window domain
    window_domain = Domain("window")

    # make initial condition
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={
            "u":0,
            "v": 0,
            "w": 0,
            "p": 0,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 100, "w": 100, "p": 100},
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic, name="ic")

    # make constraint for matching previous windows initial condition
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0, "w_prev_step_diff": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
            "w_prev_step_diff": 100,
        },
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic, name="ic")
    
    # add constraints to solver
    # inlet
    u, v, w = circular_parabola(
        Symbol("x"),
        Symbol("y"),
        Symbol("z"),
        center=inlet_center,
        normal=inlet_normal,
        radius=inlet_radius,
        max_vel=inlet_vel,
    )
    
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_mesh,
        outvar={"u": u, "v": v, "w": w},
        parameterization=time_range,
        batch_size=cfg.batch_size.inlet,
    )
    ic_domain.add_constraint(inlet, name="inlet")
    window_domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"p": 0},
        parameterization=time_range,
        batch_size=cfg.batch_size.outlet,
    )
    ic_domain.add_constraint(outlet, name="outlet")
    window_domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar={"u": 0, "v": 0, "w": 0},
        parameterization=time_range,
        batch_size=cfg.batch_size.no_slip,
    )
    ic_domain.add_constraint(no_slip, name="no_slip")
    window_domain.add_constraint(no_slip, "no_slip")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    ic_domain.add_constraint(interior, name="interior")
    window_domain.add_constraint(interior, "interior")

    # Integral Continuity 1
    integral_continuity1 = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={"normal_dot_vel": 2.540},
        batch_size=1,
        parameterization=time_range,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
    )
    ic_domain.add_constraint(integral_continuity1, name="integral_continuity1")
    window_domain.add_constraint(integral_continuity1, "integral_continuity1")

    # Integral Continuity 2
    integral_continuity2 = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_mesh,
        outvar={"normal_dot_vel": -2.540},
        batch_size=1,
        parameterization=time_range,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={"normal_dot_vel": 0.1},
    )
    ic_domain.add_constraint(integral_continuity2, name="integral_continuity2")
    window_domain.add_constraint(integral_continuity2, "integral_continuity2")

    
     # Find the bounds
    point_path = to_absolute_path("examples/aneurysm/stl_files/aneurysm_closed.stl")
    recb = mesh.Mesh.from_file(point_path)
    min_bound = np.min(recb.points, axis=0)
    max_bound = np.max(recb.points, axis=0)

    bounds = {
        "x": (min_bound[0], max_bound[0]),
        "y": (min_bound[1], max_bound[1]),
        "z": (min_bound[2], max_bound[2]),
    } 
    
    
    def mask_fn(**invar):
            invar_s = {
                key: value for key, value in invar.items() if key in ["x", "y", "z"]
            }
            # TODO... lambda x, y, z: self.interior_mesh.sdf({"x": x, "y": y, "z": z}, {})["sdf"]
            invar_p = {
                key: value for key, value in invar.items() if key not in ["x", "y", "z"]
            }
            sdf = interior_mesh.sdf(invar_s, invar_p)
            return np.less(sdf["sdf"], 0)

    voxel_inferencer = VoxelInferencer(
        bounds = [[min_bound[0], max_bound[0]], [min_bound[1], max_bound[1]], [min_bound[2], max_bound[2]]],
        npoints = [128, 128, 128],
        nodes=nodes,
        output_names=["u", "v", "w", "p"],
        export_map={"U": ["u", "v", "w"], "p": ["p"]},
        mask_fn = mask_fn,
        requires_grad=False,
        batch_size=1024,
    )
    ic_domain.add_inferencer(voxel_inferencer, "vox_inf")
    window_domain.add_constraint(voxel_inferencer, "voxel_inferencer")

    # make solver
    slv = Solver(cfg, window_domain)

    # start flow solver
    slv.solve()





if __name__ == "__main__":
    run()


