"""Backend-agnostic quantity preprocessing pipeline.

This module contains the shared logic that must run before any rendering
backend takes over: domain copy, rotation matrix computation, quantity
assembly, optional warp operations, mapping, shape validation, and
per-particle-type filtering.

Both :class:`~warpSPHPlotting.backends.matplotlib_backend.MatplotlibBackend` and
:class:`~warpSPHPlotting.backends.pyvista_backend.PyVistaBackend` (and future
backends) call :func:`prepare_particle_states` so that the business logic
for "transform raw particle data into per-type scalar arrays" lives in
exactly one place.
"""

import copy
from typing import Any, Tuple

import torch

from warpSPHCore import *

from ..enumTypes import Mapping
from ..math import buildRotationMatrix, mapQuantity
from ..options import PlottingOptions
from ..state import PlottingParticleState, assembleQuantity, filterState, rotateState


def prepare_particle_states(
    particleState: Any,
    domain: DomainDescription,
    quantity: Any,
    options: PlottingOptions,
) -> Tuple[DomainDescription, PlottingOptions, Any, Any, Any, Any]:
    """Backend-agnostic preprocessing pipeline.

    Runs the following steps in order:

    1. Deep-copy *options* so caller's instance is never mutated.
    2. Create a detached copy of the domain tensors.
    3. Build the rotation / inverse-rotation matrices from ``domain.angles``
       (if present).
    4. Assemble the per-particle quantity tensor.
    5. Rotate the particle state by the *inverse* rotation matrix.
    6. Apply any ``options.plottingOperation`` warp operations.
    7. Apply ``options.mapping`` to reduce vector → scalar when needed.
    8. Validate the resulting quantity shape.
    9. Filter into three ``PlottingParticleState`` objects by particle type.

    Parameters
    ----------
    particleState:
        Raw particle state (``warpSPHCore.ParticleState`` or duck-typed).
    domain:
        Simulation domain description.
    quantity:
        Per-particle scalar tensor or tuple of per-type tensors.
    options:
        Plotting options.  A deep copy is made internally so the original is
        never mutated.

    Returns
    -------
    domain_copy : DomainDescription
        Detached copy of the domain (with rotated extents if applicable).
    options_copy : PlottingOptions
        Deep copy of options (possibly with ``markerSize`` filled in).
    assembled : torch.Tensor
        Full per-particle assembled quantity before any further processing.
    rotated_quantities : torch.Tensor
        Full per-particle quantity array **after** operations and mapping are
        applied.  This is the array to pass to grid-mapping helpers, since they
        need all-particle quantities (not per-type slices).
    fluid_state : PlottingParticleState
        Fluid particles with processed scalar quantities.
    boundary_state : PlottingParticleState
        Boundary particles with processed scalar quantities.
    ghost_state : PlottingParticleState
        Ghost particles with processed scalar quantities.
    """
    options = copy.deepcopy(options)

    # ── 1. Domain copy with detached tensors ─────────────────────────────────
    domain_ = DomainDescription(
        min=domain.min.detach(),
        max=domain.max.detach(),
        periodic=domain.periodic.detach(),
        dim=domain.dim,
    )

    # ── 2. Rotation matrices ──────────────────────────────────────────────────
    if hasattr(domain, "angles"):
        rotMat = buildRotationMatrix(
            torch.tensor(
                domain.angles,
                dtype=domain.min.dtype,
                device=domain.min.device,
            ),
            domain.dim,
            device=domain.min.device,
            dtype=domain.min.dtype,
        )
        invRotMat = rotMat.inverse()
    else:
        rotMat = None
        invRotMat = None

    # ── 3. Assemble quantity ─────────────────────────────────────────────────
    assembled = assembleQuantity(particleState, quantity)

    # ── 4. Build plotting particle state and rotate ──────────────────────────
    plot_pstate = PlottingParticleState(
        positions=particleState.positions,
        supports=particleState.supports,
        masses=particleState.masses,
        densities=particleState.densities,
        kinds=particleState.kinds,
        quantities=assembled,
    )
    rotated = rotateState(plot_pstate, invRotMat)

    # ── 5. Apply warp operations ─────────────────────────────────────────────
    if options.plottingOperation is not None:
        ops = (
            options.plottingOperation
            if isinstance(options.plottingOperation, list)
            else [options.plottingOperation]
        )
        for op in ops:
            rotated.quantities = warpOperation(
                queryParticles=rotated,
                queryValues=rotated.quantities,
                operationProperties=op,
                adjacency=None,
                domain=domain_,
            )

    # ── 6. Apply scalar mapping ───────────────────────────────────────────────
    if options.mapping != Mapping.none:
        rotated.quantities = mapQuantity(rotated.quantities, options.mapping)

    # ── 7. Validate ────────────────────────────────────────────────────────────
    if rotated.quantities.shape[0] != rotated.positions.shape[0]:
        raise ValueError(
            f"Quantity length {rotated.quantities.shape[0]} does not match "
            f"number of particles {rotated.positions.shape[0]} after operations."
        )
    if rotated.quantities.ndim == 2:
        raise ValueError(
            f"Quantity has more than 1 component per particle "
            f"(shape: {rotated.quantities.shape}).  "
            f"Specify options.mapping to reduce to a scalar."
        )

    # ── 8. Filter by type ────────────────────────────────────────────────────
    fluid = filterState(rotated, rotated.quantities, kind=ParticleType.Fluid)
    boundary = filterState(rotated, rotated.quantities, kind=ParticleType.Boundary)
    ghost = filterState(rotated, rotated.quantities, kind=ParticleType.Ghost)

    return domain_, options, assembled, rotated.quantities, fluid, boundary, ghost
