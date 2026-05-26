"""Reusable explicit cloth sticking helpers for Isaac Sim particle cloth.

The cloth pickup demos use PhysX particle cloth for deformation, but Isaac Sim's
public tensor path gives convenient particle positions and velocities rather
than a stable per-particle contact-force manifold. Native PBD adhesion also did
not reliably lift the toy cloth from the table. The explicit sticking layer in
this module fills that gap: it detects compact compressed contact patches and
then drives only those particles in the local frame of the contact surface that
should own the grasp.

Overview of the method
----------------------
1. Scene code registers adhesive surfaces. Each surface has a world frame,
   material ranking fields, a finite contact footprint, and a contact margin.
   The utility code treats these as plain dictionaries so the toy scene and the
   full robot scene can discover surfaces differently while sharing patch logic.
2. Particle positions are tested against every active surface on the GPU with
   Torch. A particle is in a surface contact patch only when it is close to the
   surface normal plane and inside the finite surface footprint. Rectangular
   ``half_extents`` are preferred for box pads; circular ``radius`` remains
   supported for older full-scene surfaces.
3. Candidate adhesive particles must be simultaneously in two valid surface
   contacts and the two closest points on those surfaces must be within a small
   ``press_gap``. For opposed surfaces, the optional signed-normal slop requires
   particles to lie between the two surface planes rather than merely near the
   absolute planes. Without this signed test, the detector can count particles
   outside the jaw as squeezed contact, which creates stale or flying patches.
4. Valid candidates are split into grid-connected components when a cloth grid
   shape is supplied. This lets multiple folds between the same two faces become
   separate patches instead of one artificial large patch that flattens the fold.
5. Each patch stores selected particle indices plus particle positions in the
   chosen anchor surface frame. The anchor is chosen by material adhesion, then
   friction, then closest average normal distance. This keeps low-friction table
   surfaces useful as backing contacts without making the table the preferred
   owner of the grasp.
6. Projection either directly writes selected particles to their moving
   anchor-frame targets (``teleport``) or uses a capped PD-style update. Direct
   writes are simple and strong; PD is useful as an ablation for diagnosing
   energy injected by hard position projection.

Important failure modes this is meant to avoid
----------------------------------------------
- Sticking by nearest particles instead of compressed two-surface contact can
  attach cloth that is not actually squeezed by the gripper.
- Keeping bottom/table patches after lift keeps dragging table-contact cloth
  after that contact no longer exists.
- Handoff between bottom/table and inner-finger patches reuses the wrong cloth
  particles; those are different physical contact regions.
- Using one broad patch for several folds can erase buckling or force unrelated
  cloth layers to move together.
- Reading moving surface metadata from stale USD transforms while PhysX uses
  tensor-driven rigid body poses makes the sticking detector observe a different
  scene than the solver.
- Zeroing attached-particle velocity while moving their positions injects extra
  energy. Target velocity should follow the anchor-frame target motion.

The functions below intentionally avoid importing Isaac Sim. They operate on
Torch tensors and simple dictionaries. Scene code remains responsible for
discovering collision surfaces, updating their frames, getting cloth tensors,
and calling ``cloth_view.set_world_positions`` / ``set_velocities``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch


def surface_frame_tensors(surface: dict, particle_positions: torch.Tensor) -> dict:
    """Return a surface frame converted to the particle tensor device/dtype.

    ``surface["frame"]`` is expected to contain ``origin`` and a 3x3
    ``rotation`` whose columns are tangent-u, tangent-v, and outward normal.
    The returned tensors can be used directly in GPU particle calculations.
    """

    device = particle_positions.device
    dtype = particle_positions.dtype
    frame = surface["frame"]
    return {
        "origin": torch.as_tensor(frame["origin"], dtype=dtype, device=device),
        "rotation": torch.as_tensor(frame["rotation"], dtype=dtype, device=device),
    }


def surface_contact_torch(surface: dict, particle_positions: torch.Tensor) -> dict:
    """Compute per-particle geometric contact data for one adhesive surface.

    A particle is considered in contact when it is inside the surface normal
    margin and inside the tangent footprint. If ``surface["half_extents"]`` is
    present, the footprint is rectangular in local tangent coordinates; otherwise
    ``surface["radius"]`` is used as a circular footprint.
    """

    frame = surface_frame_tensors(surface, particle_positions)
    local_positions = (particle_positions - frame["origin"]) @ frame["rotation"]
    tangent_distance = torch.linalg.norm(local_positions[:, :2], dim=1)
    normal_distance = torch.abs(local_positions[:, 2])
    closest_points = (
        particle_positions
        - frame["rotation"][:, 2] * local_positions[:, 2:3]
    )
    half_extents = surface.get("half_extents")
    if half_extents is None:
        tangent_mask = tangent_distance <= float(surface["radius"])
    else:
        tangent_mask = (
            (torch.abs(local_positions[:, 0]) <= float(half_extents[0]))
            & (torch.abs(local_positions[:, 1]) <= float(half_extents[1]))
        )
    mask = (normal_distance <= float(surface["contact_margin"])) & tangent_mask
    score = normal_distance + 0.25 * tangent_distance
    return {
        "frame": frame,
        "local_positions": local_positions,
        "tangent_distance": tangent_distance,
        "signed_normal_distance": local_positions[:, 2],
        "normal_distance": normal_distance,
        "closest_points": closest_points,
        "mask": mask,
        "score": score,
    }


def surface_contacts_torch(
    active_surfaces: Sequence[dict],
    particle_positions: torch.Tensor,
) -> list[dict]:
    """Attach Torch frame/contact dictionaries to every active surface."""

    return [
        {
            **surface,
            "torch_frame": surface_frame_tensors(surface, particle_positions),
            "contact": surface_contact_torch(surface, particle_positions),
        }
        for surface in active_surfaces
    ]


def surfaces_are_opposed(
    first_surface: dict,
    second_surface: dict,
    normal_dot_max: float = -0.35,
) -> bool:
    """Return whether two surface normals roughly oppose each other."""

    first_normal = first_surface["torch_frame"]["rotation"][:, 2]
    second_normal = second_surface["torch_frame"]["rotation"][:, 2]
    return bool(torch.dot(first_normal, second_normal).item() < normal_dot_max)


def pressed_between_surfaces_torch(
    first_contact: dict,
    second_contact: dict,
    press_gap: float,
    signed_normal_slop: float | None = None,
) -> torch.Tensor:
    """Return particles geometrically squeezed between two surface contacts.

    ``press_gap`` limits the distance between closest points on the two surface
    planes. ``signed_normal_slop`` additionally requires each particle to be on
    or just inside both opposed half-spaces. Pass ``None`` to preserve the older
    gap-only full-scene behavior.
    """

    surface_gap = torch.linalg.norm(
        first_contact["closest_points"] - second_contact["closest_points"],
        dim=1,
    )
    pressed_mask = surface_gap <= float(press_gap)
    if signed_normal_slop is None:
        return pressed_mask
    between_normals = (
        first_contact["signed_normal_distance"] >= -float(signed_normal_slop)
    ) & (
        second_contact["signed_normal_distance"] >= -float(signed_normal_slop)
    )
    return pressed_mask & between_normals


def adhesive_surfaces_can_pair(
    first_surface: dict,
    second_surface: dict,
    side: str | None = None,
    *,
    require_opposed: bool = True,
) -> bool:
    """Check scene-independent rules for a usable adhesive surface pair."""

    if first_surface["kind"] != "finger" and second_surface["kind"] != "finger":
        return False
    if first_surface["prim_path"] == second_surface["prim_path"]:
        return False
    first_pair_group = first_surface.get("pair_group")
    second_pair_group = second_surface.get("pair_group")
    if first_pair_group is not None and first_pair_group == second_pair_group:
        return False
    if require_opposed and not surfaces_are_opposed(first_surface, second_surface):
        return False
    if first_surface.get("requires_closed_side") not in (None, side):
        return False
    if second_surface.get("requires_closed_side") not in (None, side):
        return False
    return True


def adhesive_pair_stats(
    first_surface: dict,
    second_surface: dict,
    press_gap: float,
    signed_normal_slop: float | None = None,
) -> dict:
    """Summarize contact and compressed-contact counts for one surface pair."""

    contact_mask = first_surface["contact"]["mask"] & second_surface["contact"]["mask"]
    contact_count = int(torch.count_nonzero(contact_mask).item())
    pair_score = first_surface["contact"]["score"] + second_surface["contact"]["score"]
    if contact_count == 0:
        min_pair_score = float(torch.min(pair_score).item())
        return {
            "contact_count": 0,
            "pressed_count": 0,
            "min_gap": None,
            "min_pair_score": min_pair_score,
            "first_min_normal": float(
                torch.min(first_surface["contact"]["normal_distance"]).item()
            ),
            "second_min_normal": float(
                torch.min(second_surface["contact"]["normal_distance"]).item()
            ),
            "first_min_tangent": float(
                torch.min(first_surface["contact"]["tangent_distance"]).item()
            ),
            "second_min_tangent": float(
                torch.min(second_surface["contact"]["tangent_distance"]).item()
            ),
            "first_radius": float(first_surface["radius"]),
            "second_radius": float(second_surface["radius"]),
        }

    surface_gap = torch.linalg.norm(
        first_surface["contact"]["closest_points"]
        - second_surface["contact"]["closest_points"],
        dim=1,
    )
    pressed_count = int(
        torch.count_nonzero(
            contact_mask
            & pressed_between_surfaces_torch(
                first_surface["contact"],
                second_surface["contact"],
                press_gap,
                signed_normal_slop,
            )
        ).item()
    )
    return {
        "contact_count": contact_count,
        "pressed_count": pressed_count,
        "min_gap": float(torch.min(surface_gap[contact_mask]).item()),
        "min_pair_score": float(torch.min(pair_score[contact_mask]).item()),
        "first_min_normal": float(
            torch.min(first_surface["contact"]["normal_distance"]).item()
        ),
        "second_min_normal": float(
            torch.min(second_surface["contact"]["normal_distance"]).item()
        ),
        "first_min_tangent": float(
            torch.min(first_surface["contact"]["tangent_distance"]).item()
        ),
        "second_min_tangent": float(
            torch.min(second_surface["contact"]["tangent_distance"]).item()
        ),
        "first_radius": float(first_surface["radius"]),
        "second_radius": float(second_surface["radius"]),
    }


def surface_candidate_distance(
    surface: dict,
    candidate_indices: torch.Tensor | None,
) -> float:
    """Return mean normal distance from a surface to candidate particles."""

    if candidate_indices is None or candidate_indices.numel() == 0:
        return float("inf")
    return float(
        torch.mean(surface["contact"]["normal_distance"][candidate_indices]).item()
    )


def choose_adhesive_anchor(
    first_surface: dict,
    second_surface: dict,
    candidate_indices: torch.Tensor,
) -> dict:
    """Choose which surface owns a patch.

    Ranking is adhesion, then friction, then closeness to the candidate
    particles. That last tiebreaker avoids arbitrary switching when two equal
    finger surfaces contact the same particles.
    """

    return max(
        (first_surface, second_surface),
        key=lambda surface: (
            surface.get("adhesion", 0.0),
            surface.get("friction", 0.0),
            -surface_candidate_distance(surface, candidate_indices),
        ),
    )


def is_finger_pinch_mode(
    mode: str,
    pinch_surface_names: Iterable[str] = ("left_inner_face", "right_inner_face"),
) -> bool:
    """Return whether a patch mode is exactly the configured pinch pair."""

    return set(mode.split("+")) == set(pinch_surface_names)


def expand_adhesive_patch_indices(
    particle_positions: torch.Tensor,
    seed_indices: torch.Tensor,
    anchor_frame: dict,
    expanded_patch_radius: float,
    expanded_patch_max_particles: int,
    eligible_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optionally grow seeds inside the anchor tangent plane.

    Expansion should normally be constrained to the same compressed-contact
    candidate set via ``eligible_indices``. Expanding beyond current contact can
    glue nearby but unsqueezed cloth and inject energy when the gripper moves.
    """

    if seed_indices.numel() == 0:
        return seed_indices
    if eligible_indices is None:
        eligible_indices = torch.arange(
            particle_positions.shape[0],
            dtype=torch.long,
            device=particle_positions.device,
        )

    local_positions = (
        particle_positions - anchor_frame["origin"]
    ) @ anchor_frame["rotation"]
    seed_local_positions = local_positions[seed_indices]
    tangent_center = torch.mean(seed_local_positions[:, :2], dim=0)
    tangent_distance = torch.linalg.norm(
        local_positions[:, :2] - tangent_center,
        dim=1,
    )
    eligible_mask = torch.zeros(
        particle_positions.shape[0],
        dtype=torch.bool,
        device=particle_positions.device,
    )
    eligible_mask[eligible_indices] = True
    expanded_mask = (
        tangent_distance <= float(expanded_patch_radius)
    ) & eligible_mask
    expanded_indices = torch.nonzero(expanded_mask, as_tuple=False).flatten()
    if expanded_indices.numel() <= seed_indices.numel():
        return seed_indices

    if expanded_indices.numel() > int(expanded_patch_max_particles):
        order = torch.argsort(tangent_distance[expanded_indices])[
            :int(expanded_patch_max_particles)
        ]
        expanded_indices = expanded_indices[order]
    return torch.unique(torch.cat((seed_indices, expanded_indices)))


def connected_component_indices_grid(
    mask: torch.Tensor,
    max_components: int,
    min_component_particles: int = 1,
    grid_shape: tuple[int, int] | None = None,
) -> list[torch.Tensor]:
    """Return largest connected components from a flattened cloth-grid mask.

    ``grid_shape`` is ``(rows, columns)``. If no matching grid shape is supplied,
    all active indices are returned as one component. The implementation stays in
    Torch tensors, so the connected-component ablation does not force a CPU copy.
    """

    if torch.count_nonzero(mask).item() == 0 or int(max_components) <= 0:
        return []
    if grid_shape is None or mask.numel() != int(grid_shape[0]) * int(grid_shape[1]):
        return [torch.nonzero(mask, as_tuple=False).flatten()]

    rows, columns = int(grid_shape[0]), int(grid_shape[1])
    inactive_label = mask.numel()
    active_grid = mask.reshape(rows, columns)
    labels = torch.arange(mask.numel(), dtype=torch.long, device=mask.device).reshape(
        rows,
        columns,
    )
    inactive_labels = torch.full_like(labels, inactive_label)
    labels = torch.where(active_grid, labels, inactive_labels)
    for _iteration in range(rows + columns):
        next_labels = labels.clone()
        next_labels[1:, :] = torch.minimum(next_labels[1:, :], labels[:-1, :])
        next_labels[:-1, :] = torch.minimum(next_labels[:-1, :], labels[1:, :])
        next_labels[:, 1:] = torch.minimum(next_labels[:, 1:], labels[:, :-1])
        next_labels[:, :-1] = torch.minimum(next_labels[:, :-1], labels[:, 1:])
        labels = torch.where(active_grid, next_labels, inactive_labels)

    flat_labels = labels.flatten()
    active_labels = flat_labels[mask]
    component_labels, component_counts = torch.unique(
        active_labels,
        sorted=False,
        return_counts=True,
    )
    count_order = torch.argsort(component_counts, descending=True)
    ordered_counts = component_counts[count_order]
    large_enough_order = count_order[
        ordered_counts >= max(int(min_component_particles), 1)
    ]
    retained_labels = component_labels[large_enough_order[:max_components]]
    return [
        torch.nonzero(flat_labels == component_label, as_tuple=False).flatten()
        for component_label in retained_labels
    ]


def make_adhesive_patch(
    first_surface: dict,
    second_surface: dict,
    particle_positions: torch.Tensor,
    candidate_indices: torch.Tensor,
    pair_score: torch.Tensor,
    *,
    expand_adhesive_patch: bool,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    required_anchor_name: str | None = None,
) -> dict | None:
    """Build one stored adhesive patch from a compressed candidate component."""

    if candidate_indices.numel() == 0:
        return None
    anchor = choose_adhesive_anchor(first_surface, second_surface, candidate_indices)
    if required_anchor_name is not None:
        surfaces_by_name = {
            first_surface["name"]: first_surface,
            second_surface["name"]: second_surface,
        }
        anchor = surfaces_by_name.get(required_anchor_name)
        if anchor is None:
            return None
    anchor_frame = anchor["torch_frame"]
    component_pair_score = pair_score[candidate_indices]
    candidate_local_positions = (
        particle_positions[candidate_indices] - anchor_frame["origin"]
    ) @ anchor_frame["rotation"]
    if adhesive_local_patch_radius > 0.0:
        best_pair_order = torch.argsort(component_pair_score)
        best_local_position = candidate_local_positions[best_pair_order[0]]
        local_distances = torch.linalg.norm(
            candidate_local_positions[:, :2] - best_local_position[:2],
            dim=1,
        )
        local_mask = local_distances <= float(adhesive_local_patch_radius)
        local_candidate_indices = candidate_indices[local_mask]
        local_pair_score = component_pair_score[local_mask]
    else:
        local_candidate_indices = candidate_indices
        local_pair_score = component_pair_score
    if local_candidate_indices.numel() == 0:
        return None

    local_order = torch.argsort(local_pair_score)
    if int(adhesive_patch_max_particles) > 0:
        local_order = local_order[:adhesive_patch_max_particles]
    seed_indices = local_candidate_indices[local_order]
    selected_indices = (
        expand_adhesive_patch_indices(
            particle_positions,
            seed_indices,
            anchor_frame,
            adhesive_expanded_patch_radius,
            adhesive_expanded_patch_max_particles,
            candidate_indices,
        )
        if expand_adhesive_patch
        else seed_indices
    )
    local_positions = (
        particle_positions[selected_indices] - anchor_frame["origin"]
    ) @ anchor_frame["rotation"]
    required_side = (
        first_surface.get("requires_closed_side")
        or second_surface.get("requires_closed_side")
    )
    return {
        "mode": f"{first_surface['name']}+{second_surface['name']}",
        "anchor_name": anchor["name"],
        "anchor_kind": anchor["kind"],
        "indices": selected_indices,
        "local_positions": local_positions,
        "requires_closed_side": required_side,
        "score": torch.mean(local_pair_score[local_order]),
        "seed_count": int(seed_indices.numel()),
        "candidate_count": int(candidate_indices.numel()),
    }


def choose_adhesive_patches_torch(
    active_surfaces: Sequence[dict],
    particle_positions: torch.Tensor,
    *,
    expand_adhesive_patch: bool,
    adhesive_pair_mode: str,
    adhesive_patch_max_particles: int,
    adhesive_local_patch_radius: float,
    adhesive_expanded_patch_radius: float,
    adhesive_expanded_patch_max_particles: int,
    adhesive_max_patches: int,
    adhesive_components_per_pair: int,
    adhesive_min_component_particles: int = 1,
    press_gap: float,
    signed_normal_slop: float | None = None,
    grid_shape: tuple[int, int] | None = None,
    side: str | None = "pinch",
    pinch_surface_names: Iterable[str] = ("left_inner_face", "right_inner_face"),
    required_anchor_name: str | None = None,
    excluded_indices: torch.Tensor | None = None,
    pair_filter: Callable[[dict, dict], bool] | None = None,
) -> list[dict]:
    """Select one or more contact-gated adhesive patches.

    ``pair_filter`` can add scene-specific restrictions after the generic pair
    checks. The returned patches are sorted by particle count, score, mode, and
    anchor name so selection is deterministic when several pairs are valid.
    """

    contacts = surface_contacts_torch(active_surfaces, particle_positions)
    excluded_mask = torch.zeros(
        particle_positions.shape[0],
        dtype=torch.bool,
        device=particle_positions.device,
    )
    if excluded_indices is not None and excluded_indices.numel() > 0:
        excluded_mask[excluded_indices] = True

    patches = []
    for first_index, first_surface in enumerate(contacts):
        for second_surface in contacts[first_index + 1:]:
            if not adhesive_surfaces_can_pair(first_surface, second_surface, side):
                continue
            if adhesive_pair_mode == "finger-pinch":
                surface_names = {first_surface["name"], second_surface["name"]}
                if surface_names != set(pinch_surface_names):
                    continue
            if pair_filter is not None and not pair_filter(first_surface, second_surface):
                continue

            combined_mask = (
                first_surface["contact"]["mask"]
                & second_surface["contact"]["mask"]
                & pressed_between_surfaces_torch(
                    first_surface["contact"],
                    second_surface["contact"],
                    press_gap,
                    signed_normal_slop,
                )
                & ~excluded_mask
            )
            pair_score = (
                first_surface["contact"]["score"]
                + second_surface["contact"]["score"]
            )
            component_indices = connected_component_indices_grid(
                combined_mask,
                adhesive_components_per_pair,
                adhesive_min_component_particles,
                grid_shape,
            )
            for candidate_indices in component_indices:
                patch = make_adhesive_patch(
                    first_surface,
                    second_surface,
                    particle_positions,
                    candidate_indices,
                    pair_score,
                    expand_adhesive_patch=expand_adhesive_patch,
                    adhesive_patch_max_particles=adhesive_patch_max_particles,
                    adhesive_local_patch_radius=adhesive_local_patch_radius,
                    adhesive_expanded_patch_radius=adhesive_expanded_patch_radius,
                    adhesive_expanded_patch_max_particles=(
                        adhesive_expanded_patch_max_particles
                    ),
                    required_anchor_name=required_anchor_name,
                )
                if patch is not None:
                    patches.append(patch)

    patches.sort(
        key=lambda patch: (
            -int(patch["indices"].numel()),
            float(patch["score"].item()),
            patch["mode"],
            patch["anchor_name"],
        )
    )
    return patches[:adhesive_max_patches]


def choose_adhesive_patch_torch(
    active_surfaces: Sequence[dict],
    particle_positions: torch.Tensor,
    **kwargs,
) -> dict | None:
    """Return the best single adhesive patch, or ``None`` if no patch exists."""

    patches = choose_adhesive_patches_torch(
        active_surfaces,
        particle_positions,
        adhesive_max_patches=1,
        adhesive_components_per_pair=1,
        **kwargs,
    )
    return patches[0] if patches else None


def choose_anchor_patches(
    active_surfaces: Sequence[dict],
    particle_positions: torch.Tensor,
    anchor_names: Sequence[str],
    *,
    excluded_indices: torch.Tensor | None = None,
    **kwargs,
) -> list[dict]:
    """Choose one disjoint patch for each requested anchor surface name."""

    patches = []
    if excluded_indices is None:
        excluded_indices = torch.empty(
            0,
            dtype=torch.long,
            device=particle_positions.device,
        )
    for anchor_name in anchor_names:
        patch = choose_adhesive_patch_torch(
            active_surfaces,
            particle_positions,
            required_anchor_name=anchor_name,
            excluded_indices=excluded_indices,
            **kwargs,
        )
        if patch is None:
            continue
        patches.append(patch)
        excluded_indices = torch.unique(
            torch.cat((excluded_indices, patch["indices"]))
        )
    return patches


def _surface_pair_normal_velocity_filter(
    patch: dict,
    surfaces_by_name: dict[str, dict],
    particle_positions: torch.Tensor,
    candidate_velocities: torch.Tensor,
    fallback_velocities: torch.Tensor,
    physics_dt: float,
) -> tuple[torch.Tensor, dict]:
    """Remove one-sided pair-normal surface velocity from cloth targets.

    The adhesive projection can safely inherit tangential motion from the anchor
    surface, such as lifting a pinched fold upward. Normal motion is different:
    if only one plane in a compressed surface pair moves along the pair normal,
    that plane is separating from or scraping across the other one and assigning
    the same normal velocity to the cloth injects release energy. Transfer the
    pair-normal component only when both planes have frame motion in the same
    normal direction; otherwise keep the cloth's existing normal velocity.
    """

    surface_names = patch.get("mode", "").split("+")
    if len(surface_names) != 2 or patch["anchor_name"] not in surface_names:
        return candidate_velocities, {
            "normal_velocity_filtered": False,
            "normal_velocity_reason": "not_surface_pair",
        }
    if any(name not in surfaces_by_name for name in surface_names):
        return candidate_velocities, {
            "normal_velocity_filtered": False,
            "normal_velocity_reason": "missing_surface",
        }

    anchor_name = patch["anchor_name"]
    other_name = surface_names[1] if surface_names[0] == anchor_name else surface_names[0]
    anchor_frame = surface_frame_tensors(
        surfaces_by_name[anchor_name],
        particle_positions,
    )
    other_frame = surface_frame_tensors(
        surfaces_by_name[other_name],
        particle_positions,
    )
    current_origins = {
        anchor_name: anchor_frame["origin"].detach().clone(),
        other_name: other_frame["origin"].detach().clone(),
    }
    previous_origins = patch.get("previous_surface_origins", {})
    previous_anchor_origin = previous_origins.get(anchor_name)
    previous_other_origin = previous_origins.get(other_name)
    if previous_anchor_origin is None or previous_other_origin is None:
        return candidate_velocities, {
            "normal_velocity_filtered": False,
            "normal_velocity_reason": "missing_previous_surface_origin",
            "current_surface_origins": current_origins,
        }

    pair_normal = anchor_frame["rotation"][:, 2]
    anchor_surface_velocity = (
        current_origins[anchor_name] - previous_anchor_origin.to(pair_normal.device)
    ) / float(physics_dt)
    other_surface_velocity = (
        current_origins[other_name] - previous_other_origin.to(pair_normal.device)
    ) / float(physics_dt)
    anchor_normal_speed = float(torch.dot(anchor_surface_velocity, pair_normal).item())
    other_normal_speed = float(torch.dot(other_surface_velocity, pair_normal).item())
    normal_epsilon = 1e-6
    same_direction = (
        abs(anchor_normal_speed) <= normal_epsilon
        or (
            abs(other_normal_speed) > normal_epsilon
            and anchor_normal_speed * other_normal_speed > 0.0
        )
    )
    if same_direction:
        return candidate_velocities, {
            "normal_velocity_filtered": False,
            "normal_velocity_reason": "paired_surface_motion",
            "current_surface_origins": current_origins,
            "anchor_normal_speed_mps": anchor_normal_speed,
            "other_normal_speed_mps": other_normal_speed,
        }

    normal = pair_normal.reshape(1, 3)
    candidate_normal = torch.sum(candidate_velocities * normal, dim=1, keepdim=True)
    fallback_normal = torch.sum(fallback_velocities * normal, dim=1, keepdim=True)
    filtered_velocities = (
        candidate_velocities
        - candidate_normal * normal
        + fallback_normal * normal
    )
    return filtered_velocities, {
        "normal_velocity_filtered": True,
        "normal_velocity_reason": "one_sided_surface_motion",
        "current_surface_origins": current_origins,
        "anchor_normal_speed_mps": anchor_normal_speed,
        "other_normal_speed_mps": other_normal_speed,
        "normal_velocity_removed_mean_mps": float(torch.mean(candidate_normal).item()),
    }


def project_attached_patches(
    cloth_view,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    particle_positions: torch.Tensor,
    surfaces_by_name: dict[str, dict],
    patches: Sequence[dict],
    *,
    physics_dt: float,
    nonanchor_velocity_damping: float,
    attached_velocity_mode: str,
    sticking_drive_mode: str,
    sticking_pd_kp: float,
    sticking_pd_kd: float,
    sticking_pd_max_speed: float,
    fold_pair_projector: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
) -> list[dict]:
    """Write attached particle position/velocity targets to a cloth view.

    ``teleport`` writes exact anchor-frame targets. ``pd`` writes a capped
    position and velocity correction. In both modes target velocity follows the
    motion of the anchor-frame target when previous targets are available.
    """

    position_targets = positions.clone()
    velocity_targets = velocities.clone() * float(nonanchor_velocity_damping)
    metrics = []
    for patch in patches:
        anchor_surface = surfaces_by_name[patch["anchor_name"]]
        anchor_frame = surface_frame_tensors(anchor_surface, particle_positions)
        target_positions = (
            patch["local_positions"] @ anchor_frame["rotation"].T
        ) + anchor_frame["origin"]
        selected_indices = patch["indices"]
        previous_target_positions = patch.get("previous_target_positions")
        if attached_velocity_mode == "zero":
            anchor_velocities = torch.zeros_like(target_positions)
        elif (
            previous_target_positions is not None
            and previous_target_positions.shape == target_positions.shape
        ):
            anchor_velocities = (
                target_positions - previous_target_positions
            ) / float(physics_dt)
        else:
            anchor_velocities = torch.zeros_like(target_positions)
        selected_velocities = velocities[0, selected_indices]
        fallback_velocities = selected_velocities * float(nonanchor_velocity_damping)
        anchor_velocities, normal_filter_info = _surface_pair_normal_velocity_filter(
            patch,
            surfaces_by_name,
            particle_positions,
            anchor_velocities,
            fallback_velocities,
            physics_dt,
        )
        current_surface_origins = normal_filter_info.pop(
            "current_surface_origins",
            None,
        )

        if sticking_drive_mode == "teleport":
            position_targets[0, selected_indices] = target_positions
            velocity_targets[0, selected_indices] = anchor_velocities
            metrics.append(
                {
                    "mode": patch["mode"],
                    "anchor_name": patch["anchor_name"],
                    "particles": int(selected_indices.numel()),
                    **normal_filter_info,
                }
            )
        else:
            selected_positions = particle_positions[selected_indices]
            position_errors = target_positions - selected_positions
            velocity_errors = anchor_velocities - selected_velocities
            position_response = min(
                max(float(sticking_pd_kp) * float(physics_dt), 0.0),
                1.0,
            )
            velocity_response = min(
                max(float(sticking_pd_kd) * float(physics_dt), 0.0),
                1.0,
            )
            raw_corrections = position_response * position_errors
            correction_speed = torch.linalg.norm(
                raw_corrections,
                dim=1,
                keepdim=True,
            ) / float(physics_dt)
            speed_scale = torch.clamp(
                float(sticking_pd_max_speed)
                / torch.clamp(correction_speed, min=1e-6),
                max=1.0,
            )
            position_corrections = raw_corrections * speed_scale
            next_positions = selected_positions + position_corrections
            correction_velocities = position_corrections / float(physics_dt)
            damping_velocities = velocity_response * velocity_errors
            next_velocities = (
                anchor_velocities
                + correction_velocities
                + damping_velocities
            )
            next_velocities, next_normal_filter_info = (
                _surface_pair_normal_velocity_filter(
                    patch,
                    surfaces_by_name,
                    particle_positions,
                    next_velocities,
                    fallback_velocities,
                    physics_dt,
                )
            )
            next_current_surface_origins = next_normal_filter_info.pop(
                "current_surface_origins",
                None,
            )
            current_surface_origins = (
                next_current_surface_origins
                if next_current_surface_origins is not None
                else current_surface_origins
            )
            next_speed = torch.linalg.norm(next_velocities, dim=1, keepdim=True)
            next_speed_scale = torch.clamp(
                float(sticking_pd_max_speed)
                / torch.clamp(next_speed, min=1e-6),
                max=1.0,
            )
            position_targets[0, selected_indices] = next_positions
            velocity_targets[0, selected_indices] = next_velocities * next_speed_scale
            metrics.append(
                {
                    "mode": patch["mode"],
                    "anchor_name": patch["anchor_name"],
                    "particles": int(selected_indices.numel()),
                    "position_error_mean_m": float(
                        torch.mean(torch.linalg.norm(position_errors, dim=1)).item()
                    ),
                    "position_error_max_m": float(
                        torch.max(torch.linalg.norm(position_errors, dim=1)).item()
                    ),
                    "correction_speed_mean_mps": float(
                        torch.mean(correction_speed).item()
                    ),
                    "correction_speed_max_mps": float(
                        torch.max(correction_speed).item()
                    ),
                    "velocity_mean_mps": float(torch.mean(next_speed).item()),
                    "velocity_max_mps": float(torch.max(next_speed).item()),
                    "speed_limited_particles": int(
                        torch.count_nonzero(next_speed_scale[:, 0] < 0.999).item()
                    ),
                    **next_normal_filter_info,
                }
            )
        patch["previous_target_positions"] = target_positions.detach().clone()
        if current_surface_origins is not None:
            patch["previous_surface_origins"] = current_surface_origins

    if fold_pair_projector is not None:
        fold_pair_projector(position_targets, velocity_targets)
    cloth_view.set_world_positions(position_targets)
    cloth_view.set_velocities(velocity_targets)
    return metrics


def attached_patch_summaries(
    patches: Sequence[dict],
    particle_positions: torch.Tensor,
) -> list[dict]:
    """Return compact JSON-serializable summaries for active patches."""

    summaries = []
    for patch in patches:
        patch_positions = particle_positions[patch["indices"]]
        patch_span = (
            torch.max(patch_positions, dim=0).values
            - torch.min(patch_positions, dim=0).values
        )
        summaries.append(
            {
                "mode": patch["mode"],
                "anchor_name": patch["anchor_name"],
                "particles": int(patch["indices"].numel()),
                "seed_count": patch.get("seed_count"),
                "candidate_count": patch.get("candidate_count"),
                "span_m": [float(value.item()) for value in patch_span],
            }
        )
    return summaries


def surface_selected_contact_diagnostic(
    surface: dict,
    selected_indices: torch.Tensor,
) -> dict:
    """Return contact-mask details for selected particles on one surface."""

    contact = surface["contact"]
    local_positions = contact["local_positions"][selected_indices]
    half_extents = surface.get("half_extents")
    normal_mask = (
        contact["normal_distance"][selected_indices]
        <= float(surface["contact_margin"])
    )
    if half_extents is None:
        tangent_mask = (
            contact["tangent_distance"][selected_indices]
            <= float(surface["radius"])
        )
    else:
        tangent_mask = (
            (torch.abs(local_positions[:, 0]) <= float(half_extents[0]))
            & (torch.abs(local_positions[:, 1]) <= float(half_extents[1]))
        )
    return {
        "name": surface["name"],
        "mask_count": int(
            torch.count_nonzero(contact["mask"][selected_indices]).item()
        ),
        "normal_count": int(torch.count_nonzero(normal_mask).item()),
        "tangent_count": int(torch.count_nonzero(tangent_mask).item()),
        "normal_abs_max_m": float(
            torch.max(contact["normal_distance"][selected_indices]).item()
        ),
        "signed_normal_min_m": float(
            torch.min(contact["signed_normal_distance"][selected_indices]).item()
        ),
        "signed_normal_max_m": float(
            torch.max(contact["signed_normal_distance"][selected_indices]).item()
        ),
        "tangent_abs_max_m": [
            float(torch.max(torch.abs(local_positions[:, axis])).item())
            for axis in range(2)
        ],
        "local_min_m": [
            float(value.item())
            for value in torch.min(local_positions, dim=0).values
        ],
        "local_max_m": [
            float(value.item())
            for value in torch.max(local_positions, dim=0).values
        ],
        "half_extents_m": list(half_extents) if half_extents is not None else None,
        "contact_margin_m": float(surface["contact_margin"]),
    }


def active_patch_contact_diagnostics(
    active_surfaces: Sequence[dict],
    particle_positions: torch.Tensor,
    patches: Sequence[dict],
    *,
    press_gap: float,
    signed_normal_slop: float | None = None,
) -> list[dict]:
    """Return compressed-contact diagnostics for every active patch."""

    if not patches:
        return []
    contacts_by_name = {
        surface["name"]: surface
        for surface in surface_contacts_torch(active_surfaces, particle_positions)
    }
    diagnostics = []
    for patch in patches:
        selected_indices = patch["indices"]
        surface_names = patch["mode"].split("+")
        patch_contacts = [
            contacts_by_name[name]
            for name in surface_names
            if name in contacts_by_name
        ]
        if len(patch_contacts) != 2 or selected_indices.numel() == 0:
            diagnostics.append(
                {
                    "mode": patch["mode"],
                    "anchor_name": patch["anchor_name"],
                    "particles": int(selected_indices.numel()),
                    "pressed_count": 0,
                    "surfaces": [],
                }
            )
            continue
        first_contact = patch_contacts[0]["contact"]
        second_contact = patch_contacts[1]["contact"]
        pair_pressed_mask = (
            first_contact["mask"]
            & second_contact["mask"]
            & pressed_between_surfaces_torch(
                first_contact,
                second_contact,
                press_gap,
                signed_normal_slop,
            )
        )
        selected_pressed = pair_pressed_mask[selected_indices]
        surface_gap = torch.linalg.norm(
            first_contact["closest_points"][selected_indices]
            - second_contact["closest_points"][selected_indices],
            dim=1,
        )
        diagnostics.append(
            {
                "mode": patch["mode"],
                "anchor_name": patch["anchor_name"],
                "particles": int(selected_indices.numel()),
                "pressed_count": int(torch.count_nonzero(selected_pressed).item()),
                "surface_gap_min_m": float(torch.min(surface_gap).item()),
                "surface_gap_max_m": float(torch.max(surface_gap).item()),
                "press_gap_m": float(press_gap),
                "surfaces": [
                    surface_selected_contact_diagnostic(surface, selected_indices)
                    for surface in patch_contacts
                ],
            }
        )
    return diagnostics


def patch_contact_keys(patches: Sequence[dict]) -> list[tuple[str, str]]:
    """Return deterministic ``(mode, anchor_name)`` keys for patch lists."""

    return sorted((patch["mode"], patch["anchor_name"]) for patch in patches)


def copy_patch_velocity_history(
    candidate_patches: Sequence[dict],
    active_patches: Sequence[dict],
) -> None:
    """Copy previous target positions between unchanged patch identities."""

    for candidate_patch in candidate_patches:
        active_patch = next(
            (
                patch
                for patch in active_patches
                if patch["mode"] == candidate_patch["mode"]
                and patch["anchor_name"] == candidate_patch["anchor_name"]
                and torch.equal(patch["indices"], candidate_patch["indices"])
            ),
            None,
        )
        if active_patch is None:
            continue
        previous_targets = active_patch.get("previous_target_positions")
        if previous_targets is not None:
            candidate_patch["previous_target_positions"] = previous_targets
        previous_surface_origins = active_patch.get("previous_surface_origins")
        if previous_surface_origins is not None:
            candidate_patch["previous_surface_origins"] = previous_surface_origins


def patch_current_pressed_mask(
    patch: dict,
    contacts_by_name: dict[str, dict],
    *,
    press_gap: float,
    signed_normal_slop: float | None = None,
) -> torch.Tensor | None:
    """Return which selected particles still satisfy the creating contact pair."""

    surface_names = patch["mode"].split("+")
    if len(surface_names) != 2:
        return None
    first_surface = contacts_by_name.get(surface_names[0])
    second_surface = contacts_by_name.get(surface_names[1])
    if first_surface is None or second_surface is None:
        return None
    selected_indices = patch["indices"]
    pair_pressed_mask = (
        first_surface["contact"]["mask"]
        & second_surface["contact"]["mask"]
        & pressed_between_surfaces_torch(
            first_surface["contact"],
            second_surface["contact"],
            press_gap,
            signed_normal_slop,
        )
    )
    return pair_pressed_mask[selected_indices]


def filter_active_patches_by_current_contact(
    active_surfaces: Sequence[dict],
    particle_positions: torch.Tensor,
    active_patches: Sequence[dict],
    *,
    press_gap: float,
    min_component_particles: int = 1,
    signed_normal_slop: float | None = None,
) -> tuple[list[dict], list[tuple[dict, int, int]]]:
    """Prune or release active patches that lost their original compression."""

    contacts_by_name = {
        surface["name"]: surface
        for surface in surface_contacts_torch(active_surfaces, particle_positions)
    }
    valid_patches = []
    release_events = []
    for patch in active_patches:
        pressed_mask = patch_current_pressed_mask(
            patch,
            contacts_by_name,
            press_gap=press_gap,
            signed_normal_slop=signed_normal_slop,
        )
        if pressed_mask is None:
            release_events.append((patch, 0, int(patch["indices"].numel())))
            continue
        pressed_count = int(torch.count_nonzero(pressed_mask).item())
        original_count = int(patch["indices"].numel())
        if pressed_count < max(int(min_component_particles), 1):
            release_events.append((patch, pressed_count, original_count))
            continue
        if pressed_count == original_count:
            valid_patches.append(patch)
            continue

        pruned_patch = {
            **patch,
            "indices": patch["indices"][pressed_mask],
            "local_positions": patch["local_positions"][pressed_mask],
            "candidate_count": pressed_count,
        }
        previous_targets = patch.get("previous_target_positions")
        if previous_targets is not None and previous_targets.shape[0] == original_count:
            pruned_patch["previous_target_positions"] = previous_targets[pressed_mask]
        valid_patches.append(pruned_patch)
        release_events.append((patch, pressed_count, original_count))
    return valid_patches, release_events
