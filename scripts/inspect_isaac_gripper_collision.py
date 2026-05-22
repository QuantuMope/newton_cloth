#!/usr/bin/env python3
"""Compare imported gripper visual and collision bounds in an Isaac USD asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def _prim_world_bounds(xform_cache, bbox_cache, prim):
    if prim.IsA(UsdGeom.Cube):
        cube = UsdGeom.Cube(prim)
        size = cube.GetSizeAttr().Get() or 2.0
        half_size = 0.5 * float(size)
        corners = [
            (x, y, z)
            for x in (-half_size, half_size)
            for y in (-half_size, half_size)
            for z in (-half_size, half_size)
        ]
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        world_points = np.asarray(
            [local_to_world.Transform(corner) for corner in corners],
            dtype=np.float64,
        )
        return world_points.min(axis=0), world_points.max(axis=0)

    if prim.IsA(UsdGeom.Mesh):
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points:
            local_to_world = xform_cache.GetLocalToWorldTransform(prim)
            world_points = np.asarray(
                [local_to_world.Transform(point) for point in points],
                dtype=np.float64,
            )
            return world_points.min(axis=0), world_points.max(axis=0)

    if not prim.IsA(UsdGeom.Boundable):
        return None
    range_ = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if range_.IsEmpty():
        return None
    return np.asarray(range_.GetMin(), dtype=np.float64), np.asarray(
        range_.GetMax(),
        dtype=np.float64,
    )


def _merge_bounds(bounds):
    valid_bounds = [bound for bound in bounds if bound is not None]
    if not valid_bounds:
        return None
    mins = np.vstack([bound[0] for bound in valid_bounds]).min(axis=0)
    maxs = np.vstack([bound[1] for bound in valid_bounds]).max(axis=0)
    return mins, maxs


def _format_bounds(bounds):
    if bounds is None:
        return None
    mins, maxs = bounds
    return {
        "min": np.round(mins, 5).tolist(),
        "max": np.round(maxs, 5).tolist(),
        "size": np.round(maxs - mins, 5).tolist(),
    }


def _link_report(stage, xform_cache, bbox_cache, link_name):
    link_prim = next((prim for prim in stage.Traverse() if prim.GetName() == link_name), None)
    if link_prim is None:
        raise RuntimeError(f"Missing link prim: {link_name}")

    visual_meshes = []
    collision_prims = []
    for prim in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(prim)
            continue
        prim_path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh) and ("/J7/" in prim_path or "/J8/" in prim_path):
            visual_meshes.append(prim)

    visual_bounds = _merge_bounds(
        [_prim_world_bounds(xform_cache, bbox_cache, prim) for prim in visual_meshes]
    )
    collision_bounds = _merge_bounds(
        [_prim_world_bounds(xform_cache, bbox_cache, prim) for prim in collision_prims]
    )
    collision_parts = []
    for prim in collision_prims:
        approximation = (
            prim.GetAttribute("physics:approximation").Get()
            if prim.HasAttribute("physics:approximation")
            else None
        )
        collision_parts.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "approximation": approximation,
                "bounds": _format_bounds(_prim_world_bounds(xform_cache, bbox_cache, prim)),
            }
        )

    return {
        "link": link_name,
        "visual_mesh_count": len(visual_meshes),
        "collision_prim_count": len(collision_prims),
        "visual_bounds": _format_bounds(visual_bounds),
        "collision_bounds": _format_bounds(collision_bounds),
        "collision_parts": collision_parts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_path", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(args.usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {args.usd_path}")
    stage.Load()

    xform_cache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    report = [
        _link_report(stage, xform_cache, bbox_cache, link_name)
        for link_name in ("left_link7", "left_link8", "right_link7", "right_link8")
    ]
    report_text = json.dumps(report, indent=2)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(report_text + "\n")
    print(report_text)


if __name__ == "__main__":
    main()
