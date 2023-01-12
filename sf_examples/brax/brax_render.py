"""
Heavily inspired by original Brax code.
https://github.com/google/brax/blob/7eaa16b4bf446b117b538dbe9c9401f97cf4afa2/brax/io/image.py

This version is significantly faster to the point where software renderer can actually be used with
render_mode=human (although still slow and in low resolution).
"""


from typing import Dict, List, Tuple

import brax
import numpy as np
from brax import math
from brax.io.image import _BASIC, _GROUND, _TARGET, _eye, _up
from brax.physics.base import vec_to_arr
from PIL import Image
from pytinyrenderer import TinyRenderCamera as Camera
from pytinyrenderer import TinyRenderLight as Light
from pytinyrenderer import TinySceneRenderer as Renderer


def _flatten_vectors(vectors):
    """Returns the flattened array of the vectors."""
    return sum(map(lambda v: [v.x, v.y, v.z], vectors), [])


def _scene(sys: brax.System) -> Tuple[Renderer, List[int], Dict]:
    """Converts a brax System and qp to a pytinyrenderer scene and instances."""
    scene = Renderer()
    extra_info = dict()
    instances = []
    offsets, rotations, body_indices = [], [], []
    mesh_geoms = {g.name: g for g in sys.config.mesh_geometries}
    for i, body in enumerate(sys.config.bodies):
        tex = _TARGET if body.name.lower() == "target" else _BASIC
        for col in body.colliders:
            col_type = col.WhichOneof("type")
            if col_type == "capsule":
                half_height = col.capsule.length / 2 - col.capsule.radius
                model = scene.create_capsule(col.capsule.radius, half_height, 2, tex.pixels, tex.width, tex.height)
            elif col_type == "box":
                hs = col.box.halfsize
                model = scene.create_cube([hs.x, hs.y, hs.z], _BASIC.pixels, tex.width, tex.height, 16.0)
            elif col_type == "sphere":
                model = scene.create_capsule(col.sphere.radius, 0, 2, tex.pixels, tex.width, _BASIC.height)
            elif col_type == "plane":
                tex = _GROUND
                model = scene.create_cube([1000.0, 1000.0, 0.0001], tex.pixels, tex.width, tex.height, 8192)
            elif col_type == "mesh":
                mesh = col.mesh
                g = mesh_geoms[mesh.name]
                scale = mesh.scale if mesh.scale else 1
                model = scene.create_mesh(
                    np.array(_flatten_vectors(g.vertices)) * scale,
                    _flatten_vectors(g.vertex_normals),
                    [0] * len(g.vertices) * 2,
                    g.faces,
                    tex.pixels,
                    tex.width,
                    tex.height,
                    1.0,
                )
            else:
                raise RuntimeError(f"unrecognized collider: {col_type}")

            instance = scene.create_object_instance(model)
            off = np.array([col.position.x, col.position.y, col.position.z])
            col_rot = math.euler_to_quat(vec_to_arr(col.rotation))
            instances.append(instance)
            offsets.append(off)
            rotations.append(np.array(col_rot))
            body_indices.append(i)

    extra_info["offsets"] = offsets
    extra_info["rotations"] = rotations
    extra_info["body_indices"] = body_indices  # to refer to the body idx in qp

    return scene, instances, extra_info


def quat_mul(u, v):
    """Multiplies two quaternions.

    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """
    return np.array(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ]
    )


def _update_scene(scene: Renderer, instances: List[int], extra_info: Dict, qp: brax.QP):
    """Updates the scene with the new qp."""

    offsets = extra_info["offsets"]
    rotations = extra_info["rotations"]
    body_indices = extra_info["body_indices"]

    np_pos = np.array(qp.pos)
    np_rot = np.array(qp.rot)

    for i, instance in enumerate(instances):
        body = body_indices[i]
        off = offsets[i]
        col_rot = rotations[i]
        pos = np_pos[body] + math.rotate(off, qp.rot[body])
        rot = quat_mul(np_rot[body], col_rot)
        scene.set_object_position(instances[i], list(pos))
        scene.set_object_orientation(instances[i], [rot[1], rot[2], rot[3], rot[0]])


def create_scene(qp, sys):
    if (len(qp.pos.shape), len(qp.rot.shape)) != (2, 2):
        raise RuntimeError("unexpected shape in qp")
    scene, instances, extra_info = _scene(sys)
    return instances, scene, extra_info


def create_camera(height, width, qp, ssaa, sys, target):
    eye, up = _eye(sys, qp), _up(sys)
    hfov = 58.0
    vfov = hfov * height / width
    camera = Camera(
        viewWidth=width * ssaa, viewHeight=height * ssaa, position=eye, target=target, up=up, hfov=hfov, vfov=vfov
    )
    return camera


def create_light(target):
    direction = [0.57735, -0.57735, 0.57735]
    light = Light(direction=direction, ambient=0.8, diffuse=0.8, specular=0.6, shadowmap_center=target)
    return light


class BraxRenderer:
    def __init__(self, env, render_mode: str, brax_video_res_px: int = 200):
        self.env = env
        self.screen = None
        self.render_mode = render_mode

        self.instances, self.extra_info, self.scene = None, None, None

        self.ssaa = 2  # supersampling factor
        self.width = self.height = brax_video_res_px  # anything higher is super slow because CPU renderer :|

    # noinspection PyProtectedMember
    def render(self):
        sys = self.env._env.sys

        from brax import jumpy as jp

        qp = jp.take(self.env._state.qp, 0)

        if self.scene is None:
            self.instances, self.scene, self.extra_info = create_scene(qp, sys)

        _update_scene(self.scene, self.instances, self.extra_info, qp)

        target = [qp.pos[0, 0], qp.pos[0, 1], 0]
        light = create_light(target)
        camera = create_camera(self.height, self.width, qp, self.ssaa, sys, target)

        img = self.scene.get_camera_image(self.instances, light, camera).rgb

        arr = np.reshape(np.array(img, dtype=np.uint8), (camera.view_height, camera.view_width, -1))
        if self.ssaa > 1:
            arr = np.asarray(Image.fromarray(arr).resize((self.width, self.height)))

        if self.render_mode == "human":
            import pygame

            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.width, self.height))

            pygame.surfarray.blit_array(self.screen, arr.swapaxes(0, 1))
            pygame.display.update()

        return arr
