import math
import numpy as np
from lark import Lark, Transformer, v_args, LarkError
import copy


def lin_p(p_a, p_b, weight):
    p_a, p_b = np.asarray(p_a), np.asarray(p_b)
    return p_a + (p_b - p_a) * float(weight)

def orthogonal_projection_onto_plane(point, plane_p1, plane_p2, plane_p3):
    point = np.asarray(point)
    plane_p1, plane_p2, plane_p3 = np.asarray(plane_p1), np.asarray(plane_p2), np.asarray(plane_p3)
    v1 = plane_p2 - plane_p1
    v2 = plane_p3 - plane_p1
    normal = np.cross(v1, v2)
    norm_mag = np.linalg.norm(normal)
    if norm_mag == 0: return point
    normal = normal / norm_mag
    return point - np.dot(point - plane_p1, normal) * normal

def line_plane_intersection(line_p1, line_p2, plane_p1, plane_p2, plane_p3):
    line_p1, line_p2 = np.asarray(line_p1), np.asarray(line_p2)
    plane_p1, plane_p2, plane_p3 = np.asarray(plane_p1), np.asarray(plane_p2), np.asarray(plane_p3)
    line_dir = line_p2 - line_p1
    v1 = plane_p2 - plane_p1
    v2 = plane_p3 - plane_p1
    plane_normal = np.cross(v1, v2)
    norm_mag = np.linalg.norm(plane_normal)
    if norm_mag == 0: return None
    plane_normal = plane_normal / norm_mag
    dot_product = np.dot(line_dir, plane_normal)
    if abs(dot_product) < 1e-9: return None
    t = np.dot(plane_normal, plane_p1 - line_p1) / dot_product
    return line_p1 + t * line_dir
