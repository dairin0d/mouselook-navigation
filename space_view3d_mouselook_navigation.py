#  ***** BEGIN GPL LICENSE BLOCK *****
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  ***** END GPL LICENSE BLOCK *****

# <pep8 compliant>

bl_info = {
    "name": "Mouselook Navigation",
    "description": "Integrated 3D view navigation",
    "author": "dairin0d",
    "version": (0, 9, 3),
    "blender": (2, 7, 0),
    "location": "View3D > MMB/Scrollwheel",
    "warning": "",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/"
        "Scripts/3D_interaction/MouselookNavigation",
    "tracker_url": "https://developer.blender.org/??????????????????????????",
    "category": "3D View"}
#============================================================================#

import bpy
import bgl
import bmesh

from mathutils import Color, Vector, Matrix, Quaternion, Euler

from bpy_extras.view3d_utils import (region_2d_to_location_3d,
                                     location_3d_to_region_2d,
                                     region_2d_to_vector_3d,
                                     region_2d_to_origin_3d,
                                     )

import math
import time

"""
Note: due to the use of timer, operator consumes more resources than Blender's default
TODO:
* correct & stable collision detection?
* Blender's trackball
* ortho-grid/quadview-clip/projection-name display is not updated
"""

class SmartView3D:
    def __init__(self, context=None):
        if context is None:
            context = bpy.context
        self.userprefs = bpy.context.user_preferences
        self.region = context.region # expected type: Region
        self.space_data = context.space_data # expected type: SpaceView3D
        self.region_data = context.region_data # expected type: RegionView3D
        self.use_camera_axes = False
        self.use_viewpoint =  False
        
        r = self.region
        x0, y0, x1, y1 = r.x, r.y, r.x+r.width, r.y+r.height
        self.region_rect = [Vector((x0, y0)), Vector((x1-x0, y1-y0))]
        for r in context.area.regions:
            if r.type == 'TOOLS':
                x0 = r.x + r.width
            elif r.type == 'UI':
                x1 = r.x
        self.clickable_region_rect = [Vector((x0, y0)), Vector((x1-x0, y1-y0))]
    
    def __get(self):
        return self.space_data.lock_cursor
    def __set(self, value):
        self.space_data.lock_cursor = value
    lock_cursor = property(__get, __set)
    
    def __get(self):
        return self.space_data.lock_object
    def __set(self, value):
        self.space_data.lock_object = value
    lock_object = property(__get, __set)
    
    def __get(self):
        return self.space_data.lock_bone
    def __set(self, value):
        self.space_data.lock_bone = value
    lock_bone_name = property(__get, __set)
    
    def __get(self):
        v3d = self.space_data
        obj = v3d.lock_object
        if obj and (obj.type == 'ARMATURE') and v3d.lock_bone:
            try:
                if obj.mode == 'EDIT':
                    return obj.data.edit_bones[v3d.lock_bone]
                else:
                    return obj.data.bones[v3d.lock_bone]
            except:
                pass
        return None
    def __set(self, value):
        self.space_data.lock_bone = (value.name if value else "")
    lock_bone = property(__get, __set)
    
    def __get(self):
        return self.space_data.lock_camera
    def __set(self, value):
        self.space_data.lock_camera = value
    lock_camera = property(__get, __set)
    
    def __get(self):
        return self.userprefs.view.use_camera_lock_parent
    def __set(self, value):
        self.userprefs.view.use_camera_lock_parent = value
    lock_camera_parent = property(__get, __set)
    
    def __get(self):
        return self.space_data.region_3d
    region_3d = property(__get)
    
    def __get(self):
        return self.region_data == self.space_data.region_3d
    is_region_3d = property(__get)
    
    # 0: bottom left (Front Ortho)
    # 1: top left (Top Ortho)
    # 2: bottom right (Right Ortho)
    # 3: top right (User Persp)
    def __get(self):
        return self.space_data.region_quadviews
    quadviews = property(__get)
    
    def __get(self):
        return len(self.space_data.region_quadviews) != 0
    quadview_enabled = property(__get)
    
    def __get(self):
        return self.region_data.lock_rotation
    def __set(self, value):
        self.region_data.lock_rotation = value
    quadview_lock = property(__get, __set)
    
    def __get(self):
        return self.region_data.show_sync_view
    def __set(self, value):
        self.region_data.show_sync_view = value
    quadview_sync = property(__get, __set)
    
    def __get(self):
        return self.region_data.view_camera_offset
    def __set(self, value):
        self.region_data.view_camera_offset = value
    camera_offset = property(__get, __set)
    
    def __get(self):
        return self.region_data.view_camera_zoom
    def __set(self, value):
        self.region_data.view_camera_zoom = value
    camera_zoom = property(__get, __set)
    
    def __get(self):
        return self.space_data.camera
    def __set(self, value):
        self.space_data.camera = value
    camera = property(__get, __set)
    
    def __get(self):
        if self.is_camera and (self.camera.type == 'CAMERA'):
            return self.camera.data.lens
        else:
            return self.space_data.lens
    def __set(self, value):
        if self.is_camera and (self.camera.type == 'CAMERA'):
            if self.lock_camera:
                self.camera.data.lens = value
        else:
            self.space_data.lens = value
    lens = property(__get, __set)
    
    def __get(self):
        if self.is_camera and (self.camera.type == 'CAMERA'):
            return self.camera.data.clip_start
        else:
            return self.space_data.clip_start
    def __set(self, value):
        if self.is_camera and (self.camera.type == 'CAMERA'):
            if self.lock_camera:
                self.camera.data.clip_start = value
        else:
            self.space_data.clip_start = value
    clip_start = property(__get, __set)
    
    def __get(self):
        if self.is_camera and (self.camera.type == 'CAMERA'):
            return self.camera.data.clip_end
        else:
            return self.space_data.clip_end
    def __set(self, value):
        if self.is_camera and (self.camera.type == 'CAMERA'):
            if self.lock_camera:
                self.camera.data.clip_end = value
        else:
            self.space_data.clip_end = value
    clip_end = property(__get, __set)
    
    def __get(self):
        return ((self.region_data.view_perspective == 'CAMERA') and bool(self.space_data.camera))
    def __set(self, value):
        if value and self.space_data.camera:
            self.region_data.view_perspective = 'CAMERA'
        elif self.region_data.is_perspective:
            self.region_data.view_perspective = 'PERSP'
        else:
            self.region_data.view_perspective = 'ORTHO'
    is_camera = property(__get, __set)
    
    def __get(self):
        if self.is_camera:
            if (self.camera.type == 'CAMERA'):
                return self.camera.data.type != 'ORTHO'
            else:
                return True
        else:
            return self.region_data.is_perspective
    def __set(self, value):
        if self.is_camera:
            if self.lock_camera:
                if (self.camera.type == 'CAMERA'):
                    cam_data = self.camera.data
                    old_value = (cam_data.type != 'ORTHO')
                    if value != old_value:
                        if cam_data.type == 'ORTHO':
                            cam_data.type = 'PERSP'
                        else:
                            cam_data.type = 'ORTHO'
        elif self.is_region_3d or not self.quadview_lock:
            self.region_data.is_perspective = value
            if value:
                self.region_data.view_perspective = 'PERSP'
            else:
                self.region_data.view_perspective = 'ORTHO'
    is_perspective = property(__get, __set)
    
    def __get(self):
        return self.region_data.view_distance
    def __set(self, value):
        if self.quadview_sync and (not self.is_region_3d):
            quadviews = self.quadviews
            quadviews[0].view_distance = value
            quadviews[0].update()
            quadviews[1].view_distance = value
            quadviews[1].update()
            quadviews[2].view_distance = value
            quadviews[2].update()
        else:
            self.region_data.view_distance = value
            self.region_data.update()
    raw_distance = property(__get, __set)
    
    def __get(self):
        return self.region_data.view_location.copy()
    def __set(self, value):
        if self.quadview_sync and (not self.is_region_3d):
            quadviews = self.quadviews
            quadviews[0].view_location = value.copy()
            quadviews[0].update()
            quadviews[1].view_location = value.copy()
            quadviews[1].update()
            quadviews[2].view_location = value.copy()
            quadviews[2].update()
        else:
            self.region_data.view_location = value.copy()
            self.region_data.update()
    raw_location = property(__get, __set)
    
    def __get(self):
        value = self.region_data.view_rotation.copy()
        if not self.use_camera_axes:
            value = value * Quaternion((1, 0, 0), -math.pi*0.5)
        return value
    def __set(self, value):
        if not self.use_camera_axes:
            value = value * Quaternion((1, 0, 0), math.pi*0.5)
        if self.is_region_3d or (not self.quadview_lock):
            self.region_data.view_rotation = value.copy()
            self.region_data.update()
    raw_rotation = property(__get, __set)
    
    def __get(self):
        return (self.viewpoint if self.use_viewpoint else self.focus)
    def __set(self, value):
        if self.use_viewpoint:
            self.viewpoint = value
        else:
            self.focus = value
    pivot = property(__get, __set)
    
    def __get(self):
        if self.is_camera and (self.camera.type == 'CAMERA') and (self.camera.data.type == 'ORTHO'):
            return self.camera.data.ortho_scale
        else:
            return self.raw_distance
    def __set(self, value):
        pivot = self.pivot
        value = max(value, 1e-12) # just to be sure that it's never zero or negative
        if self.is_camera and (self.camera.type == 'CAMERA') and (self.camera.data.type == 'ORTHO'):
            if self.lock_camera:
                self.camera.data.ortho_scale = value
        else:
            self.raw_distance = value
        self.pivot = pivot
    distance = property(__get, __set)
    
    def __set_cam_matrix(self, m):
        cam = self.space_data.camera
        if self.lock_camera_parent:
            max_parent = cam
            while True:
                if (max_parent.parent is None) or (max_parent.parent_type == 'VERTEX'):
                    break # 'VERTEX' isn't a rigidbody-type transform
                max_parent = max_parent.parent
            cm_inv = cam.matrix_world.inverted_safe()
            pm = cm_inv * max_parent.matrix_world
            max_parent.matrix_world = m * pm
        else:
            cam.matrix_world = m
    
    def __get(self):
        v3d = self.space_data
        rv3d = self.region_data
        if self.is_camera:
            m = v3d.camera.matrix_world
            return m.translation + self.forward * rv3d.view_distance
        elif v3d.lock_object:
            obj = self.lock_object
            bone = self.lock_bone
            m = obj.matrix_world
            if bone:
                m = m * (bone.matrix if obj.mode == 'EDIT' else bone.matrix_local)
            return m.translation.copy()
        elif v3d.lock_cursor:
            return v3d.cursor_location.copy()
        else:
            return self.raw_location
    def __set(self, value):
        v3d = self.space_data
        rv3d = self.region_data
        if self.is_camera:
            if self.lock_camera:
                m = v3d.camera.matrix_world.copy()
                m.translation = value - self.forward * rv3d.view_distance
                self.__set_cam_matrix(m)
        elif v3d.lock_object:
            pass
        elif v3d.lock_cursor:
            pass
        else:
            self.raw_location = value
    focus = property(__get, __set)
    # TODO: quadview "box" movement mode (3 views move synchronously)
    # in Camera View, shift+drag changes view_camera_offset
    # and mouse wheel changes view_camera_zoom (unless lock_camera is True)
    
    # Camera (and viewport): -Z is forward, Y is up, X is right
    def __get(self):
        v3d = self.space_data
        rv3d = self.region_data
        if self.is_camera:
            value = v3d.camera.matrix_world.to_quaternion()
            if not self.use_camera_axes:
                value = value * Quaternion((1, 0, 0), -math.pi*0.5)
        else:
            value = self.raw_rotation
        return value
    def __set(self, value):
        v3d = self.space_data
        rv3d = self.region_data
        pivot = self.pivot
        if self.is_camera:
            if not self.use_camera_axes:
                value = value * Quaternion((1, 0, 0), math.pi*0.5)
            if self.lock_camera:
                LRS = v3d.camera.matrix_world.decompose()
                m = MatrixLRS(LRS[0], value, LRS[2])
                forward = -m.col[2].to_3d().normalized() # in camera axes, forward is -Z
                m.translation = self.focus - forward * rv3d.view_distance
                self.__set_cam_matrix(m)
        else:
            self.raw_rotation = value
        self.pivot = pivot
    rotation = property(__get, __set)
    
    def __get(self): # in object axes
        world_x = Vector((1, 0, 0))
        world_z = Vector((0, 0, 1))
        
        x = self.right # right
        y = self.forward # forward
        z = self.up # up
        
        if abs(y.z) > (1 - 1e-12): # sufficiently close to vertical
            roll = 0.0
            xdir = x.copy()
        else:
            xdir = y.cross(world_z)
            rollPos = angle_signed(-y, x, xdir, 0.0)
            rollNeg = angle_signed(-y, x, -xdir, 0.0)
            if abs(rollNeg) < abs(rollPos):
                roll = rollNeg
                xdir = -xdir
            else:
                roll = rollPos
        xdir = Vector((xdir.x, xdir.y, 0)).normalized()
        
        yaw = angle_signed(-world_z, xdir, world_x, 0.0)
        
        zdir = xdir.cross(y).normalized()
        pitch = angle_signed(-xdir, zdir, world_z, 0.0)
        
        return Euler((pitch, roll, yaw), 'YXZ')
    def __set(self, value): # in object axes
        rot_x = Quaternion((1, 0, 0), value.x)
        rot_y = Quaternion((0, 1, 0), value.y)
        rot_z = Quaternion((0, 0, 1), value.z)
        rot = rot_z * rot_x * rot_y
        if self.use_camera_axes:
            rot = rot * Quaternion((1, 0, 0), math.pi*0.5)
        self.rotation = rot
    turntable_euler = property(__get, __set)
    
    def __get(self):
        v3d = self.space_data
        rv3d = self.region_data
        if self.is_camera:
            return v3d.camera.matrix_world.translation.copy()
        else:
            return self.focus - self.forward * rv3d.view_distance
    def __set(self, value):
        self.focus = self.focus + (value - self.viewpoint)
    viewpoint = property(__get, __set)
    
    def __get(self, viewpoint=False):
        m = self.rotation.to_matrix()
        m.resize_4x4()
        m.translation = (self.viewpoint if viewpoint else self.focus)
        return m
    def __set(self, m, viewpoint=False):
        if viewpoint:
            self.viewpoint = m.translation.copy()
        else:
            self.focus = m.translation.copy()
        self.rotation = m.to_quaternion()
    matrix = property(__get, __set)
    
    def __get_axis(self, x, y, z):
        rot = self.rotation
        if self.use_camera_axes:
            rot = rot * Quaternion((1, 0, 0), -math.pi*0.5)
        return (rot * Vector((x, y, z))).normalized()
    forward = property(lambda self: self.__get_axis(0, 1, 0))
    back = property(lambda self: self.__get_axis(0, -1, 0))
    up = property(lambda self: self.__get_axis(0, 0, 1))
    down = property(lambda self: self.__get_axis(0, 0, -1))
    left = property(lambda self: self.__get_axis(-1, 0, 0))
    right = property(lambda self: self.__get_axis(1, 0, 0))
    
    def project(self, pos, align=False): # 0,0 means region's bottom left corner
        region = self.region
        rv3d = self.region_data
        xy = location_3d_to_region_2d(region, rv3d, pos.copy())
        if align:
            xy = snap_pixel_vector(xy)
        return xy
    
    def unproject(self, xy, pos=None, align=False): # 0,0 means region's bottom left corner
        if align:
            xy = snap_pixel_vector(xy)
        if pos is None:
            pos = self.focus
        elif isinstance(pos, (int, float)):
            pos = self.zbuf_range[2] + self.forward * pos
        region = self.region
        rv3d = self.region_data
        return region_2d_to_location_3d(region, rv3d, xy.copy(), pos.copy())
    
    def ray(self, xy): # 0,0 means region's bottom left corner
        region = self.region
        rv3d = self.region_data
        
        view_dir = self.forward
        near, far, origin = self.zbuf_range
        near = origin + view_dir * near
        far = origin + view_dir * far
        
        a = region_2d_to_location_3d(region, rv3d, xy.copy(), near)
        b = region_2d_to_location_3d(region, rv3d, xy.copy(), far)
        return a, b
    
    def read_zbuffer(self, xy, wh=(1, 1)): # xy is in window coordinates!
        if isinstance(wh, (int, float)):
            wh = (wh, wh)
        elif len(wh) < 2:
            wh = (wh[0], wh[0])
        x, y, w, h = int(xy[0]), int(xy[1]), int(wh[0]), int(wh[1])
        zbuf = bgl.Buffer(bgl.GL_FLOAT, [w*h])
        bgl.glReadPixels(x, y, w, h, bgl.GL_DEPTH_COMPONENT, bgl.GL_FLOAT, zbuf)
        return zbuf
    
    def zbuf_to_depth(self, zbuf):
        near, far, origin = self.zbuf_range
        depth_linear = zbuf*far + (1.0 - zbuf)*near
        if self.is_perspective:
            return (far * near) / (zbuf*near + (1.0 - zbuf)*far)
        else:
            return zbuf*far + (1.0 - zbuf)*near
    
    def depth(self, xy, region_coords=True):
        if region_coords: # convert to window coords
            xy = xy + Vector((self.region.x, self.region.y))
        return self.zbuf_to_depth(self.read_zbuffer(xy)[0])
    
    def __get(self):
        rv3d = self.region_data
        if rv3d.is_perspective or (rv3d.view_perspective == 'CAMERA'):
            return (self.clip_start, self.clip_end, self.viewpoint)
        return (-self.clip_end*0.5, self.clip_end*0.5, self.focus)
    zbuf_range = property(__get)
    
    del __get
    del __set

def MatrixLRS(L, R, S):
    m = R.to_matrix().to_4x4()
    m.col[0] *= S.x
    m.col[1] *= S.y
    m.col[2] *= S.z
    m.translation = L
    return m

def angle_signed(n, v0, v1, fallback=None):
    angle = v0.angle(v1, fallback)
    if (angle != fallback) and (angle > 0):
        angle *= math.copysign(1.0, v0.cross(v1).dot(n))
    return angle

def snap_pixel_vector(v): # to have 2d-stable 3d drawings
    return Vector((round(v.x)+0.5, round(v.y)+0.5))

# Virtual Trackball by Gavin Bell
# Ok, simulate a track-ball.  Project the points onto the virtual
# trackball, then figure out the axis of rotation, which is the cross
# product of P1 P2 and O P1 (O is the center of the ball, 0,0,0)
# Note:  This is a deformed trackball-- is a trackball in the center,
# but is deformed into a hyperbolic sheet of rotation away from the
# center.  This particular function was chosen after trying out
# several variations.
#
# It is assumed that the arguments to this routine are in the range
# (-1.0 ... 1.0)
def trackball(p1x, p1y, p2x, p2y, TRACKBALLSIZE=1.0):
    #"""
    #if (p1x == p2x) and (p1y == p2y):
    #    return Quaternion() # Zero rotation
    
    # First, figure out z-coordinates for projection of P1 and P2 to deformed sphere
    p1 = Vector((p1x, p1y, tb_project_to_sphere(TRACKBALLSIZE, p1x, p1y)))
    p2 = Vector((p2x, p2y, tb_project_to_sphere(TRACKBALLSIZE, p2x, p2y)))
    
    # Now, we want the cross product of P1 and P2
    a = p2.cross(p1) # vcross(p2,p1,a); # Axis of rotation
    
    # Figure out how much to rotate around that axis.
    d = p1 - p2
    t = d.magnitude / (2.0*TRACKBALLSIZE)
    
    # Avoid problems with out-of-control values...
    t = min(max(t, -1.0), 1.0)
    #phi = 2.0 * math.asin(t) # how much to rotate about axis
    phi = 2.0 * t # how much to rotate about axis
    
    return Quaternion(a, phi)

# Project an x,y pair onto a sphere of radius r OR a hyperbolic sheet
# if we are away from the center of the sphere.
def tb_project_to_sphere(r, x, y):
    d = math.sqrt(x*x + y*y)
    if (d < r * math.sqrt(0.5)): # Inside sphere
        z = math.sqrt(r*r - d*d)
    else: # On hyperbola
        t = r / math.sqrt(2)
        z = t*t / d
    return z

# Loosely based on "Improved Collision detection and Response" by Kasper Fauerby
# http://www.peroxide.dk/papers/collision/collision.pdf
def apply_collisions(scene, p_head, v, view_height, is_crouching, parallel, max_slides):
    head_h = 0.75 # relative to total height
    char_h = view_height / head_h
    char_r = char_h * 0.5
    if is_crouching:
        char_h *= 0.5
    
    p_base = p_head - Vector((0, 0, char_h*head_h))
    p_top = p_base + Vector((0, 0, char_h))
    p_center = (p_base + p_top) * 0.5
    
    e2w = Matrix.Identity(3)
    e2w.col[0] = (char_r, 0, 0)
    e2w.col[1] = (0, char_r, 0)
    e2w.col[2] = (0, 0, char_h*0.5)
    e2w.resize_4x4()
    
    subdivs=8
    max_cnt=16
    
    collided = False
    new_center = p_center
    
    v = v.copy()
    
    while max_slides >= 0:
        e2w.translation = new_center
        w2e = e2w.inverted_safe()
        
        #ray_origin = (None if parallel else p_center)
        ray_origin = (head_h-0.5 if parallel else p_head)
        
        p0 = new_center
        d, p, c, n = ellipsoid_sweep(scene, e2w, w2e, v, ray_origin, subdivs, max_cnt)
        new_center = p
        
        if d is None: # didn't colliside with anything
            break
        
        """
        if (d < 1.0) and (not parallel):
            ce = w2e * c
            ne = -ce.normalized()
            pe = ce + ne
            p = e2w * pe
            new_center = p
            n = (e2w * Vector((ne.x, ne.y, ne.z, 0))).to_3d().normalized()
        """
        
        v = v - (p - p0) # subtract moved distance from velocity
        v = v + v.project(n) # project velocity to the sliding plane
        
        collided = True
        max_slides -= 1
    
    return new_center - p_center, collided

def ellipsoid_sweep(scene, e2w, w2e, v, ray_origin, subdivs=8, max_cnt=16):
    v = (w2e * Vector((v.x, v.y, v.z, 0))).to_3d()
    n = v.normalized()
    min_d = None
    d_n = None
    d_p = None
    d_c = None
    contacts_count = 0
    
    use_avg = False
    
    is_parallel = not isinstance(ray_origin, Vector)
    if is_parallel:
        max_cnt = 0 # sides aren't needed for parallel rays
    
    for p1 in ellipsoid_sweep_rays(e2w, v, subdivs, max_cnt):
        if is_parallel:
            p0 = w2e * p1
            p0 = p0 - p0.project(n) - (n * ray_origin)
            p0 = e2w * p0
        else:
            p0 = ray_origin.copy()
        
        raycast_result = scene.ray_cast(p0, p1)
        if raycast_result[0]:
            rn = raycast_result[4]
            rn = (w2e * Vector((rn.x, rn.y, rn.z, 0))).to_3d()
            p = w2e * raycast_result[3]
            L = p.dot(n)
            r = p - L*n
            h = math.sqrt(max(1.0 - r.length_squared, 0.0))
            if h > 0.1: # ignore almost tangential collisions
                d = L - h # distance of impact
                if not use_avg:
                    if (min_d is None) or (d < min_d):
                        min_d = d
                        d_p = n * min_d # stopping point
                        d_c = p # contact point
                        #d_n = (d_p - d_c) # contact normal
                        d_n = rn
                else:
                    if (min_d is None):
                        d_p = Vector()
                        d_c = Vector()
                        d_n = Vector()
                    min_d = d
                    d_p += n * min_d # stopping point
                    d_c += p # contact point
                    #d_n += (d_p - d_c) # contact normal
                    d_n = rn
                contacts_count += 1
    
    if min_d is None:
        return (None, e2w * v, None, None)
    
    if use_avg:
        d_p = d_p * (1.0 / contacts_count)
        d_c = d_c * (1.0 / contacts_count)
        d_n = d_n * (1.0 / contacts_count)
    
    d_p = e2w * d_p
    d_c = e2w * d_c
    d_n = (e2w * Vector((d_n.x, d_n.y, d_n.z, 0))).to_3d().normalized()
    return (min_d, d_p, d_c, d_n)

def ellipsoid_sweep_rays(e2w, v, subdivs=8, max_cnt=16):
    n = v.normalized()
    t1 = n.orthogonal()
    t2 = n.cross(t1)
    
    full_circle = 2*math.pi
    quarter_circle = 0.5*math.pi
    arc_step = full_circle / subdivs
    v_len = v.magnitude
    v_cnt = min(int(math.ceil(v_len / arc_step)), max_cnt)
    a_cnt = max(int(math.ceil(quarter_circle / arc_step)), 1)
    
    for i_v in range(v_cnt):
        c_n = (i_v / v_cnt) * v_len
        r_cnt = subdivs
        for i_r in range(r_cnt):
            angle = (i_r / r_cnt) * full_circle
            c_t1 = math.cos(angle)
            c_t2 = math.sin(angle)
            ray = c_n*n + c_t1*t1 + c_t2*t2
            yield (e2w * ray)
    
    for i_a in range(a_cnt+1):
        c_a = math.sin((i_a / a_cnt) * quarter_circle)
        r_t = math.sqrt(1 - c_a*c_a)
        c_n = v_len + c_a
        r_cnt = max(int(math.ceil((full_circle * r_t) / arc_step)), 1)
        for i_r in range(r_cnt):
            angle = (i_r / r_cnt) * full_circle
            c_t1 = math.cos(angle) * r_t
            c_t2 = math.sin(angle) * r_t
            ray = c_n*n + c_t1*t1 + c_t2*t2
            yield (e2w * ray)

def calc_selection_center(context): # View3D area is assumed
    context_mode = context.mode
    active_object = context.active_object
    m = (active_object.matrix_world if active_object else None)
    positions = []
    
    if (context_mode == 'OBJECT') or (not active_object):
        m = None
        positions.extend(obj.matrix_world.translation for obj in context.selected_objects)
    elif context_mode == 'EDIT_MESH':
        bm = bmesh.from_edit_mesh(active_object.data)
        if bm.select_history and (len(bm.select_history) < len(bm.verts)/4):
            verts = set()
            for elem in bm.select_history:
                if isinstance(elem, bmesh.types.BMVert):
                    verts.add(elem)
                else:
                    verts.update(elem.verts)
            positions.extend(v.co for v in verts)
        else:
            positions.extend(v.co for v in bm.verts if v.select)
    elif context_mode in {'EDIT_CURVE', 'EDIT_SURFACE'}:
        for spline in active_object.data.splines:
            for point in spline.bezier_points:
                if point.select_control_point:
                    positions.append(point.co)
                else:
                    if point.select_left_handle:
                        positions.append(point.handle_left)
                    if point.select_right_handle:
                        positions.append(point.handle_right)
            positions.extend(point.co for point in spline.points if point.select)
    elif context_mode == 'EDIT_METABALL':
        active_elem = active_object.data.elements.active
        if active_elem:
            positions.append(active_elem.co)
        # Currently there is no API for element.select
        #positions.extend(elem.co for elem in active_object.data.elements if elem.select)
    elif context_mode == 'EDIT_LATTICE':
        positions.extend(point.co for point in active_object.data.points if point.select)
    elif context_mode == 'EDIT_ARMATURE':
        for bone in active_object.data.edit_bones:
            if bone.select_head:
                positions.append(bone.head)
            if bone.select_tail:
                positions.append(bone.tail)
    elif context_mode == 'POSE':
        # consider only topmost parents
        bones = set(bone for bone in active_object.data.bones if bone.select)
        parents = set(bone for bone in bones if not bones.intersection(bone.parent_recursive))
        positions.extend(bone.matrix_local.translation for bone in parents)
    elif context_mode == 'EDIT_TEXT':
        # Blender considers only caret position as the selection center
        # But TextCurve has no API for text edit mode
        positions.append(Vector()) # use active object's position
    elif context_mode == 'PARTICLE':
        positions.append(Vector()) # use active object's position
    elif context_mode in {'SCULPT', 'PAINT_WEIGHT', 'PAINT_VERTEX', 'PAINT_TEXTURE'}:
        # last stroke position? (at least in sculpt mode, when Rotate Around Selection
        # is enabled, the view rotates around the average/center of last stroke)
        # This information is not available in Python, though
        positions.append(Vector()) # use active object's position
    
    if len(positions) == 0:
        return None
    
    n_positions = len(positions)
    if m is not None:
        positions = (m * p for p in positions)
    
    return sum(positions, Vector()) * (1.0 / n_positions)

class InputKeyMonitor:
    all_keys = {'NONE', 'LEFTMOUSE', 'MIDDLEMOUSE', 'RIGHTMOUSE', 'BUTTON4MOUSE', 'BUTTON5MOUSE', 'BUTTON6MOUSE', 'BUTTON7MOUSE', 'ACTIONMOUSE', 'SELECTMOUSE', 'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE', 'EVT_TWEAK_L', 'EVT_TWEAK_M', 'EVT_TWEAK_R', 'EVT_TWEAK_A', 'EVT_TWEAK_S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'LEFT_CTRL', 'LEFT_ALT', 'LEFT_SHIFT', 'RIGHT_ALT', 'RIGHT_CTRL', 'RIGHT_SHIFT', 'OSKEY', 'GRLESS', 'ESC', 'TAB', 'RET', 'SPACE', 'LINE_FEED', 'BACK_SPACE', 'DEL', 'SEMI_COLON', 'PERIOD', 'COMMA', 'QUOTE', 'ACCENT_GRAVE', 'MINUS', 'SLASH', 'BACK_SLASH', 'EQUAL', 'LEFT_BRACKET', 'RIGHT_BRACKET', 'LEFT_ARROW', 'DOWN_ARROW', 'RIGHT_ARROW', 'UP_ARROW', 'NUMPAD_2', 'NUMPAD_4', 'NUMPAD_6', 'NUMPAD_8', 'NUMPAD_1', 'NUMPAD_3', 'NUMPAD_5', 'NUMPAD_7', 'NUMPAD_9', 'NUMPAD_PERIOD', 'NUMPAD_SLASH', 'NUMPAD_ASTERIX', 'NUMPAD_0', 'NUMPAD_MINUS', 'NUMPAD_ENTER', 'NUMPAD_PLUS', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'PAUSE', 'INSERT', 'HOME', 'PAGE_UP', 'PAGE_DOWN', 'END', 'MEDIA_PLAY', 'MEDIA_STOP', 'MEDIA_FIRST', 'MEDIA_LAST', 'TEXTINPUT', 'WINDOW_DEACTIVATE', 'TIMER', 'TIMER0', 'TIMER1', 'TIMER2', 'TIMER_JOBS', 'TIMER_AUTOSAVE', 'TIMER_REPORT', 'TIMERREGION', 'NDOF_MOTION', 'NDOF_BUTTON_MENU', 'NDOF_BUTTON_FIT', 'NDOF_BUTTON_TOP', 'NDOF_BUTTON_BOTTOM', 'NDOF_BUTTON_LEFT', 'NDOF_BUTTON_RIGHT', 'NDOF_BUTTON_FRONT', 'NDOF_BUTTON_BACK', 'NDOF_BUTTON_ISO1', 'NDOF_BUTTON_ISO2', 'NDOF_BUTTON_ROLL_CW', 'NDOF_BUTTON_ROLL_CCW', 'NDOF_BUTTON_SPIN_CW', 'NDOF_BUTTON_SPIN_CCW', 'NDOF_BUTTON_TILT_CW', 'NDOF_BUTTON_TILT_CCW', 'NDOF_BUTTON_ROTATE', 'NDOF_BUTTON_PANZOOM', 'NDOF_BUTTON_DOMINANT', 'NDOF_BUTTON_PLUS', 'NDOF_BUTTON_MINUS', 'NDOF_BUTTON_ESC', 'NDOF_BUTTON_ALT', 'NDOF_BUTTON_SHIFT', 'NDOF_BUTTON_CTRL', 'NDOF_BUTTON_1', 'NDOF_BUTTON_2', 'NDOF_BUTTON_3', 'NDOF_BUTTON_4', 'NDOF_BUTTON_5', 'NDOF_BUTTON_6', 'NDOF_BUTTON_7', 'NDOF_BUTTON_8', 'NDOF_BUTTON_9', 'NDOF_BUTTON_10', 'NDOF_BUTTON_A', 'NDOF_BUTTON_B', 'NDOF_BUTTON_C'}
    all_modifiers = {'alt', 'ctrl', 'oskey', 'shift'}
    all_events = {'ANY', 'NOTHING', 'PRESS', 'RELEASE', 'CLICK', 'DOUBLE_CLICK', 'NORTH', 'NORTH_EAST', 'EAST', 'SOUTH_EAST', 'SOUTH', 'SOUTH_WEST', 'WEST', 'NORTH_WEST'}
    
    def __init__(self, event=None):
        self.event = ""
        self.states = {}
        self.invoke_key = 'NONE'
        self.invoke_event = 'NONE'
        if event is not None:
            self.invoke_key = event.type
            self.invoke_event = event.value
            self.update(event)
    
    def __getitem__(self, name):
        if ":" in name:
            return self.event == name
        return self.states.setdefault(name, False)
    
    def __setitem__(self, name, state):
        self.states[name] = state
    
    def update(self, event):
        if (event.value == 'PRESS') or (event.value == 'DOUBLE_CLICK'):
            self.states[event.type] = True
        elif event.value == 'RELEASE':
            self.states[event.type] = False
        
        self.states['alt'] = event.alt
        self.states['ctrl'] = event.ctrl
        self.states['oskey'] = event.oskey
        self.states['shift'] = event.shift
        
        self.event = event.type+":"+event.value
    
    def keychecker(self, keys):
        km = self
        keys = self.parse_keys(keys)
        def check(state=True):
            for key in keys:
                if key.startswith("!"):
                    if km[key[1:]] != state:
                        return True
                else:
                    if km[key] == state:
                        return True
            return False
        check.is_event = ((":" in keys[0]) if keys else False)
        return check
    
    def combine_key_parts(self, key, keyset, use_invoke_key=False):
        elements = key.split()
        combined0 = "".join(elements)
        combined1 = "_".join(elements)
        
        if use_invoke_key and (combined0 == "{INVOKEKEY}"):
            return self.invoke_key
        
        if combined0 in keyset:
            return combined0
        elif combined1 in keyset:
            return combined1
        
        return ""
    
    def parse_keys(self, keys_string):
        parts = keys_string.split(":")
        keys_string = parts[0]
        event_id = ""
        if len(parts) > 1:
            event_id = self.combine_key_parts(parts[1].upper(), self.all_events)
            if event_id:
                event_id = ":"+event_id
        
        keys = []
        for key in keys_string.split(","):
            key = key.strip()
            is_negative = key.startswith("!")
            prefix = ""
            if is_negative:
                key = key[1:]
                prefix = "!"
            
            key_id = self.combine_key_parts(key.upper(), self.all_keys, True)
            modifier_id = self.combine_key_parts(key.lower(), self.all_modifiers)
            
            if key_id:
                keys.append(prefix+key_id+event_id)
            elif modifier_id:
                if len(event_id) != 0:
                    modifier_id = modifier_id.upper()
                    if modifier_id == 'OSKEY': # has no left/right/ndof variants
                        keys.append(prefix+modifier_id+event_id)
                    else:
                        keys.append(prefix+"LEFT_"+modifier_id+event_id)
                        keys.append(prefix+"RIGHT_"+modifier_id+event_id)
                        keys.append(prefix+"NDOF_BUTTON_"+modifier_id+event_id)
                else:
                    keys.append(prefix+modifier_id)
        
        return keys

class ModeStack:
    def __init__(self, keys, transitions, default_mode, mode='NONE'):
        self.keys = keys
        self.prev_state = {}
        self.transitions = set(transitions)
        self.mode = mode
        self.default_mode = default_mode
        self.stack = [self.default_mode] # default mode should always be in the stack!
    
    def update(self):
        for name in self.keys:
            keychecker = self.keys[name]
            is_on = int(keychecker())
            
            if keychecker.is_event:
                delta_on = is_on * (-1 if name in self.stack else 1)
            else:
                delta_on = is_on - self.prev_state.get(name, 0)
                self.prev_state[name] = is_on
            
            if delta_on > 0:
                if self.transition_allowed(self.mode, name):
                    self.remove(name)
                    self.stack.append(name) # move to top
                    self.mode = name
            elif delta_on < 0:
                if self.mode != name:
                    self.remove(name)
                else:
                    self.find_transition()
    
    def remove(self, name):
        if name in self.stack:
            self.stack.remove(name)
    
    def find_transition(self):
        for i in range(len(self.stack)-1, -1, -1):
            name = self.stack[i]
            if self.transition_allowed(self.mode, name):
                self.mode = name
                self.stack = self.stack[:i+1]
    
    def transition_allowed(self, mode0, mode1):
        is_allowed = (mode0+":"+mode1) in self.transitions
        is_allowed |= (mode1+":"+mode0) in self.transitions
        return is_allowed
    
    def add_transitions(self, transitions):
        self.transitions.update(transitions)
    
    def remove_transitions(self, transitions):
        self.transitions.difference_update(transitions)

class MouselookNavigation(bpy.types.Operator):
    """Mouselook navigation"""
    bl_idname = "view3d.mouselook_navigation"
    bl_label = "Mouselook navigation"
    bl_options = {'GRAB_POINTER', 'BLOCKING'} # IMPORTANT! otherwise Continuous Grab won't work
    
    modes = ['ORBIT', 'PAN', 'DOLLY', 'ZOOM', 'FLY', 'FPS']
    transitions = ['NONE:ORBIT', 'NONE:PAN', 'NONE:DOLLY', 'NONE:ZOOM', 'NONE:FLY', 'NONE:FPS', 'ORBIT:PAN', 'ORBIT:DOLLY', 'ORBIT:ZOOM', 'ORBIT:FLY', 'ORBIT:FPS', 'PAN:DOLLY', 'PAN:ZOOM', 'DOLLY:ZOOM', 'FLY:FPS']
    
    default_mode = bpy.props.EnumProperty(items=[(m, m, m) for m in modes], name="Default mode", description="Default mode", default='ORBIT')
    allowed_transitions = bpy.props.EnumProperty(items=[(t, t, t) for t in transitions], name="Transitions", description="Allowed transitions between modes", default=set(transitions), options={'ENUM_FLAG'})
    
    zbrush_mode = bpy.props.BoolProperty(name="ZBrush mode", description="The operator would be invoked only if mouse is over empty space or close to region border", default=False)
    
    ortho_unrotate = bpy.props.BoolProperty(name="Ortho unrotate", description="In Ortho mode, rotation is abandoned if another mode is selected", default=True)
    
    def _keyprop(name, default_keys):
        return bpy.props.StringProperty(name=name, description=name, default=default_keys)
    str_keys_confirm = _keyprop("Confirm", "Ret, Numpad Enter, Left Mouse")
    str_keys_cancel = _keyprop("Cancel", "Esc, Right Mouse")
    str_keys_rotmode_switch = _keyprop("Rotation Mode Switch", "Space: Press")
    str_keys_origin_mouse = _keyprop("Origin: Mouse", "")
    str_keys_origin_selection = _keyprop("Origin: Selection", "")
    str_keys_orbit = _keyprop("Orbit", "") # main operator key (MMB) by default
    str_keys_orbit_snap = _keyprop("Orbit Snap", "Alt")
    str_keys_pan = _keyprop("Pan", "Shift")
    str_keys_dolly = _keyprop("Dolly", "")
    str_keys_zoom = _keyprop("Zoom", "Ctrl")
    str_keys_fly = _keyprop("Fly", "{Invoke key}: Double click")
    str_keys_fps = _keyprop("Walk", "Tab: Press")
    str_keys_FPS_forward = _keyprop("FPS forward", "W")
    str_keys_FPS_back = _keyprop("FPS back", "S")
    str_keys_FPS_left = _keyprop("FPS left", "A")
    str_keys_FPS_right = _keyprop("FPS right", "D")
    str_keys_FPS_up = _keyprop("FPS up", "E, R")
    str_keys_FPS_down = _keyprop("FPS down", "Q, F")
    str_keys_fps_acceleration = _keyprop("FPS fast", "Shift")
    str_keys_fps_slowdown = _keyprop("FPS slow", "Ctrl")
    str_keys_fps_crouch = _keyprop("FPS crouch", "Ctrl")
    str_keys_fps_jump = _keyprop("FPS jump", "Space")
    str_keys_fps_teleport = _keyprop("FPS teleport", "{Invoke key}, V")
    
    def create_keycheckers(self, event):
        self.keys_invoke = self.km.keychecker(event.type)
        if event.value in {'RELEASE', 'CLICK'}:
            self.keys_invoke_confirm = self.km.keychecker(event.type+":PRESS")
        else:
            self.keys_invoke_confirm = self.km.keychecker(event.type+":RELEASE")
        self.keys_confirm = self.km.keychecker(self.str_keys_confirm)
        self.keys_cancel = self.km.keychecker(self.str_keys_cancel)
        self.keys_rotmode_switch = self.km.keychecker(self.str_keys_rotmode_switch)
        self.keys_origin_mouse = self.km.keychecker(self.str_keys_origin_mouse)
        self.keys_origin_selection = self.km.keychecker(self.str_keys_origin_selection)
        self.keys_orbit = self.km.keychecker(self.str_keys_orbit)
        self.keys_orbit_snap = self.km.keychecker(self.str_keys_orbit_snap)
        self.keys_pan = self.km.keychecker(self.str_keys_pan)
        self.keys_dolly = self.km.keychecker(self.str_keys_dolly)
        self.keys_zoom = self.km.keychecker(self.str_keys_zoom)
        self.keys_fly = self.km.keychecker(self.str_keys_fly)
        self.keys_fps = self.km.keychecker(self.str_keys_fps)
        self.keys_FPS_forward = self.km.keychecker(self.str_keys_FPS_forward)
        self.keys_FPS_back = self.km.keychecker(self.str_keys_FPS_back)
        self.keys_FPS_left = self.km.keychecker(self.str_keys_FPS_left)
        self.keys_FPS_right = self.km.keychecker(self.str_keys_FPS_right)
        self.keys_FPS_up = self.km.keychecker(self.str_keys_FPS_up)
        self.keys_FPS_down = self.km.keychecker(self.str_keys_FPS_down)
        self.keys_fps_acceleration = self.km.keychecker(self.str_keys_fps_acceleration)
        self.keys_fps_slowdown = self.km.keychecker(self.str_keys_fps_slowdown)
        self.keys_fps_crouch = self.km.keychecker(self.str_keys_fps_crouch)
        self.keys_fps_jump = self.km.keychecker(self.str_keys_fps_jump)
        self.keys_fps_teleport = self.km.keychecker(self.str_keys_fps_teleport)
    
    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        settings = wm.mouselook_navigation_runtime_settings
        if not settings.is_enabled:
            return False
        return (context.space_data.type == 'VIEW_3D')
    
    def modal(self, context, event):
        try:
            return self.modal_main(context, event)
        except:
            # If anything fails, at least dispose the resources
            self.cleanup(context)
            raise
    
    def modal_main(self, context, event):
        region = context.region
        v3d = context.space_data
        rv3d = context.region_data
        
        # Sometimes ZBuffer gets cleared for some reason,
        # so we need to wait at least 1 frame to get depth
        if self.delayed_mouse_depth is not None:
            if self.delayed_mouse_depth[0] > 0:
                self.process_delayed_depth()
            else:
                return {'RUNNING_MODAL'}
        
        region_pos = self.sv.region_rect[0]
        region_size = self.sv.region_rect[1]
        
        userprefs = context.user_preferences
        drag_threshold = userprefs.inputs.drag_threshold
        tweak_threshold = userprefs.inputs.tweak_threshold
        mouse_double_click_time = userprefs.inputs.mouse_double_click_time / 1000.0
        rotate_method = userprefs.inputs.view_rotate_method
        invert_mouse_zoom = userprefs.inputs.invert_mouse_zoom
        invert_wheel_zoom = userprefs.inputs.invert_zoom_wheel
        use_zoom_to_mouse = userprefs.view.use_zoom_to_mouse
        use_auto_perspective = userprefs.view.use_auto_perspective
        
        use_zoom_to_mouse |= self.force_origin_mouse
        use_auto_perspective &= self.rotation_snap_autoperspective
        
        walk_prefs = userprefs.inputs.walk_navigation
        teleport_time = walk_prefs.teleport_time
        walk_speed_factor = walk_prefs.walk_speed_factor
        use_gravity = walk_prefs.use_gravity
        view_height = walk_prefs.view_height
        jump_height = walk_prefs.jump_height
        
        self.km.update(event)
        mouse_prev = Vector((event.mouse_prev_x, event.mouse_prev_y))
        mouse = Vector((event.mouse_x, event.mouse_y))
        mouse_offset = mouse - self.mouse0
        mouse_delta = mouse - mouse_prev
        mouse_region = mouse - region_pos
        
        # Attempt to match Blender's default speeds
        ZOOM_SPEED_COEF = 0.77
        TRACKBALL_SPEED_COEF = 0.35
        TURNTABLE_SPEED_COEF = 0.62
        
        clock = time.clock()
        dt = 0.01
        speed_move = 2.5 * self.sv.distance# * dt # use realtime dt
        speed_zoom = ZOOM_SPEED_COEF * dt
        speed_rot = TRACKBALL_SPEED_COEF * dt
        speed_euler = Vector((-1, 1)) * TURNTABLE_SPEED_COEF * dt
        speed_autolevel = 1 * dt
        
        if invert_mouse_zoom:
            speed_zoom *= -1
        
        speed_move *= self.fps_speed_modifier
        speed_zoom *= self.zoom_speed_modifier
        speed_rot *= self.rotation_speed_modifier
        speed_euler *= self.rotation_speed_modifier
        speed_autolevel *= self.autolevel_speed_modifier
        
        confirm = self.keys_confirm()
        cancel = self.keys_cancel()
        
        is_orbit_snap = False
        trackball_mode = self.trackball_mode
        
        self.mode_stack.update()
        mode = self.mode_stack.mode
        
        if self.explicit_orbit_origin is not None:
            m_ofs = self.sv.matrix
            m_ofs.translation = self.explicit_orbit_origin
            m_ofs_inv = m_ofs.inverted_safe()
        
        if (mode == 'FLY') or (mode == 'FPS'):
            if self.sv.is_region_3d or not self.sv.quadview_lock:
                self.explicit_orbit_origin = None
                self.sv.is_perspective = True
                self.sv.lock_cursor = False
                self.sv.lock_object = None
                self.sv.use_viewpoint = True
                trackball_mode = 'CENTER'
                
                mode = 'ORBIT'
                
                move_vector = self.FPS_move_vector()
                
                if self.mode_stack.mode == 'FPS':
                    if move_vector.z != 0: # turn off gravity if manual up/down is used
                        use_gravity = False
                        walk_prefs.use_gravity = use_gravity
                    elif self.keys_fps_jump():
                        use_gravity = True
                        walk_prefs.use_gravity = use_gravity
                    
                    rotate_method = 'TURNTABLE'
                    min_speed_autolevel = 30 * dt
                    speed_autolevel = max(speed_autolevel, min_speed_autolevel)
                    
                    self.update_fly_speed(event, True)
                    
                    if not self.keys_fps_teleport():
                        self.teleport_allowed = True
                    
                    if self.teleport_allowed and self.keys_fps_teleport():
                        ray_data = self.sv.ray(self.sv.project(self.sv.focus))
                        raycast_result = context.scene.ray_cast(ray_data[0], ray_data[1])
                        if raycast_result[0]:
                            normal = raycast_result[4]
                            if normal.dot(ray_data[1] - ray_data[0]) > 0:
                                normal = -normal
                            self.teleport_time_start = clock
                            self.teleport_pos = raycast_result[3] + normal * view_height
                            self.teleport_pos_start = self.sv.viewpoint
                    
                    if move_vector.magnitude > 0:
                        self.teleport_pos = None
                else:
                    use_gravity = False
                    
                    self.update_fly_speed(event, (move_vector.magnitude > 0))
                    
                    if (not self.keys_invoke.is_event) and self.keys_invoke():
                        self.fly_speed = Vector()
                        mode = 'PAN'
                
                self.rotate_method = rotate_method # used for FPS horizontal
                
                if (event.type == 'MOUSEMOVE') or (event.type == 'INBETWEEN_MOUSEMOVE'):
                    if mode == 'ORBIT':
                        if (rotate_method == 'TURNTABLE') or is_orbit_snap:
                            self.change_euler(mouse_delta.y * speed_euler.y, mouse_delta.x * speed_euler.x, 0)
                        else: # 'TRACKBALL'
                            self.change_rot_mouse(mouse_delta, mouse, speed_rot, trackball_mode)
                    elif mode == 'PAN':
                        self.change_pos_mouse(mouse_delta, False)
                
                mode = self.mode_stack.mode # for display in header
                
                self.pos = self.sv.focus
        else:
            self.sv.use_viewpoint = False
            use_gravity = False
            self.teleport_pos = None
            self.teleport_allowed = False
            
            confirm |= self.keys_invoke_confirm()
            
            if self.keys_rotmode_switch():
                if rotate_method == 'TURNTABLE':
                    rotate_method = 'TRACKBALL'
                else:
                    rotate_method = 'TURNTABLE'
                userprefs.inputs.view_rotate_method = rotate_method
            self.rotate_method = rotate_method # used for FPS horizontal
            
            is_orbit_snap = self.keys_orbit_snap()
            delta_orbit_snap = int(is_orbit_snap) - int(self.prev_orbit_snap)
            self.prev_orbit_snap = is_orbit_snap
            if delta_orbit_snap < 0:
                self.euler = self.sv.turntable_euler
                self.rot = self.sv.rotation
            
            if not self.sv.is_perspective:
                if mode == 'DOLLY':
                    mode = 'ZOOM'
                
                # The goal is to make it easy to pan view without accidentally rotating it
                if self.ortho_unrotate:
                    if mode in ('PAN', 'DOLLY', 'ZOOM'):
                        # forbid transitions back to orbit
                        self.mode_stack.remove_transitions({'ORBIT:PAN', 'ORBIT:DOLLY', 'ORBIT:ZOOM'})
                        self.reset_rotation(rotate_method, use_auto_perspective)
            
            if (event.type == 'MOUSEMOVE') or (event.type == 'INBETWEEN_MOUSEMOVE'):
                if mode == 'ORBIT':
                    # snapping trackball rotation is problematic (I don't know how to do it)
                    if (rotate_method == 'TURNTABLE') or is_orbit_snap:
                        self.change_euler(mouse_delta.y * speed_euler.y, mouse_delta.x * speed_euler.x, 0)
                    else: # 'TRACKBALL'
                        self.change_rot_mouse(mouse_delta, mouse, speed_rot, trackball_mode)
                    
                    if use_auto_perspective:
                        self.sv.is_perspective = not is_orbit_snap
                    
                    if is_orbit_snap:
                        self.snap_rotation(self.rotation_snap_subdivs)
                elif mode == 'PAN':
                    self.change_pos_mouse(mouse_delta, False)
                elif mode == 'DOLLY':
                    self.change_pos_mouse(mouse_delta, True)
                elif mode == 'ZOOM':
                    self.change_distance((mouse_delta.y - mouse_delta.x) * speed_zoom, use_zoom_to_mouse)
        
        if event.type.startswith('TIMER'):
            dt = clock - self.clock
            self.clock = clock
            
            if speed_autolevel > 0:
                if (not is_orbit_snap) or (mode != 'ORBIT'):
                    if rotate_method == 'TURNTABLE':
                        self.change_euler(0, 0, speed_autolevel, False)
                    elif self.autolevel_trackball:
                        speed_autolevel *= 1.0 - abs(self.sv.forward.z)
                        self.change_euler(0, 0, speed_autolevel, self.autolevel_trackball_up)
            
            if self.teleport_pos is None:
                speed_move *= dt
                
                fps_speed = self.calc_FPS_speed(walk_speed_factor)
                abs_speed = Vector()
                if fps_speed.magnitude > 0:
                    if not self.sv.is_perspective:
                        self.change_distance(fps_speed.y * speed_zoom*(-4), use_zoom_to_mouse)
                        fps_speed.y = 0
                    abs_speed = self.abs_fps_speed(fps_speed.x, fps_speed.y, fps_speed.z, speed_move, use_gravity)
                
                if use_gravity:
                    gravity = -9.91
                    self.velocity.z *= 0.999 # dampen
                    self.velocity.z += gravity * dt
                    is_jump = self.keys_fps_jump()
                    if is_jump:
                        if self.velocity.z < 0:
                            self.velocity.z *= 0.9
                        if not self.prev_jump:
                            self.velocity.z += jump_height
                        self.velocity.z += (abs(gravity) + jump_height) * dt
                    self.prev_jump = is_jump
                    
                    is_crouching = self.keys_fps_crouch()
                    
                    pos0 = self.sv.viewpoint
                    pos = pos0.copy()
                    
                    v0 = abs_speed
                    v = abs_speed
                    #v, collided = apply_collisions(context.scene, pos, v0, view_height, is_crouching, False, 1)
                    pos += v
                    
                    v0 = self.velocity * dt
                    v, collided = apply_collisions(context.scene, pos, v0, view_height, is_crouching, True, 0)
                    if collided:
                        self.velocity = Vector()
                    pos += v
                    
                    abs_speed = pos - pos0
                else:
                    self.velocity = Vector()
            else:
                p0 = self.sv.viewpoint
                t = (clock - self.teleport_time_start) + dt # +dt to move immediately
                if t >= teleport_time:
                    p1 = self.teleport_pos
                    self.teleport_pos = None
                else:
                    t = t / teleport_time
                    p1 = self.teleport_pos * t + self.teleport_pos_start * (1.0 - t)
                abs_speed = p1 - p0
            
            if abs_speed.magnitude > 0:
                self.change_pos(abs_speed)
            
            context.area.tag_redraw()
        
        if self.explicit_orbit_origin is not None:
            pre_rotate_focus = m_ofs_inv * self.pos
            m_ofs = self.sv.matrix
            m_ofs.translation = self.explicit_orbit_origin
            self.pos = m_ofs * pre_rotate_focus
            self.sv.focus = self.pos
        
        txt = "{} (zoom={})".format(mode, self.sv.distance)
        context.area.header_text_set(txt)
        
        if confirm:
            self.cleanup(context)
            return {'FINISHED'}
        elif cancel:
            self.revert_changes()
            self.cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def update_fly_speed(self, event, dont_fly=False):
        wheel_up = int(event.type == 'WHEELUPMOUSE')
        wheel_down = int(event.type == 'WHEELDOWNMOUSE')
        wheel_delta = wheel_up - wheel_down
        
        if dont_fly:
            self.fly_speed = Vector() # stop (FPS overrides flight)
            self.change_distance(wheel_delta*0.5)
        else:
            fwd_speed = self.fly_speed.y
            if (wheel_delta * fwd_speed < 0) and (abs(fwd_speed) >= 2):
                wheel_delta *= 2 # quick direction reversal
            fwd_speed = min(max(fwd_speed + wheel_delta, -9), 9)
            fwd_speed = round(fwd_speed) # avoid accumulation errors
            self.fly_speed.y = fwd_speed
    
    def FPS_move_vector(self):
        move_forward = self.keys_FPS_forward()
        move_back = self.keys_FPS_back()
        move_left = self.keys_FPS_left()
        move_right = self.keys_FPS_right()
        move_up = self.keys_FPS_up()
        move_down = self.keys_FPS_down()
        
        move_x = int(move_right) - int(move_left)
        move_y = int(move_forward) - int(move_back)
        move_z = int(move_up) - int(move_down)
        
        return Vector((move_x, move_y, move_z))
    
    def calc_FPS_speed(self, walk_speed_factor=5):
        move_vector = self.FPS_move_vector()
        
        movement_accelerate = self.keys_fps_acceleration()
        movement_slowdown = self.keys_fps_slowdown()
        move_speedup = int(movement_accelerate) - int(movement_slowdown)
        if self.mode_stack.mode in {'PAN', 'DOLLY', 'ZOOM'}:
            move_speedup = 0
        
        fps_speed = move_vector * (walk_speed_factor ** move_speedup)
        
        if fps_speed.magnitude == 0:
            fps_speed = self.fly_speed.copy()
            fps_speed.x = self.calc_fly_speed(fps_speed.x)
            fps_speed.y = self.calc_fly_speed(fps_speed.y)
            fps_speed.z = self.calc_fly_speed(fps_speed.z)
        
        return fps_speed
    
    def calc_fly_speed(self, v, k=2):
        if round(v) == 0:
            return 0
        return math.copysign(2 ** (abs(v) - k), v)
    
    def change_distance(self, delta, to_explicit_origin=False):
        log_zoom = math.log(max(self.sv.distance, self.min_distance), 2)
        self.sv.distance = math.pow(2, log_zoom + delta)
        if to_explicit_origin and (self.explicit_orbit_origin is not None):
            dst = self.explicit_orbit_origin
            offset = self.pos - dst
            log_zoom = math.log(max(offset.magnitude, self.min_distance), 2)
            offset = offset.normalized() * math.pow(2, log_zoom + delta)
            self.pos = dst + offset
            self.sv.focus = self.pos
    
    def abs_fps_speed(self, dx, dy, dz, speed=1.0, use_gravity=False):
        xdir, ydir, zdir = self.sv.right, self.sv.forward, self.sv.up
        fps_horizontal = (self.fps_horizontal or use_gravity) and self.sv.is_perspective
        if (self.rotate_method == 'TURNTABLE') and fps_horizontal:
            ysign = (-1.0 if zdir.z < 0 else 1.0)
            zdir = Vector((0, 0, 1))
            ydir = Quaternion(zdir, self.euler.z) * Vector((0, 1, 0))
            xdir = ydir.cross(zdir)
            ydir *= ysign
        return (xdir*dx + ydir*dy + zdir*dz) * speed
    
    def change_pos(self, abs_speed):
        self.pos += abs_speed
        self.sv.focus = self.pos
    
    def change_pos_mouse(self, mouse_delta, is_dolly=False):
        self.pos += self.mouse_delta_movement(mouse_delta, is_dolly)
        self.sv.focus = self.pos
    
    def mouse_delta_movement(self, mouse_delta, is_dolly=False):
        region = self.sv.region
        region_center = Vector((region.width*0.5, region.height*0.5))
        p0 = self.sv.unproject(region_center)
        p1 = self.sv.unproject(region_center - mouse_delta)
        pd = p1 - p0
        if is_dolly:
            pd_x = pd.dot(self.sv.right)
            pd_y = pd.dot(self.sv.up)
            pd = (self.sv.right * pd_x) + (self.sv.forward * pd_y)
        return pd
    
    def reset_rotation(self, rotate_method, use_auto_perspective):
        self.rot = self.rot0.copy()
        self.euler = self.euler0.copy()
        if rotate_method == 'TURNTABLE':
            self.sv.turntable_euler = self.euler # for turntable
        else:
            self.sv.rotation = self.rot # for trackball
        
        if use_auto_perspective:
            self.sv.is_perspective = self._perspective0
    
    def snap_rotation(self, n=1):
        grid = math.pi*0.5 / n
        euler = self.euler.copy()
        euler.x = round(euler.x / grid) * grid
        euler.y = round(euler.y / grid) * grid
        euler.z = round(euler.z / grid) * grid
        self.sv.turntable_euler = euler
        self.rot = self.sv.rotation
    
    def change_euler(self, ex, ey, ez, always_up=False):
        self.euler.x += ex
        self.euler.z += ey
        if always_up and (self.sv.up.z < 0) or (abs(self.euler.y) > math.pi*0.5):
            _pi = math.copysign(math.pi, self.euler.y)
            self.euler.y = _pi - (_pi - self.euler.y) * math.pow(2, -abs(ez))
        else:
            self.euler.y *= math.pow(2, -abs(ez))
        self.sv.turntable_euler = self.euler
        self.rot = self.sv.rotation # update other representation
    
    def change_rot_mouse(self, mouse_delta, mouse, speed_rot, trackball_mode):
        if trackball_mode == 'CENTER':
            mouse_delta *= speed_rot
            spin = -((self.sv.right * mouse_delta.x) + (self.sv.up * mouse_delta.y)).normalized()
            axis = spin.cross(self.sv.forward)
            self.rot = Quaternion(axis, mouse_delta.magnitude) * self.rot
        elif trackball_mode == 'WRAPPED':
            mouse_delta *= speed_rot
            cdir = Vector((0, -1, 0))
            tv, x_neg, y_neg = self.trackball_vector(mouse)
            r = cdir.rotation_difference(tv)
            spin = r * Vector((mouse_delta.x, 0, mouse_delta.y))
            axis = spin.cross(tv)
            axis = self.sv.matrix.to_3x3() * axis
            self.rot = Quaternion(axis, mouse_delta.magnitude) * self.rot
        else:
            # Glitchy/buggy. Consult with Dalai Felinto?
            region = self.sv.region
            mouse -= Vector((region.x, region.y))
            halfsize = Vector((region.width, region.height))*0.5
            p1 = (mouse - mouse_delta) - halfsize
            p2 = (mouse) - halfsize
            p1 = Vector((p1.x/halfsize.x, p1.y/halfsize.y))
            p2 = Vector((p2.x/halfsize.x, p2.y/halfsize.y))
            q = trackball(p1.x, p1.y, p2.x, p2.y, 1.1)
            axis, angle = q.to_axis_angle()
            axis = self.sv.matrix.to_3x3() * axis
            q = Quaternion(axis, angle * speed_rot*200)
            self.rot = q * self.rot
        self.rot.normalize()
        self.sv.rotation = self.rot # update other representation
        self.euler = self.sv.turntable_euler # update other representation
    
    def _wrap_xy(self, xy, m=1):
        region = self.sv.region
        x = xy.x % (region.width*m)
        y = xy.y % (region.height*m)
        return Vector((x, y))
    def trackball_vector(self, xy):
        region = self.sv.region
        region_halfsize = Vector((region.width*0.5, region.height*0.5))
        radius = region_halfsize.magnitude * 1.1
        xy -= Vector((region.x, region.y)) # convert to region coords
        xy = self._wrap_xy(xy, 2)
        x_neg = (xy.x >= region.width)
        y_neg = (xy.y >= region.height)
        xy = self._wrap_xy(xy)
        xy -= region_halfsize # make relative to center
        xy *= (1.0/radius) # normalize
        z = math.sqrt(1.0 - xy.length_squared)
        return Vector((xy.x, -z, xy.y)).normalized(), x_neg, y_neg
    
    def calc_zbrush_border(self, scale=0.05, abs_min=16):
        clickable_region_size = self.sv.clickable_region_rect[1]
        wrk_sz = min(clickable_region_size.x, clickable_region_size.y)
        return max(wrk_sz*scale, abs_min)
    
    def process_delayed_depth(self):
        redraws_count, mouse, mouse_region = self.delayed_mouse_depth
        
        zbuf = self.sv.read_zbuffer(mouse)[0]
        zcam = self.sv.zbuf_to_depth(zbuf)
        
        if zbuf < 1.0:
            self.explicit_orbit_origin = self.sv.unproject(mouse_region, zcam)
            if self.sv.is_perspective:
                # Blender adjusts distance so that focus and z-point lie in the same plane
                viewpoint = self.sv.viewpoint
                self.sv.distance = zcam
                self.sv.viewpoint = viewpoint
                # Update memorized values
                self._distance0 = self.sv.distance
                self._pos0 = self.sv.focus
                self.pos = self._pos0.copy()
        else:
            self.explicit_orbit_origin = self.sv.unproject(mouse_region)
        
        self.delayed_mouse_depth = None
    
    def invoke(self, context, event):
        wm = context.window_manager
        userprefs = context.user_preferences
        addon_prefs = userprefs.addons[__name__].preferences
        region = context.region
        v3d = context.space_data
        rv3d = context.region_data
        
        if event.value == 'RELEASE':
            # 'ANY' is useful for click+doubleclick, but release is not intended
            # IMPORTANT: self.bl_idname is NOT the same as class.bl_idname!
            for kmi in KeyMapItemSearch(MouselookNavigation.bl_idname):
                if (kmi.type == event.type) and (kmi.value == 'ANY'):
                    return {'CANCELLED'}
        
        self.sv = SmartView3D(context)
        
        region_pos = self.sv.region_rect[0]
        region_size = self.sv.region_rect[1]
        clickable_region_pos = self.sv.clickable_region_rect[0]
        clickable_region_size = self.sv.clickable_region_rect[1]
        
        self.zbrush_border = self.calc_zbrush_border()
        
        self.km = InputKeyMonitor(event)
        self.create_keycheckers(event)
        mouse_prev = Vector((event.mouse_prev_x, event.mouse_prev_y))
        mouse = Vector((event.mouse_x, event.mouse_y))
        mouse_delta = mouse - mouse_prev
        mouse_region = mouse - region_pos
        mouse_clickable_region = mouse - clickable_region_pos
        
        zbuf = self.sv.read_zbuffer(mouse)[0]
        zcam = self.sv.zbuf_to_depth(zbuf)
        ray_data = self.sv.ray(mouse_region)
        raycast_result = context.scene.ray_cast(ray_data[0], ray_data[1])
        
        self.force_origin_mouse = self.keys_origin_mouse()
        self.force_origin_selection = self.keys_origin_selection()
        use_origin_mouse = userprefs.view.use_mouse_depth_navigate
        use_origin_selection = userprefs.view.use_rotate_around_active
        if self.force_origin_selection:
            use_origin_selection = True
            use_origin_mouse = False
        elif self.force_origin_mouse:
            use_origin_selection = False
            use_origin_mouse = True
        
        self.delayed_mouse_depth = None
        self.explicit_orbit_origin = None
        if use_origin_selection:
            self.explicit_orbit_origin = calc_selection_center(context)
        elif use_origin_mouse:
            self.delayed_mouse_depth = [0, mouse, mouse_region]
            #self.process_delayed_depth()
        
        mode_keys = {'ORBIT':self.keys_orbit, 'PAN':self.keys_pan, 'DOLLY':self.keys_dolly, 'ZOOM':self.keys_zoom, 'FLY':self.keys_fly, 'FPS':self.keys_fps}
        self.mode_stack = ModeStack(mode_keys, self.allowed_transitions, self.default_mode, 'NONE')
        self.mode_stack.update()
        if self.mode_stack.mode == 'NONE':
            if self.zbrush_mode:
                # In Sculpt mode, zbuffer seems to be cleared!
                # Also, zbuf can be written by non-geometry, which is probably not desirable
                is_over_obj = raycast_result[0]# or (zbuf < 1.0)
                mouse_region_11 = clickable_region_size - mouse_clickable_region
                wrk_x = min(mouse_clickable_region.x, mouse_region_11.x)
                wrk_y = min(mouse_clickable_region.y, mouse_region_11.y)
                wrk_pos = min(wrk_x, wrk_y)
                if is_over_obj and (wrk_pos > self.zbrush_border):
                    return {'PASS_THROUGH'}
            self.mode_stack.mode = self.default_mode
        
        if addon_prefs.use_blender_colors:
            try:
                view_overlay_color = userprefs.themes[0].view_3d.view_overlay
            except:
                view_overlay_color = Color((0,0,0))
            self.color_crosshair_visible = view_overlay_color
            self.color_crosshair_obscured = view_overlay_color
            self.color_zbrush_border = view_overlay_color
        else:
            self.color_crosshair_visible = addon_prefs.color_crosshair_visible
            self.color_crosshair_obscured = addon_prefs.color_crosshair_obscured
            self.color_zbrush_border = addon_prefs.color_zbrush_border
        self.show_crosshair = addon_prefs.show_crosshair
        self.show_zbrush_border = addon_prefs.show_zbrush_border
        
        settings = wm.mouselook_navigation_runtime_settings
        settings = addon_prefs
        self.fps_horizontal = settings.fps_horizontal
        self.trackball_mode = settings.trackball_mode
        self.fps_speed_modifier = settings.fps_speed_modifier
        self.zoom_speed_modifier = settings.zoom_speed_modifier
        self.rotation_snap_subdivs = settings.rotation_snap_subdivs
        self.rotation_snap_autoperspective = settings.rotation_snap_autoperspective
        self.rotation_speed_modifier = settings.rotation_speed_modifier
        self.autolevel_trackball = settings.autolevel_trackball
        self.autolevel_trackball_up = settings.autolevel_trackball_up
        self.autolevel_speed_modifier = settings.autolevel_speed_modifier
        
        self.prev_orbit_snap = False
        self.min_distance = 2 ** -10
        
        self.fly_speed = Vector()
        
        self._clock0 = time.clock()
        self._continuous0 = userprefs.inputs.use_mouse_continuous
        self._mouse0 = Vector((event.mouse_x, event.mouse_y))
        self._perspective0 = self.sv.is_perspective
        self._distance0 = self.sv.distance
        self._pos0 = self.sv.focus
        self._rot0 = self.sv.rotation
        self._euler0 = self.sv.turntable_euler
        
        self.mouse0 = self._mouse0.copy()
        self.clock0 = self._clock0
        self.pos = self._pos0.copy()
        self.rot0 = self._rot0.copy()
        self.rot = self.rot0.copy()
        self.euler0 = self._euler0.copy()
        self.euler = self.euler0.copy()
        
        self.clock = self.clock0
        self.velocity = Vector()
        self.prev_jump = False
        self.teleport_pos = None
        self.teleport_pos_start = None
        self.teleport_time_start = -1
        self.teleport_allowed = False
        
        userprefs.inputs.use_mouse_continuous = True
        
        self.register_handlers(context)
        
        # We need the view to redraw so that crosshair would appear
        # immediately after user presses MMB
        context.area.header_text_set()
        context.area.tag_redraw()
        
        return {'RUNNING_MODAL'}
    
    def revert_changes(self):
        self.sv.use_viewpoint = False
        self.sv.rotation = self._rot0
        self.sv.distance = self._distance0
        self.sv.focus = self._pos0
        self.sv.is_perspective = self._perspective0
    
    def cleanup(self, context):
        userprefs = context.user_preferences
        userprefs.inputs.use_mouse_continuous = self._continuous0
        
        self.unregister_handlers(context)
        
        # We need the view to redraw so that crosshair would disappear
        # immediately after user releases MMB
        context.area.header_text_set()
        context.area.tag_redraw()
    
    def register_handlers(self, context):
        wm = context.window_manager
        wm.modal_handler_add(self)
        self._timer = wm.event_timer_add(0.01, context.window)
        self._handle_view = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_view, (self, context), 'WINDOW', 'POST_VIEW')
        self._handle_px = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback_px, (self, context), 'WINDOW', 'POST_PIXEL')
    
    def unregister_handlers(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
        if self._handle_view is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_view, 'WINDOW')
        if self._handle_px is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_px, 'WINDOW')


def draw_crosshair(self, context, use_focus):
    userprefs = context.user_preferences
    region = context.region
    v3d = context.space_data
    rv3d = context.region_data
    
    if self.sv.is_camera and not self.sv.lock_camera:
        return # camera can't be manipulated, so crosshair is meaningless here
    
    alpha = 1.0
    has_explicit_origin = (self.explicit_orbit_origin is not None)
    if use_focus:
        if has_explicit_origin:
            alpha = 0.4
        
        focus_proj = self.sv.project(self.sv.focus)
        if focus_proj is None:
            return
        
        region_center = Vector((region.width*0.5, region.height*0.5))
        if (focus_proj - region_center).magnitude < 2:
            focus_proj = region_center
        
        if self.sv.is_camera and (not self.sv.is_perspective): # Somewhy Blender behaves like this
            focus_proj = region_center # in case camera has non-zero shift
    elif has_explicit_origin:
        focus_proj = self.sv.project(self.explicit_orbit_origin)
        if focus_proj is None:
            return
    else:
        return
    
    focus_proj = snap_pixel_vector(focus_proj)
    
    near, far, origin = self.sv.zbuf_range
    dist = (self.sv.focus - origin).magnitude
    if self.sv.is_perspective:
        dist = min(max(dist, near*1.01), far*0.99)
    else:
        dist = min(max(dist, near*0.99 + far*0.01), far*0.99 + near*0.01)
    
    l0, l1 = 16, 25
    lines = [(Vector((0, l0)), Vector((0, l1))), (Vector((0, -l0)), Vector((0, -l1))),
             (Vector((l0, 0)), Vector((l1, 0))), (Vector((-l0, 0)), Vector((-l1, 0)))]
    lines = [(self.sv.unproject(p0 + focus_proj, dist, True),
              self.sv.unproject(p1 + focus_proj, dist, True)) for p0, p1 in lines]
    
    depth_test_prev = gl_get(bgl.GL_DEPTH_TEST)
    depth_func_prev = gl_get(bgl.GL_DEPTH_FUNC)
    depth_mask_prev = gl_get(bgl.GL_DEPTH_WRITEMASK)
    line_stipple_prev = gl_get(bgl.GL_LINE_STIPPLE)
    color_prev = gl_get(bgl.GL_COLOR)
    blend_prev = gl_get(bgl.GL_BLEND)
    line_width_prev = gl_get(bgl.GL_LINE_WIDTH)
    
    gl_enable(bgl.GL_BLEND, True)
    gl_enable(bgl.GL_LINE_STIPPLE, False)
    gl_enable(bgl.GL_DEPTH_WRITEMASK, False)
    gl_enable(bgl.GL_DEPTH_TEST, True)
    
    color = self.color_crosshair_visible
    bgl.glDepthFunc(bgl.GL_LEQUAL)
    bgl.glColor4f(color[0], color[1], color[2], 1.0*alpha)
    bgl.glLineWidth(1)
    bgl.glBegin(bgl.GL_LINES)
    for p0, p1 in lines:
        bgl.glVertex3f(p0[0], p0[1], p0[2])
        bgl.glVertex3f(p1[0], p1[1], p1[2])
    bgl.glEnd()
    
    color = self.color_crosshair_obscured
    bgl.glDepthFunc(bgl.GL_GREATER)
    bgl.glColor4f(color[0], color[1], color[2], 0.35*alpha)
    bgl.glLineWidth(3)
    bgl.glBegin(bgl.GL_LINES)
    for p0, p1 in lines:
        bgl.glVertex3f(p0[0], p0[1], p0[2])
        bgl.glVertex3f(p1[0], p1[1], p1[2])
    bgl.glEnd()
    
    
    gl_enable(bgl.GL_DEPTH_TEST, depth_test_prev)
    gl_enable(bgl.GL_DEPTH_WRITEMASK, depth_mask_prev)
    bgl.glDepthFunc(depth_func_prev)
    gl_enable(bgl.GL_LINE_STIPPLE, line_stipple_prev)
    bgl.glColor4f(color_prev[0], color_prev[1], color_prev[2], color_prev[3])
    gl_enable(bgl.GL_BLEND, blend_prev)
    bgl.glLineWidth(line_width_prev)

def draw_callback_view(self, context):
    userprefs = context.user_preferences
    region = context.region
    v3d = context.space_data
    rv3d = context.region_data
    
    if self.sv.region_data != context.region_data:
        return
    
    if self.show_crosshair:
        draw_crosshair(self, context, False)
        draw_crosshair(self, context, True)

def draw_callback_px(self, context):
    userprefs = context.user_preferences
    region = context.region
    v3d = context.space_data
    rv3d = context.region_data
    
    if self.sv.region_data != context.region_data:
        return
    
    if self.delayed_mouse_depth is not None:
        self.delayed_mouse_depth[0] += 1 # increment redraws counter
    
    region_pos = self.sv.region_rect[0]
    region_size = self.sv.region_rect[1]
    clickable_region_pos = self.sv.clickable_region_rect[0]
    clickable_region_size = self.sv.clickable_region_rect[1]
    
    if self.zbrush_mode and self.show_zbrush_border:
        blend_prev = gl_get(bgl.GL_BLEND)
        gl_enable(bgl.GL_BLEND, True)
        x, y = clickable_region_pos - region_pos
        w, h = clickable_region_size
        border = self.zbrush_border
        color = self.color_zbrush_border
        bgl.glColor4f(color[0], color[1], color[2], 0.5)
        bgl.glBegin(bgl.GL_LINE_LOOP)
        bgl.glVertex2f(x + border, y + border)
        bgl.glVertex2f(x + w-border, y + border)
        bgl.glVertex2f(x + w-border, y + h-border)
        bgl.glVertex2f(x + border, y + h-border)
        bgl.glEnd()
        gl_enable(bgl.GL_BLEND, blend_prev)


# OpenGl helper functions/data
gl_state_info = {
    bgl.GL_MATRIX_MODE:(bgl.GL_INT, 1),
    bgl.GL_PROJECTION_MATRIX:(bgl.GL_DOUBLE, 16),
    bgl.GL_LINE_WIDTH:(bgl.GL_FLOAT, 1),
    bgl.GL_BLEND:(bgl.GL_BYTE, 1),
    bgl.GL_LINE_STIPPLE:(bgl.GL_BYTE, 1),
    bgl.GL_COLOR:(bgl.GL_FLOAT, 4),
    bgl.GL_SMOOTH:(bgl.GL_BYTE, 1),
    bgl.GL_DEPTH_TEST:(bgl.GL_BYTE, 1),
    bgl.GL_DEPTH_FUNC:(bgl.GL_INT, 1),
    bgl.GL_DEPTH_WRITEMASK:(bgl.GL_BYTE, 1),
}
gl_type_getters = {
    bgl.GL_INT:bgl.glGetIntegerv,
    bgl.GL_DOUBLE:bgl.glGetFloatv, # ?
    bgl.GL_FLOAT:bgl.glGetFloatv,
    #bgl.GL_BYTE:bgl.glGetFloatv, # Why GetFloat for getting byte???
    bgl.GL_BYTE:bgl.glGetBooleanv, # maybe like that?
}

def gl_get(state_id):
    type, size = gl_state_info[state_id]
    buf = bgl.Buffer(type, [size])
    gl_type_getters[type](state_id, buf)
    return (buf if (len(buf) != 1) else buf[0])

def gl_enable(state_id, enable):
    if enable:
        bgl.glEnable(state_id)
    else:
        bgl.glDisable(state_id)

def gl_matrix_to_buffer(m, dtype=bgl.GL_FLOAT):
    tempMat = [m[i][j] for i in range(4) for j in range(4)]
    return bgl.Buffer(dtype, 16, tempMat)


def KeyMapItemSearch(idname, place=None):
    if isinstance(place, bpy.types.KeyMap):
        for kmi in place.keymap_items:
            if kmi.idname == idname:
                yield kmi
    elif isinstance(place, bpy.types.KeyConfig):
        for keymap in place.keymaps:
            for kmi in KeyMapItemSearch(idname, keymap):
                yield kmi
    else:
        wm = bpy.context.window_manager
        for keyconfig in wm.keyconfigs:
            for kmi in KeyMapItemSearch(idname, keyconfig):
                yield kmi

def IsKeyMapItemEvent(kmi, event):
    event_any = (event.shift or event.ctrl or event.alt or event.oskey)
    event_key_modifier = 'NONE' # no such info in event
    return ((kmi.type == event.type) and
            (kmi.value == event.value) and
            (kmi.shift == event.shift) and
            (kmi.ctrl == event.ctrl) and
            (kmi.alt == event.alt) and
            (kmi.oskey == event.oskey) and
            (kmi.any == event_any) and
            (kmi.key_modifier == event_key_modifier))

def register_keymap_for_mode(mode_name):
    wm = bpy.context.window_manager
    try:
        km = wm.keyconfigs.addon.keymaps[mode_name]
    except:
        km = wm.keyconfigs.addon.keymaps.new(mode_name, space_type='VIEW_3D', region_type='WINDOW')
    kmi = km.keymap_items.new(MouselookNavigation.bl_idname, 'MIDDLEMOUSE', 'ANY', any=True, head=True)

def unregister_keymap_for_mode(mode_name):
    wm = bpy.context.window_manager
    try:
        km = wm.keyconfigs.addon.keymaps[mode_name]
    except:
        return
    for kmi in list(km.keymap_items):
        if kmi.idname == MouselookNavigation.bl_idname:
            km.keymap_items.remove(kmi)

def register_keymaps():
    register_keymap_for_mode('3D View')
    #register_keymap_for_mode('Sculpt')

def unregister_keymaps():
    unregister_keymap_for_mode('3D View')
    #unregister_keymap_for_mode('Sculpt')

class VIEW3D_PT_mouselook_navigation(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_label = "Mouselook Nav."
    
    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        settings = wm.mouselook_navigation_runtime_settings
        
        userprefs = context.user_preferences
        addon_prefs = userprefs.addons[__name__].preferences
        settings = addon_prefs
        
        col = layout.column(True)
        col.prop(settings, "zoom_speed_modifier")
        col.prop(settings, "rotation_speed_modifier")
        col.prop(settings, "fps_speed_modifier")
        
        layout.prop(settings, "fps_horizontal")
        
        box = layout.box()
        row = box.row()
        row.label(text="Orbit snap")
        row.prop(settings, "rotation_snap_autoperspective", text="To Ortho", toggle=True)
        box.prop(settings, "rotation_snap_subdivs", text="Subdivs")
        
        box = layout.box()
        row = box.row()
        row.label(text="Trackball")
        row.prop(settings, "trackball_mode", text="")
        row = box.row(True)
        row.prop(settings, "autolevel_trackball", text="Autolevel", toggle=True)
        cell = row.row(True)
        cell.active = settings.autolevel_trackball
        cell.prop(settings, "autolevel_trackball_up", text="Up", toggle=True)
        
        layout.prop(settings, "autolevel_speed_modifier")
    
    def draw_header(self, context):
        layout = self.layout
        wm = context.window_manager
        settings = wm.mouselook_navigation_runtime_settings
        layout.prop(settings, "is_enabled", text="")

class MouselookNavigationRuntimeSettings(bpy.types.PropertyGroup):
    is_enabled = bpy.props.BoolProperty(name="Enabled", description="Enable/disable Mouselook Navigation", default=True, options={'HIDDEN'})
    '''
    zoom_speed_modifier = bpy.props.FloatProperty(name="Zoom speed", description="Zooming speed", default=1.0)
    rotation_speed_modifier = bpy.props.FloatProperty(name="Rotation speed", description="Rotation speed", default=1.0)
    fps_speed_modifier = bpy.props.FloatProperty(name="FPS speed", description="FPS movement speed", default=1.0)
    fps_horizontal = bpy.props.BoolProperty(name="FPS horizontal", description="Force forward/backward to be in horizontal plane, and up/down to be vertical", default=False)
    trackball_mode = bpy.props.EnumProperty(name="Trackball mode", description="Rotation algorithm used in trackball mode", default='WRAPPED', items=[('BLENDER', 'Blender', 'Blender (buggy!)'), ('WRAPPED', 'Wrapped', 'Wrapped'), ('CENTER', 'Center', 'Center')])
    rotation_snap_subdivs = bpy.props.IntProperty(name="Orbit snap subdivs", description="Intermediate angles used when snapping (1: 90°, 2: 45°, 3: 30°, etc.)", default=1, min=1)
    rotation_snap_autoperspective = bpy.props.BoolProperty(name="Orbit snap->ortho", description="If Auto Perspective is enabled, rotation snapping will automatically switch the view to Ortho", default=True)
    autolevel_trackball = bpy.props.BoolProperty(name="Trackball Autolevel", description="Autolevel in Trackball mode", default=False)
    autolevel_trackball_up = bpy.props.BoolProperty(name="Trackball Autolevel up", description="Try to autolevel 'upright' in Trackball mode", default=False)
    autolevel_speed_modifier = bpy.props.FloatProperty(name="Autolevel speed", description="Autoleveling speed", default=0.0, min=0.0)
    '''

class ThisAddonPreferences(bpy.types.AddonPreferences):
    # this must match the addon name, use '__package__'
    # when defining this in a submodule of a python package.
    bl_idname = __name__
    
    show_crosshair = bpy.props.BoolProperty(name="Show Crosshair", default=True)
    show_zbrush_border = bpy.props.BoolProperty(name="Show ZBrush border", default=True)
    use_blender_colors = bpy.props.BoolProperty(name="Use Blender's colors", default=True)
    color_crosshair_visible = bpy.props.FloatVectorProperty(name="Crosshair (visible)", default=(0.0, 0.0, 0.0), subtype='COLOR', min=0.0, max=1.0)
    color_crosshair_obscured = bpy.props.FloatVectorProperty(name="Crosshair (obscured)", default=(0.0, 0.0, 0.0), subtype='COLOR', min=0.0, max=1.0)
    color_zbrush_border = bpy.props.FloatVectorProperty(name="ZBrush border", default=(0.0, 0.0, 0.0), subtype='COLOR', min=0.0, max=1.0)
    
    #is_enabled = bpy.props.BoolProperty(name="Enabled", description="Enable/disable Mouselook Navigation", default=True, options={'HIDDEN'})
    zoom_speed_modifier = bpy.props.FloatProperty(name="Zoom speed", description="Zooming speed", default=1.0)
    rotation_speed_modifier = bpy.props.FloatProperty(name="Rotation speed", description="Rotation speed", default=1.0)
    fps_speed_modifier = bpy.props.FloatProperty(name="FPS speed", description="FPS movement speed", default=1.0)
    fps_horizontal = bpy.props.BoolProperty(name="FPS horizontal", description="Force forward/backward to be in horizontal plane, and up/down to be vertical", default=False)
    trackball_mode = bpy.props.EnumProperty(name="Trackball mode", description="Rotation algorithm used in trackball mode", default='WRAPPED', items=[('BLENDER', 'Blender', 'Blender (buggy!)'), ('WRAPPED', 'Wrapped', 'Wrapped'), ('CENTER', 'Center', 'Center')])
    rotation_snap_subdivs = bpy.props.IntProperty(name="Orbit snap subdivs", description="Intermediate angles used when snapping (1: 90°, 2: 45°, 3: 30°, etc.)", default=1, min=1)
    rotation_snap_autoperspective = bpy.props.BoolProperty(name="Orbit snap->ortho", description="If Auto Perspective is enabled, rotation snapping will automatically switch the view to Ortho", default=True)
    autolevel_trackball = bpy.props.BoolProperty(name="Trackball Autolevel", description="Autolevel in Trackball mode", default=False)
    autolevel_trackball_up = bpy.props.BoolProperty(name="Trackball Autolevel up", description="Try to autolevel 'upright' in Trackball mode", default=False)
    autolevel_speed_modifier = bpy.props.FloatProperty(name="Autolevel speed", description="Autoleveling speed", default=0.0, min=0.0)
    
    def draw(self, context):
        layout = self.layout
        row = layout.row()
        col = row.column()
        col.prop(self, "use_blender_colors")
        col.prop(self, "show_crosshair")
        col.prop(self, "show_zbrush_border")
        col = row.column()
        col.active = not self.use_blender_colors
        col.row().prop(self, "color_crosshair_visible")
        col.row().prop(self, "color_crosshair_obscured")
        col.row().prop(self, "color_zbrush_border")

def register():
    bpy.utils.register_class(ThisAddonPreferences)
    
    bpy.utils.register_class(MouselookNavigation)
    register_keymaps()
    
    bpy.utils.register_class(MouselookNavigationRuntimeSettings)
    bpy.types.WindowManager.mouselook_navigation_runtime_settings = \
        bpy.props.PointerProperty(type=MouselookNavigationRuntimeSettings)
    bpy.utils.register_class(VIEW3D_PT_mouselook_navigation)

def unregister():
    bpy.utils.unregister_class(VIEW3D_PT_mouselook_navigation)
    if hasattr(bpy.types.WindowManager, "mouselook_navigation_runtime_settings"):
        del bpy.types.WindowManager.mouselook_navigation_runtime_settings
    bpy.utils.unregister_class(MouselookNavigationRuntimeSettings)
    
    unregister_keymaps()
    bpy.utils.unregister_class(MouselookNavigation)
    
    bpy.utils.unregister_class(ThisAddonPreferences)

if __name__ == "__main__":
    unregister_keymaps()
    register()
