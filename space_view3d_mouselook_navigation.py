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
    "version": (0, 3, 0),
    "blender": (2, 7, 0),
    "location": "View3D > MMB/Scrollwheel",
    "warning": "",
    "wiki_url": "http://wiki.blender.org/index.php/Extensions:2.6/Py/"
        "Scripts/3D_interaction/??????????????????????????????????????????????????????????",
    "tracker_url": "https://developer.blender.org/????????????????????????????????????????",
    "category": "3D View"}
#============================================================================#

import bpy
import bgl

from mathutils import Vector, Matrix, Quaternion, Euler

from bpy_extras.view3d_utils import (region_2d_to_location_3d,
                                     location_3d_to_region_2d,
                                     region_2d_to_vector_3d,
                                     region_2d_to_origin_3d,
                                     )

import math
import time

# Middle mouse short click = exit on second click
# Middle mouse drag = exit on release (smart toggle)
# Middle mouse on raycastable surface ("Auto Depth") = orbit around it
# ("Zoom To Mouse Position")
# ("Rotate Around The Selection")
# ESC = cancel & return to previous orientation
# WS AD RF | P: L" IK | UpDown LeftRight PgupPgdown = navigation, fps-style
# Q | { = switch Camera/Gravity coordsystem
# Space = switch ortho/perspective (or LMB/RMB?)
# Left|Right mouse = switch turntable/FPS mode (or Space?)
#    (is there a need for a "trackball" mode?)
# Hold LMB/RMB = pan?
# Shift = pan (move in the camera plane) (a [smart] toggle?)
# Ctrl = move closer to/farther from the orbiting point
# Alt = discrete rotation/movement
# ??? = navigate faster/slower?
# ??? = show/hide crosshair?

"""
Aside from cursor/object lock, there are following "rotate around" options:
* orbiting point (crosshair)
* camera origin (FPS mode)
* selection (manipulator location)
* raycasted point on surface
* active/specific object/element location
* cursor

User may or may not want the camera to center on that point.

Zoom options:
* zoom to orbit point
** zoom to mouse (orbit point just shifts accordingly)
* zoom to current point
* zoom the orbit point to/away from camera

Maybe like this?
Ctrl+scrollwheel = zoom to mouse
Ctrl+MMB = rotate around raycasted point
MMB+Ctrl = zoom to/from orbit point

"""

# also:
# lock to cursor/object
# emulate middle button? (Alt+LMB) (may be automatic)
# invert zoom wheel direction? (may be automatic)
# continuous grab (can be handy for the implementation)?
# double click time (can be handy for the implementation)?
# orbit style
# zoom style

# shift+scrollwheel = move Z
# shift+scrollwheel = move Y
# in fly mode ortho auto-switches to perspective
#   (can be replaced by discrete movement, e.g. shift+alt)

# space_data.region_3d : 3D region in this space, in case of quad view the camera region
# space_data.region_quadviews : 3D regions (the third one defines quad view settings, the forth one is same as ‘region_3d’)

"""
class SpaceRegionView3D_Wrapper:
    def __init__(self, region, space_data, region_data):
        self.region = region # expected type: Region
        self.space_data = space_data # expected type: SpaceView3D
        self.region_data = region_data # expected type: RegionView3D
    
    # === SpaceView3D === #
    @property
    def local_view(self):
        return self.space_data.local_view
    @local_view.setter
    def local_view(self, value):
        self.space_data.local_view = value
    
    @property
    def pivot_point(self):
        return self.space_data.pivot_point
    @pivot_point.setter
    def pivot_point(self, value):
        self.space_data.pivot_point = value
    
    @property
    def transform_orientation(self):
        return self.space_data.transform_orientation
    @transform_orientation.setter
    def transform_orientation(self, value):
        self.space_data.transform_orientation = value
    
    @property
    def grid_scale(self):
        return self.space_data.grid_scale
    @grid_scale.setter
    def grid_scale(self, value):
        self.space_data.grid_scale = value
    
    # === RegionView3D === #
    @property
    def perspective_matrix(self):
        return self.region_data.perspective_matrix
    
    @property
    def view_matrix(self):
        return self.region_data.view_matrix
    @view_matrix.setter
    def view_matrix(self, value):
        self.region_data.view_matrix = value
    
    def update(self):
        self.region_data.update()
"""


class SmartView3D:
    def __init__(self, region, space_data, region_data):
        self.region = region # expected type: Region
        self.space_data = space_data # expected type: SpaceView3D
        self.region_data = region_data # expected type: RegionView3D
        self.use_camera_axes = False
    
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
                    if cam_data.type == 'ORTHO':
                        cam_data.type = 'PERSP'
                    else:
                        cam_data.type = 'ORTHO'
        else:
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
        if self.is_camera and (self.camera.type == 'CAMERA') and (self.camera.data.type == 'ORTHO'):
            return self.camera.data.ortho_scale
        else:
            return self.raw_distance
    def __set(self, value):
        if self.is_camera and (self.camera.type == 'CAMERA') and (self.camera.data.type == 'ORTHO'):
            if self.lock_camera:
                self.camera.data.ortho_scale = value
        else:
            self.raw_distance = value
    distance = property(__get, __set)
    
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
                m = v3d.camera.matrix_world
                m.translation = value - self.forward * rv3d.view_distance
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
        if self.is_camera:
            if not self.use_camera_axes:
                value = value * Quaternion((1, 0, 0), math.pi*0.5)
            if self.lock_camera:
                #m = v3d.camera.matrix_world
                #focus = m.translation + self.forward * rv3d.view_distance
                LRS = v3d.camera.matrix_world.decompose()
                m = MatrixLRS(LRS[0], value, LRS[2])
                forward = -m.col[2].to_3d().normalized() # in camera axes, forward is -Z
                m.translation = self.focus - forward * rv3d.view_distance
                v3d.camera.matrix_world = m
        else:
            self.raw_rotation = value
    rotation = property(__get, __set)
    # TODO: "rotation around viewpoint" mode
    # TODO: "rotation around arbitrary point"
    
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
            pos = self.viewpoint + self.forward * pos
        region = self.region
        rv3d = self.region_data
        return region_2d_to_location_3d(region, rv3d, xy.copy(), pos.copy())
    
    def ray(self, xy): # 0,0 means region's bottom left corner
        region = self.region
        v3d = self.space_data
        rv3d = self.region_data
        
        viewPos = self.viewpoint
        viewDir = self.forward
        
        near = viewPos + viewDir * self.clip_start
        far = viewPos + viewDir * self.clip_end
        
        a = region_2d_to_location_3d(region, rv3d, xy.copy(), near)
        b = region_2d_to_location_3d(region, rv3d, xy.copy(), far)
        
        # When viewed from in-scene camera, near and far
        # planes clip geometry even in orthographic mode.
        clip = rv3d.is_perspective or (rv3d.view_perspective == 'CAMERA')
        
        return a, b, clip
    
    def read_zbuffer(self, xy, wh=(1, 1)): # xy is in window coordinates!
        if isinstance(wh, (int, float)):
            wh = (wh, wh)
        elif len(wh) < 2:
            wh = (wh[0], wh[0])
        x, y, w, h = int(xy[0]), int(xy[1]), int(wh[0]), int(wh[1])
        zbuf = bgl.Buffer(bgl.GL_FLOAT, [w*h])
        bgl.glReadPixels(x, y, w, h, bgl.GL_DEPTH_COMPONENT, bgl.GL_FLOAT, zbuf)
        return zbuf.to_list()
    
    def zbuf_to_depth(self, zbuf):
        near = self.clip_start
        far = self.clip_end
        return (far * near) / (zbuf * (far - near) - far)
    
    def depth(self, xy, region_coords=True):
        if region_coords: # convert to window coords
            xy = xy + Vector((self.region.x, self.region.y))
        return self.zbuf_to_depth(self.read_zbuffer(xy)[0])
    
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
    #"""
    
    '''
    #float phi, si, q1[4], dvec[3], newvec[3];
    
    newvec = Vector((p2x, p2y, -tb_project_to_sphere(TRACKBALLSIZE, p2x, p2y))
    #calctrackballvec(&vod->ar->winrct, x, y, newvec);
    
    dvec = newvec - vod->trackvec
    #sub_v3_v3v3(dvec, newvec, vod->trackvec);
    
    si = dvec.magnitude / (2.0 * TRACKBALLSIZE)
    
    a = vod->trackvec.cross(newvec)
    #cross_v3_v3v3(q1 + 1, vod->trackvec, newvec);
    a.normalize()
    #normalize_v3(q1 + 1);
    
    # Allow for rotation beyond the interval [-pi, pi]
    while (si > 1.0):
        si -= 2.0
    
    # This relation is used instead of
    # - phi = asin(si) so that the angle
    # - of rotation is linearly proportional
    # - to the distance that the mouse is
    # - dragged.
    phi = si * (math.pi / 2.0)
    
    return Quaternion(a, phi)
    #q1[0] = math.cos(phi)
    #mul_v3_fl(q1 + 1, math.sin(phi))
    #mul_qt_qtqt(vod->viewquat, q1, vod->oldquat);
    '''


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

class InputKeyMonitor:
    all_keys = {'NONE', 'LEFTMOUSE', 'MIDDLEMOUSE', 'RIGHTMOUSE', 'BUTTON4MOUSE', 'BUTTON5MOUSE', 'BUTTON6MOUSE', 'BUTTON7MOUSE', 'ACTIONMOUSE', 'SELECTMOUSE', 'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'WHEELINMOUSE', 'WHEELOUTMOUSE', 'EVT_TWEAK_L', 'EVT_TWEAK_M', 'EVT_TWEAK_R', 'EVT_TWEAK_A', 'EVT_TWEAK_S', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE', 'LEFT_CTRL', 'LEFT_ALT', 'LEFT_SHIFT', 'RIGHT_ALT', 'RIGHT_CTRL', 'RIGHT_SHIFT', 'OSKEY', 'GRLESS', 'ESC', 'TAB', 'RET', 'SPACE', 'LINE_FEED', 'BACK_SPACE', 'DEL', 'SEMI_COLON', 'PERIOD', 'COMMA', 'QUOTE', 'ACCENT_GRAVE', 'MINUS', 'SLASH', 'BACK_SLASH', 'EQUAL', 'LEFT_BRACKET', 'RIGHT_BRACKET', 'LEFT_ARROW', 'DOWN_ARROW', 'RIGHT_ARROW', 'UP_ARROW', 'NUMPAD_2', 'NUMPAD_4', 'NUMPAD_6', 'NUMPAD_8', 'NUMPAD_1', 'NUMPAD_3', 'NUMPAD_5', 'NUMPAD_7', 'NUMPAD_9', 'NUMPAD_PERIOD', 'NUMPAD_SLASH', 'NUMPAD_ASTERIX', 'NUMPAD_0', 'NUMPAD_MINUS', 'NUMPAD_ENTER', 'NUMPAD_PLUS', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'PAUSE', 'INSERT', 'HOME', 'PAGE_UP', 'PAGE_DOWN', 'END', 'MEDIA_PLAY', 'MEDIA_STOP', 'MEDIA_FIRST', 'MEDIA_LAST', 'TEXTINPUT', 'WINDOW_DEACTIVATE', 'TIMER', 'TIMER0', 'TIMER1', 'TIMER2', 'TIMER_JOBS', 'TIMER_AUTOSAVE', 'TIMER_REPORT', 'TIMERREGION', 'NDOF_MOTION', 'NDOF_BUTTON_MENU', 'NDOF_BUTTON_FIT', 'NDOF_BUTTON_TOP', 'NDOF_BUTTON_BOTTOM', 'NDOF_BUTTON_LEFT', 'NDOF_BUTTON_RIGHT', 'NDOF_BUTTON_FRONT', 'NDOF_BUTTON_BACK', 'NDOF_BUTTON_ISO1', 'NDOF_BUTTON_ISO2', 'NDOF_BUTTON_ROLL_CW', 'NDOF_BUTTON_ROLL_CCW', 'NDOF_BUTTON_SPIN_CW', 'NDOF_BUTTON_SPIN_CCW', 'NDOF_BUTTON_TILT_CW', 'NDOF_BUTTON_TILT_CCW', 'NDOF_BUTTON_ROTATE', 'NDOF_BUTTON_PANZOOM', 'NDOF_BUTTON_DOMINANT', 'NDOF_BUTTON_PLUS', 'NDOF_BUTTON_MINUS', 'NDOF_BUTTON_ESC', 'NDOF_BUTTON_ALT', 'NDOF_BUTTON_SHIFT', 'NDOF_BUTTON_CTRL', 'NDOF_BUTTON_1', 'NDOF_BUTTON_2', 'NDOF_BUTTON_3', 'NDOF_BUTTON_4', 'NDOF_BUTTON_5', 'NDOF_BUTTON_6', 'NDOF_BUTTON_7', 'NDOF_BUTTON_8', 'NDOF_BUTTON_9', 'NDOF_BUTTON_10', 'NDOF_BUTTON_A', 'NDOF_BUTTON_B', 'NDOF_BUTTON_C'}
    all_modifiers = {'alt', 'ctrl', 'oskey', 'shift'}
    all_events = {'ANY', 'NOTHING', 'PRESS', 'RELEASE', 'CLICK', 'DOUBLE_CLICK', 'NORTH', 'NORTH_EAST', 'EAST', 'SOUTH_EAST', 'SOUTH', 'SOUTH_WEST', 'WEST', 'NORTH_WEST'}
    
    def __init__(self, event=None):
        self.event = ""
        self.states = {}
        if event is not None:
            self.update(event)
    
    def __getitem__(self, name):
        if ":" in name:
            return self.event == name
        return self.states.setdefault(name, False)
    
    def __setitem__(self, name, state):
        self.states[name] = state
    
    def update(self, event):
        if event.value == 'PRESS':
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
        return check
    
    @staticmethod
    def combine_key_parts(key, keyset):
        elements = key.split()
        combined0 = "".join(elements)
        combined1 = "_".join(elements)
        if combined0 in keyset:
            return combined0
        elif combined1 in keyset:
            return combined1
        return ""
    
    @classmethod
    def parse_keys(cls, keys_string):
        parts = keys_string.split(":")
        keys_string = parts[0]
        event_id = ""
        if len(parts) > 1:
            event_id = cls.combine_key_parts(parts[1].upper(), cls.all_events)
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
            
            key_id = cls.combine_key_parts(key.upper(), cls.all_keys)
            modifier_id = cls.combine_key_parts(key.lower(), cls.all_modifiers)
            
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

class MouselookNavigation(bpy.types.Operator):
    """Mouselook navigation"""
    bl_idname = "view3d.mouselook_navigation"
    bl_label = "Mouselook navigation"
    bl_options = {'GRAB_POINTER', 'BLOCKING'} # IMPORTANT! otherwise Continuous Grab won't work
    
    modes = ['ORBIT', 'PAN', 'DOLLY', 'ZOOM', 'FLY', 'FPS']
    transitions = ['NONE:ORBIT', 'NONE:PAN', 'NONE:DOLLY', 'NONE:ZOOM', 'NONE:FLY', 'NONE:FPS', 'ORBIT:PAN', 'ORBIT:DOLLY', 'ORBIT:ZOOM', 'ORBIT:FLY', 'ORBIT:FPS', 'PAN:DOLLY', 'PAN:ZOOM', 'DOLLY:ZOOM', 'FLY:FPS']
    _transitions = ['NONE:ORBIT', 'NONE:PAN', 'NONE:DOLLY', 'NONE:ZOOM', 'ORBIT:FLY', 'ORBIT:FPS', 'PAN:DOLLY', 'PAN:ZOOM', 'DOLLY:ZOOM', 'FLY:FPS']
    
    default_mode = bpy.props.EnumProperty(items=[(m, m, m) for m in modes], name="Default mode", description="Default mode", default='ORBIT')
    allowed_transitions = bpy.props.EnumProperty(items=[(t, t, t) for t in transitions], name="Transitions", description="Allowed transitions between modes", default=set(_transitions), options={'ENUM_FLAG'})
    
    zbrush_mode = bpy.props.BoolProperty(name="ZBrush mode", description="The operator would be invoked only if mouse is over empty space or close to region border", default=True)
    
    ortho_unrotate = bpy.props.BoolProperty(name="Ortho unrotate", description="In Ortho mode, rotation is abandoned if another mode is selected", default=True)
    
    def _keyprop(name, default_keys):
        return bpy.props.StringProperty(name=name, description=name, default=default_keys)
    str_keys_confirm = _keyprop("Confirm", "Ret, Numpad Enter")
    str_keys_cancel = _keyprop("Cancel", "Esc")
    str_keys_rotmode_switch = _keyprop("Rotation Mode Switch", "Tab: Press")
    str_keys_orbit = _keyprop("Orbit", "") # main operator key (MMB) by default
    str_keys_orbit_snap = _keyprop("Orbit Snap", "Shift")
    #str_keys_pan = _keyprop("Pan", "Right Mouse, Shift")
    #str_keys_dolly = _keyprop("Dolly", "Ctrl")
    #str_keys_zoom = _keyprop("Zoom", "Left Mouse, Alt")
    str_keys_pan = _keyprop("Pan", "Alt")
    str_keys_dolly = _keyprop("Dolly", "")
    #str_keys_zoom = _keyprop("Zoom", "!Alt")
    str_keys_zoom = _keyprop("Zoom", "Ctrl")
    str_keys_FPS_forward = _keyprop("FPS forward", "W")
    str_keys_FPS_back = _keyprop("FPS back", "S")
    str_keys_FPS_left = _keyprop("FPS left", "A")
    str_keys_FPS_right = _keyprop("FPS right", "D")
    str_keys_FPS_up = _keyprop("FPS up", "E, R")
    str_keys_FPS_down = _keyprop("FPS down", "Q, F")
    str_keys_fps_acceleration = _keyprop("FPS fast", "")
    str_keys_fps_slowdown = _keyprop("FPS slow", "")
    
    def create_keycheckers(self, event):
        self.keys_invoke = self.km.keychecker(event.type)
        self.keys_confirm = self.km.keychecker(self.str_keys_confirm)
        self.keys_cancel = self.km.keychecker(self.str_keys_cancel)
        self.keys_rotmode_switch = self.km.keychecker(self.str_keys_rotmode_switch)
        self.keys_orbit = self.km.keychecker(self.str_keys_orbit)
        self.keys_orbit_snap = self.km.keychecker(self.str_keys_orbit_snap)
        self.keys_pan = self.km.keychecker(self.str_keys_pan)
        self.keys_dolly = self.km.keychecker(self.str_keys_dolly)
        self.keys_zoom = self.km.keychecker(self.str_keys_zoom)
        self.keys_FPS_forward = self.km.keychecker(self.str_keys_FPS_forward)
        self.keys_FPS_back = self.km.keychecker(self.str_keys_FPS_back)
        self.keys_FPS_left = self.km.keychecker(self.str_keys_FPS_left)
        self.keys_FPS_right = self.km.keychecker(self.str_keys_FPS_right)
        self.keys_FPS_up = self.km.keychecker(self.str_keys_FPS_up)
        self.keys_FPS_down = self.km.keychecker(self.str_keys_FPS_down)
        self.keys_fps_acceleration = self.km.keychecker(self.str_keys_fps_acceleration)
        self.keys_fps_slowdown = self.km.keychecker(self.str_keys_fps_slowdown)
    
    @classmethod
    def poll(cls, context):
        wm = context.window_manager
        if not wm.mouselook_navigation_runtime_settings.is_enabled:
            return False
        return (context.space_data.type == 'VIEW_3D')
    
    def execute(self, context):
        v3d = context.space_data
        rv3d = v3d.region_3d
        
        #rv3d.view_location = self._initial_location + Vector(self.offset)
    
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
        
        userprefs = context.user_preferences
        drag_threshold = userprefs.inputs.drag_threshold
        tweak_threshold = userprefs.inputs.tweak_threshold
        mouse_double_click_time = userprefs.inputs.mouse_double_click_time / 1000.0
        rotate_method = userprefs.inputs.view_rotate_method
        invert_mouse_zoom = userprefs.inputs.invert_mouse_zoom
        use_auto_perspective = userprefs.view.use_auto_perspective
        
        self.km.update(event)
        mouse_prev = Vector((event.mouse_prev_x, event.mouse_prev_y))
        mouse = Vector((event.mouse_x, event.mouse_y))
        mouse_offset = mouse - self.mouse0
        mouse_delta = mouse - mouse_prev
        
        clock = time.clock()
        dt = 0.01
        speed_move = 2.5 * self.sv.distance * dt
        speed_zoom = 1 * dt
        speed_rot = 1 * dt
        speed_euler = Vector((-1, 1)) * dt
        speed_autolevel = 1 * dt
        
        if invert_mouse_zoom:
            speed_zoom *= -1
        
        speed_move *= self.fps_speed_modifier
        speed_zoom *= self.zoom_speed_modifier
        speed_rot *= self.rotation_speed_modifier
        speed_euler *= self.rotation_speed_modifier
        speed_autolevel *= self.autolevel_speed_modifier
        
        confirm = self.keys_confirm() or self.keys_invoke(False)
        cancel = self.keys_cancel()
        
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
        
        fps_speed = self.calc_FPS_speed()
        
        self.detect_mode_changes()
        
        mode = self.mode
        if not self.sv.is_perspective:
            if mode == 'DOLLY':
                mode = 'ZOOM'
            
            # The goal is to make it easy to pan view without accidentally rotating it
            if self.ortho_unrotate:
                if mode in ('PAN', 'DOLLY', 'ZOOM'):
                    # forbid transitions back to orbit
                    self.allowed_transitions = self.allowed_transitions.difference(
                        {'ORBIT:PAN', 'ORBIT:DOLLY', 'ORBIT:ZOOM'})
                    # snap to original orientation
                    self.rot = self.rot0.copy()
                    self.euler = self.euler0.copy()
                    if rotate_method == 'TURNTABLE':
                        self.sv.turntable_euler = self.euler # for turntable
                    else:
                        self.sv.rotation = self.rot # for trackball
        
        if (event.type == 'MOUSEMOVE') or (event.type == 'INBETWEEN_MOUSEMOVE'):
            #zbuf = self.sv.read_zbuffer(mouse)[0]
            #zcam = self.sv.zbuf_to_depth(zbuf)
            
            if mode == 'ORBIT':
                # snapping trackball rotation is problematic (I don't know how to do it)
                if (rotate_method == 'TURNTABLE') or is_orbit_snap:
                    self.change_euler(mouse_delta.y * speed_euler.y, mouse_delta.x * speed_euler.x, 0)
                else: # 'TRACKBALL'
                    self.change_rot_mouse(mouse_delta, mouse, speed_rot)
                if is_orbit_snap:
                    if self.rotation_snap_autoperspective and use_auto_perspective:
                        print(self.sv.is_perspective)
                        self.sv.is_perspective = False
                    self.snap_rotation(self.rotation_snap_subdivs)
            elif mode == 'PAN':
                self.change_pos_mouse(mouse_delta, False)
            elif mode == 'DOLLY':
                self.change_pos_mouse(mouse_delta, True)
            elif mode == 'ZOOM':
                self.change_distance((mouse_delta.y - mouse_delta.x) * speed_zoom)
        
        if event.type.startswith('TIMER'):
            if speed_autolevel > 0: #rotate_method == 'TURNTABLE':
                if (mode != 'ORBIT') or (not is_orbit_snap):
                    if rotate_method == 'TURNTABLE':
                        self.change_euler(0, 0, speed_autolevel, False)
                    elif self.autolevel_trackball:
                        speed_autolevel *= 1.0 - abs(self.sv.forward.z)
                        self.change_euler(0, 0, speed_autolevel, self.autolevel_trackball_up)
            
            if fps_speed.magnitude > 0:
                if not self.sv.is_perspective:
                    self.change_distance(fps_speed.y * speed_zoom*(-4))
                    fps_speed.y = 0
                self.change_pos(fps_speed.x, fps_speed.y, fps_speed.z, speed_move)
            
            #txt = "{} {} {} {}".format(event.type, move_x, move_y, move_z)
            #context.area.header_text_set("%s" % (str(self.pos)))
            #context.area.header_text_set(txt)
            #print(txt)
            
            context.area.tag_redraw()
        
        if confirm:
            self.cleanup(context)
            return {'FINISHED'}
        elif cancel:
            self.revert_changes()
            self.cleanup(context)
            return {'CANCELLED'}
        
        return {'RUNNING_MODAL'}
    
    def calc_FPS_speed(self):
        move_forward = self.keys_FPS_forward()
        move_back = self.keys_FPS_back()
        move_left = self.keys_FPS_left()
        move_right = self.keys_FPS_right()
        move_up = self.keys_FPS_up()
        move_down = self.keys_FPS_down()
        
        move_x = int(move_right) - int(move_left)
        move_y = int(move_forward) - int(move_back)
        move_z = int(move_up) - int(move_down)
        
        movement_accelerate = self.keys_fps_acceleration()
        movement_slowdown = self.keys_fps_slowdown()
        
        move_speedup = int(movement_accelerate) - int(movement_slowdown)
        
        return Vector((move_x, move_y, move_z)) * (5 ** move_speedup)
    
    def change_distance(self, delta):
        log_zoom = math.log(max(self.sv.distance, self.min_distance), 2)
        self.sv.distance = math.pow(2, log_zoom + delta)
    
    def change_pos(self, dx, dy, dz, speed=1.0):
        xdir, ydir, zdir = self.sv.right, self.sv.forward, self.sv.up
        if (self.rotate_method == 'TURNTABLE') and self.fps_horizontal:
            ysign = (-1.0 if zdir.z < 0 else 1.0)
            zdir = Vector((0, 0, 1))
            ydir = Quaternion(zdir, self.euler.z) * Vector((0, 1, 0))
            xdir = ydir.cross(zdir)
            ydir *= ysign
        self.pos += (xdir*dx + ydir*dy + zdir*dz) * speed
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
    
    def change_rot_mouse(self, mouse_delta, mouse, speed_rot):
        if self.trackball_mode == 'CENTER':
            mouse_delta *= speed_rot
            spin = self.mouse_delta_movement(mouse_delta).normalized()
            axis = spin.cross(self.sv.forward)
            self.rot = Quaternion(axis, mouse_delta.magnitude) * self.rot
        elif self.trackball_mode == 'WRAPPED':
            mouse_delta *= speed_rot
            cdir = Vector((0, -1, 0))
            tv, x_neg, y_neg = self.trackball_vector(mouse)
            r = cdir.rotation_difference(tv)
            spin = r * Vector((mouse_delta.x, 0, mouse_delta.y))
            axis = spin.cross(tv)
            axis = self.sv.matrix.to_3x3() * axis
            self.rot = Quaternion(axis, mouse_delta.magnitude) * self.rot
        else:
            region = self.sv.region
            mouse -= Vector((region.x, region.y))
            halfsize = Vector((region.width, region.height))*0.5
            p1 = (mouse - mouse_delta) - halfsize
            p2 = (mouse) - halfsize
            p1 = Vector((p1.x/halfsize.x, p1.y/halfsize.y))
            p2 = Vector((p2.x/halfsize.x, p2.y/halfsize.y))
            q = trackball(p1.x, p1.y, p2.x, p2.y, 1.1)
            
            """
            TRACKBALLSIZE = 1.1
            #region = self.sv.region
            #halfsize = Vector((region.width, region.height))*0.5
            
            pmouse = mouse - mouse_delta
            px = (halfsize.x - pmouse.x) / (region.width / 4)
            py = (halfsize.y - pmouse.y) / (region.height / 2)
            #vod_trackvec = Vector((px, py, -tb_project_to_sphere(TRACKBALLSIZE, px, py)))
            vod_trackvec = Vector((px, -tb_project_to_sphere(TRACKBALLSIZE, px, py), py))
            #vod_trackvec = self.vod_trackvec
            
            #x = BLI_rcti_cent_x(rect) - mx;
            #x /= (float)(BLI_rcti_size_x(rect) / 4);
            #y = BLI_rcti_cent_y(rect) - my;
            #y /= (float)(BLI_rcti_size_y(rect) / 2);
            
            x = (halfsize.x - mouse.x) / (region.width / 4)
            y = (halfsize.y - mouse.y) / (region.height / 2)
            
            #float phi, si, q1[4], dvec[3], newvec[3];
            
            #newvec = Vector((x, y, -tb_project_to_sphere(TRACKBALLSIZE, x, y)))
            newvec = Vector((x, -tb_project_to_sphere(TRACKBALLSIZE, x, y), y))
            #calctrackballvec(&vod->ar->winrct, x, y, newvec);
            
            dvec = newvec - vod_trackvec
            #sub_v3_v3v3(dvec, newvec, vod->trackvec);
            
            si = dvec.magnitude / (2.0 * TRACKBALLSIZE)
            
            a = vod_trackvec.cross(newvec)
            #cross_v3_v3v3(q1 + 1, vod->trackvec, newvec);
            a.normalize()
            #normalize_v3(q1 + 1);
            
            # Allow for rotation beyond the interval [-pi, pi]
            while (si > 1.0):
                si -= 2.0
            
            # This relation is used instead of
            # - phi = asin(si) so that the angle
            # - of rotation is linearly proportional
            # - to the distance that the mouse is
            # - dragged.
            phi = si * (math.pi / 2.0)
            
            #q = Quaternion(a, phi)
            #q1[0] = math.cos(phi)
            #mul_v3_fl(q1 + 1, math.sin(phi))
            #mul_qt_qtqt(vod->viewquat, q1, vod->oldquat);
            
            #vod_trackvec.y = 0
            #newvec.y = 0
            
            vod_trackvec.normalize()
            newvec.normalize()
            #print("{} -> {}".format(vod_trackvec, newvec))
            #q = vod_trackvec.rotation_difference(newvec)
            q = newvec.rotation_difference(vod_trackvec)
            """
            
            axis, angle = q.to_axis_angle()
            #print("q: {}, {}".format(axis, angle))
            axis = self.sv.matrix.to_3x3() * axis
            q = Quaternion(axis, angle * speed_rot*200)
            self.rot = q * self.rot
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
    
    def detect_mode_changes(self):
        for name in self.mode_keys:
            self.detect_mode_change(name)
    def detect_mode_change(self, name):
        is_on = self.mode_keys[name]()
        delta_on = int(is_on) - int(self.mode_prev_state[name])
        
        if delta_on > 0:
            if self.transition_allowed(self.mode, name):
                self.mode_stack_remove(name)
                self.modes_stack.append(name) # move to top
                self.mode = name
        elif delta_on < 0:
            if self.mode != name:
                self.mode_stack_remove(name)
            else:
                self.find_allowed_transition()
        
        self.mode_prev_state[name] = is_on
    
    def mode_stack_remove(self, name):
        if name in self.modes_stack:
            self.modes_stack.remove(name)
    
    def find_allowed_transition(self):
        for i in range(len(self.modes_stack)-1, -1, -1):
            name = self.modes_stack[i]
            if self.transition_allowed(self.mode, name):
                self.mode = name
                self.modes_stack = self.modes_stack[:i+1]
    
    def transition_allowed(self, mode0, mode1):
        is_allowed = (mode0+":"+mode1) in self.allowed_transitions
        is_allowed |= (mode1+":"+mode0) in self.allowed_transitions
        return is_allowed
    
    def calc_zbrush_border(self):
        region = self.sv.region
        wrk_sz = min(region.width, region.height)
        return max(wrk_sz*0.05, 16)
    
    def invoke(self, context, event):
        wm = context.window_manager
        userprefs = context.user_preferences
        region = context.region
        v3d = context.space_data
        rv3d = context.region_data
        
        region_pos = Vector((region.x, region.y))
        region_size = Vector((region.width, region.height))
        
        self.km = InputKeyMonitor(event)
        mouse_prev = Vector((event.mouse_prev_x, event.mouse_prev_y))
        mouse = Vector((event.mouse_x, event.mouse_y))
        mouse_delta = mouse - mouse_prev
        mouse_region = mouse - region_pos
        
        self.sv = SmartView3D(context.region, context.space_data, context.region_data)
        zbuf = self.sv.read_zbuffer(mouse)[0]
        zcam = self.sv.zbuf_to_depth(zbuf)
        ray_data = self.sv.ray(mouse_region)
        raycast_result = context.scene.ray_cast(ray_data[0], ray_data[1])
        
        self.create_keycheckers(event)
        
        
        TRACKBALLSIZE = 1.1
        region = self.sv.region
        halfsize = Vector((region.width, region.height))*0.5
        
        x = (halfsize.x - mouse.x) / (region.width / 4)
        y = (halfsize.y - mouse.y) / (region.height / 2)
        self.vod_trackvec = Vector((x, y, -tb_project_to_sphere(TRACKBALLSIZE, x, y)))
        
        
        
        #if rv3d.view_perspective == 'CAMERA':
        #    rv3d.view_perspective = 'PERSP'
        
        self.mode = 'NONE'
        self.modes_stack = []
        self.mode_keys = {'ORBIT':self.keys_orbit, 'PAN':self.keys_pan, 'DOLLY':self.keys_dolly, 'ZOOM':self.keys_zoom}
        self.mode_prev_state = {'ORBIT':False, 'PAN':False, 'DOLLY':False, 'ZOOM':False}
        self.detect_mode_changes()
        if self.mode == 'NONE':
            if self.zbrush_mode:
                # In Sculpt mode, zbuffer seems to be cleared!
                # Also, zbuf can be written by non-geometry, which is probably not desirable
                is_over_obj = raycast_result[0]# or (zbuf < 1.0)
                mouse_region_11 = region_size - mouse_region
                wrk_x = min(mouse_region.x, mouse_region_11.x)
                wrk_y = min(mouse_region.y, mouse_region_11.y)
                wrk_pos = min(wrk_x, wrk_y)
                if is_over_obj and (wrk_pos > self.calc_zbrush_border()):
                    return {'PASS_THROUGH'}
            self.mode = self.default_mode
            self.modes_stack = [self.mode]
        
        self.fps_horizontal = wm.mouselook_navigation_runtime_settings.fps_horizontal
        self.trackball_mode = wm.mouselook_navigation_runtime_settings.trackball_mode
        self.fps_speed_modifier = wm.mouselook_navigation_runtime_settings.fps_speed_modifier
        self.zoom_speed_modifier = wm.mouselook_navigation_runtime_settings.zoom_speed_modifier
        self.rotation_snap_subdivs = wm.mouselook_navigation_runtime_settings.rotation_snap_subdivs
        self.rotation_snap_autoperspective = wm.mouselook_navigation_runtime_settings.rotation_snap_autoperspective
        self.rotation_speed_modifier = wm.mouselook_navigation_runtime_settings.rotation_speed_modifier
        self.autolevel_trackball = wm.mouselook_navigation_runtime_settings.autolevel_trackball
        self.autolevel_trackball_up = wm.mouselook_navigation_runtime_settings.autolevel_trackball_up
        self.autolevel_speed_modifier = wm.mouselook_navigation_runtime_settings.autolevel_speed_modifier
        
        self.prev_orbit_snap = False
        self.min_distance = 2 ** -10
        self.dolly_mode = True # make this a user preference
        self.threshold_reached = False
        self.prev_pan_dolly = None
        self.not_pan_dolly_yet = True
        self.pan_dolly_snaps_back = False
        
        self._clock0 = time.clock()
        self._continuous0 = userprefs.inputs.use_mouse_continuous
        self._mouse0 = Vector((event.mouse_x, event.mouse_y))
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
        
        userprefs.inputs.use_mouse_continuous = True
        
        self.register_handlers(context)
        
        # We need the view to redraw so that crosshair would appear
        # immediately after user presses MMB
        context.area.header_text_set()
        context.area.tag_redraw()
        
        return {'RUNNING_MODAL'}
    
    def revert_changes(self):
        self.sv.rotation = self._rot0
        self.sv.distance = self._distance0
        self.sv.focus = self._pos0
    
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
        wm.event_timer_remove(self._timer)
        if self._handle_view is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_view, 'WINDOW')
        if self._handle_px is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle_px, 'WINDOW')


def draw_crosshair(self, context):
    region = context.region
    v3d = context.space_data
    rv3d = context.region_data
    
    if self.sv.region_data != context.region_data:
        return
    
    if self.sv.is_camera and not self.sv.lock_camera:
        return # camera can't be manipulated, so crosshair is meaningless here
    
    userprefs = context.user_preferences
    color = userprefs.themes[0].view_3d.view_overlay
    
    depth_test_prev = gl_get(bgl.GL_DEPTH_TEST)
    depth_func_prev = gl_get(bgl.GL_DEPTH_FUNC)
    depth_mask_prev = gl_get(bgl.GL_DEPTH_WRITEMASK)
    line_stipple_prev = gl_get(bgl.GL_LINE_STIPPLE)
    color_prev = gl_get(bgl.GL_COLOR)
    blend_prev = gl_get(bgl.GL_BLEND)
    line_width_prev = gl_get(bgl.GL_LINE_WIDTH)
    
    region_center = Vector((region.width*0.5, region.height*0.5))
    focus_proj = self.sv.project(self.sv.focus)
    if (focus_proj - region_center).magnitude < 2:
        focus_proj = region_center
    if self.sv.is_camera and (not self.sv.is_perspective): # Somewhy Blender behaves like this
        focus_proj = region_center # in case camera has non-zero shift
    focus_proj = snap_pixel_vector(focus_proj)
    
    alpha = 1.0
    if self.pan_dolly_snaps_back:
        alpha = 0.35
    
    l0, l1 = 16, 25
    lines = [(Vector((0, l0)), Vector((0, l1))), (Vector((0, -l0)), Vector((0, -l1))),
             (Vector((l0, 0)), Vector((l1, 0))), (Vector((-l0, 0)), Vector((-l1, 0)))]
    dist = min(max(self.sv.distance, self.sv.clip_start*1.01), self.sv.clip_end*0.99)
    lines = [(self.sv.unproject(p0 + focus_proj, dist, True),
              self.sv.unproject(p1 + focus_proj, dist, True)) for p0, p1 in lines]
    
    gl_enable(bgl.GL_BLEND, True)
    gl_enable(bgl.GL_LINE_STIPPLE, False)
    gl_enable(bgl.GL_DEPTH_WRITEMASK, False)
    gl_enable(bgl.GL_DEPTH_TEST, True)
    
    bgl.glDepthFunc(bgl.GL_LEQUAL)
    bgl.glColor4f(color[0], color[1], color[2], 1.0*alpha)
    bgl.glLineWidth(1)
    bgl.glBegin(bgl.GL_LINES)
    for p0, p1 in lines:
        bgl.glVertex3f(p0[0], p0[1], p0[2])
        bgl.glVertex3f(p1[0], p1[1], p1[2])
    bgl.glEnd()
    
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
    draw_crosshair(self, context)

def draw_callback_px(self, context):
    region = context.region
    v3d = context.space_data
    rv3d = context.region_data
    
    if self.sv.region_data != context.region_data:
        return
    
    if self.zbrush_mode:
        blend_prev = gl_get(bgl.GL_BLEND)
        gl_enable(bgl.GL_BLEND, True)
        w, h = float(region.width), float(region.height)
        border = self.calc_zbrush_border()
        userprefs = context.user_preferences
        color = userprefs.themes[0].view_3d.view_overlay
        bgl.glColor4f(color[0], color[1], color[2], 0.5)
        bgl.glBegin(bgl.GL_LINE_LOOP)
        bgl.glVertex2f(border, border)
        bgl.glVertex2f(w-border, border)
        bgl.glVertex2f(w-border, h-border)
        bgl.glVertex2f(border, h-border)
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
    kmi = km.keymap_items.new(MouselookNavigation.bl_idname, 'MIDDLEMOUSE', 'PRESS', any=True, head=True)
    #kmi = km.keymap_items.new(MouselookNavigation.bl_idname, 'LEFTMOUSE', 'PRESS', any=True, head=True)

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
    bl_label = "Mouselook Navigation"
    
    """
    @classmethod
    def poll(cls, context):
        return (context.space_data and context.active_object)
    """
    
    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        layout.prop(wm.mouselook_navigation_runtime_settings, "is_enabled")
        layout.prop(wm.mouselook_navigation_runtime_settings, "fps_horizontal")
        layout.prop(wm.mouselook_navigation_runtime_settings, "fps_speed_modifier")
        layout.prop(wm.mouselook_navigation_runtime_settings, "zoom_speed_modifier")
        layout.prop(wm.mouselook_navigation_runtime_settings, "trackball_mode")
        layout.prop(wm.mouselook_navigation_runtime_settings, "rotation_snap_subdivs")
        layout.prop(wm.mouselook_navigation_runtime_settings, "rotation_snap_autoperspective")
        layout.prop(wm.mouselook_navigation_runtime_settings, "rotation_speed_modifier")
        layout.prop(wm.mouselook_navigation_runtime_settings, "autolevel_trackball")
        layout.prop(wm.mouselook_navigation_runtime_settings, "autolevel_trackball_up")
        layout.prop(wm.mouselook_navigation_runtime_settings, "autolevel_speed_modifier")

class MouselookNavigationRuntimeSettings(bpy.types.PropertyGroup):
    is_enabled = bpy.props.BoolProperty(name="Enabled", default=True, options={'HIDDEN'})
    fps_horizontal = bpy.props.BoolProperty(name="FPS horizontal", default=False)
    fps_speed_modifier = bpy.props.FloatProperty(name="FPS speed", default=1.0)
    zoom_speed_modifier = bpy.props.FloatProperty(name="Zoom speed", default=1.0)
    trackball_mode = bpy.props.EnumProperty(items=[('BLENDER', 'Blender', 'Blender'), ('WRAPPED', 'Wrapped', 'Wrapped'), ('CENTER', 'Center', 'Center')], name="Trackball mode", default='WRAPPED')
    rotation_snap_subdivs = bpy.props.IntProperty(name="Orbit snap subdivs", default=1, min=1)
    rotation_snap_autoperspective = bpy.props.BoolProperty(name="Orbit snap->ortho", default=True)
    rotation_speed_modifier = bpy.props.FloatProperty(name="Rotation speed", default=1.0)
    autolevel_trackball = bpy.props.BoolProperty(name="Trackball Autolevel", default=False)
    autolevel_trackball_up = bpy.props.BoolProperty(name="Trackball Autolevel up", default=False)
    autolevel_speed_modifier = bpy.props.FloatProperty(name="Autolevel speed", description="Autoleveling speed", default=0.0, min=0.0)

def register():
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

if __name__ == "__main__":
    unregister_keymaps()
    register()
