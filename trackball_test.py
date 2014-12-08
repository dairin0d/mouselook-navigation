import bpy
from mathutils import Vector, Quaternion

import math

### UTILITY STUFF ###
USER_TRACKBALL = True

def WM_event_add_modal_handler(C, op):
    wm = C.window_manager
    wm.modal_handler_add(op)

def ED_region_tag_redraw(ar): # ar is region
    ar.tag_redraw()# I'm guessing here

def BLI_rcti_cent_x(rect): # Center of the rect?
    return (rect.x + rect.width) * 0.5 # I'm guessing here

def BLI_rcti_cent_y(rect): # Center of the rect?
    return (rect.y + rect.height) * 0.5 # I'm guessing here

def BLI_rcti_size_x(rect): # Size of the rect?
    return rect.width # I'm guessing here

def BLI_rcti_size_y(rect): # Size of the rect?
    return rect.height # I'm guessing here

sqrt = math.sqrt

M_PI = math.pi
M_SQRT1_2 = math.sqrt(0.5)
M_SQRT2 = math.sqrt(2)
#############################################################################


### calctrackballvec() ###
"""
#define TRACKBALLSIZE  (1.1)

static void calctrackballvec(const rcti *rect, int mx, int my, float vec[3])
{
	float x, y, radius, d, z, t;

	radius = TRACKBALLSIZE;

	/* normalize x and y */
	x = BLI_rcti_cent_x(rect) - mx;
	x /= (float)(BLI_rcti_size_x(rect) / 4);
	y = BLI_rcti_cent_y(rect) - my;
	y /= (float)(BLI_rcti_size_y(rect) / 2);

	d = sqrt(x * x + y * y);
	if (d < radius * (float)M_SQRT1_2) { /* Inside sphere */
		z = sqrt(radius * radius - d * d);
	}
	else { /* On hyperbola */
		t = radius / (float)M_SQRT2;
		z = t * t / d;
	}

	vec[0] = x;
	vec[1] = y;
	vec[2] = -z;     /* yah yah! */
}
"""

TRACKBALLSIZE = 1.1

#static void calctrackballvec(const rcti *rect, int mx, int my, float vec[3])
def calctrackballvec(rect, mx, my, vec):
    #float x, y, radius, d, z, t;
    
    radius = TRACKBALLSIZE
    
    # normalize x and y
    x = BLI_rcti_cent_x(rect) - mx
    x /= float(BLI_rcti_size_x(rect) / 4)
    y = BLI_rcti_cent_y(rect) - my
    y /= float(BLI_rcti_size_y(rect) / 2)
    
    d = sqrt(x * x + y * y)
    if (d < radius * M_SQRT1_2): # Inside sphere
        z = sqrt(radius * radius - d * d)
    else: # On hyperbola
        t = radius / M_SQRT2
        z = t * t / d
    
    vec[0] = x
    vec[1] = y
    vec[2] = -z # yah yah!
#############################################################################


### viewrotate_apply() ###
"""
static void viewrotate_apply(ViewOpsData *vod, int x, int y)
{
	RegionView3D *rv3d = vod->rv3d;

	rv3d->view = RV3D_VIEW_USER; /* need to reset every time because of view snapping */

	if (U.flag & USER_TRACKBALL) {
		float phi, si, q1[4], dvec[3], newvec[3];

		calctrackballvec(&vod->ar->winrct, x, y, newvec);

		sub_v3_v3v3(dvec, newvec, vod->trackvec);

		si = len_v3(dvec);
		si /= (float)(2.0 * TRACKBALLSIZE);

		cross_v3_v3v3(q1 + 1, vod->trackvec, newvec);
		normalize_v3(q1 + 1);

		/* Allow for rotation beyond the interval [-pi, pi] */
		while (si > 1.0f)
			si -= 2.0f;

		/* This relation is used instead of
		 * - phi = asin(si) so that the angle
		 * - of rotation is linearly proportional
		 * - to the distance that the mouse is
		 * - dragged. */
		phi = si * (float)(M_PI / 2.0);

		q1[0] = cos(phi);
		mul_v3_fl(q1 + 1, sin(phi));
		mul_qt_qtqt(vod->viewquat, q1, vod->oldquat);

		if (vod->use_dyn_ofs) {
			/* compute the post multiplication quat, to rotate the offset correctly */
			conjugate_qt_qt(q1, vod->oldquat);
			mul_qt_qtqt(q1, q1, vod->viewquat);

			conjugate_qt(q1); /* conj == inv for unit quat */
			copy_v3_v3(rv3d->ofs, vod->ofs);
			sub_v3_v3(rv3d->ofs, vod->dyn_ofs);
			mul_qt_v3(q1, rv3d->ofs);
			add_v3_v3(rv3d->ofs, vod->dyn_ofs);
		}
	}
	else {
		/* New turntable view code by John Aughey */
		float q1[4];
		float m[3][3];
		float m_inv[3][3];
		const float zvec_global[3] = {0.0f, 0.0f, 1.0f};
		float xaxis[3];

		/* Sensitivity will control how fast the viewport rotates.  0.007 was
		 * obtained experimentally by looking at viewport rotation sensitivities
		 * on other modeling programs. */
		/* Perhaps this should be a configurable user parameter. */
		const float sensitivity = 0.007f;

		/* Get the 3x3 matrix and its inverse from the quaternion */
		quat_to_mat3(m, vod->viewquat);
		invert_m3_m3(m_inv, m);

		/* avoid gimble lock */
#if 1
		if (len_squared_v3v3(zvec_global, m_inv[2]) > 0.001f) {
			float fac;
			cross_v3_v3v3(xaxis, zvec_global, m_inv[2]);
			if (dot_v3v3(xaxis, m_inv[0]) < 0) {
				negate_v3(xaxis);
			}
			fac = angle_normalized_v3v3(zvec_global, m_inv[2]) / (float)M_PI;
			fac = fabsf(fac - 0.5f) * 2;
			fac = fac * fac;
			interp_v3_v3v3(xaxis, xaxis, m_inv[0], fac);
		}
		else {
			copy_v3_v3(xaxis, m_inv[0]);
		}
#else
		copy_v3_v3(xaxis, m_inv[0]);
#endif

		/* Determine the direction of the x vector (for rotating up and down) */
		/* This can likely be computed directly from the quaternion. */

		/* Perform the up/down rotation */
		axis_angle_to_quat(q1, xaxis, sensitivity * -(y - vod->oldy));
		mul_qt_qtqt(vod->viewquat, vod->viewquat, q1);

		if (vod->use_dyn_ofs) {
			conjugate_qt(q1); /* conj == inv for unit quat */
			sub_v3_v3(rv3d->ofs, vod->dyn_ofs);
			mul_qt_v3(q1, rv3d->ofs);
			add_v3_v3(rv3d->ofs, vod->dyn_ofs);
		}

		/* Perform the orbital rotation */
		axis_angle_to_quat(q1, zvec_global, sensitivity * vod->reverse * (x - vod->oldx));
		mul_qt_qtqt(vod->viewquat, vod->viewquat, q1);

		if (vod->use_dyn_ofs) {
			conjugate_qt(q1);
			sub_v3_v3(rv3d->ofs, vod->dyn_ofs);
			mul_qt_v3(q1, rv3d->ofs);
			add_v3_v3(rv3d->ofs, vod->dyn_ofs);
		}
	}

	/* check for view snap */
	if (vod->axis_snap) {
		int i;
		float viewquat_inv[4];
		float zaxis[3] = {0, 0, 1};
		invert_qt_qt(viewquat_inv, vod->viewquat);

		mul_qt_v3(viewquat_inv, zaxis);

		for (i = 0; i < NUM_SNAP_QUATS; i++) {

			float view = (int)snapquats[i][4];
			float viewquat_inv_test[4];
			float zaxis_test[3] = {0, 0, 1};

			invert_qt_qt(viewquat_inv_test, snapquats[i]);
			mul_qt_v3(viewquat_inv_test, zaxis_test);
			
			if (angle_v3v3(zaxis_test, zaxis) < DEG2RADF(45 / 3)) {
				/* find the best roll */
				float quat_roll[4], quat_final[4], quat_best[4];
				float viewquat_align[4]; /* viewquat aligned to zaxis_test */
				float viewquat_align_inv[4]; /* viewquat aligned to zaxis_test */
				float best_angle = FLT_MAX;
				int j;

				/* viewquat_align is the original viewquat aligned to the snapped axis
				 * for testing roll */
				rotation_between_vecs_to_quat(viewquat_align, zaxis_test, zaxis);
				normalize_qt(viewquat_align);
				mul_qt_qtqt(viewquat_align, vod->viewquat, viewquat_align);
				normalize_qt(viewquat_align);
				invert_qt_qt(viewquat_align_inv, viewquat_align);

				/* find best roll */
				for (j = 0; j < 8; j++) {
					float angle;
					float xaxis1[3] = {1, 0, 0};
					float xaxis2[3] = {1, 0, 0};
					float quat_final_inv[4];

					axis_angle_to_quat(quat_roll, zaxis_test, (float)j * DEG2RADF(45.0f));
					normalize_qt(quat_roll);

					mul_qt_qtqt(quat_final, snapquats[i], quat_roll);
					normalize_qt(quat_final);
					
					/* compare 2 vector angles to find the least roll */
					invert_qt_qt(quat_final_inv, quat_final);
					mul_qt_v3(viewquat_align_inv, xaxis1);
					mul_qt_v3(quat_final_inv, xaxis2);
					angle = angle_v3v3(xaxis1, xaxis2);

					if (angle <= best_angle) {
						best_angle = angle;
						copy_qt_qt(quat_best, quat_final);
						if (j) view = 0;  /* view grid assumes certain up axis */
					}
				}

				copy_qt_qt(vod->viewquat, quat_best);
				rv3d->view = view; /* if we snap to a rolled camera the grid is invalid */

				break;
			}
		}
	}
	vod->oldx = x;
	vod->oldy = y;

	/* avoid precision loss over time */
	normalize_qt(vod->viewquat);

	/* use a working copy so view rotation locking doesnt overwrite the locked
	 * rotation back into the view we calculate with */
	copy_qt_qt(rv3d->viewquat, vod->viewquat);

	ED_view3d_camera_lock_sync(vod->v3d, rv3d);

	ED_region_tag_redraw(vod->ar);
}
"""

#static void viewrotate_apply(ViewOpsData *vod, int x, int y)
def viewrotate_apply(vod, x, y):
    #RegionView3D *rv3d = vod->rv3d;
    rv3d = vod.rv3d
    
    # I don't know of any Python equivalent for this line:
    #rv3d->view = RV3D_VIEW_USER; /* need to reset every time because of view snapping */
    
    #if (U.flag & USER_TRACKBALL) {
    if (USER_TRACKBALL):
        #float phi, si, q1[4], dvec[3], newvec[3];
        phi = 0.0
        si = 0.0
        q1 = Quaternion()
        dvec = Vector((0,0,0))
        newvec = Vector((0,0,0))
        
        #calctrackballvec(&vod->ar->winrct, x, y, newvec);
        calctrackballvec(vod.ar, x, y, newvec)
        
        #sub_v3_v3v3(dvec, newvec, vod->trackvec);
        dvec = newvec - vod.trackvec
        
        #si = len_v3(dvec);
        #si /= (float)(2.0 * TRACKBALLSIZE);
        si = dvec.magnitude
        si /= (2.0 * TRACKBALLSIZE)
        
        #cross_v3_v3v3(q1 + 1, vod->trackvec, newvec);
        #normalize_v3(q1 + 1);
        q1_axis = vod.trackvec.cross(newvec) # I'm guessing here
        q1_axis.normalize()
        
        # Allow for rotation beyond the interval [-pi, pi]
        while (si > 1.0):
            si -= 2.0
        
        # This relation is used instead of
        # phi = asin(si) so that the angle of
        # rotation is linearly proportional to
        # the distance that the mouse is dragged.
        phi = si * (M_PI / 2.0)
        
        #q1[0] = cos(phi);
        #mul_v3_fl(q1 + 1, sin(phi));
        #mul_qt_qtqt(vod->viewquat, q1, vod->oldquat);
        q1 = Quaternion(q1_axis, phi)
        vod.viewquat = q1 * vod.oldquat
        
        #if (vod->use_dyn_ofs) {
        if vod.use_dyn_ofs:
            pass # irrelevant code (doesn't matter in this test)
    else:
        pass # irrelevant code (doesn't matter in this test)
    
    # check for view snap
    #if (vod->axis_snap) {
    if vod.axis_snap:
        pass # irrelevant code (doesn't matter in this test)
    
    vod.oldx = x
    vod.oldy = y
    
    # avoid precision loss over time
    vod.viewquat.normalize()
    
    # use a working copy so view rotation locking doesnt overwrite
    # the locked rotation back into the view we calculate with
    #copy_qt_qt(rv3d->viewquat, vod->viewquat);
    rv3d.view_rotation = vod.viewquat.copy()
    
    #ED_view3d_camera_lock_sync(vod.v3d, rv3d) # irrelevant in this test
    
    ED_region_tag_redraw(vod.ar)
#############################################################################


### viewops_data_create(), viewops_data_free() ###
"""
static void viewops_data_create(bContext *C, wmOperator *op, const wmEvent *event)
{
	static float lastofs[3] = {0, 0, 0};
	RegionView3D *rv3d;
	ViewOpsData *vod = MEM_callocN(sizeof(ViewOpsData), "viewops data");

	/* store data */
	op->customdata = vod;
	vod->sa = CTX_wm_area(C);
	vod->ar = CTX_wm_region(C);
	vod->v3d = vod->sa->spacedata.first;
	vod->rv3d = rv3d = vod->ar->regiondata;

	/* set the view from the camera, if view locking is enabled.
	 * we may want to make this optional but for now its needed always */
	ED_view3d_camera_lock_init(vod->v3d, vod->rv3d);

	vod->dist_prev = rv3d->dist;
	vod->camzoom_prev = rv3d->camzoom;
	copy_qt_qt(vod->viewquat, rv3d->viewquat);
	copy_qt_qt(vod->oldquat, rv3d->viewquat);
	vod->origx = vod->oldx = event->x;
	vod->origy = vod->oldy = event->y;
	vod->origkey = event->type; /* the key that triggered the operator.  */
	vod->use_dyn_ofs = (U.uiflag & USER_ORBIT_SELECTION) != 0;
	copy_v3_v3(vod->ofs, rv3d->ofs);

	if (vod->use_dyn_ofs) {
		Scene *scene = CTX_data_scene(C);
		Object *ob = OBACT;

		if (ob && (ob->mode & OB_MODE_ALL_PAINT) && (BKE_object_pose_armature_get(ob) == NULL)) {
			/* in case of sculpting use last average stroke position as a rotation
			 * center, in other cases it's not clear what rotation center shall be
			 * so just rotate around object origin
			 */
			if (ob->mode & OB_MODE_SCULPT) {
				float stroke[3];
				ED_sculpt_get_average_stroke(ob, stroke);
				copy_v3_v3(lastofs, stroke);
			}
			else {
				copy_v3_v3(lastofs, ob->obmat[3]);
			}
		}
		else {
			/* If there's no selection, lastofs is unmodified and last value since static */
			calculateTransformCenter(C, V3D_CENTROID, lastofs, NULL);
		}

		negate_v3_v3(vod->dyn_ofs, lastofs);
	}
	else if (U.uiflag & USER_ZBUF_ORBIT) {
		Scene *scene = CTX_data_scene(C);

		view3d_operator_needs_opengl(C); /* needed for zbuf drawing */

		if ((vod->use_dyn_ofs = ED_view3d_autodist(scene, vod->ar, vod->v3d, event->mval, vod->dyn_ofs, true))) {
			if (rv3d->is_persp) {
				float my_origin[3]; /* original G.vd->ofs */
				float my_pivot[3]; /* view */
				float dvec[3];

				/* locals for dist correction */
				float mat[3][3];
				float upvec[3];

				negate_v3_v3(my_origin, rv3d->ofs);             /* ofs is flipped */

				/* Set the dist value to be the distance from this 3d point
				 * this means youll always be able to zoom into it and panning wont go bad when dist was zero */

				/* remove dist value */
				upvec[0] = upvec[1] = 0;
				upvec[2] = rv3d->dist;
				copy_m3_m4(mat, rv3d->viewinv);

				mul_m3_v3(mat, upvec);
				sub_v3_v3v3(my_pivot, rv3d->ofs, upvec);
				negate_v3(my_pivot);                /* ofs is flipped */

				/* find a new ofs value that is along the view axis (rather than the mouse location) */
				closest_to_line_v3(dvec, vod->dyn_ofs, my_pivot, my_origin);
				vod->dist_prev = rv3d->dist = len_v3v3(my_pivot, dvec);

				negate_v3_v3(rv3d->ofs, dvec);
			}
			negate_v3(vod->dyn_ofs);
			copy_v3_v3(vod->ofs, rv3d->ofs);
		}
	}

	{
		/* for dolly */
		const float mval_f[2] = {(float)event->mval[0],
		                         (float)event->mval[1]};
		ED_view3d_win_to_vector(vod->ar, mval_f, vod->mousevec);
	}

	/* lookup, we don't pass on v3d to prevent confusement */
	vod->grid = vod->v3d->grid;
	vod->far = vod->v3d->far;

	calctrackballvec(&vod->ar->winrct, event->x, event->y, vod->trackvec);

	{
		float tvec[3];
		negate_v3_v3(tvec, rv3d->ofs);
		vod->zfac = ED_view3d_calc_zfac(rv3d, tvec, NULL);
	}

	vod->reverse = 1.0f;
	if (rv3d->persmat[2][1] < 0.0f)
		vod->reverse = -1.0f;

	rv3d->rflag |= RV3D_NAVIGATING;
}

static void viewops_data_free(bContext *C, wmOperator *op)
{
	ARegion *ar;
	Paint *p = BKE_paint_get_active_from_context(C);

	if (op->customdata) {
		ViewOpsData *vod = op->customdata;
		ar = vod->ar;
		vod->rv3d->rflag &= ~RV3D_NAVIGATING;

		if (vod->timer)
			WM_event_remove_timer(CTX_wm_manager(C), vod->timer->win, vod->timer);

		MEM_freeN(vod);
		op->customdata = NULL;
	}
	else {
		ar = CTX_wm_region(C);
	}

	if (p && (p->flags & PAINT_FAST_NAVIGATE))
		ED_region_tag_redraw(ar);
}
"""

class ViewOpsData:
    def __init__(self):
        self.ar = None
        self.rv3d = None
        self.viewquat = None
        self.oldquat = None
        self.oldx = 0
        self.oldy = 0
        self.trackvec = Vector((0,0,0))
        self.use_dyn_ofs = False # just a stub here
        self.axis_snap = False # just a stub here

#static void viewops_data_create(bContext *C, wmOperator *op, const wmEvent *event)
def viewops_data_create(C, op, event):
    #ViewOpsData *vod = MEM_callocN(sizeof(ViewOpsData), "viewops data");
    #op->customdata = vod;
    vod = ViewOpsData()
    op.customdata = vod
    
    vod.ar = C.region # vod->ar = CTX_wm_region(C);
    vod.rv3d = C.region_data # vod->rv3d = rv3d = vod->ar->regiondata;
    
    rv3d = vod.rv3d;
    
    #copy_qt_qt(vod->viewquat, rv3d->viewquat);
    #copy_qt_qt(vod->oldquat, rv3d->viewquat);
    vod.viewquat = rv3d.view_rotation.copy()
    vod.oldquat = rv3d.view_rotation.copy()
    
    #vod->origx = vod->oldx = event->x;
    #vod->origy = vod->oldy = event->y;
    vod.oldx = event.mouse_x
    vod.oldy = event.mouse_y
    
    #calctrackballvec(&vod->ar->winrct, event->x, event->y, vod->trackvec);
    calctrackballvec(vod.ar, event.mouse_x, event.mouse_y, vod.trackvec);

#static void viewops_data_free(bContext *C, wmOperator *op)
def viewops_data_free(C, op):
    #Paint *p = BKE_paint_get_active_from_context(C);
    
    #if (op->customdata) {
    if hasattr(op, "customdata"):
        vod = op.customdata #ViewOpsData *vod = op->customdata;
        ar = vod.ar #ar = vod->ar;
        #vod->rv3d->rflag &= ~RV3D_NAVIGATING;
        #if (vod->timer) WM_event_remove_timer(CTX_wm_manager(C), vod->timer->win, vod->timer);
        #MEM_freeN(vod);
        #op->customdata = NULL;
    else:
        ar = C.region #ar = CTX_wm_region(C);
    
    #if (p && (p->flags & PAINT_FAST_NAVIGATE))
        #ED_region_tag_redraw(ar);
    ED_region_tag_redraw(ar)
#############################################################################


### viewrotate_modal() ###
"""
static int viewrotate_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
	ViewOpsData *vod = op->customdata;
	short event_code = VIEW_PASS;

	/* execute the events */
	if (event->type == MOUSEMOVE) {
		event_code = VIEW_APPLY;
	}
	else if (event->type == EVT_MODAL_MAP) {
		switch (event->val) {
			case VIEW_MODAL_CONFIRM:
				event_code = VIEW_CONFIRM;
				break;
			case VIEWROT_MODAL_AXIS_SNAP_ENABLE:
				vod->axis_snap = true;
				event_code = VIEW_APPLY;
				break;
			case VIEWROT_MODAL_AXIS_SNAP_DISABLE:
				vod->axis_snap = false;
				event_code = VIEW_APPLY;
				break;
			case VIEWROT_MODAL_SWITCH_ZOOM:
				WM_operator_name_call(C, "VIEW3D_OT_zoom", WM_OP_INVOKE_DEFAULT, NULL);
				event_code = VIEW_CONFIRM;
				break;
			case VIEWROT_MODAL_SWITCH_MOVE:
				WM_operator_name_call(C, "VIEW3D_OT_move", WM_OP_INVOKE_DEFAULT, NULL);
				event_code = VIEW_CONFIRM;
				break;
		}
	}
	else if (event->type == vod->origkey && event->val == KM_RELEASE) {
		event_code = VIEW_CONFIRM;
	}

	if (event_code == VIEW_APPLY) {
		viewrotate_apply(vod, event->x, event->y);
	}
	else if (event_code == VIEW_CONFIRM) {
		ED_view3d_depth_tag_update(vod->rv3d);
		viewops_data_free(C, op);

		return OPERATOR_FINISHED;
	}

	return OPERATOR_RUNNING_MODAL;
}
"""

#static int viewrotate_modal(bContext *C, wmOperator *op, const wmEvent *event)
def viewrotate_modal(C, op, event):
    #ViewOpsData *vod = op->customdata;
    vod = op.customdata
    
    #short event_code = VIEW_PASS;
    event_code = 'VIEW_PASS'
    
    # execute the events
    if event.type == 'MOUSEMOVE':
        event_code = 'VIEW_APPLY'
    else:
        # The original logic in this section is irrelevant,
        # so I just put there some way to stop the operator
        if event.type == 'LEFTMOUSE':
            event_code = 'VIEW_CONFIRM'
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            event_code = 'VIEW_CONFIRM'
    
    if (event_code == 'VIEW_APPLY'):
        #viewrotate_apply(vod, event->x, event->y);
        viewrotate_apply(vod, event.mouse_x, event.mouse_y)
    elif (event_code == 'VIEW_CONFIRM'):
        #ED_view3d_depth_tag_update(vod->rv3d); # what does this function do?
        viewops_data_free(C, op)
        
        return {'FINISHED'}
    
    return {'RUNNING_MODAL'}
#############################################################################


### viewrotate_invoke() ###
"""
static int viewrotate_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
	ViewOpsData *vod;
	RegionView3D *rv3d;

	/* makes op->customdata */
	viewops_data_create(C, op, event);
	vod = op->customdata;
	rv3d = vod->rv3d;

	if (rv3d->viewlock) { /* poll should check but in some cases fails, see poll func for details */
		viewops_data_free(C, op);
		return OPERATOR_PASS_THROUGH;
	}

	/* switch from camera view when: */
	if (rv3d->persp != RV3D_PERSP) {

		if (U.uiflag & USER_AUTOPERSP) {
			if (!ED_view3d_camera_lock_check(vod->v3d, vod->rv3d)) {
				rv3d->persp = RV3D_PERSP;
			}
		}
		else if (rv3d->persp == RV3D_CAMOB) {

			/* changed since 2.4x, use the camera view */
			if (vod->v3d->camera) {
				rv3d->dist = ED_view3d_offset_distance(vod->v3d->camera->obmat, rv3d->ofs, VIEW3D_DIST_FALLBACK);
				ED_view3d_from_object(vod->v3d->camera, rv3d->ofs, rv3d->viewquat, &rv3d->dist, NULL);
			}

			if (!ED_view3d_camera_lock_check(vod->v3d, vod->rv3d)) {
				rv3d->persp = rv3d->lpersp;
			}
		}
		ED_region_tag_redraw(vod->ar);
	}
	
	if (event->type == MOUSEPAN) {
		/* Rotate direction we keep always same */
		if (U.uiflag2 & USER_TRACKPAD_NATURAL)
			viewrotate_apply(vod, 2 * event->x - event->prevx, 2 * event->y - event->prevy);
		else
			viewrotate_apply(vod, event->prevx, event->prevy);
			
		ED_view3d_depth_tag_update(rv3d);
		
		viewops_data_free(C, op);
		
		return OPERATOR_FINISHED;
	}
	else if (event->type == MOUSEROTATE) {
		/* MOUSEROTATE performs orbital rotation, so y axis delta is set to 0 */
		viewrotate_apply(vod, event->prevx, event->y);
		ED_view3d_depth_tag_update(rv3d);
		
		viewops_data_free(C, op);
		
		return OPERATOR_FINISHED;
	}
	else {
		/* add temp handler */
		WM_event_add_modal_handler(C, op);

		return OPERATOR_RUNNING_MODAL;
	}
}
"""

#static int viewrotate_invoke(bContext *C, wmOperator *op, const wmEvent *event)
def viewrotate_invoke(C, op, event):
    #ViewOpsData *vod;
    #RegionView3D *rv3d;
    
    # makes op->customdata
    viewops_data_create(C, op, event)
    vod = op.customdata
    rv3d = vod.rv3d
    
    #if (rv3d->viewlock) { /* poll should check but in some cases fails, see poll func for details */
    if (rv3d.lock_rotation):
        pass # irrelevant code (doesn't matter in this test)
    
    # switch from camera view when:
    #if (rv3d->persp != RV3D_PERSP) {
    if (rv3d.view_perspective != 'PERSP'):
        pass # irrelevant code (doesn't matter in this test)
    
    event_type = None # for this test
    
    #if (event->type == MOUSEPAN) {
    if event_type == 'MOUSEPAN':
        pass # irrelevant code (doesn't matter in this test)
    #else if (event->type == MOUSEROTATE) {
    elif event_type == 'MOUSEROTATE': # This seems like a 'one-off' case?
        pass # irrelevant code (doesn't matter in this test)
    else:
        # add temp handler
        WM_event_add_modal_handler(C, op);
        
        return {'RUNNING_MODAL'}
#############################################################################


### REGISTRATION STUFF ###
class TrackballTestOperator(bpy.types.Operator):
    """Trackball test"""
    bl_idname = "view3d.trackball_test"
    bl_label = "Trackball test"
    
    def modal(self, context, event):
        return viewrotate_modal(context, self, event)
    
    def invoke(self, context, event):
        return viewrotate_invoke(context, self, event)

def register():
    bpy.utils.register_class(TrackballTestOperator)

def unregister():
    bpy.utils.unregister_class(TrackballTestOperator)

if __name__ == "__main__":
    register()
