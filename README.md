Mouselook navigation
====================
Extra 3D view navigation features

Right now, the addon auto-registers a keymap on Middle Mouse Button with some default control scheme (which is subject to change), but the control scheme can be adjusted via operator properties (see the corresponding keymap).

Operator properties:
* Default mode -- the operator would start in this mode if other mode keys are not pressed
* Transitions -- alowed transitions between modes
* ZBrush mode -- if enabled, the operator's default mode will be invoked only if there is no geometry under the mouse, or if the mouse is sufficiently close to the 3D View border
* Ortho unrotate -- if enabled and view projection is Orthographic, switching from Orbit to Pan/Dolly/Zoom will snap view rotation to its initial value and will disable switching to Orbit until the operator has finished.
* Confirm -- key(s) that confirm changes to view
* Cancel -- key(s) that cancel changes to view
* Rotation Mode Switch -- key(s) that confirm switch between Turntable and Trackball rotation modes
* Orbit -- key(s) that switch to Orbit mode
* Orbit Snap -- key(s) that switch rotation snapping
* Pan -- key(s) that switch to Pan mode
* Dolly -- key(s) that switch to Dolly mode
* Zoom -- key(s) that switch to Zoom mode
* FPS forward -- key(s) that move view in forward FPS direction
* FPS back -- key(s) that move view in backward FPS direction
* FPS left -- key(s) that move view in left FPS direction
* FPS right -- key(s) that move view in right FPS direction
* FPS up -- key(s) that move view in upward FPS direction
* FPS down -- key(s) that move view in downward FPS direction
* FPS fast -- key(s) that switch FPS movement to a faster speed
* FPS slow -- key(s) that switch FPS movement to a slower speed

Options in N-panel ("Mouselook navigation" subpanel)
* Enabled -- enables or disables the mouselook navigation operator
* FPS horizontal -- if enabled, WASD keys move view in horizontal plane, and ER/QF move view in world's vertical direction
* FPS speed -- speed multiplier for FPS movement
* Zoom speed -- speed multiplier for zooming
* Trackball mode -- what trackball algorithm to use
  * Center -- rotation depends only on mouse speed and not on mouse position; has the most stable and predictable behavior
  * Wrapped -- like Center, but rotation depends on mouse position
  * Blender -- uses the same trackball algorithm as in Blender (in theory. In practice I haven't figured out how to make it behave exactly like in Blender)
* Orbit snap subdivs -- number of intermediate angles to which view rotation can be snapped (1 snaps to each 90 degrees, 2 snaps to each 45 degrees, and so on)
* Orbit snap->ortho -- if Auto Perspective is enabled in user preferences, rotation snapping would switch view to Orthographic mode
* Rotation speed -- speed multiplier for rotation
* Trackball Autolevel -- enables or disables autolevelling in trackball mode
* Trackball Autolevel up -- if enabled, autolevelling would always try to orient view's up axis to world's up axis in trackball mode
* Autolevel speed -- speed of autolevelling (autolevelling decreases view tilt over time)

Known issues/missing features:
* Blender trackball mode doesn't actually behave like in Blender
* "Auto Depth", "Rotate Around Selection", "Camera Parent Lock" are not yet implemented
* Full FPS/Fly navigation

About:
This addon grew out from the ideas I had about making MMB (Middle Mouse Button) navigation in Blender more convenient for myself. These were:
* Be able to cancel MMB rotation, with the view returning to where it was
* Ability to move in First Person Shooter style (using WASD control scheme) while orbiting
* Ability to freeily switch between orbit/pan/dolly/zoom and be able to do it with just the mouse (e.g. MMB is orbit, MMB+LMB is zoom, MMB+ RMB is pan)
* Have a 3D crosshair in MMB navigation to indicate where the oribiting origin is
* Make it less easy to accidentaly rotate view in Ortho mode
I also think that Blender's native Fly/FPS navigation operators have some room for improvement, but I didn't get to implement that yet.
I was also considering some sort of "view history stack", but there seem to be addons for that already.

moth3r suggested some additional features, like ZBrush mode (which seems to be especially useful for tablet users), some control setups, and provided a lot of useful feedback. Thanks!
