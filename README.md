## Extra 3D view navigation features

## Authors:

dairin0d - original author - developer

Ivan Santic (MOTH3R) - coauthor, added idea (ZBrush mode nav), beta-tester

## Description

The addon attempts to provide better usability and customization of basic 3D viewport navigation (in particular, ZBrush mode and FPS-like movement).
Still highly unpolished and a work-in-progress, but should be suitable for the evaluation of its design ideas.

## About:

This add-on grew out from the ideas I had about making MMB (Middle Mouse Button) navigation in Blender more convenient for myself. These were:
* Be able to cancel MMB rotation, with the view returning to where it was
* Ability to move in First Person Shooter style (using WASD control scheme) while orbiting
* Ability to freely switch between orbit/pan/dolly/zoom and be able to do it with just the mouse (e.g. MMB is orbit, MMB+LMB is zoom, MMB+RMB is pan)
* Have a 3D crosshair in MMB navigation to indicate where the orbiting origin is
* Make it less easy to accidentally rotate view in Ortho mode

moth3r suggested some additional features, like ZBrush mode (which seems to be especially useful for tablet users), some control setups, and provided a lot of useful feedback. Thanks!

## Usage

Right now, the addon auto-registers a keymap on Middle Mouse Button with some default control scheme (which is subject to change), but the control scheme can be adjusted via operator properties (see the corresponding keymap).

Right now, the default control scheme is this:
* Confirm: Return/Enter, left mouse
* Cancel: Esc, right mouse
* Rotation mode switch: Space
* Rotate: middle mouse
* Snap rotation: middle mouse + Alt
* Move (pan): middle mouse + Shift
* Zoom: middle mouse + Ctrl
* Fly: middle mouse double-click
* Walk: Tab

FPS navigation
* Forward: W
* Backward: S
* Left: A
* Right: D
* Up: E, R
* Down: Q, F
* Faster speed: Shift
* Slower speed: Ctrl
* Crouch: Ctrl
* Jump: Space
* Teleport: middle mouse, V

## Operator properties:
* Default mode - the operator would start in this mode if other mode keys are not pressed
* Transitions - allowed transitions between modes
* ZBrush mode - if enabled, the operator's default mode will be invoked only if there is no geometry under the mouse, or if the mouse is sufficiently close to the 3D View border
* Ortho unrotate - if enabled and view projection is Orthographic, switching from Orbit to Pan/Dolly/Zoom will snap view rotation to its initial value and will disable switching to Orbit until the operator has finished
* Confirm - key(s) that confirm changes to view
* Cancel - key(s) that cancel changes to view
* Rotation Mode Switch - key(s) that confirm switch between Turntable and Trackball rotation modes
* Origin: Mouse - key(s) that force Auto Depth option for the duration of the operator
* Origin: Selection - key(s) that force Rotate Around Selection option for the duration of the operator
* Orbit - key(s) that switch to Orbit mode
* Orbit Snap - key(s) that switch rotation snapping
* Pan - key(s) that switch to Pan mode
* Dolly - key(s) that switch to Dolly mode
* Zoom - key(s) that switch to Zoom mode
* Fly - key(s) that switch to Fly mode
* Walk - key(s) that switch to Walk mode

## FPS (First Person shooter) properties:
* FPS forward - key(s) that move view in forward FPS direction
* FPS back - key(s) that move view in backward FPS direction
* FPS left - key(s) that move view in left FPS direction
* FPS right - key(s) that move view in right FPS direction
* FPS up - key(s) that move view in upward FPS direction (automatically switches off gravity)
* FPS down - key(s) that move view in downward FPS direction (automatically switches off gravity)
* FPS fast - key(s) that switch FPS movement to a faster speed
* FPS slow - key(s) that switch FPS movement to a slower speed
* FPS crouch - key(s) for "crouching" in Walk mode (when gravity is on)
* FPS jump - key(s) for "jumping"/"jetpack" in Walk mode (automatically switches on gravity)
* FPS teleport - key(s) for "teleporting"/"grappling hook" in Walk mode
* FPS horizontal (in N-panel) - if enabled, WASD keys move view in horizontal plane, and ER/QF move view in world's vertical direction
* FPS speed (in N-panel) - speed multiplier for FPS movement

## N-panel ("Mouse-look navigation" sub-panel):
* Enabled - enables or disables the mouse-look navigation operator
* Zoom speed - speed multiplier for zooming
* Trackball mode - what trackball algorithm to use
 * Center - rotation depends only on mouse speed and not on mouse position; has the most stable and predictable behavior
 * Wrapped - like Center, but rotation depends on mouse position
 * Blender - uses the same trackball algorithm as in Blender (in theory. In practice I haven't figured out how to make it behave exactly like in Blender)
* Orbit snap subdivs - number of intermediate angles to which view rotation can be snapped (1 snaps to each 90 degrees, 2 snaps to each 45 degrees, and so on)
* Orbit snap->ortho - if Auto Perspective is enabled in user preferences, rotation snapping would switch view to Orthographic mode
* Rotation speed - speed multiplier for rotation
* Trackball Auto-level - enables or disables autolevelling in the trackball mode
* Trackball Auto-level up - if enabled, autolevelling would always try to orient view's up axis to world's up axis in trackball mode
* Autolevel speed - speed of autolevelling (autolevelling decreases tilt view over time)

## Known issues/missing features:
* Blender trackball mode doesn't actually behave like in Blender
* Ortho-grid/quadview-clip/projection-name display is not updated
* Auto Depth for Ortho mode (don't know how to calculate correct position from zbuf in ortho mode)
* Rotate Around Selection might work incorrectly in scuplt/paint modes and when editing metaballs or text-curves
* Due to the use of timer, operator consumes more resources than Blender's default
