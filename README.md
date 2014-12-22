## Extra 3D view navigation features

## Authors:

dairin0d - original author - developer

Ivan Santic (MOTH3R) - coauthor, added idea (ZBrush mode nav), beta-tester

## Description:

The addon attempts to provide better usability and customization of basic 3D viewport navigation (in particular, ZBrush mode and FPS-like movement). It's an alternative to Blender's default orbit/pan/zoom/dolly/fly/walk navigation.

Most notable features:
* ZBrush mode - mostly useful for tablet users, since it allows to use the same mouse button both for painting/sculpting and for navigation (depending on whether you clicked on geometry or on background)
* Easy switching between navigation modes without exiting the operator
* Changes to viewport can be cancelled from any mode
* FPS-like movement is available in all navigation modes
* Crosshair is visible in all modes and has a different look when obscured
* Option to more easily prevent accidental viewport rotation in Ortho projection
* Different turntable/trackball algorithms and different fly mode (more FPS-like) 

## About:

This add-on grew out from the ideas I had about making MMB (Middle Mouse Button) navigation in Blender more convenient for myself. These were:
* Be able to cancel MMB rotation, with the view returning to where it was
* Ability to move in First Person Shooter style (using WASD control scheme) while orbiting
* Ability to freely switch between orbit/pan/dolly/zoom and be able to do it with just the mouse (e.g. MMB is orbit, MMB+LMB is zoom, MMB+RMB is pan)
* Have a 3D crosshair in MMB navigation to indicate where the orbiting origin is
* Make it less easy to accidentally rotate view in Ortho mode

moth3r suggested some additional features, like ZBrush mode (which seems to be especially useful for tablet users), some control setups, and provided a lot of useful feedback. Thanks!

## Documentation:

See the corresponding wiki page, it has illustrations:

http://wiki.blender.org/index.php/Extensions:2.6/Py/Scripts/3D_interaction/MouselookNavigation
