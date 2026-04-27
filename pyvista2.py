from __future__ import annotations

import pyvista as pv
from pyvista import examples

pl = pv.Plotter(shape=(2, 2))

pl.subplot(0, 0)
pl.add_text('Render Window 0', font_size=30)
globe = examples.load_globe()
texture = examples.load_globe_texture()
pl.add_mesh(globe, texture=texture)

pl.subplot(0, 1)
pl.add_text('Render Window 1', font_size=30)
pl.add_mesh(pv.Cube(), show_edges=True, color='lightblue')

pl.subplot(1, 0)
pl.add_text('Render Window 2', font_size=30)
sphere = pv.Sphere()
pl.add_mesh(sphere, scalars=sphere.points[:, 2])
pl.add_scalar_bar('Z')
# pl.add_axes()
pl.add_axes(interactive=True)

pl.subplot(1, 1)
pl.add_text('Render Window 3', font_size=30)
pl.add_mesh(pv.Cone(), color='g', show_edges=True)
pl.show_bounds(all_edges=True)

# Display the window
pl.show()