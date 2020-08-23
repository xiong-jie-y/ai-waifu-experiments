#%%
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gdk

#%%
# full screenshot
window = Gdk.get_default_root_window()
pb = Gdk.pixbuf_get_from_window(window, *window.get_geometry())
pb.savev("full.png", "png", (), ())

# screenshots for all windows
window = Gdk.get_default_root_window()
screen = window.get_screen()
typ = window.get_type_hint()
for i, w in enumerate(screen.get_window_stack()):
    pb = Gdk.pixbuf_get_from_window(w, *w.get_geometry())
    pb.savev("{}.png".format(i), "png", (), ())

#%%
# screenshot active window
screen = Gdk.get_default_root_window().get_screen()
w = screen.get_active_window()
pb = Gdk.pixbuf_get_from_window(w, *w.get_geometry())
pb.savev("active.png", "png", (), ())

#%%
# screenshots for all windows
window = Gdk.get_default_root_window()
screen = window.get_screen()
typ = window.get_type_hint()


# %%
import re
[a for a in dir(screen.get_window_stack()[0]) if re.search(r"title", a)]
screen.get_window_stack()[0].get_property('name')

# %%
def get_window_screen(window_id):
    window = Gdk.get_default_root_window()
    screen = window.get_screen()
    typ = window.get_type_hint()
    for i, w in enumerate(screen.get_window_stack()):
        if w.get_xid() == int(window_id, 16):
            pb = Gdk.pixbuf_get_from_window(w, *w.get_geometry())
            # pb.savev("chosen_window.png".format(i), "png", (), ())
            return pb
    return None

#%%
from gi.repository import GdkPixbuf
import numpy
def array_from_pixbuf(p):
    " convert from GdkPixbuf to numpy array"
    w,h,c,r=(p.get_width(), p.get_height(), p.get_n_channels(), p.get_rowstride())
    assert p.get_colorspace() == GdkPixbuf.Colorspace.RGB
    assert p.get_bits_per_sample() == 8
    if  p.get_has_alpha():
        assert c == 4
    else:
        assert c == 3
    assert r >= w * c
    a=numpy.frombuffer(p.get_pixels(),dtype=numpy.uint8)
    if a.shape[0] == w*c*h:
        return a.reshape( (h, w, c) )
    else:
        b=numpy.zeros((h,w*c),'uint8')
        for j in range(h):
            b[j,:]=a[r*j:r*j+w*c]
        return b.reshape( (h, w, c) )

# array_from_pixbuf(get_window_screen('0x500000b')).shape

# %%
import pyfakewebcam
import numpy as np

camera = pyfakewebcam.FakeWebcam('/dev/video0', 640, 480)

#%%
from PIL import Image
import time

window_id = '0x2a0000b'

while True:
    screen_pixbuf = get_window_screen(window_id)
    if screen_pixbuf is not None:
        screen_np = array_from_pixbuf(screen_pixbuf)
        screen_np = np.asarray(Image.fromarray(screen_np).resize((640, 480)))

        camera.schedule_frame(screen_np)
        time.sleep(1/30.0)


# %%
