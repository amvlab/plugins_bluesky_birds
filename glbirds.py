""" Bird traffic gui plugin 

Developed by amvlab in collaboration with Dr. Isabel Metz from DLR
"""
import numpy as np

from bluesky.ui.qtgl import glhelpers as glh
from bluesky import settings
from bluesky.ui import palette
from bluesky.tools.aero import ft, kts
from bluesky.network import context as ctx
from bluesky.network.subscriber import subscriber

# Register settings defaults
settings.set_variable_defaults(text_size=13, bird_size=10)

palette.set_default_colours(
    bird=(255, 0, 0),
    angry_bird=(255, 160, 0)
)

# Static
MAX_NBIRDS = 10000

### Initialization function of your plugin.
def init_plugin():

    config = {
        'plugin_name':     'BIRDGUI',
        'plugin_type':     'gui',
        }

    return config

# Bird gl traffic class
class BirdTraffic(glh.RenderObject, layer=100):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initialized = False

        self.bird_hdg = glh.GLBuffer()
        self.bird_lat = glh.GLBuffer()
        self.bird_lon = glh.GLBuffer()
        self.bird_alt = glh.GLBuffer()
        self.bird_color = glh.GLBuffer()
        self.bird_lbl = glh.GLBuffer()
        self.bird_symbol = glh.VertexArrayObject(glh.gl.GL_TRIANGLE_FAN)
        self.birdlabels = glh.Text(settings.text_size, (8, 3))
        self.nbirds = 0
        self.show_lbl = True

    def create(self):
        bird_size = settings.bird_size
        self.bird_hdg.create(MAX_NBIRDS * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.bird_lat.create(MAX_NBIRDS * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.bird_lon.create(MAX_NBIRDS * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.bird_alt.create(MAX_NBIRDS * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.bird_color.create(MAX_NBIRDS * 4, glh.GLBuffer.UsagePattern.StreamDraw)
        self.bird_lbl.create(MAX_NBIRDS * 24, glh.GLBuffer.UsagePattern.StreamDraw)


        # ------- Bird triangle fan -------------------------
        birdvertices = np.array([(0.0, 0.0),                     # 1
                            (0.7 * bird_size, -0.1 * bird_size),   # wing, 2 
                            (0.7 * bird_size, 0.1 * bird_size),    # wing, 3
                            (0.25 * bird_size, 0.1 * bird_size),    # trans, 4
                            (0.1 * bird_size, 0.1 * bird_size),    # trans 5
                            (0.05 * bird_size, 0.5 * bird_size),    #head. 6
                            (-0.05 * bird_size, 0.5 * bird_size),    #head. 7
                            (-0.1 * bird_size, 0.1 * bird_size),    # trans 8
                            (-0.25 * bird_size, 0.1 * bird_size),    # trans, 9
                            (-0.7 * bird_size, 0.1 * bird_size),    # wing, 10
                            (-0.7 * bird_size, -0.1 * bird_size),   # wing, 11
                            (-0.25 * bird_size, -0.1 * bird_size),   # trans, 12 
                            (0, -0.5 * bird_size),   # tail, 13
                            (0.25 * bird_size, -0.1 * bird_size),   # trans, 14 
                            (0.7 * bird_size, -0.1 * bird_size)],
                        dtype=np.float32)

        self.bird_symbol.create(vertex=birdvertices)

        self.bird_symbol.set_attribs(lat=self.bird_lat, lon=self.bird_lon, color=self.bird_color,
                                   orientation=self.bird_hdg, instance_divisor=1)

        self.birdlabels.create(self.bird_lbl, self.bird_lat, self.bird_lon, self.bird_color,
                             (bird_size, -0.5 * bird_size), instanced=True)

    def draw(self):
        if self.nbirds:
            self.bird_symbol.draw(n_instances=self.nbirds)

            if self.show_lbl:
                self.birdlabels.draw(n_instances=self.nbirds)


    @subscriber(topic='BIRDDATA',  actonly=True)
    def bird_catcher(self, data):
        ''' Receive bird data from bluesky Simulation Node. '''
        # if not self.initialized:
        #     return
                
        if ctx.action == ctx.action.Reset or ctx.action == ctx.action.ActChange:
            self.nbirds = 0
            return
        
        elif ctx.action == ctx.action.Replace:
            self.update_bird_data(data)


    def update_bird_data(self, data):

        # get bird data
        bird_id = data.id
        bird_type = data.type
        bird_lat = data.lat
        bird_lon = data.lon
        bird_hdg = data.hdg
        bird_alt = data.alt
        bird_vs = data.vs
        bird_hs = data.hs
        bird_lbl_type = data.lbl_type
        angry_birds = data.angry_birds

        # update buffers
        self.nbirds = len(bird_lat)
        self.bird_lat.update(np.array(bird_lat, dtype=np.float32))
        self.bird_lon.update(np.array(bird_lon, dtype=np.float32))
        self.bird_hdg.update(np.array(bird_hdg, dtype=np.float32))
        self.bird_alt.update(np.array(bird_alt, dtype=np.float32))

        # colors
        rawlabel = ''
        color = np.empty(
            (min(self.nbirds, MAX_NBIRDS), 4), dtype=np.uint8)
        rgb_bird = palette.bird

        zdata = zip(bird_id, bird_type, bird_alt, bird_vs, bird_hs, angry_birds)
        for i, (id, b_type, alt, vs, hs, angry_bird) in enumerate(zdata):
            if i >= MAX_NBIRDS:
                break

            # Make label
            if self.show_lbl:
                if bird_lbl_type == 'id':
                    rawlabel += '%-8s' % id[:8]
                else:
                    rawlabel += '%-8s' % b_type[:8]
                    
                rawlabel += '%-5d' % int(alt / ft + 0.5)
                vsarrow = 30 if vs > 0.25 else 31 if vs < -0.25 else 32
                rawlabel += '%1s  %-8d' % (chr(vsarrow),
                                            int(hs / kts + 0.5))
            
            # check if bird is angry
            if angry_bird:
                color[i, :] = palette.angry_bird + (255,)

            else:
                # Set default color
                color[i, :] = tuple(rgb_bird) + (255,)

        
        # update bird label
        self.bird_color.update(color)
        self.bird_lbl.update(np.array(rawlabel.encode('utf8'), dtype=np.bytes_))
