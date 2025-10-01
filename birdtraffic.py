''' Bird traffic simulation plugin 

Jointly developed by amvlab and Dr. Isabel Metz from DLR
'''
import numpy as np

from bluesky import stack
from bluesky.tools.aero import ft, kts
from bluesky import core, stack
from bluesky.network.publisher import state_publisher

# Update rate of bird update messages [Hz]
BIRDUPDATE_RATE = 5

def init_plugin():

    config = {
        # The name of your plugin
        'plugin_name'      : 'BIRDSIM',
        'plugin_type'      : 'sim',
        'update_interval'  :  1.0,
        'update'           : bird_traf.update,
        'reset'            : bird_traf.reset
        }

    return config


class BirdTraffic(core.Entity):

    def __init__(self):
        super().__init__()

        self.nbird = 0 # number of birds
        
        # initialize bid array
        self.id      = []  # identifier (string)
        self.type    = []  # bird type (string)

        # Positions
        self.lat     = np.array([], dtype=float)  # latitude [deg]
        self.lon     = np.array([], dtype=float)  # longitude [deg]
        self.alt     = np.array([], dtype=float)  # altitude [m]
        self.hdg     = np.array([], dtype=float)  # traffic heading [deg]

        # Velocities
        self.hs     = np.array([], dtype=float)   # horizontal airspeed [m/s]
        self.vs     = np.array([], dtype=float)  # vertical speed [m/s]

        # Boolean to indicate if the bird is in danger of striking an aircraft
        # At the moment this just changes the color of the bird.
        self.angry_birds = np.array([], dtype=bool)

        # Label type to show in gui. Default is 'id;
        self.labels = ["id", "type"]
        self.lbl_type  = self.labels[0]
   
    def create_bird(self, birdid:str, birdtype:str="goose", birdlat: float=52., birdlon: float=4., birdhdg: float=None, birdalt: float=0, 
                    birdspd: float=0):
        
        # add one bird
        n = 1

        # increase number of birds
        self.nbird += n

        # get position of bird
        birdlat = np.array(n * [birdlat])
        birdlon = np.array(n * [birdlon])

        # Limit longitude to [-180.0, 180.0]
        birdlon[birdlon > 180.0] -= 360.0
        birdlon[birdlon < -180.0] += 360.0

        # add to birdinfo to lists
        self.id.append(birdid)
        self.type.append(birdtype)

        # Positions
        self.lat = np.append(self.lat, birdlat)
        self.lon = np.append(self.lon, birdlon)
        self.alt = np.append(self.alt, birdalt)

        # Heading
        self.hdg = np.append(self.hdg, birdhdg)

        # Velocities
        self.hs = np.append(self.hs, birdspd)
        vs = 0
        self.vs = np.append(self.vs, vs)

        # set danger of striking an aircraft to 
        self.angry_birds = np.append(self.angry_birds, False)

    def id2idx(self, birdid: str):
        """Find index of bird id"""

        return self.id.index(birdid.upper())
    
    def remove_bird(self, birdid: str):
        print ("we attempt to delete birdie ", birdid)
        
        index_to_remove = self.id2idx(birdid)
        
        
        self.nbird = self.nbird - 1 # number of birds
        
        # basic info

        del self.id [index_to_remove]   # identifier (string)
        del self.type[index_to_remove]   # bird type (string)

        # Positions
        self.lat              = np.delete(self.lat, index_to_remove)
        self.lon              = np.delete(self.lon, index_to_remove)  
        self.alt              = np.delete(self.alt, index_to_remove)       
        self.hdg              = np.delete(self.hdg, index_to_remove)  
        

        # Velocities
        self.hs     = np.delete(self.hs, index_to_remove)    # horizontal airspeed [m/s]
        self.vs     = np.delete(self.vs, index_to_remove)   # vertical speed [m/s]

        return    
    
    def reset(self):

        # clear all TODO: copy traffarrays
        self.nbird = 0 # number of birds
        
        # initialize bid array
        self.id      = []  # identifier (string)
        self.type    = []  # bird type (string)

        # Positions
        self.lat     = np.array([], dtype=float)  # latitude [deg]
        self.lon     = np.array([], dtype=float)  # longitude [deg]
        self.alt     = np.array([], dtype=float)  # altitude [m]
        self.hdg     = np.array([], dtype=float)  # traffic heading [deg]

        # Velocities
        self.hs     = np.array([], dtype=float)   # horizontal airspeed [m/s]
        self.vs     = np.array([], dtype=float)  # vertical speed [m/s]

        # danger of striking an aircraft
        self.angry_birds = np.array([], dtype=bool)
        
        return

    def update(self):
        pass
        return

    @state_publisher(topic='BIRDDATA', dt=1000 // BIRDUPDATE_RATE)
    def release_birds(self):
        '''release them to the visual world '''

        data = dict()
        data['id']         = self.id
        data['type']       = self.type
        data['lat']        = self.lat
        data['lon']        = self.lon
        data['alt']        = self.alt
        data['hdg']        = self.hdg
        data['vs']         = self.vs
        data['hs']         = self.hs
        data['angry_birds']= self.angry_birds
        data['lbl_type']   = self.lbl_type

        return data

    @stack.command(name='CREBIRD')
    def CREBIRD(self, birdid: str, birdtype: str="goose", birdlat: float=52., birdlon: float=4., birdhdg: float=None, birdalt: float=0, 
                birdspd: float = 0):
        ''' CREBIRD birdid,type,lat,lon,hdg,alt,spd '''
        # correct some argument units
        birdspd *= kts
        birdalt *= ft

        # create the bird
        self.create_bird(birdid, birdtype, birdlat, birdlon, birdhdg, birdalt, birdspd)

    @stack.command(name='BIRDLABEL')
    def birdlabel(self):
        ''' BIRDLABEL'''
        # choose the other option
        self.lbl_type = self.labels[0] if self.lbl_type == self.labels[1] else self.labels[1]

    @stack.command(name = 'DELBIRD')
    def DELBIRD(self, birdid: str):
        ''' DELBIRD birdid '''
        # bird left the area, landed or was eaten by an aircraft
        self.remove_bird(birdid)


bird_traf = BirdTraffic()
