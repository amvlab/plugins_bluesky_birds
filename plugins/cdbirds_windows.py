# -*- coding: utf-8 -*-
"""
Bird conflict detection simulation plugin

Jointly developed by amvlab and Dr. Isabel Metz from DLR
"""

import numpy as np
import os
import pandas as pd

import bluesky as bs
from bluesky import core, stack
from bluesky.tools.aero import ft, kts
from bluesky.stack.cmdparser import append_commands

from bluesky.plugins.birdtraffic import bird_traf

dir = os.path.dirname(__file__)

def init_plugin():
    
    global bird_cdr
    
    bird_cdr = Conflict_Detection_Birds()

    
    config = {
        # The name of your plugin
        'plugin_name'      : 'BIRDCDR',
        'plugin_type'      : 'sim',
        'update'           : bird_cdr.conflict_detection,
        }

    return config


@stack.command(name = 'LOGNAME')
def LOGNAME(filename):
    '''when we want to load a scenario'''

    # bird left the area, landed or was eaten by an aircraft
    bird_cdr.set_logname(filename) 
    
    return

@stack.command(name = 'CRE_BIRDAC')
def CRE_BIRDAC(acid, actype: str="B744", aclat: float=52., aclon: float=4., achdg: float=None, acalt: float=0,  
        acspd: float = 0):
    """CREM2 acid, type, [latlon], [hdg], [alt], [spd], prio"""
    ## DEPRECATED!!!
    # Creates an aircrft, but also assigns priority
    # Convert stuff for bs.traf.cre

    # correct some argument units
    acspd *= kts
    acalt *= ft
        
    # First create the aircraft

    bs.traf.cre(acid, actype, aclat, aclon, achdg, acalt, acspd)

    
    
    
    # Then assign its collision envelope
    idx = bs.traf.id.index(acid)

    
    coll_rad, coll_height, coll_sweep = bird_cdr.assign_envelope(actype)

    
  #  bs.traf.priority[idx] = prio

    # you can just do this
    bird_cdr.ac_collision_radius[idx] = coll_rad
    bird_cdr.ac_collision_height[idx] = coll_height
    bird_cdr.ac_collision_sweep[idx]  = coll_sweep
    
    # bs.traf.ac_collision_radius[idx] = 342.
    # bs.traf.ac_collision_height[idx] = coll_height
    # bird_cdr.traf.ac_collision_radius[idx] = 18.
    # bird_cdr.traf.ac_collision_height[idx] = coll_height


    # add path plan for specific aircraft
    #bs.traf.path_plans[-1] = path_plan_dict[ACID]

    return


class Conflict_Detection_Birds(core.Entity):
    
    def __init__(self):
        
        super().__init__()
        ''' get our birdie info from birdtraffic'''
                
        with self.settrafarrays():
            
            self.ac_collision_radius = np.array([], dtype=int)
            self.ac_collision_height = np.array([], dtype=int)
            self.ac_collision_sweep = np.array([], dtype=int)

        self.counter_strikes = 0
        self.counter = 0
        
        # in case of many birds, it is wise to use the function quickdist
        # which needs a coslat value. This one is for Eindhoven airport, NL
        self.coslat = np.cos(np.radians(51.4192475))
        # Create datalog instance

        # self.log = Datalog()
        self.reset()
        
        
        # for logging we need a filename. If we use the IC, the name is given
        # this is a placeholder to avoid errors in case no logname is provided in the 
        # scenario
        self.filename2save = 'ZZZZ_bird_logging'
        
        
        ''' to perform CD, we need to know whether the bird enters the 
        safety envelope. For that purpose, we need to define the expansion of the 
        safety envelope. Here, the required parameters are initialized. In "add_envelope", 
        they will be assigned to newly created aircraft'''
    


        return
    
    def create(self, n=1):
        super().create(n)

        self.ac_collision_radius[-n:] = 20. # unit is m
        self.ac_collision_height[-n:] = 1.42 # unit is m
        self.ac_collision_sweep[-n:]  = 24. # degrees  

    def assign_envelope(self, ac_type):
        
        ### now we have hard-coded radii and heights. Add a function here
        ### to assign the values depending on the aircraft type
        # different numbers to init for checking only
        coll_rad = 42.
        coll_height = 1.49
        
        # sweep is relevant for fixed-wing aircraft to ensure that
        # only collisions to the front of the aircraft are counted
        # if you want the full circle, use 90 for your aircraft types
        coll_sweep = 26.
        
        return coll_rad, coll_height, coll_sweep
 
    def set_logname(self, filename):
        # assigning the actual logname for the logfile recording collision parameters.
        
        self.filename2save = filename    
        
        return
    
    def reset(self):

        
        self.counter = self.counter + 1

        self.filename_set = False
        
        
        return        
     
    def conflict_detection(self):
 
        # only run if there are birds and aircraft
        if len(bird_traf.id) == 0 or bs.traf.ntraf == 0:
            return

        # once we have the first birdies, we would like to have a filename to store
        if not self.filename_set:
            bird_traf.filename_judihui = self.filename2save
            self.filename_set = True

        # bring input to correct format
        lat_birds, lon_birds, alt_birds, lat_aircraft, lon_aircraft, alt_aircraft = self.reshape()       

        # first filter:lateral distance
        dxy = self.distance(np.radians(lat_birds), np.radians(lon_birds), np.radians(lat_aircraft), np.radians(lon_aircraft))
        #dxyqwik = self.quick_distance(np.radians(lat_birds), np.radians(lon_birds), np.radians(lat_aircraft), np.radians(lon_aircraft))
        #print simt, bs.traf.id, bird_traf.id, "dist",dxy, "kwik", dxyqwik, "deltaalt",abs(alt_birds - alt_aircraft), "alt_b",alt_birds, "alt_ac",alt_aircraft
        #print "lat b", lat_birds, "lon b", lon_birds, "lat ac", lat_aircraft, "lon ac", lon_aircraft
        #print "rad b", bird_traf.collision_radius, "rad ac", bs.traf.ac_collision_radius,  "h ac", bs.traf.ac_collision_height
        #print
        
        # is this already in the dangerous area?
        # input for ac is already radius (diameter/2)
        # for birds we are fixed now: 0.5m individuals, 5m flocks

        c_rad_birds = bird_traf.collision_radius.reshape(len(bird_traf.collision_radius), 1)
        c_rad_ac = self.ac_collision_radius.reshape(1,len(self.ac_collision_radius))
      #  print ('dxy', dxy, 'c_rad_birds', c_rad_birds, 'c_rad_ac', c_rad_ac, 'sum', c_rad_birds + c_rad_ac)
        dangerous_dist = (dxy <= c_rad_birds + c_rad_ac)*1.
       # print ('dxy', dxy, 'c_rad_birds', c_rad_birds, 'c_rad_ac', c_rad_ac, 'sum', c_rad_birds + c_rad_ac)
        
        # only continue for bird-ac combinations where the lateral distance is too small
        if len(np.where(np.any(dangerous_dist == 1. , axis = 1) == True)[0]) > 0 and \
           len(np.where(np.any(dangerous_dist == 1. , axis = 0) == True)[0]) > 0 : 
              # print "dangerous dist"
              

               
               # filter
               alt_birds = alt_birds[np.where(np.any(dangerous_dist == 1. , axis = 1) == True)[0]]
               alt_aircraft = alt_aircraft[0][np.where(np.any(dangerous_dist == 1., axis = 0) == True)[0]]
               collision_height = self.ac_collision_height[np.where(np.any(dangerous_dist == 1. , axis = 0) == True)[0]]                
               
               # filter
               lat_birds    = lat_birds[np.where(np.any(dangerous_dist == 1. , axis = 1) == True)[0]]
               lon_birds    = lon_birds[np.where(np.any(dangerous_dist == 1. , axis = 1) == True)[0]]
               lat_aircraft = lat_aircraft[0][np.where(np.any(dangerous_dist == 1., axis = 0) == True)[0]]
               lon_aircraft = lon_aircraft[0][np.where(np.any(dangerous_dist == 1., axis = 0) == True)[0]]
               
               
               
               
               # is a list and has therefore to be converted 
               id_ac = np.array(bs.traf.id)
               # used for later
               sweep = self.ac_collision_sweep[np.where(np.any(dangerous_dist == 1. , axis = 0) == True)[0]]
               hdg   = bs.traf.hdg[np.where(np.any(dangerous_dist == 1. , axis = 0) == True)[0]]
               id_ac = id_ac[np.where(np.any(dangerous_dist == 1. , axis = 0) == True)[0]]
               id_bird = bird_traf.id[np.where(np.any(dangerous_dist == 1. , axis = 1) == True)[0]]




               # altiutde difference: 
               # only birds in the same plane as the aircraft are interesting
               # input is already ac_height/2
               dangerous_alt = (abs(alt_birds - alt_aircraft) <= collision_height)*1.
              # print ('alt bird', alt_birds, 'alt ac',alt_aircraft,'collheight', collision_height, 'dangerous alt', dangerous_alt )
               
               # only continue if there are bird-ac combinations within 
               #dangerous distance AND in the same altitude band
    
                # only continue if any birds and aircraft are in the same altitude layer
               if len(np.where(np.any(dangerous_alt == 1. , axis = 1) == True)[0]) > 0 and \
                  len(np.where(np.any(dangerous_alt == 1. , axis = 0) == True)[0]) > 0 :
                    los_bird_idx = np.where(np.any(dangerous_alt == 1., axis=1) == True)[0]
                    los_ac_idx = np.where(np.any(dangerous_alt == 1., axis=0) == True)[0]
                    
                    
                    # filter
                    lat_birds    = lat_birds[np.where(np.any(dangerous_alt == 1. , axis = 1) == True)[0]]
                    lon_birds    = lon_birds[np.where(np.any(dangerous_alt == 1. , axis = 1) == True)[0]]
                    

                    lat_aircraft = lat_aircraft[np.where(np.any(dangerous_alt == 1., axis = 0) == True)[0]]
                    lon_aircraft = lon_aircraft[np.where(np.any(dangerous_alt == 1., axis = 0) == True)[0]]
                    

                    sweep        = sweep[np.where(np.any(dangerous_alt == 1. , axis = 0) == True)[0]]
                    hdg          = hdg[np.where(np.any(dangerous_alt == 1. , axis = 0) == True)[0]]
                    id_ac        = id_ac[np.where(np.any(dangerous_alt == 1. , axis = 0) == True)[0]]
                    id_bird      = id_bird[np.where(np.any(dangerous_alt == 1. , axis = 1) == True)[0]]

            
                    # bearing between bird and aircraft
                    bearing = self.bearing(np.radians(lat_aircraft), np.radians(lon_aircraft), np.radians(lat_birds), np.radians(lon_birds))
                    #print "simt", simt
                    #print "bearing", bearing, "ac pos", lat_aircraft, lon_aircraft, "bird pos", lat_birds, lon_birds
                    # top view of the aircraft: bird strikes only occurr if 
                    # they take place in the front half (end is wingtip)
                    # relative values required
                    pacman_high = ( 90. + sweep)
                    pacman_low  = (-90. - sweep)
                    #print "pacmaaaan", pacman_low, pacman_high
                    # explanation in method
                    delta_heading = ((((hdg - bearing)%360.) + 180. + 360.)% 360.) - 180.        
                   # print ("delta heading", delta_heading)
                
                    # and is it within the front area of the aircraft?
                    # then we have a strike!
                    pacman = ((delta_heading > pacman_low) & (delta_heading < pacman_high) )* 1.
                    #print "pacman", pacman
                    # which birds were hit? 

                    id_hit_birds = id_bird[np.where(np.any(pacman ==1., axis = 1) == True)[0]]
                    id_hit_ac = id_ac[np.where(np.any(pacman ==1., axis = 0) == True)[0]]
                    #print
                    #print "hdg aircraft", hdg
                    #print "pacman", pacman_high, pacman_low, "delta hdg", delta_heading, "bearing",bearing
                    
                    # only continue if there was a strike
                    if len(id_hit_birds) > 0:
                        strike_time = bs.sim.simt

                        idx_birds_hit = []
                        bird_data = []
                        lat_birds = []
                        lon_birds = []
                        
                        for identity in id_hit_birds:
                            # this is the index in the class birds
                            index_birds = int(np.where(bird_traf.id == float(identity))[0][0])
                            idx_birds_hit.append(index_birds) 
                            lat_birds.append(bird_traf.lat[index_birds])
                            lon_birds.append(bird_traf.lon[index_birds])
                            
                            bird_data.append(str(bird_traf.id[index_birds]) + ' \t ' +  str(bird_traf.tas[index_birds]) \
                                              + ' \t ' + str(bird_traf.lat[index_birds]) + ' \t ' +  str(bird_traf.lon[index_birds]) \
                                              + ' \t ' + str(bird_traf.alt[index_birds]) + ' \t ' + str(bird_traf.bird_size[index_birds])\
                                              + ' \t ' + str(bird_traf.collision_radius[index_birds]) + ' \t ' + str(bird_traf.no_inds[index_birds]) \
                                              + ' \t ' + str(bird_traf.flock_flag[index_birds]))   
                        
                            # log data
                            #self.log.write(str(strike_time), "BIRD", str(bird_traf.id[index_birds]),\
                            #                str(bird_traf.tas[index_birds]), str(bird_traf.lat[index_birds]), \
                             #               str(bird_traf.lon[index_birds]), str(bird_traf.alt[index_birds]), \
                             #               str(bird_traf.cat[index_birds]), str(bird_traf.flock_flag[index_birds]), \
                               #             "BUFFER")
                                                                    
                            
                        
                        # remove them        
                        bird_traf.remove_bird(idx_birds_hit)

          
                        # store IDs of hit aircraft
                        #id_hit_ac = id_ac[np.where(np.any(pacman ==1., axis = 0) == True)[0]]

                        
                        # increase counter
                        self.counter_strikes = self.counter_strikes + len(id_hit_ac)

                        #  store the aircraft id's of the hit aircraft - preparation
                        to_mark = []
                        for identity in id_hit_ac:

                            if identity in bs.traf.id:
                                to_mark.append(int(np.where(np.array(bs.traf.id) == identity)[0][0]))
                        to_mark = np.unique(to_mark)
                       
                        # store the aircraft indices of the hit aircraft - execution 
                        # idx is the index of the array: 0:n
                        # pos is the value of to_mark at the index - in this case
                        # it marks the position of the aircraft within the hit_ac array
                        
                        for idx in to_mark:

                            
                            ac_data = str(bs.traf.id[idx]) + ' \t ' + str(bs.traf.tas[idx]) \
                             + ' \t ' + str(bs.traf.lat[idx]) + ' \t ' + str(bs.traf.lon[idx]) \
                             + ' \t ' + str(bs.traf.alt[idx]) + ' \t ' + str(bs.traf.type[idx])
                            
                            # which bird did this aircraft hit?
                            # determination via lat-lon difference (max. 0.001 resp.)
                            for i in range(len(idx_birds_hit)):

                                if (abs(bs.traf.lat[idx] - lat_birds[i]) < 0.001) and \
                                    (abs(bs.traf.lon[idx] - lon_birds[i] < 0.001)):

                                        #self.log.write(bird_traf.filename2save, str(strike_time), ac_data, bird_data[i], "collision")
                                        self.log.write(bird_traf.filename_judihui, str(strike_time), ac_data, bird_data[i], "collision")


                                        
                            # log data
                           # self.log.write(str(strike_time), "AIRCRAFT", str(bs.traf.id[idx]), str(bs.traf.tas[idx]), \
                            #               str(bs.traf.lat[idx]), str(bs.traf.lon[idx]), str(bs.traf.alt[idx]), str(bs.traf.type[idx]), \
                             #              str(bs.traf.orig[idx]), str(bs.traf.dest[idx]))
                                           
                         # write: str(strike_time), aircraft, bird --> header true/false?



                        # store the aircraft id's as well
                        # even if an aircraft has more than one strike: we only want to store it's id once
                        # np.in1d: is arg1 in arg2?
                        #new_id_hit_ac = id_hit_ac[np.where(np.in1d( id_hit_ac, bs.traf.nr_strikes, invert = True))]        
                        #bs.traf.nr_strikes = np.append(bs.traf.nr_strikes, new_id_hit_ac)
                        #print "traf_strikes", bs.traf.nr_strikes

                    # log data of collision
                        

                        #self.log.save(bird_traf.filename2save) 
                        self.log.save(bird_traf.filename_judihui)
                        print('saved', bird_traf.filename_judihui)

    
        return 



    # format input for calculation
    # hint: name the reshapes differently oooooooor make an individual module
    # individual module might have adavantages as there are inputs from traf. and from birds.
    # height and radius of aircraft: store in input_files or find ways to get it via other parameters
    def reshape(self):


        # birds are the columns, aircraft are the rows
        lat_birds = bird_traf.lat.reshape((len(bird_traf.lat),1))
        lon_birds = bird_traf.lon.reshape((len(bird_traf.lon),1))
        alt_birds = bird_traf.alt.reshape((len(bird_traf.alt),1))

        
        lat_aircraft = bs.traf.lat.reshape((1,len(bs.traf.lat)))
        lon_aircraft = bs.traf.lon.reshape((1,len(bs.traf.lon)))
        alt_aircraft = bs.traf.alt.reshape((1,len(bs.traf.alt)))        
        
        
        
        return lat_birds, lon_birds, alt_birds, lat_aircraft, lon_aircraft, alt_aircraft
        
        
    # use the haversine formula to calculate the distance between birds and ac    
    # input is already in radians
    def distance(self, lat_birds, lon_birds, lat_ac, lon_ac):
        
        a = np.sin((lat_birds-lat_ac)/2)*np.sin((lat_birds-lat_ac)/2) + \
            np.cos(lat_birds)*np.cos(lat_ac)*np.sin((lon_birds-lon_ac)/2)*np.sin((lon_birds-lon_ac)/2)
   
        c= 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance= bird_traf.earth_radius*c # 6317000m corresponds to the earth radius

        
        return distance


    def quick_distance(self,lat_birds, lon_birds, lat_ac, lon_ac):
        '''a bit less accurracy but sooooo much faster '''
        dx = bird_traf.earth_radius * (lon_birds - lon_ac) * self.coslat
        dy = bird_traf.earth_radius * (lat_birds - lat_ac)
        distance = np.sqrt(dx*dx + dy*dy)
        
        return distance


    def bearing(self, lat1, lon1, lat2, lon2):
    
        deltal = lon2-lon1
    
    # calculate runway bearing
        bearing = np.arctan2(np.sin(deltal)*np.cos(lat2), (np.cos(lat1)*np.sin(lat2)-
                np.sin(lat1)*np.cos(lat2)*np.cos(deltal)))
        
        # normalize to 0-360 degrees
        bearing = (np.degrees(bearing)+360)%360
        
        return bearing


class Datalog():
    def __init__(self):
        print ("we are in CDR datalog")

        self.buffer=[]
        
        #filename will be set in first run
        self.filename_flag = False

         
        return
    
    def write(self, filename,time, ac_data, bird_data, occurrence_type):


        # filename[5:15] is the date
        self.buffer.append( filename +" \t "  + time +" \t " + ac_data + " \t " + bird_data + '\t' + occurrence_type + chr(13) + chr(10))
       
        return

    def save(self, filename):
        
        # files are saved per airport. Hence only create a new file if 
        # no file for this airport exists
        log_file = os.path.join(dir, "bird_CDR_log/"  + filename + ".txt" )
        print ('in save, logpath is', log_file)
        if not os.path.isfile(log_file):  
           # log_file = "log/" + filename_def + ".txt"
           # self.log_file = os.path.join(dir, log_file)
            #print "INIT", filename_def, filename, log_file
            
            with open(log_file, "a") as writeto:
                writeto.write('date \t time \t id_ac \t tas \t lat \t lon \t alt \t type \t id_bird \t tas \t lat \t lon \t alt \t size \t coll_rad \t number \t flock_flag \t occurrence type \n')
        
        
       # if not self.filename_flag:
          #  filename = "log/" + filename + ".txt"
            
           # self.filename = os.path.join(dir,  filename)

            # write the header            
            #with open(self.filename, "a") as writeto:
            #    writeto.write('time \t id_ac \t tas \t lat \t lon \t alt \t type \t orig \t dest id_bird \t tas \t lat \t lon \t alt \t size \t coll_rad \t number \t flock_flag \n')

            #self.filename_flag = True

# Write buffer to file 

        with open(log_file, "a") as writeto:
            for i in range(len(self.buffer)):

                writeto.write(self.buffer[i])


        self.buffer = []    
        


        
        
        
        
        return
   
