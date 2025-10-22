"""
Bird traffic simulation plugin

Jointly developed by amvlab and Dr. Isabel Metz from DLR
"""

import numpy as np
import os
import pandas as pd

import bluesky as bs
from bluesky import stack
from bluesky.tools.aero import ft, kts
from bluesky import core, stack
from bluesky.network.publisher import state_publisher

# Update rate of bird update messages [Hz]
BIRDUPDATE_RATE = 1


def init_plugin():

    bird_traf = BirdTraffic()

    config = {
        # The name of your plugin
        "plugin_name": "BIRDSIM",
        "plugin_type": "sim",
        "update": bird_traf.update,
        "reset": bird_traf.reset,
    }
    return config


class BirdTraffic(core.Entity):

    def __init__(self):
        """Initialize the BirdTraffic entity with default parameters and reset all bird data."""
        super().__init__()

        # to find bird files, we need to know the current directory
        self.dir = os.path.dirname(__file__)

        # go back to the roots
        self.reset()

        # Label type to show in gui. Default is 'id;
        self.labels = ["id", "type"]
        self.lbl_type = self.labels[0]

        # some global parameters
        self.earth_radius = 6371000.0  # Earth radius in m

    def update(self):
        """Update bird positions either from a loaded bird movement file or individually created."""

        if self.is_loading_bird_movements:
            self.update_bird_movements()

        else:
            self.update_position_individuals()

    def load_bird_movements(self, filename):
        """Load bird movement data from a CSV file and prepare it for simulation.

        Args:
            filename: Name of the CSV file (without extension) containing bird movement data
        """
        stack.echo(f"Bird file {filename} loaded!")

        self.is_loading_bird_movements = True
        filename2use = filename + ".csv"
        filename2read = os.path.join(self.dir, filename2use)

        # give the user the choice to pass a bird movement file with or without header
        data = pd.read_csv(filename2read, engine="python", index_col=False)

        if not "no_individuals" in data.columns:
            data.columns = [
                "id",
                "time",
                "lon",
                "lat",
                "alt",
                "cat",
                "no_individuals",
                "flock_flag",
                "id1",
                "trk",
                "spd",
            ]

        # ensure that input data is sorted first by id and then by time
        data = data.sort_values(["id", "time"], ascending=[True, True])
        # for updating positions during the simulation, we always need the current and next position
        data["timeshift"] = data["time"].shift(-1)
        data["latshift"] = data["lat"].shift(-1)
        data["lonshift"] = data["lon"].shift(-1)
        data["altshift"] = data["alt"].shift(-1)

        # now we need the right order in time only
        data = data.sort_values(by="time")
        # feed the data to the simulation
        self._assign_values(data)

    def update_bird_movements(self):
        """Update bird positions from loaded movement file data based on simulation time."""

        # only required if there is at least one bird left to be simulated
        # if no birds are left, we can get out again
        if len(self.input_time) < 1 or bs.sim.simt > self.input_time[-1]:
            self.reset()
            return

        # all birds which had a new position since the last update are considered
        idx_time_passed = np.argmax(self.input_time > bs.sim.simt)

        # bird info to the check-function
        if idx_time_passed > 0:
            input_time = self.input_time[:idx_time_passed]
            input_id1 = self.input_id1[:idx_time_passed]
            input_id2 = self.input_id2[:idx_time_passed]
            input_bird_size = self.input_bird_size[:idx_time_passed]
            input_no_inds = self.input_no_inds[:idx_time_passed]
            input_flock_flag = self.input_flock_flag[:idx_time_passed]
            input_alt = self.input_alt[:idx_time_passed]
            input_lat = self.input_lat[:idx_time_passed]
            input_lon = self.input_lon[:idx_time_passed]
            input_spd = self.input_spd[:idx_time_passed]
            input_trk = self.input_trk[:idx_time_passed]

            input_time_next = self.input_time_next[:idx_time_passed]
            input_lat_next = self.input_lat_next[:idx_time_passed]
            input_lon_next = self.input_lon_next[:idx_time_passed]
            input_alt_next = self.input_alt_next[:idx_time_passed]

            # this function tests if the bird id passed are already known
            # to the simulation. The ones which appear new, will be initiated
            self._check_bird(
                input_id1, input_bird_size, input_no_inds, input_flock_flag, input_alt
            )

            # remove birds which have reached their last position in the recording
            id_to_remove = input_id1[np.where(input_id1 != input_id2)[0]]

            if len(id_to_remove) > 0:
                to_remove = []

                for identity in id_to_remove:

                    if identity in self.id:

                        to_remove = to_remove + list(np.where(self.id == identity)[0])

                # delete their data
                self.remove_bird(to_remove)

            # position update
            # explanation np.ndenumerate: index is the iterator through the array
            # while id1 is the actual value.
            # e.g. ([4,3,7]): index = 0,1,2, id1 = 4,3,7

            for index, id1 in np.ndenumerate(input_id1):
                index_to_replace = np.where(self.id == id1)[0]

                self.last_ts[index_to_replace] = input_time[index]
                self.last_lat[index_to_replace] = input_lat[index]
                self.last_lon[index_to_replace] = input_lon[index]
                self.last_alt[index_to_replace] = input_alt[index]

                self.next_ts[index_to_replace] = input_time_next[index]
                self.next_lat[index_to_replace] = input_lat_next[index]
                self.next_lon[index_to_replace] = input_lon_next[index]
                self.next_alt[index_to_replace] = input_alt_next[index]

                self.gs[index_to_replace] = input_spd[index]
                self.trk[index_to_replace] = input_trk[index]

            # delete processed data to save space
            self._remove_input(idx_time_passed)

        # update bird positions
        self.update_position_bird_movements()

    ######################################################################################################
    ######################################################################################################

    # CREATE INDIVIDUAL BIRDS

    ######################################################################################################
    ######################################################################################################

    def create_individual(
        self,
        birdid,
        birdtype,
        flock_flag,
        bird_size,
        no_inds,
        birdlat,
        birdlon,
        birdtrk,
        birdalt,
        birdspd,
    ):
        """Creating solo birdies for testing"""

        # add one bird object
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
        self.id = np.append(self.id, birdid)
        self.flock_flag = np.append(self.flock_flag, flock_flag)
        self.type = np.append(self.type, birdtype)
        self.bird_size = np.append(self.bird_size, bird_size)
        self.no_inds = np.append(self.no_inds, no_inds)
        self.angry_birds = np.append(self.angry_birds, False)

        # Positions
        self.lat = np.append(self.lat, birdlat)
        self.lon = np.append(self.lon, birdlon)
        self.alt = np.append(self.alt, birdalt)

        # for linear extrapolation of bird position
        self.last_ts = np.append(self.last_ts, bs.sim.simt)
        self.last_lat = np.append(self.last_lat, birdlat)
        self.last_lon = np.append(self.last_lon, birdlon)
        self.last_alt = np.append(self.last_alt, birdalt)

        # placeholders only (required wehn using bird movement files)
        self.next_ts = np.append(self.next_ts, -999)
        self.next_lat = np.append(self.next_lat, -999)
        self.next_lon = np.append(self.next_lon, -999)
        self.next_alt = np.append(self.next_alt, -999)

        # Heading
        self.trk = np.append(self.trk, birdtrk)

        # Velocities
        self.gs = np.append(self.gs, birdspd)
        # zero for now
        self.vs = np.append(self.vs, 0.0)

        # for wildlife strike detection. Values obtained from and documented in
        # https://doi.org/10.3390/aerospace5040112
        if no_inds == 1:

            if bird_size == 1:
                self.collision_radius = np.append(self.collision_radius, 0.5 * 0.32)
            elif bird_size == 2:
                self.collision_radius = np.append(self.collision_radius, 0.5 * 0.68)
            elif bird_size == 3:
                self.collision_radius = np.append(self.collision_radius, 0.5 * 1.40)
            else:
                stack.echo(f"Solo birdie size not captured, {bird_size}")

        else:

            if bird_size == 1:
                self.collision_radius = np.append(
                    self.collision_radius, np.sqrt(no_inds) * 0.5 * 0.32 + 0.06
                )
            elif bird_size == 2:
                self.collision_radius = np.append(
                    self.collision_radius, np.sqrt(no_inds) * 0.5 * 0.68 + 0.16
                )
            elif bird_size == 3:
                self.collision_radius = np.append(
                    self.collision_radius, np.sqrt(no_inds) * 0.5 * 1.40 + 0.41
                )
            else:
                stack.echo(f"Gang birdie size not captured, {bird_size}")

    ##############
    # UPDATE POSITION
    ##############

    def update_position_bird_movements(self):
        """Update bird positions using linear interpolation between recorded positions."""

        # simply linear interpolation between two given recorded positions

        # timedelta between last and next position
        entire_delta_t = self.next_ts - self.last_ts
        delta_time_now = bs.sim.simt - self.last_ts

        self.lat = (self.next_lat - self.last_lat) * (
            delta_time_now / entire_delta_t
        ) + self.last_lat
        self.lon = (self.next_lon - self.last_lon) * (
            delta_time_now / entire_delta_t
        ) + self.last_lon

        # altitude only to be interpolated if it actually is changing
        altchange = self.last_alt != self.next_alt

        if np.any(altchange != 0.0):
            idx4altchange = np.where(altchange != 0.0)
            self.alt[idx4altchange] = (
                self.next_alt[idx4altchange] - self.last_alt[idx4altchange]
            ) * (
                delta_time_now[idx4altchange] / entire_delta_t[idx4altchange]
            ) + self.last_alt[
                idx4altchange
            ]

        self.vs = (self.next_alt - self.last_alt) / (entire_delta_t)

    def update_position_individuals(self):
        """Update positions of individual birds based on their current speed and heading.

        Uses current position, speed, and heading to extrapolate new position.
        Handles coordinate conversion between radians and degrees.
        """
        delta_time_now = bs.sim.simt - self.last_ts

        delta_distance = self.gs * delta_time_now

        # linear extrapolation based on current position and time/distance covered
        # since last position
        self.lat, self.lon = self._extrapolate_position(
            delta_time_now, delta_distance, self.lat, self.lon, self.trk
        )

        # at the end, set current time as last timestep for extrapolation in
        # the next update round:
        # correct input format required, self.last_ts is an array
        self.last_ts = np.ones(len(self.last_ts)) * bs.sim.simt

    ######################################################################################################
    ######################################################################################################

    # DELETE BIRDS FROM SIMULATION
    # a) because they left the area
    # b) because an aircraft had them for breakfast

    ######################################################################################################
    ######################################################################################################

    def remove_bird(self, index_to_remove):
        """Remove bird from simulation and clean up all associated data.

        Args:
            index_to_remove: Index or array of indices of birds to remove
        """

        # as soon as a bird leaves the simulation, its information has to be removed
        # idx is the index, where the bird info is stored per list

        # list of removed birds
        self.removed_id = np.append(self.removed_id, self.id[index_to_remove])

        self.nbird = self.nbird - 1  # number of birds
        # basic info
        self.id = np.delete(self.id, index_to_remove)
        self.type = np.delete(self.type, index_to_remove)
        self.flock_flag = np.delete(self.flock_flag, index_to_remove)
        self.bird_size = np.delete(self.bird_size, index_to_remove)
        self.no_inds = np.delete(self.no_inds, index_to_remove)
        self.angry_birds = np.delete(self.angry_birds, index_to_remove)
        self.collision_radius = np.delete(self.collision_radius, index_to_remove)

        # Positions
        self.lat = np.delete(self.lat, index_to_remove)
        self.lon = np.delete(self.lon, index_to_remove)
        self.alt = np.delete(self.alt, index_to_remove)
        self.trk = np.delete(self.trk, index_to_remove)

        # Velocities
        self.gs = np.delete(self.gs, index_to_remove)  # horizontal airspeed [m/s]
        self.vs = np.delete(self.vs, index_to_remove)  # vertical speed [m/s]

        self.last_ts = np.delete(self.last_ts, index_to_remove)
        self.last_lat = np.delete(self.last_lat, index_to_remove)
        self.last_lon = np.delete(self.last_lon, index_to_remove)
        self.last_alt = np.delete(self.last_alt, index_to_remove)

        self.next_ts = np.delete(self.next_ts, index_to_remove)
        self.next_lat = np.delete(self.next_lat, index_to_remove)
        self.next_lon = np.delete(self.next_lon, index_to_remove)
        self.next_alt = np.delete(self.next_alt, index_to_remove)

    def reset(self):
        """Reset all bird data and initialize empty arrays for bird properties."""

        self.nbird = 0  # number of birds
        self.is_loading_bird_movements = False

        # initialize bird array
        self.id = np.array([], dtype=int)
        self.type = np.array([], dtype=int)
        self.flock_flag = np.array([], dtype=int)
        self.bird_size = np.array([], dtype=int)
        self.no_inds = np.array([], dtype=int)
        self.collision_radius = np.array([])

        # Boolean to indicate if the bird is in danger of striking an aircraft
        # At the moment this just changes the color of the bird.
        self.angry_birds = np.array([], dtype=bool)

        # Positions
        self.lat = np.array([], dtype=float)  # latitude [deg]
        self.lon = np.array([], dtype=float)  # longitude [deg]
        self.alt = np.array([], dtype=float)  # altitude [m]
        self.trk = np.array([], dtype=float)  # traffic track [deg]

        # Velocities
        self.gs = np.array([], dtype=float)  # horizontal airspeed [m/s]
        self.vs = np.array([], dtype=float)  # vertical speed [m/s]

        # for extrapolation of positions
        self.input_time = np.array([])

        # values for calculation
        self.last_ts = np.array([])
        self.last_lat = np.array([])
        self.last_lon = np.array([])
        self.last_alt = np.array([])

        self.next_ts = np.array([])
        self.next_lat = np.array([])
        self.next_lon = np.array([])
        self.next_alt = np.array([])

        self.input_time_next = np.array([])
        self.input_lat_next = np.array([])
        self.input_lon_next = np.array([])
        self.input_alt_next = np.array([])

        # birds which experienced a collision don't fly anymore
        # (when looking into wildlife strikes)
        self.removed_id = np.array([])

    @stack.command(name="CREBIRD")
    def CREBIRD(
        self,
        birdid,
        birdtype: str = "pelican",
        bird_size: int = 3,
        no_inds: int = 1,
        birdlat: float = 52.0,
        birdlon: float = 4.0,
        birdtrk: float = None,
        birdalt: float = 0,
        birdspd: float = 0,
    ):
        # correct some argument units
        if no_inds > 1:
            flock_flag = False
        else:
            flock_flag = True

        birdspd *= kts
        birdalt *= ft

        # create the bird
        self.create_individual(
            birdid,
            birdtype,
            flock_flag,
            bird_size,
            no_inds,
            birdlat,
            birdlon,
            birdtrk,
            birdalt,
            birdspd,
        )

    @stack.command(name="DELBIRD")
    def DELBIRD(self, birdid):
        # bird left the area, landed or was eaten by an aircraft

        # remove_bird needs an array index - convert
        index_to_remove = self._id2idx(birdid)

        self.remove_bird(index_to_remove)

    @stack.command(name="BIRDS")
    def BIRDS(self, filename):
        """when we want to load a csv file with bird movements"""

        self.load_bird_movements(filename)

    @stack.command(name="BIRDLABEL")
    def birdlabel(self):
        """BIRDLABEL"""
        # choose the other option
        self.lbl_type = (
            self.labels[0] if self.lbl_type == self.labels[1] else self.labels[1]
        )

    def _id2idx(self, birdid):
        """Find index of bird id"""

        return np.where(self.id == np.char.upper(np.array(birdid)))[0][0]

    def _assign_values(self, data):
        """Assign values from bird movement data to internal arrays.

        Args:
            data: Pandas DataFrame containing bird movement data
        """
        self.input_id1 = np.array(pd.to_numeric(data["id"])).astype(int)
        self.input_id2 = np.array(pd.to_numeric(data["id1"])).astype(int)
        self.input_lat = np.array(pd.to_numeric(data["lat"]))
        self.input_lon = np.array(pd.to_numeric(data["lon"]))
        self.input_spd = np.array(pd.to_numeric(data["spd"]))
        self.input_trk = np.array(pd.to_numeric(data["trk"]))
        self.input_alt = np.array(pd.to_numeric(data["alt"]))

        self.input_bird_size = np.array(pd.to_numeric(data["cat"])).astype(int)
        self.input_flock_flag = np.array(data["flock_flag"])
        self.input_no_inds = np.array(data["no_individuals"]).astype(int)

        self.input_time = np.array(pd.to_numeric(data["time"]))

        self.input_time_next = np.array(pd.to_numeric(data["timeshift"]))
        self.input_lat_next = np.array(pd.to_numeric(data["latshift"]))
        self.input_lon_next = np.array(pd.to_numeric(data["lonshift"]))
        self.input_alt_next = np.array(pd.to_numeric(data["altshift"]))

    def _extrapolate_position(self, delta_t, delta_s, lat_in, lon_in, trk):
        """Extrapolate bird position using haversine formula.

        Args:
            delta_t: Time delta (not used in current implementation)
            delta_s: Distance delta in meters
            lat_in: Current latitude in degrees
            lon_in: Current longitude in degrees
            trk: Track/heading in degrees

        Returns:
            tuple: New latitude and longitude in degrees
        """
        theta = np.radians(trk)
        last_lat = np.radians(lat_in)
        last_lon = np.radians(lon_in)

        d = delta_s / self.earth_radius

        # calculate the next position with the haversine function
        # check http://www.movable-type.co.uk/scripts/latlong.html for reference

        lat_pos = np.arcsin(
            np.sin(last_lat) * np.cos(d) + np.cos(last_lat) * np.sin(d) * np.cos(theta)
        )

        lon_pos = last_lon + np.arctan2(
            np.sin(theta) * np.sin(d) * np.cos(last_lat),
            np.cos(d) - np.sin(last_lat) * np.sin(lat_pos),
        )

        lat_expol = np.degrees(lat_pos)
        lon_expol = np.degrees(lon_pos)

        return lat_expol, lon_expol

    def _check_bird(
        self, input_id1, input_bird_size, input_no_inds, input_flock_flag, input_alt
    ):
        """Check if birds are already known to simulation and create new ones if needed.

        Args:
            input_id1: Array of bird IDs
            input_bird_size: Array of bird size categories
            input_no_inds: Array of number of individuals per bird group
            input_flock_flag: Array of flock flags
            input_alt: Array of altitudes
        """

        # test 1: not in removed_id: if an avian radar bird was eaten by
        # an aircraft, it still might have track data. But as it has been eaten,
        # it can't fly anymore

        # WARNING: THis bird_idx_to add does refer to the idx in the list "input_id1", not the position
        # in the input list containing all birdies
        bird_idx_to_add = np.where(
            np.in1d(input_id1, self.removed_id, invert=True)
            & (np.in1d(input_id1, self.id, invert=True))
        )[0]

        # np.where idx is not in removed or id
        # append the values with these idxs

        # collision radius of birds: f(span, size, number)
        add_no_inds = input_no_inds[bird_idx_to_add]
        add_bird_size = input_bird_size[bird_idx_to_add]

        # spans are - UPDATE FOR PUBLIC VERSION
        # small: 0.34 m
        # medium: 0.69 m
        # large: 1.43 m

        # radius for protected zone around birds
        add_radius = np.zeros(len(bird_idx_to_add))

        # *0.5 because span is diameter and we need radius
        # reference for values: https://doi.org/10.3390/aerospace5040112
        # first the flocks, solo flyers will then be overwritten
        # small
        add_radius[add_bird_size == 1] = (
            np.sqrt(add_no_inds[add_bird_size == 1]) * 0.5 * 0.32
        ) + 0.06
        # medium
        add_radius[add_bird_size == 2] = (
            np.sqrt(add_no_inds[add_bird_size == 2]) * 0.5 * 0.68
        ) + 0.16
        # flock flock
        add_radius[add_bird_size == 3] = (
            np.sqrt(add_no_inds[add_bird_size == 3]) * 0.5 * 1.40
        ) + 0.41

        # now update for solo flyers
        add_radius[np.where((add_bird_size == 1) & (add_no_inds == 1))] = 0.5 * 0.32
        add_radius[np.where((add_bird_size == 2) & (add_no_inds == 1))] = 0.5 * 0.68
        add_radius[np.where((add_bird_size == 3) & (add_no_inds == 1))] = 0.5 * 1.40

        self.id = np.append(self.id, input_id1[bird_idx_to_add])
        self.bird_size = np.append(self.bird_size, add_bird_size)
        self.no_inds = np.append(self.no_inds, add_no_inds)
        self.flock_flag = np.append(self.flock_flag, input_flock_flag[bird_idx_to_add])
        self.type = np.append(self.bird_size, add_bird_size)
        self.angry_birds = np.append(self.angry_birds, False)
        # and a placeholder for all the other items
        self.last_ts = np.append(self.last_ts, np.zeros([len(bird_idx_to_add)]))
        self.last_lat = np.append(self.last_lat, np.zeros([len(bird_idx_to_add)]))
        self.last_lon = np.append(self.last_lon, np.zeros([len(bird_idx_to_add)]))
        self.last_alt = np.append(self.last_alt, np.zeros([len(bird_idx_to_add)]))

        self.next_ts = np.append(self.next_ts, np.zeros([len(bird_idx_to_add)]))
        self.next_lat = np.append(self.next_lat, np.zeros([len(bird_idx_to_add)]))
        self.next_lon = np.append(self.next_lon, np.zeros([len(bird_idx_to_add)]))
        self.next_alt = np.append(self.next_alt, np.zeros([len(bird_idx_to_add)]))

        self.lat = np.append(self.lat, np.zeros([len(bird_idx_to_add)]))
        self.lon = np.append(self.lon, np.zeros([len(bird_idx_to_add)]))
        self.gs = np.append(self.gs, np.zeros([len(bird_idx_to_add)]))
        self.vs = np.append(self.vs, np.zeros([len(bird_idx_to_add)]))
        self.trk = np.append(self.trk, np.zeros([len(bird_idx_to_add)]))

        self.alt = np.append(self.alt, input_alt[bird_idx_to_add])
        self.collision_radius = np.append(self.collision_radius, add_radius)

    def _remove_input(self, no_to_remove):
        """Remove processed input data to save memory.

        Args:
            no_to_remove: Number of elements to remove from the beginning of input arrays
        """

        # remove the info we already looked at
        # these are the first x elements. So the array now starts at the position [element]+1

        self.input_time = self.input_time[no_to_remove:]
        self.input_id1 = self.input_id1[no_to_remove:]
        self.input_id2 = self.input_id2[no_to_remove:]
        self.input_lat = self.input_lat[no_to_remove:]
        self.input_lon = self.input_lon[no_to_remove:]
        self.input_spd = self.input_spd[no_to_remove:]
        self.input_trk = self.input_trk[no_to_remove:]
        self.input_alt = self.input_alt[no_to_remove:]
        self.input_flock_flag = self.input_flock_flag[no_to_remove:]
        self.input_bird_size = self.input_bird_size[no_to_remove:]
        self.input_no_inds = self.input_no_inds[no_to_remove:]

        self.input_time_next = self.input_time_next[no_to_remove:]
        self.input_lat_next = self.input_lat_next[no_to_remove:]
        self.input_lon_next = self.input_lon_next[no_to_remove:]
        self.input_alt_next = self.input_alt_next[no_to_remove:]

    @state_publisher(topic="BIRDDATA", dt=1000 // BIRDUPDATE_RATE)
    def release_birds(self):
        """release them to the visual world"""

        data = dict()
        # id is necessary for some gui stuff
        data["id"] = self.id
        # id for now
        data["type"] = self.id
        data["lat"] = self.lat
        data["lon"] = self.lon
        data["alt"] = self.alt
        data["trk"] = self.trk
        data["gs"] = self.gs
        data["vs"] = self.vs
        data["lbl_type"] = self.lbl_type
        data["angry_birds"] = self.angry_birds

        return data
