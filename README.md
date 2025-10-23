# BlueSky Birds Plugin



https://github.com/user-attachments/assets/b43ee9d3-bbc6-4dfc-8499-43ab9e1faf2f



A BlueSky simulation plugin that adds realistic bird traffic simulation and visualization capabilities to the [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) air traffic management simulator.

Jointly developed by [amvlab](https://amvlab.eu) and [Dr. Isabel Metz](https://www.linkedin.com/in/isabel-c-metz/) from DLR.

## Overview

This plugin extends BlueSky with bird traffic simulation functionality, allowing researchers and aviation professionals to study bird-aircraft interactions and wildlife hazard management in a controlled simulation environment.

## Features

- **Bird Traffic Simulation** (`plugins/birdtraffic.py`): Core simulation engine for bird movement and behaviour
- **Real-time Visualization** (`plugins/glbirds.py`): OpenGL-based rendering of birds in BlueSky's Qt6-based GUI
- **Demo Scenarios** Two demo scenarios are included. One loads bird movements with pre-determined tracks. The other creates an individual bird which moves in the set direction.

## Components

### Plugins

To use these plugins place the contents of the `plugins` directory wherever your BlueSky instance looks for plugins.

   - #### [birdtraffic.py](https://github.com/amvlab/plugins_bluesky_birds/blob/main/plugins/birdtraffic.py)
     The core simulation module that handles the bird traffic arrays. These are similar to BlueSky traffic arrays.

   - #### [glbirds.py](https://github.com/amvlab/plugins_bluesky_birds/blob/main/plugins/glbirds.py)
     The visualization module providing OpenGL-based bird rendering and integration with BlueSky's Qt6-based GUI.

### Scenarios

Place the contents of the `scenarios` directory whereever your BlueSky instance looks for scenarios.

   - #### [demo1.scn](https://github.com/amvlab/plugins_bluesky_birds/blob/main/scenarios/demo1.scn)
     This demo loads pre-determined bird movements from `plugins/bird_movements/sample_birds.csv`.

   - #### [demo2.scn](https://github.com/amvlab/plugins_bluesky_birds/blob/main/scenarios/demo1.scn)
     This demo creates a single bird with a given speed, heading, and altitude.

## Installation

1. Ensure you have BlueSky version 1.1.0 installed and configured
2. Clone this repository into your BlueSky plugins directory
3. The plugins will be automatically detected by BlueSky on startup

No additional dependencies are required beyond a standard BlueSky installation.

## Usage

### Loading the Bird Traffic Simulation Plugin

The bird traffic simulation can be loaded in multiple ways:

1. **Automatically via settings**: Add the bird traffic plugin to your BlueSky `settings.cfg` file:
   ```
   enabled_plugins = ['BIRDSIM']
   ```
   Then start BlueSky - the bird traffic simulation will be automatically loaded

2. **Manually from console or scenario**:
   ```
   PLUGIN LOAD BIRDSIM
   ```

### Loading the Bird Visualization Plugin

The bird visualization must currently be loaded manually from the BlueSky console or from a scenario:

1. Load the GUI plugin:
   ```
   PLUGIN LOAD BIRDGUI
   ```

2. Add the bird visualization:
   ```
   ADDVIS BIRDTRAFFIC
   ```

Once both components are loaded, any created birds will appear in the simulation environment. Note that this plugin cannot be loaded from `settings.cfg`

### Bird Commands

**CREBIRD** - Create a bird in the simulation
```
CREBIRD birdid,type,lat,lon,hdg,alt,spd
```
- `birdid`: Unique bird identifier
- `type`: Bird type (default: pelican)
- `bird_size`: 1= small (e.g. a Sparrow), 2 = medium (e.g. a duck), 3 = large (e.g. a pelican). 
- `no_inds`: how many birds are flying together in that bird id
- `lat`: Latitude [degrees]
- `lon`: Longitude [degrees]
- `trk`: track [degrees]
- `alt`: altitude [feet]
- `spd`: speed in [knots]

Example:
```
CREBIRD HAWK01,hawk,52.3676,4.9041,090,500,15
```

**DELBIRD** - Remove a bird from the simulation
```
DELBIRD birdid
```
- `birdid`: Unique bird identifier

Example:
```
DELBIRD HAWK01
```

**BIRDS** - Open a `csv` file with predefined bird tracks, e.g. from avian radar, weather radar or GPS tracking
Refer to this [publication](https://doi.org/10.3390/aerospace5040112) for documentation of integrating avian and weather radar data

```
BIRDS filepath
```
- `filepath`: place the bird track csv inside the plugins directory and give the path, e.g. `birds_movements/sample_birds`

Example:
```
BIRDS plugins/bird_movements/sample_birds
```

#### Bird tracks file format

This file must be a `csv` file. Ech individual line refers to a bird position at a given time. It is optional to include the header column names but they must have the following fields: 

- 'id', 'time', 'lon','lat', 'alt', 'cat', 'no_individuals', 'flock_flag','id1', 'hdg', 'spd'

Each field is explained below:

- `id`: Unique bird identifier
- `time`: Simulation time of recorded Position [seconds]
- `lon`: longitude [decimal degree]
- `lat`: latitude [decimal degree]
- `alt`: altitude [metres]
- `cat`: bird category or type, e.g. [small, medium, large] or [goose, hawk, pelican]
- `no_individuals`: how many birds are represented by the bird id [-]
- `flock_flag`: if the bird id represents a single bird or a flock [-]
- `id1`: identical to 'id' EXCEPT in the last line of that bird. There, 'id1' must differ to get the bird id removed from the simulation
- `trk`: track of bird [degrees] 
- `spd`: Speed of bird [metres per second]

In the line with the last recorded bird position, 'id1' must differ from 'id' to delete the bird from the simulation after their last recorded position.

An example scenario that loads pre-deterimned bird tracks can be seen in [demo1.scn](https://github.com/amvlab/plugins_bluesky_birds/blob/main/scenarios/demo1.scn). This scenario calls the bird file found in `plugins/bird_movements/sample_birds.csv`.

This scenario can be loaded in BlueSky by entering the follwoing command in the BlueSky console.

```
IC demo1
```

**BIRDLABEL** "Toggle between bird label display modes (ID or type).
```
BIRDLABEL
```
- No arguments

Example:
```
BIRDLABEL
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
