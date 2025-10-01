# BlueSky Birds Plugin

A BlueSky simulation plugin that adds realistic bird traffic simulation and visualization capabilities to the [BlueSky](https://github.com/TUDelft-CNS-ATM/bluesky) air traffic management simulator.

Jointly developed by [amvlab](https://amvlab.eu) and [Dr. Isabel Metz](https://www.linkedin.com/in/isabel-c-metz/) from DLR.

## Overview

This plugin extends BlueSky with bird traffic simulation functionality, allowing researchers and aviation professionals to study bird-aircraft interactions and wildlife hazard management in a controlled simulation environment.

## Features

- **Bird Traffic Simulation** (`birdtraffic.py`): Core simulation engine for bird movement and behavior
- **Real-time Visualization** (`glbirds.py`): OpenGL-based rendering of birds in BlueSky's Qt6-based GUI

## Components

### birdtraffic.py
The core simulation module that handles the bird traffic arrays. These are similar to BlueSky traffic arrays.

### glbirds.py
The visualization module providing OpenGL-based bird rendering and integration with BlueSky's Qt6-based GUI.

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

Once both components are loaded, any created birds will appear in the simulation environment.

### Bird Commands

**CREBIRD** - Create a bird in the simulation
```
CREBIRD birdid,type,lat,lon,hdg,alt,spd
```
- `birdid`: Unique bird identifier
- `type`: Bird type (default: goose)
- `lat,lon`: Position coordinates
- `hdg`: Heading in degrees
- `alt`: Altitude in feet
- `spd`: Speed in knots

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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
