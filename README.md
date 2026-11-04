# Multi-Material CT Simulation and Synthetic Data Generation

An interactive, high-fidelity X-ray Computed Tomography (CT) simulation and synthetic dataset generation platform built with **gVXR (gVirtualXRay)** and **Streamlit**.

## Features

- **Multi-Material Mesh Support**: Load and assemble multi-part CAD models (.stl) with distinct chemical compositions (elements, compounds, mixtures) and densities.
- **X-Ray Source Modeling**: 
  - Monochromatic beam modeling
  - Polychromatic beam modeling via Kramers' Law spectrum generation
  - Beam filtration support (Al, Cu, Sn, etc.)
- **Detector Configuration**: Flat-panel detector setup with configurable dimensions, pixel spacing, and source-to-isocenter / isocenter-to-detector distances.
- **CT Acquisition Simulation**: 
  - Configurable rotation angles and projection counts (up to 360°)
  - Poisson quantum noise simulation
  - Sinogram slice extraction and live projection previews
- **Synthetic Data Export**:
  - 16-bit / 32-bit TIFF projection stacks
  - Sinogram images
  - Comprehensive JSON metadata and scan configuration
- **Interactive Web Interface**: Streamlit-based UI with interactive 3D visualization, auto-framing calculation, and real-time background execution.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jaypalod734/Multi-Material-CT-Simulation-and-Synthetic-Data-Generation.git
   cd Multi-Material-CT-Simulation-and-Synthetic-Data-Generation
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Generate sample STL phantoms (optional):**
   ```bash
   python generate_sample_stls.py
   ```

2. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```

3. Configure your scanner geometry, assign materials to mesh components, set acquisition parameters, and run the simulation.

## Project Structure

```
├── .streamlit/             # Streamlit configuration
├── app.py                  # Main Streamlit application
├── config.py               # Simulation configuration dataclasses
├── generate_sample_stls.py # Generator for multi-material calibration phantoms
├── materials_db.json       # Predefined materials database
├── mesh_manager.py         # Mesh transformation and material assignment
├── output_manager.py       # TIFF stack, sinogram, and metadata saving
├── requirements.txt        # Python package dependencies
├── run_sim_task.py         # Subprocess simulation runner
├── scanner_setup.py        # gVXR source and detector geometry setup
├── simulation.py           # Core simulation execution engine
└── utils.py                # Helper utilities and spectra calculations
```

## License

This project is licensed under the MIT License.
