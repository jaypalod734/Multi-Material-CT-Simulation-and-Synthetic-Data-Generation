import os
import json
import math
import struct
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from config import SimulationConfig


def load_materials_db(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the preset materials database from JSON."""
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "materials_db.json")
    
    if not os.path.exists(db_path):
        return {}
    
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("materials", {})


def generate_kramers_spectrum(
    kvp: float = 100.0,
    min_kev: float = 20.0,
    num_bins: int = 25,
    total_photons: int = 50000,
    anode_z: int = 74  # Tungsten (W)
) -> List[Tuple[float, int]]:
    """
    Generate a discrete polychromatic X-ray energy spectrum using Kramers' law
    with low-energy cutoff filtration:
        I(E) proportional to Z * (E_max - E) * T_inherent(E)
    
    Returns a list of tuples: [(energy_in_keV, photon_count), ...]
    """
    if kvp <= min_kev:
        raise ValueError(f"Tube potential kVp ({kvp} kV) must be greater than min_kev ({min_kev} keV)")
    
    energies = [min_kev + (i + 0.5) * (kvp - min_kev) / num_bins for i in range(num_bins)]
    weights = []
    
    for e in energies:
        # Kramers Bremsstrahlung intensity: proportional to (E_max - E)
        brem = anode_z * (kvp - e)
        # Approximate low-energy window / inherent filtration transmission:
        # Low energy photons are heavily attenuated by beryllium/tube glass window
        window_transmission = 1.0 / (1.0 + math.exp(-0.4 * (e - min_kev * 1.3)))
        intensity = max(0.0, brem * window_transmission)
        
        # Characteristic K-alpha / K-beta emission lines for Tungsten target around 59 keV and 67 keV
        if anode_z == 74 and kvp > 69.5:
            if 57.0 <= e <= 60.0:
                intensity += brem * 0.4
            elif 66.0 <= e <= 69.0:
                intensity += brem * 0.25
                
        weights.append(intensity)
    
    sum_w = sum(weights)
    # Prevent division by zero
    if sum_w <= 0:
        sum_w = 1.0
        
    spectrum = []
    # Distribute the total photons across energy bins according to their normalized weight
    for e, w in zip(energies, weights):
        count = int(round((w / sum_w) * total_photons))
        if count > 0:
            spectrum.append((round(e, 2), count))
            
    return spectrum


def validate_config(config: SimulationConfig) -> List[str]:
    """Validate simulation configuration parameters. Returns list of error messages (empty if valid)."""
    errors = []
    
    # Check parts
    if not config.parts:
        errors.append("No 3D parts specified for the simulation.")
    else:
        for p in config.parts:
            if not os.path.exists(p.stl_path):
                errors.append(f"STL file for part '{p.label}' does not exist: {p.stl_path}")
            if p.density <= 0:
                errors.append(f"Density for part '{p.label}' must be positive, got {p.density}")
                
    # Check source
    if config.source.beam_type not in ["monochromatic", "polychromatic"]:
        errors.append(f"Unknown beam type: {config.source.beam_type}")
    if config.source.mono_energy_kev <= 0:
        errors.append(f"Monochromatic energy must be positive, got {config.source.mono_energy_kev}")
    if config.source.poly_tube_voltage_kv <= config.source.poly_min_energy_kev:
        errors.append(f"kVp ({config.source.poly_tube_voltage_kv}) must be greater than min energy ({config.source.poly_min_energy_kev})")
        
    # Check detector
    if config.detector.rows <= 0 or config.detector.cols <= 0:
        errors.append("Detector rows and columns must be positive integers.")
    if config.detector.pixel_size_x <= 0 or config.detector.pixel_size_y <= 0:
        errors.append("Pixel sizes must be positive.")
        
    # Check acquisition
    if config.acquisition.num_projections <= 0:
        errors.append("Number of projections must be greater than zero.")
    if config.acquisition.total_angle_deg <= 0:
        errors.append("Total rotation angle must be greater than zero.")
        
    return errors


def estimate_output_size(rows: int, cols: int, num_projections: int, output_format: str = "uint16") -> str:
    """Calculate and format estimated storage requirement for projection stack."""
    bytes_per_pixel = 2 if output_format == "uint16" else 4
    total_bytes = rows * cols * num_projections * bytes_per_pixel
    
    if total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.2f} KB"
    elif total_bytes < 1024 * 1024 * 1024:
        return f"{total_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{total_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_stl_bounds_cm(stl_path: str, unit_length: str = "mm") -> Tuple[float, float, float]:
    """
    Compute (dx, dy, dz) centered half-extents in cm directly from STL file
    without initializing OpenGL / GLFW contexts.
    """
    scale_to_cm = 0.1 if unit_length == "mm" else 1.0
    if unit_length == "m":
        scale_to_cm = 100.0
    elif unit_length == "um":
        scale_to_cm = 0.0001
        
    file_size = os.path.getsize(stl_path)
    if file_size > 84:
        with open(stl_path, "rb") as f:
            f.read(80)
            n_tri = struct.unpack("<I", f.read(4))[0]
            if abs(file_size - (84 + n_tri * 50)) <= 2:
                dt = np.dtype([
                    ('normal', '<f4', (3,)),
                    ('v1', '<f4', (3,)),
                    ('v2', '<f4', (3,)),
                    ('v3', '<f4', (3,)),
                    ('attr', '<u2')
                ])
                data = np.fromfile(f, dtype=dt, count=n_tri)
                all_verts = np.concatenate([data['v1'], data['v2'], data['v3']], axis=0)
                min_c = all_verts.min(axis=0) * scale_to_cm
                max_c = all_verts.max(axis=0) * scale_to_cm
                dx = float((max_c[0] - min_c[0]) / 2.0)
                dy = float((max_c[1] - min_c[1]) / 2.0)
                dz = float((max_c[2] - min_c[2]) / 2.0)
                return dx, dy, dz
                
    # Fallback for ASCII STL files
    xs, ys, zs = [], [], []
    with open(stl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                if len(parts) >= 4:
                    xs.append(float(parts[1]) * scale_to_cm)
                    ys.append(float(parts[2]) * scale_to_cm)
                    zs.append(float(parts[3]) * scale_to_cm)
    if xs:
        dx = float((max(xs) - min(xs)) / 2.0)
        dy = float((max(ys) - min(ys)) / 2.0)
        dz = float((max_c[2] - min_c[2]) / 2.0 if 'max_c' in locals() else (max(zs) - min(zs)) / 2.0)
        return dx, dy, dz
        
    return 1.0, 1.0, 1.0


def calculate_auto_framing(
    stl_path: str,
    sod_cm: float = 40.0,
    sdd_cm: float = 55.0,
    rows: int = 256,
    cols: int = 256,
    unit_length: str = "mm",
    margin: float = 1.20
) -> float:
    """
    Automatically calculate optimal detector pixel size (in mm)
    so that ANY arbitrary STL mesh fits completely inside the FOV across 360° rotation.
    
    Formula:
        R_max = sqrt(dx^2 + dy^2)  (in rotation plane)
        Footprint = 2 * R_max * (SDD / SOD) * margin
        pixel_size = Footprint / resolution
    """
    dx, dy, dz = get_stl_bounds_cm(stl_path, unit_length=unit_length)

    # Maximum radius from rotation axis (Z)
    r_max = math.sqrt(dx * dx + dy * dy)
    diameter = 2.0 * r_max
    height = 2.0 * dz

    # Cone-beam geometric magnification M = SDD / SOD
    magnification = sdd_cm / sod_cm

    # Projected detector footprint needed
    needed_width_cm = diameter * magnification * margin
    needed_height_cm = height * magnification * margin
    max_needed_cm = max(needed_width_cm, needed_height_cm)

    # Pixel size in mm
    pixel_size_mm = (max_needed_cm * 10.0) / min(rows, cols)
    return round(float(pixel_size_mm), 2)
