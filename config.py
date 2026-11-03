from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import os

# @dataclass automatically generates __init__(), __repr__(), and __eq__() methods

@dataclass
class PartConfig:
    """Configuration for a single 3D mesh part in the simulation."""
    label: str
    stl_path: str
    material_type: str = "element"  # element, compound, etc
    material_identifier: str = "Al"  # eg- Al,Cu
    density: float = 2.70           # in g/cm^3
    unit_length: str = "mm"         # STL mesh unit: mm,cm,m
    center_mesh: bool = True        
    scale: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    translate: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # in mm
    rotation: Optional[List[float]] = None  # [angle(in deg), rx, ry, rz]


@dataclass
class FilterConfig:
    """Physical beam filtration specifications."""
    material: str = "Al"  # Element symbol
    thickness: float = 1.0  
    unit: str = "mm"


@dataclass
class SourceConfig:
    """X-ray source beam geometry and spectral parameters."""
    position: List[float] = field(default_factory=lambda: [-40.0, 0.0, 0.0])  # [x,y,z] in unit specified by position_unit
    position_unit: str = "cm"
    source_shape: str = "point"  # "point" or "parallel"
    beam_type: str = "monochromatic"  # "monochromatic" or "polychromatic"
    
    # Monochromatic parameters
    mono_energy_kev: float = 80.0
    mono_photon_count: int = 1000
    
    # Polychromatic parameters
    poly_tube_voltage_kv: float = 100.0  # Peak tube potential (kVp)
    poly_energy_bins: int = 25           # Discretization bins for spectrum
    poly_min_energy_kev: float = 20.0    # Lower cutoff energy
    poly_anode_material: str = "W"       # Tungsten target
    poly_total_photons: int = 50000      # Total photons across spectrum
    
    # Beam filters
    filters: List[FilterConfig] = field(default_factory=list)


@dataclass
class DetectorConfig:
    """Detector array geometry, orientation, and resolution."""
    position: List[float] = field(default_factory=lambda: [15.0, 0.0, 0.0])  # [x,y,z] in position_unit
    position_unit: str = "cm"
    up_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    rows: int = 512
    cols: int = 512
    pixel_size_x: float = 0.4
    pixel_size_y: float = 0.4
    pixel_size_unit: str = "mm"


@dataclass
class AcquisitionConfig:
    """CT acquisition trajectory, rotation, noise, and output format settings."""
    num_projections: int = 180
    total_angle_deg: float = 360.0
    rotation_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    output_format: str = "uint16"  # "uint16" or "float32"
    sinogram_row: Optional[int] = None  # None defaults to centre row 
    apply_poisson_noise: bool = False
    photon_noise_scale: float = 50000.0  # Incident photon flux for Poisson noise


@dataclass
class SimulationConfig:
    """Root configuration holding all scanner, object, and acquisition parameters."""
    output_dir: str = "output"   # Output directory where projection TIFFs, sinograms, and metadata will be saved
    parts: List[PartConfig] = field(default_factory=list)
    source: SourceConfig = field(default_factory=SourceConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)

    def to_dict(self) -> Dict[str, Any]:   # Converts dataclass objects to standard dictionary
        return asdict(self)

    @classmethod
    #Converts Dictionary to dataclass objects
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationConfig":
        if "configuration" in data:  #Cases where dictionary is wrapped in metadata file
            data = data["configuration"]
            
        parts = [PartConfig(**p) for p in data.get("parts", [])]

        #Extracting parameters and reconstructing objects
        source_data = data.get("source", {})
        filters_data = source_data.get("filters", [])
        filters = [FilterConfig(**f) for f in filters_data]
        source_kwargs = {k: v for k, v in source_data.items() if k != "filters"}
        source = SourceConfig(**source_kwargs, filters=filters)
        
        detector = DetectorConfig(**data.get("detector", {}))
        acquisition = AcquisitionConfig(**data.get("acquisition", {}))

        # Return a new instance of SimulationConfig
        return cls(
            output_dir=data.get("output_dir", "output"),
            parts=parts,
            source=source,
            detector=detector,
            acquisition=acquisition
        )

    # Save to & Load from JSON on disk
    def to_json(self, file_path: str, indent: int = 2) -> None:
        # Create destination directory if it doesn't already exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True) 
        # Writing dictionary to JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    # Read the JSON file and reconstruct SimulationConfig object
    def from_json(cls, file_path: str) -> "SimulationConfig":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
