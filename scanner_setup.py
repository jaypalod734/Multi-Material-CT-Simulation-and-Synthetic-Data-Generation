from typing import List, Tuple
from gvxrPython3 import gvxr
from config import SourceConfig, DetectorConfig
from utils import generate_kramers_spectrum


class ScannerSetup:
    """Configures X-ray beam source, filters, and flat-panel detector in gVXR."""

    @staticmethod
    def setup_source(config: SourceConfig) -> List[Tuple[float, int]]:
        """ Configure beam geometry, spectral distribution, and filtration.
        
        Returns: List of energy bins used: [(energy_kev, photon_count), ...]  """
        # Set source position
        gvxr.setSourcePosition(
            float(config.position[0]),
            float(config.position[1]),
            float(config.position[2]),
            config.position_unit
        )

        # Source shape (point source or parallel beam)
        if config.source_shape.lower() == "parallel":
            gvxr.useParallelBeam()
        else:
            gvxr.usePointSource()

        # Reset spectrum and inherent filters
        gvxr.resetBeamSpectrum()
        gvxr.clearFiltration()

        used_bins: List[Tuple[float, int]] = []
        theoretical_i0 = 0.0

        # Configure spectrum: Monochromatic or Polychromatic
        if config.beam_type.lower() == "polychromatic":
            used_bins = generate_kramers_spectrum(
                kvp=config.poly_tube_voltage_kv,
                min_kev=config.poly_min_energy_kev,
                num_bins=config.poly_energy_bins,
                total_photons=config.poly_total_photons
            )
            for energy_kev, count in used_bins:
                if hasattr(gvxr, "addEnergyBinToSpectrumPerPixelAtSDD"):
                    gvxr.addEnergyBinToSpectrumPerPixelAtSDD(energy_kev, "keV", count)
                else:
                    gvxr.addEnergyBinToSpectrum(energy_kev, "keV", count)
                theoretical_i0 += energy_kev
            print(f"[ScannerSetup] Polychromatic spectrum configured: {len(used_bins)} bins, "
                  f"{config.poly_tube_voltage_kv} kVp, theoretical I0={theoretical_i0:.1f}")
        else:
            # Monochromatic beam
            energy_kev = float(config.mono_energy_kev)
            count = int(config.mono_photon_count)
            if hasattr(gvxr, "setMonoChromaticPerPixelAtSDD"):
                gvxr.setMonoChromaticPerPixelAtSDD(energy_kev, "keV", count)
            else:
                gvxr.setMonoChromatic(energy_kev, "keV", count)
            used_bins = [(energy_kev, count)]
            theoretical_i0 = energy_kev
            print(f"[ScannerSetup] Monochromatic beam configured: {energy_kev} keV, theoretical I0={theoretical_i0:.1f}")

        # Apply physical beam filters (eg- 1 mm Al, 0.1 mm Cu)
        for flt in config.filters:
            gvxr.addFilter(flt.material, float(flt.thickness), flt.unit)
            print(f"[ScannerSetup] Added filter: {flt.thickness} {flt.unit} of {flt.material}")

        return used_bins, theoretical_i0

    @staticmethod
    def setup_detector(config: DetectorConfig) -> None:
        """Configure flat-panel detector position, orientation, resolution, and pixel spacing."""
        # Set detector position
        gvxr.setDetectorPosition(
            float(config.position[0]),
            float(config.position[1]),
            float(config.position[2]),
            config.position_unit
        )

        # Set the 3D orientation vector
        gvxr.setDetectorUpVector(
            float(config.up_vector[0]),
            float(config.up_vector[1]),
            float(config.up_vector[2])
        )

        # Set the detector resolution (number of pixel columns and rows)
        gvxr.setDetectorNumberOfPixels(int(config.cols), int(config.rows))

        # Set the physical width, height, and unit of each pixel
        gvxr.setDetectorPixelSize(
            float(config.pixel_size_x),
            float(config.pixel_size_y),
            config.pixel_size_unit
        )

        # Log the detector configuration details and position for confirmation
        print(f"[ScannerSetup] Detector configured: {config.cols}x{config.rows} pixels "
              f"({config.pixel_size_x}x{config.pixel_size_y} {config.pixel_size_unit}) "
              f"at position {config.position} {config.position_unit}")
