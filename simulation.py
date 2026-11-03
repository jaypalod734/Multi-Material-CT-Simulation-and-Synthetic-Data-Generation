import os
import time
from typing import Dict, Any, Optional, Callable
import numpy as np
import tifffile
from gvxrPython3 import gvxr

from config import SimulationConfig
from utils import validate_config
from scanner_setup import ScannerSetup
from mesh_manager import MeshManager
from output_manager import OutputManager


class SimulationEngine:
    """Core simulation loop coordinating scanner setup, mesh loading, CT acquisition, and output."""

    @staticmethod
    def initialize_context() -> None:
        """Initialize OpenGL interactive/compute context if not already established."""
        try:
            renderer = gvxr.getOpenGlRenderer()
        except Exception:
            renderer = "Unknown"

        if renderer == "Unknown" or not renderer:
            print("[SimulationEngine] Initializing OpenGL context...")
            gvxr.createOpenGLContext()
            renderer = gvxr.getOpenGlRenderer()
            
        print(f"[SimulationEngine] OpenGL context active on: {renderer}")

    @classmethod
    def run_simulation(
        cls,
        config: SimulationConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute a full multi-material CT acquisition run.

        Arguements:
            config: SimulationConfig defining geometry, materials, and acquisition parameters.
            progress_callback: Optional callback functions(current_proj, total_proj, message_str)
            
        Returns:
            Dictionary containing run summary, paths, and timings.
        """
        # 1. Validate inputs
        errors = validate_config(config)
        if errors:
            raise ValueError(f"Simulation Configuration Error(s):\n - " + "\n - ".join(errors))

        start_time = time.time()
        output_dir = os.path.abspath(config.output_dir)
        OutputManager.ensure_output_dir(output_dir)

        # 2. Context Initialization
        cls.initialize_context()

        # 3. Clean scene before setup
        try:
            gvxr.resetSceneTransformation()
            MeshManager.clear_scene_meshes()
        except Exception:
            pass

        # 4. Scanner Setup (Source, Spectrum & Detector)
        used_bins, theoretical_i0 = ScannerSetup.setup_source(config.source)
        ScannerSetup.setup_detector(config.detector)

        rows = int(config.detector.rows)
        cols = int(config.detector.cols)

        # 5. Load Multi-Material 3D STL Meshes
        loaded_parts = MeshManager.load_parts(config.parts)

        # 6. Prepare CT Acquisition
        num_angles = int(config.acquisition.num_projections)
        total_angle = float(config.acquisition.total_angle_deg)
        angle_step = total_angle / num_angles
        axis = config.acquisition.rotation_axis

        sinogram_row = config.acquisition.sinogram_row
        if sinogram_row is None:
            sinogram_row = rows // 2
        else:
            sinogram_row = int(sinogram_row)

        sinogram_slices = []
        saved_projection_files = []

        print(f"[SimulationEngine] Starting CT scan: {num_angles} projections over {total_angle}° "
              f"(step: {angle_step:.2f}°)...")

        # 7. Multi-Angle Acquisition Loop
        i0_val = theoretical_i0 if theoretical_i0 > 0 else 80.0

        for i in range(num_angles):
            current_angle = i * angle_step

            # Compute X-ray projection for current turntable angle
            raw_proj = np.asarray(gvxr.computeXRayImage()).reshape((rows, cols))

            # Calibrate I0 on first projection using the actual incident beam in the air field
            if i == 0:
                measured_max = float(np.max(raw_proj))
                if measured_max > 0:
                    i0_val = measured_max
                else:
                    i0_val = theoretical_i0 if theoretical_i0 > 0 else 80.0
                print(f"[SimulationEngine] Incident beam flux I0 calibrated: {i0_val:.2f} (measured incident flat-field)")

            # Normalize transmission: T = I / I0 in [0.0, 1.0]
            transmission = np.clip(raw_proj / i0_val, 0.0, 1.0)

            # Apply realistic Poisson photon counting noise if enabled
            if config.acquisition.apply_poisson_noise:
                flux = float(config.acquisition.photon_noise_scale)
                expected_photons = transmission * flux
                noisy_photons = np.random.poisson(expected_photons)
                transmission_measured = np.clip(noisy_photons / flux, 0.0, 1.0)
            else:
                transmission_measured = transmission

            # Save projection TIFF
            proj_filename = f"proj_{i:04d}.tif"
            proj_filepath = os.path.join(output_dir, proj_filename)
            OutputManager.save_projection_tiff(
                proj_filepath,
                transmission_measured,
                output_format=config.acquisition.output_format
            )
            saved_projection_files.append(proj_filepath)

            # Calculate attenuation line integral for sinogram slice: -ln(I/I0)
            attenuation_slice = -np.log(np.clip(transmission_measured[sinogram_row, :], 1e-6, 1.0))
            sinogram_slices.append(attenuation_slice)

            # Progress notification
            if progress_callback:
                progress_callback(
                    i + 1,
                    num_angles,
                    f"Acquired angle {current_angle:.1f}° ({i+1}/{num_angles})"
                )

            # Rotate scene turntable around specified rotation axis
            gvxr.rotateScene(angle_step, axis[0], axis[1], axis[2])

        # 9. Build and Save 2D Sinogram
        sinogram = np.array(sinogram_slices, dtype=np.float32)  # Shape: (num_angles, cols)
        sinogram_path = os.path.join(output_dir, "sinogram.tif")
        OutputManager.save_sinogram(sinogram_path, sinogram)
        print(f"[SimulationEngine] Sinogram saved: {sinogram_path} (shape={sinogram.shape})")

        # 10. Generate Visual Preview Figure
        preview_path = os.path.join(output_dir, "preview.png")
        proj_0_preview = tifffile.imread(saved_projection_files[0])
        beam_label = f"({config.source.beam_type.capitalize()}" + (
            f" + Poisson noise)" if config.acquisition.apply_poisson_noise else ")"
        )
        OutputManager.save_preview_figure(
            preview_path,
            proj_preview=proj_0_preview,
            sinogram=sinogram,
            slice_row=sinogram_row,
            title_suffix=beam_label
        )
        print(f"[SimulationEngine] Preview plot saved: {preview_path}")

        # 11. Reset Scene Transformation and Clean Up
        try:
            gvxr.resetSceneTransformation()
            MeshManager.clear_scene_meshes()
        except Exception:
            pass

        duration = time.time() - start_time
        fps = num_angles / max(duration, 0.001)

        # 12. Save Metadata JSON
        meta_summary = {
            "duration_seconds": round(duration, 3),
            "projections_per_second": round(fps, 2),
            "num_projections": num_angles,
            "sinogram_slice_row": sinogram_row,
            "beam_type": config.source.beam_type,
            "energy_bins": used_bins,
            "poisson_noise_enabled": config.acquisition.apply_poisson_noise,
            "loaded_mesh_parts": loaded_parts
        }
        metadata_path = os.path.join(output_dir, "simulation_config.json")
        OutputManager.save_metadata(metadata_path, config, extra_info=meta_summary)
        print(f"[SimulationEngine] Metadata recorded: {metadata_path}")
        print(f"[SimulationEngine] Completed in {duration:.2f}s ({fps:.1f} proj/sec)")

        return {
            "output_dir": output_dir,
            "preview_path": preview_path,
            "sinogram_path": sinogram_path,
            "metadata_path": metadata_path,
            "num_projections": num_angles,
            "duration_sec": duration,
            "fps": fps
        }
