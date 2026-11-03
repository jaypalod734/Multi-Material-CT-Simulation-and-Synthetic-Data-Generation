import os
import json
import time
import datetime
from typing import Dict, Any, Optional
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from config import SimulationConfig


class OutputManager:
    """Handles saving projection TIFF stacks, sinograms, metadata JSON, and preview figures."""

    @staticmethod
    def ensure_output_dir(output_dir: str) -> str:
        """Create output directory if it does not already exist."""
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @staticmethod
    def save_projection_tiff(file_path: str, image: np.ndarray, output_format: str = "uint16") -> None:
        """Save a single projection image as TIFF."""
        if output_format == "uint16":
            # Ensure uint16 range [0, 65535]
            if image.dtype != np.uint16:
                norm = np.clip(image, 0.0, 1.0)
                data = (norm * 65535.0).astype(np.uint16)
            else:
                data = image
            tifffile.imwrite(file_path, data)
        else:
            # float32 raw / attenuation
            data = image.astype(np.float32)
            tifffile.imwrite(file_path, data)

    @staticmethod
    def save_sinogram(file_path: str, sinogram: np.ndarray) -> None:
        """Save 2D sinogram array as a 32-bit floating point TIFF."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        tifffile.imwrite(file_path, sinogram.astype(np.float32))

    @staticmethod
    def save_metadata(
        file_path: str,
        config: SimulationConfig,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save simulation metadata and run parameters to JSON alongside images."""
        meta = {
            "timestamp": datetime.datetime.now().isoformat(),
            "configuration": config.to_dict(),
            "execution_summary": extra_info or {}
        }
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def save_preview_figure(
        file_path: str,
        proj_preview: np.ndarray,
        sinogram: np.ndarray,
        slice_row: int,
        title_suffix: str = ""
    ) -> None:
        """
        Generate and save a 3-panel visual preview:
        1. Projection 0° with sinogram slice indicator
        2. Sinogram (Rotation Angle vs Detector Column)
        3. 1D attenuation profile along the slice
        """
        fig = plt.figure(figsize=(15, 4.5))

        # Panel 1: First Projection
        ax1 = fig.add_subplot(1, 3, 1)
        im1 = ax1.imshow(proj_preview, cmap="gray")
        ax1.axhline(slice_row, color="red", linestyle="--", alpha=0.85, label=f"Slice row {slice_row}")
        ax1.set_title(f"Projection 0° {title_suffix}")
        ax1.set_xlabel("Detector Column (px)")
        ax1.set_ylabel("Detector Row (px)")
        ax1.legend(loc="lower right", fontsize=8)
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Panel 2: Sinogram
        ax2 = fig.add_subplot(1, 3, 2)
        im2 = ax2.imshow(sinogram, cmap="inferno", aspect="auto", extent=[0, sinogram.shape[1], 360, 0])
        ax2.set_title("Sinogram (Central Slice)")
        ax2.set_xlabel("Detector Column (px)")
        ax2.set_ylabel("Rotation Angle (°)")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Attenuation $-\\ln(I/I_0)$")

        # Panel 3: 1D Profile at Angle 0°
        ax3 = fig.add_subplot(1, 3, 3)
        profile = sinogram[0, :]
        ax3.plot(profile, color="#0284c7", lw=1.5)
        ax3.fill_between(range(len(profile)), profile, color="#0284c7", alpha=0.15)
        ax3.set_title("1D Attenuation Profile (0°)")
        ax3.set_xlabel("Detector Column (px)")
        ax3.set_ylabel("Attenuation Line Integral")
        ax3.grid(True, linestyle=":", alpha=0.6)

        # Adjust subplot padding to prevent label overlap
        plt.tight_layout()

        # Ensure the destination folder exists before saving
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        # Attempt to save the figure
        try:
            plt.savefig(file_path, dpi=150)
        # If the file is open or access is denied, save with a unique timestamp instead
        except (PermissionError, OSError):
            base, ext = os.path.splitext(file_path)
            alt_path = f"{base}_{int(time.time())}{ext}"
            plt.savefig(alt_path, dpi=150)
        finally:
            plt.close(fig)
