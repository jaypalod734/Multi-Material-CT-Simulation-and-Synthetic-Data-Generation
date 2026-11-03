from typing import List, Dict, Any
from gvxrPython3 import gvxr
from config import PartConfig


class MeshManager:
    """Manages loading 3D STL meshes, spatial placement, and multi-material property assignment."""

    @staticmethod
    def clear_scene_meshes() -> None:
        # Remove previously loaded meshes from gVXR scene graph 
        try:
            gvxr.removePolygonMeshesFromXRayRenderer()
            gvxr.removePolygonMeshesFromSceneGraph()
        except Exception:
            pass

    @staticmethod
    def load_part(part: PartConfig) -> Dict[str, Any]:
        """
        Load a single STL mesh into gVXR and assign its transformation and material.
        
        Arguments:
            part: PartConfig defining STL path, position, scale, material and density.
            
        Returns:
            Dictionary with bounding box and loaded part details.
        """
        # Load mesh file into gVXR node
        gvxr.loadMeshFile(part.label, part.stl_path, part.unit_length)
        
        # Geometry transformations
        if part.center_mesh:
            gvxr.moveToCentre(part.label)
            
        if part.scale != [1.0, 1.0, 1.0]:
            gvxr.scaleNode(part.label, part.scale[0], part.scale[1], part.scale[2])
            
        if any(t != 0.0 for t in part.translate):
            gvxr.translateNode(
                part.label,
                part.translate[0],
                part.translate[1],
                part.translate[2],
                part.unit_length
            )
            
        if part.rotation is not None and len(part.rotation) == 4:
            # rotation: [angle(in deg), rx, ry, rz]
            gvxr.rotateNode(
                part.label,
                part.rotation[0],
                part.rotation[1],
                part.rotation[2],
                part.rotation[3]
            )
            
        # Assign material properties
        mat_type = part.material_type.lower()
        if mat_type == "element":
            gvxr.setElement(part.label, part.material_identifier)
        elif mat_type == "compound":
            gvxr.setCompound(part.label, part.material_identifier)
        else:
            gvxr.setElement(part.label, part.material_identifier)
            
        gvxr.setDensity(part.label, float(part.density), "g/cm3")
        
        # Fetch bounding box for inspection/verification
        bbox = gvxr.getNodeAndChildrenBoundingBox(part.label)
        
        return {
            "label": part.label,
            "stl_path": part.stl_path,
            "material_type": part.material_type,
            "material": part.material_identifier,
            "density": part.density,
            "bounding_box_cm": bbox
        }

    @classmethod
    def load_parts(cls, parts: List[PartConfig]) -> List[Dict[str, Any]]:
        #Load and configure a list of multi-material mesh parts
        cls.clear_scene_meshes()
        summary = []
        for part in parts:
            info = cls.load_part(part)
            summary.append(info)
            print(f"[MeshManager] Loaded '{part.label}': Material={part.material_identifier} ({part.material_type}), "
                  f"Density={part.density} g/cm3")
        return summary
