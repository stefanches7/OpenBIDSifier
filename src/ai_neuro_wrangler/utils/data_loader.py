"""
Data loading utilities for neuroimaging files.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Any


def load_nifti(file_path: str) -> Tuple[np.ndarray, Any]:
    """
    Load NIfTI neuroimaging file.
    
    Args:
        file_path: Path to NIfTI file
        
    Returns:
        Tuple of (data array, header/affine info)
    """
    try:
        import nibabel as nib
        
        img = nib.load(file_path)
        data = img.get_fdata()
        header = img.header
        affine = img.affine
        
        return data, {"header": header, "affine": affine}
    
    except ImportError:
        raise ImportError(
            "nibabel is required for loading NIfTI files. "
            "Install it with: pip install nibabel"
        )


def save_nifti(
    data: np.ndarray,
    file_path: str,
    header_info: Optional[Any] = None,
    affine: Optional[np.ndarray] = None
) -> None:
    """
    Save data as NIfTI file.
    
    Args:
        data: Data array to save
        file_path: Output file path
        header_info: Optional header information
        affine: Optional affine transformation matrix
    """
    try:
        import nibabel as nib
        
        if affine is None:
            affine = np.eye(4)
        
        img = nib.Nifti1Image(data, affine)
        
        if header_info is not None and hasattr(header_info, 'header'):
            # Copy header information if available
            img.header.set_data_dtype(data.dtype)
        
        nib.save(img, file_path)
    
    except ImportError:
        raise ImportError(
            "nibabel is required for saving NIfTI files. "
            "Install it with: pip install nibabel"
        )


def validate_nifti_path(file_path: str) -> bool:
    """
    Validate if path points to a NIfTI file.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if valid NIfTI file path
    """
    path = Path(file_path)
    
    if not path.exists():
        return False
    
    valid_extensions = ['.nii', '.nii.gz']
    return any(str(path).endswith(ext) for ext in valid_extensions)
