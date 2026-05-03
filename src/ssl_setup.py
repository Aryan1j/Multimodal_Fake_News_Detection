"""
SSL Setup - Fixes SSL certificate issues for HuggingFace downloads on macOS.

This module must be called BEFORE importing transformers in train/eval/inference paths.
Set environment variable PROJECT_DISABLE_CERTIFI_SSL=1 to skip patching.
"""

import os
import ssl


def apply_certifi_for_hf_downloads():
    """
    Apply SSL certificate fixes for HuggingFace model downloads.
    
    On macOS, HuggingFace downloads may fail with SSL certificate errors.
    This function attempts to fix that by:
    1. Trying to use truststore for system certificate injection
    2. Falling back to certifi certificates if truststore fails
    
    Set PROJECT_DISABLE_CERTIFI_SSL=1 to skip this patching entirely.
    """
    # Check if patching is disabled
    if os.environ.get("PROJECT_DISABLE_CERTIFI_SSL") == "1":
        return
    
    try:
        # First, try truststore for system certificate injection
        import truststore
        truststore.inject_into_ssl()
        return
    except (ImportError, Exception):
        pass
    
    try:
        # Fall back to certifi certificates
        import certifi
        
        # Create a custom SSL context with certifi certificates
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        # Monkey-patch ssl to use certifi by default
        _original_create_default_context = ssl.create_default_context
        
        def _patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, 
                                            cafile=None, capath=None, cadata=None):
            if cafile is None and capath is None and cadata is None:
                cafile = certifi.where()
            return _original_create_default_context(
                purpose, cafile=cafile, capath=capath, cadata=cadata
            )
        
        ssl.create_default_context = _patched_create_default_context
        
    except ImportError:
        # certifi not available, skip patching
        pass
    except Exception as e:
        # Log but don't fail - SSL issues may not affect all systems
        print(f"Warning: SSL patching failed: {e}")


# Auto-apply on import if this module is imported directly
if __name__ != "__main__":
    apply_certifi_for_hf_downloads()
