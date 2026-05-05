
import os
import ssl


def apply_certifi_for_hf_downloads():
    

    if os.environ.get("PROJECT_DISABLE_CERTIFI_SSL") == "1":
        return
    
    try:
       
        import truststore
        truststore.inject_into_ssl()
        return
    except (ImportError, Exception):
        pass
    
    try:
        
        import certifi
        
        
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        
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
       
        pass
    except Exception as e:
      
        print(f"Warning: SSL patching failed: {e}")



if __name__ != "__main__":
    apply_certifi_for_hf_downloads()
