import importlib

class ConfigManager:
    _instance = None
    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self.__class__._instance is not None:
            raise Exception("This class is a singleton!")
        self.em_module = None

    def set_method(self, method):
        print(f"Setting method: {method}")
        try:
            module_name = f'.EM_onestep_{method}' if method != 'default' else '.EM_onestep'
            self.em_module = importlib.import_module(module_name, package='guided_diffusion')
            print(f"Module loaded successfully: {module_name}")
        except ImportError as e:
            print(f"Failed to import the module for method {method}: {str(e)}")
            raise ImportError(f"Failed to import the module for method {method}: {str(e)}")

    def get_EM_functions(self):
        if self.em_module and hasattr(self.em_module, 'EM_Initial') and hasattr(self.em_module, 'EM_onestep'):
            return self.em_module.EM_Initial, self.em_module.EM_onestep
        else:
            print("Failed to fetch EM functions.")
        raise Exception("EM functions are not available, check method setup.")
