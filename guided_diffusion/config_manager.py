import importlib

class ConfigManager:
    _instance = None
    method = None

    @classmethod
    def getInstance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if ConfigManager._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ConfigManager._instance = self

    def set_method(self, method):
        self.method = method
        print(method)
        if method == 'GEM':
            module_name = '.EM_onestep_GEM'
        elif method == 'smooth':
            module_name = '.EM_onestep_smooth'
        else:
            module_name = '.EM_onestep'  # Default or other cases

        self.em_module = importlib.import_module(module_name, package='guided_diffusion')

    def get_EM_functions(self):
        try:
            return self.em_module.EM_Initial, self.em_module.EM_onestep
        except AttributeError:
            raise Exception("The method has not been set or an invalid method was provided.")
