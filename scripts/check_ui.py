import importlib, traceback
try:
    importlib.import_module('server.ui')
    print('server.ui imported successfully')
except Exception:
    traceback.print_exc()