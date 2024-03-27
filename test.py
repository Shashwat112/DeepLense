import pkgutil
import tests
import importlib

def main():
    for i,(_, module_name, _) in enumerate(pkgutil.iter_modules(tests.__path__)):
        print(f'\nTest Case {i+1}: ',end='')
        importlib.import_module(f'tests.{module_name}').run()
    print()

if __name__=='__main__':
    main()