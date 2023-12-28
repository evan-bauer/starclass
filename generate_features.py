from zipfile import ZipFile, ZIP_DEFLATED
import os, sys
from modules.starclass import MakeStardict

def main():
    if len(sys.argv) <= 4 or len(sys.argv)>5:
        print("usage: generate_features.py cluster truncate verbose must_not_exist")
        print("python3 generate_features.py ngc6819 True True True")
    cluster, truncate, verbose, must_not_exist=sys.argv[1:]
    def export():
        MakeStardict(export_all=True, cluster=str(cluster), truncate=truncate, verbose=verbose, must_not_exist=must_not_exist)
    export()
    
if __name__ == '__main__':
     main()