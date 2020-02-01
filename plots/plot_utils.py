import glob
import os
import shutil

import matplotlib


def copy_plot_themes():
    stylelib_path = os.path.join(matplotlib.get_configdir(), "stylelib")
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    file_list = glob.glob(os.path.join(curr_dir, '*.mplstyle'))

    try:
        if not os.path.exists(stylelib_path):
            os.makedirs(stylelib_path)
        for f in file_list:
            fname = os.path.basename(f)
            shutil.copy2(f, os.path.join(stylelib_path, fname))
        print("Themes copied to %s" % stylelib_path)
    except:
        print("An error occured! Try to manually copy the *.mplstyle files. E.g.:\nmkdir -p %s && cp *.mplstyle %s" % (
        stylelib_path, stylelib_path))
