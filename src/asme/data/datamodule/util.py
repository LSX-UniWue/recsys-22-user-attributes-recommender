import os, shutil


# This is a substitute for shutil.copytree for python <= 3.7 since it does not provide the dirs_exist_okay parameter.
def copytree(src, dst,dirs_exist_ok=False):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            os.makedirs(d, exist_ok=dirs_exist_ok)
            copytree(s, d, dirs_exist_ok)
        else:
            dest_dir = os.path.dirname(d)
            if not os.path.exists(dest_dir):
                os.makedirs(d, exist_ok=dirs_exist_ok)
            shutil.copy2(s, d)
