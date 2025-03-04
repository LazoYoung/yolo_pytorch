import os
import scipy.io as sio


metafile = os.path.join("dataset", "ILSVRC2012_devkit_t12", "data", "meta.mat")
meta = sio.loadmat(metafile)["synsets"]
print(meta)