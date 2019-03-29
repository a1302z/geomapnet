import set_paths
import dataset_loaders.deeploc as dl
d = dl.DeepLoc("../data/deepslam_data/DeepLoc", train=True, semantic=True)
#d = dl.DeepLoc("../data/deepslam_data/DeepLoc", train=False, scene='all', semantic=True)
print((d[0])[0])
