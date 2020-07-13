import numpy as np
import matplotlib.pyplot as plt
import trackpy as tp
import nplocate as nl
try:
    import tcc_python_scripts.tcc.wrapper as tw
    TCC = True
except ImportError:
    TCC = False



diameter = 11
img = np.load('image.npy')
xyz = tp.locate(img, diameter=diameter)
xyz = np.array(xyz)[:, :3]
xyz_tp = xyz.copy()

should_add = True
while should_add:
    n0 = xyz.shape[0]
    xyz = nl.add(xyz, img, diameter * 2, diameter, lambda im : np.array(tp.locate(im, diameter))[:, :3])
    n1 = xyz.shape[0]
    should_add = n1 > n0
xyz = nl.refine(xyz, img, diameter * 2, diameter)

print(f"\nParticle found: trackpy, {xyz_tp.shape[0]}; nplocate refined, {xyz.shape[0]}\n")

if TCC:
    tcc = tw.TCCWrapper()
    tcc.set_tcc_executable_directory('/usr/local/bin')
    tcc.input_parameters['Run']['frames'] = 1
    tcc.input_parameters['Box']['box_type'] = 1
    tcc.input_parameters['Simulation']['rcutAA'] = 20.0
    tcc.input_parameters['Simulation']['rcutAB'] = 20.0
    tcc.input_parameters['Simulation']['rcutBB'] = 20.0
    tcc.input_parameters['Simulation']['PBCs'] = 0
    tcc.input_parameters['Simulation']['voronoi_parameter'] = 0.82
    tcc.input_parameters['Simulation']['analyse_all_clusters'] = 1
    box = img.shape
    result_nl = tcc.run(box, xyz)
    result_tp = tcc.run(box, xyz_tp)
    print("TCC result trackpy")
    print(result_tp.head())
    print("\n" * 2)
    print("TCC result nplocate")
    print(result_nl.head())
