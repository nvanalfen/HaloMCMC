from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import TrivialPhaseSpace, ZuMandelbaum15Cens, ZuMandelbaum15Sats, \
                                        Leauthaud11Cens, Leauthaud11Sats, Zheng07Cens, Zheng07Sats, \
                                        NFWPhaseSpace, SubhaloPhaseSpace
from halotools.empirical_models import NFWPhaseSpace, SubhaloPhaseSpace, Tinker13Cens, Tinker13QuiescentSats, \
                                        TrivialProfile, Tinker13ActiveSats
from halotools_ia.ia_models.ia_model_components import CentralAlignment, RandomAlignment, RadialSatelliteAlignment, \
                                                        HybridSatelliteAlignment, MajorAxisSatelliteAlignment, SatelliteAlignment, \
                                                        SubhaloAlignment
from halotools_ia.ia_models.ia_strength_models import RadialSatelliteAlignmentStrengthAlternate

from halotools_ia.ia_models.nfw_phase_space import AnisotropicNFWPhaseSpace

from intrinsic_alignments.ia_models.occupation_models import SubHaloPositions, IsotropicSubhaloPositions, SemiIsotropicSubhaloPositions

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from halotools.mock_observables import tpcf

from halotools.sim_manager import HaloTableCache
from halotools.sim_manager import CachedHaloCatalog
from halotools_ia.correlation_functions import ed_3d, ee_3d

from halotools.utils import crossmatch

import time

import emcee
from multiprocessing import Pool

import sys
import os

import warnings
warnings.filterwarnings("ignore")

##### FUNCTIONS

def get_coords_and_orientations(model_instance, correlation_group="all"):
    if correlation_group == "all":
        x = model_instance.mock.galaxy_table["x"]
        y = model_instance.mock.galaxy_table["y"]
        z = model_instance.mock.galaxy_table["z"]
        axis_x = model_instance.mock.galaxy_table["galaxy_axisA_x"]
        axis_y = model_instance.mock.galaxy_table["galaxy_axisA_y"]
        axis_z = model_instance.mock.galaxy_table["galaxy_axisA_z"]
        coords = np.array( [x,y,z] ).T
        orientations = np.array( [axis_x,axis_y,axis_z] ).T
        return coords, orientations, coords
    elif correlation_group == "censat":
        sat_cut = model_instance.mock.galaxy_table[model_instance.mock.galaxy_table["gal_type"]=="satellites"]
        cen_cut = model_instance.mock.galaxy_table[model_instance.mock.galaxy_table["gal_type"]=="centrals"]
        satx = sat_cut["x"]
        saty = sat_cut["y"]
        satz = sat_cut["z"]
        cenx = cen_cut["x"]
        ceny = cen_cut["y"]
        cenz = cen_cut["z"]
        axis_x = sat_cut["galaxy_axisA_x"]
        axis_y = sat_cut["galaxy_axisA_y"]
        axis_z = sat_cut["galaxy_axisA_z"]
        sat_coords = np.array( [satx,saty,satz] ).T
        cen_coords = np.array( [cenx,ceny,cenz] ).T
        sat_orientations = np.array( [axis_x,axis_y,axis_z] ).T
        return sat_coords, sat_orientations, cen_coords
    elif correlation_group == "satcen":
        sat_cut = model_instance.mock.galaxy_table[model_instance.mock.galaxy_table["gal_type"]=="satellites"]
        cen_cut = model_instance.mock.galaxy_table[model_instance.mock.galaxy_table["gal_type"]=="centrals"]
        satx = sat_cut["x"]
        saty = sat_cut["y"]
        satz = sat_cut["z"]
        cenx = cen_cut["x"]
        ceny = cen_cut["y"]
        cenz = cen_cut["z"]
        axis_x = cen_cut["galaxy_axisA_x"]
        axis_y = cen_cut["galaxy_axisA_y"]
        axis_z = cen_cut["galaxy_axisA_z"]
        sat_coords = np.array( [satx,saty,satz] ).T
        cen_coords = np.array( [cenx,ceny,cenz] ).T
        cen_orientations = np.array( [axis_x,axis_y,axis_z] ).T
        return sat_coords, cen_orientations, cen_coords
    elif correlation_group == "cencen":
        cen_cut = model_instance.mock.galaxy_table[model_instance.mock.galaxy_table["gal_type"]=="centrals"]
        cenx = cen_cut["x"]
        ceny = cen_cut["y"]
        cenz = cen_cut["z"]
        axis_x = cen_cut["galaxy_axisA_x"]
        axis_y = cen_cut["galaxy_axisA_y"]
        axis_z = cen_cut["galaxy_axisA_z"]
        cen_coords = np.array( [cenx,ceny,cenz] ).T
        cen_orientations = np.array( [axis_x,axis_y,axis_z] ).T
        return cen_coords, cen_orientations, cen_coords
    else:
        sat_cut = model_instance.mock.galaxy_table[model_instance.mock.galaxy_table["gal_type"]=="satellites"]
        satx = sat_cut["x"]
        saty = sat_cut["y"]
        satz = sat_cut["z"]
        axis_x = sat_cut["galaxy_axisA_x"]
        axis_y = sat_cut["galaxy_axisA_y"]
        axis_z = sat_cut["galaxy_axisA_z"]
        sat_coords = np.array( [satx,saty,satz] ).T
        sat_orientations = np.array( [axis_x,axis_y,axis_z] ).T
        return sat_coords, sat_orientations, sat_coords

# Eliminate halos with 0 for halo_axisA_x(,y,z)
def mask_bad_halocat(halocat):
    bad_mask = (halocat.halo_table["halo_axisA_x"] == 0) & (halocat.halo_table["halo_axisA_y"] == 0) & (halocat.halo_table["halo_axisA_z"] == 0)
    halocat._halo_table = halocat.halo_table[ ~bad_mask ]

def get_model(ind=None):
    if not ind is None:
        return models[ind]
        
    ind = model_ind[0]
    model = models[ind]
    ind += 1
    model_ind[0] = ind % len(models)
    return model
    
def get_correlation(sats_alignment, cens_alignment, correlation_group, ind):
    model_instance = get_model(ind)
    
    # Reassign a and gamma for RadialSatellitesAlignmentStrength
    model_instance.model_dictionary["satellites_orientation"].param_dict["satellite_alignment_strength"] = sats_alignment
    model_instance.model_dictionary["centrals_orientation"].param_dict["central_alignment_strength"] = cens_alignment
        
    model_instance._input_model_dictionary["satellites_orientation"].assign_satellite_orientation( table=model_instance.mock.galaxy_table )
    model_instance._input_model_dictionary["centrals_orientation"].assign_central_orientation( table=model_instance.mock.galaxy_table )
    
    # Perform correlation functions on galaxies
    coords1, orientations, coords2 = get_coords_and_orientations(model_instance, correlation_group=correlation_group)
    #galaxy_coords, galaxy_orientations = get_galaxy_coordinates_and_orientations(model_instance, halocat)
    #galaxy_omega, galaxy_eta, galaxy_xi = galaxy_alignment_correlations(galaxy_coords, galaxy_orientations, rbins)
    omega = ed_3d( coords1, orientations, coords2, rbins, period=halocat.Lbox )
    
    return omega
    
def log_prob(theta, inv_cov, x, y, halocat, rbins, split, front, correlation_group, cores):
    model_instance = get_model()
    if len(theta) == 2:
        satellite_alignment_strength, central_alignment_strength = theta
    else:
        satellite_alignment_strength = theta
        central_alignment_strength = 1

    avg_runs = 10
    omegas = []
    
    params = [ ( satellite_alignment_strength, central_alignment_strength, correlation_group, ind ) for ind in range(avg_runs) ]
    
    pool = Pool(cores)
    omegas = pool.starmap( get_correlation, params )
    
    omegas = np.array( omegas )
    omega = np.mean( omegas, axis=0 )
        
    if front:
        diff = omega[:split] - y[:split]
    else:
        diff = omega[split:] - y[split:]
    
    return -0.5 * np.dot( diff, np.dot( inv_cov, diff ) )

global_nums = []

def string_to_bool(value):
    return value == "1" or value.lower() == "true"
    
def read_variables(f_name):
    vars = {}
    f = open(f_name)
    for line in f:
        if line.strip() != '':
            key, value = [ el.strip() for el in line.split(":->:") ]
            vars[key] = value
    
    storage_location = vars["storage_location"]
    split = int( vars["split"] )
    front = string_to_bool( vars["front"] )
    correlation_group = vars["correlation_group"]
    cov_f_name = vars[ "cov_f_name" ]
    truth_f_name = vars["truth_f_name"]
    sample_name = vars["sample_name"]
    cores = int( vars["cores"] )
    
    return storage_location, split, front, correlation_group, sample_name, cov_f_name, truth_f_name, cores

def parse_args():
    job = sys.argv[1]
    variable_f_name = sys.argv[2]
    
    return job, variable_f_name

models = np.repeat(None, 10)
model_ind = np.array([0])
    
if __name__ == "__main__":
    job, variable_f_name =  parse_args()
    storage_location, split, front, correlation_group, sample_name, cov_f_name, truth_f_name, cores = \
                        read_variables( variable_f_name )

    truth_mean = np.load(truth_f_name)

    rbins = np.logspace(-1,1.4,20)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0

    cache = HaloTableCache()

    halocat = CachedHaloCatalog(simname='bolshoi', halo_finder='rockstar', redshift=0, version_name='halotools_v0p4')
    mask_bad_halocat(halocat)

    # MODELS
    cens_occ_model = Zheng07Cens()
    #cens_occ_model = Leauthaud11Cens()
    cens_prof_model = TrivialPhaseSpace()
    cens_orientation = CentralAlignment()
    prof_args = ("satellites", np.logspace(10.5, 15.2, 15))
    sats_occ_model = Zheng07Sats()
    #sats_occ_model = Leauthaud11Sats()
    sats_prof_model = SubhaloPhaseSpace(*prof_args)
    sats_orientation = SubhaloAlignment(satellite_alignment_strength=1, halocat=halocat)
    
    if sample_name == 'sample_1':
        print("A")
        cens_occ_model.param_dict['logMmin'] = 12.54
        cens_occ_model.param_dict['sigma_logM'] = 0.26

        sats_occ_model.param_dict['alpha'] = 1.0
        sats_occ_model.param_dict['logM0'] = 12.68
        sats_occ_model.param_dict['logM1'] = 13.48

        cens_orientation.param_dict['central_alignment_strength'] = 0.755
        sats_orientation.param_dict['satellite_alignment_strength'] = 0.279
    elif sample_name == 'sample_2':
        print("B")
        cens_occ_model.param_dict['logMmin'] = 11.93
        cens_occ_model.param_dict['sigma_logM'] = 0.26

        sats_occ_model.param_dict['alpha'] = 1.0
        sats_occ_model.param_dict['logM0'] = 12.05
        sats_occ_model.param_dict['logM1'] = 12.85

        cens_orientation.param_dict['central_alignment_strength'] = 0.64
        sats_orientation.param_dict['satellite_alignment_strength'] = 0.084
    elif sample_name =='sample_3':
        print("C")
        cens_occ_model.param_dict['logMmin'] = 11.61
        cens_occ_model.param_dict['sigma_logM'] = 0.26

        sats_occ_model.param_dict['alpha'] = 1.0
        sats_occ_model.param_dict['logM0'] = 11.8
        sats_occ_model.param_dict['logM1'] = 12.6

        cens_orientation.param_dict['central_alignment_strength'] = 0.57172919
        sats_orientation.param_dict['satellite_alignment_strength'] = 0.01995
    
    for i in range(len(models)):

        model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                         centrals_profile = cens_prof_model,
                                         satellites_occupation = sats_occ_model,
                                         satellites_profile = sats_prof_model,
                                         centrals_orientation = cens_orientation,
                                         satellites_orientation = sats_orientation,
                                         model_feature_calling_sequence = (
                                         'centrals_occupation',
                                         'centrals_profile',
                                         'satellites_occupation',
                                         'satellites_profile',
                                         'centrals_orientation',
                                         'satellites_orientation')
                                        )

        print(i)
        model_instance.populate_mock(halocat, seed=132358712)
        models[i] = model_instance

    ndim, nwalkers = 2, 5
    #ndim, nwalkers = 1, 4

    p0 = 2*((np.random.rand(nwalkers, ndim)) - 0.5)

    cov = np.load(cov_f_name)
    n = 5*5*5
    p = len(rbin_centers)

    factor = (n-p-2)/(n-1)
    
    if front:
        cov = cov[:split,:split]
    else:
        cov = cov[split:,split:]
        
    inv_cov = np.linalg.inv(cov)
    # Include the factor from the paper
    inv_cov *= factor

    try:
        f_name = os.path.join(storage_location,"MCMC_"+job+".h5")
        backend = emcee.backends.HDFBackend(f_name)
        args = [inv_cov, rbin_centers, truth_mean, halocat, rbins, split, front, correlation_group, cores]
        moves = [emcee.moves.StretchMove(a=2),emcee.moves.StretchMove(a=1.1),emcee.moves.StretchMove(a=1.5),emcee.moves.StretchMove(a=1.3)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=args, backend=backend, moves=moves)
        sampler.run_mcmc(p0, 10000, store=True, progress=True)
        #    print(time.time()-start)
    
    except Exception as e:
        print(e)
