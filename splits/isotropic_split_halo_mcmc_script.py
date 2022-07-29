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
    
def get_correlation(a, gamma, correlation_group, ind):
    model_instance = get_model(ind)
    
    # Reassign a and gamma for RadialSatellitesAlignmentStrength
    model_instance.model_dictionary["satellites_radial_alignment_strength"].param_dict["a"] = a
    model_instance.model_dictionary["satellites_radial_alignment_strength"].param_dict["gamma"] = gamma

    model_instance.model_dictionary["satellites_radial_alignment_strength"].assign_satellite_alignment_strength( table=model_instance.mock.galaxy_table )
        
    model_instance._input_model_dictionary["satellites_orientation"].assign_satellite_orientation( table=model_instance.mock.galaxy_table )
    model_instance._input_model_dictionary["centrals_orientation"].assign_central_orientation( table=model_instance.mock.galaxy_table )
    
    # Perform correlation functions on galaxies
    coords1, orientations, coords2 = get_coords_and_orientations(model_instance, correlation_group=correlation_group)
    #galaxy_coords, galaxy_orientations = get_galaxy_coordinates_and_orientations(model_instance, halocat)
    #galaxy_omega, galaxy_eta, galaxy_xi = galaxy_alignment_correlations(galaxy_coords, galaxy_orientations, rbins)
    omega = ed_3d( coords1, orientations, coords2, rbins, period=halocat.Lbox )
    
    return omega
    
def log_prob(theta, inv_cov, x, y, halocat, rbins, split, front, correlation_group):
    start=time.time()
    if len(theta) == 2:
        a, gamma = theta
    else:
        a = theta
        gamma = 0

    if a < -5.0 or a > 5.0:
        return -np.inf

    avg_runs = 10
    
    params = [ ( a, gamma, correlation_group, ind ) for ind in range(avg_runs) ]
    
    pool = Pool()
    omegas = pool.starmap( get_correlation, params )
    
    omegas = np.array( omegas )
    omega = np.mean( omegas, axis=0 )
        
    if front:
        diff = omega[:split] - y[:split]
    else:
        diff = omega[split:] - y[split:]

    print(time.time()-start)
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
    jackknife_cov = vars[ "jackknife_cov" ]
    if jackknife_cov.lower() == "none":
            jackknife_cov = None
    front = string_to_bool( vars["front"] )
    correlation_group = vars["correlation_group"]
    values_f_name = vars[ "values_f_name" ]
    truth_f_name = vars["truth_f_name"]
    
    return storage_location, split, front, correlation_group, jackknife_cov, values_f_name, truth_f_name

def parse_args():
    job = sys.argv[1]
    variable_f_name = sys.argv[2]
    
    return job, variable_f_name

models = np.repeat(None, 10)
model_ind = np.array([0])
    
if __name__ == "__main__":
    job, variable_f_name =  parse_args()
    storage_location, split, front, correlation_group, jackknife_cov, values_f_name, truth_f_name = \
                        read_variables( variable_f_name )

    truth_df = pd.read_csv(truth_f_name, index_col=False)
    #eta_df = pd.read_csv("Halo_eta_w_false_subhalo.csv", index_col=False)
    #xi_df = pd.read_csv("Halo_xi_w_false_subhalo.csv", index_col=False)

    truth_mean = np.array( truth_df.mean() )
    
    #omega_std = np.array( omega_df.std() )
    #eta_mean = np.array( eta_df.mean() )
    #eta_std = np.array( eta_df.std() )
    #xi_mean = np.array( xi_df.mean() )
    #xi_std = np.array( xi_df.std() )

    rbins = np.logspace(-1,1.4,20)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0

    cache = HaloTableCache()
    #for entry in cache.log: print(entry)

    #halocat = CachedHaloCatalog(simname='multidark', redshift=0)
    halocat = CachedHaloCatalog(simname='bolshoi', halo_finder='rockstar', redshift=0, version_name='halotools_v0p4')
    mask_bad_halocat(halocat)

    # MODELS
    cens_occ_model = Leauthaud11Cens
    #cens_occ_model = SubHaloPositions
    cens_prof_model = TrivialPhaseSpace
    cens_orientation = CentralAlignment
    sats_occ_model = Leauthaud11Sats
    #sat_occ_model = SubHaloPositions
    sats_prof_model1 = NFWPhaseSpace
    prof_args1 = ("satellites", np.logspace(10.5, 15.2, 15))
    #sats_orientation1 = SubhaloAlignment(satellite_alignment_strength=0.5, halocat=halocat)
    sats_orientation2 = RadialSatelliteAlignment
    #sats_orientation1 = SubhaloAlignment
    sats_strength = RadialSatelliteAlignmentStrengthAlternate()
    Lbox = halocat.Lbox
    sats_strength.inherit_halocat_properties(Lbox=Lbox)

    central_alignment = 1
    
    for i in range(len(models)):

        model_instance = HodModelFactory(centrals_occupation = cens_occ_model(),
                                         centrals_profile = cens_prof_model(),
                                         satellites_occupation = sats_occ_model(),
                                         satellites_profile = sats_prof_model1(*prof_args1),
                                         satellites_radial_alignment_strength = sats_strength,
                                         centrals_orientation = cens_orientation(alignment_strength=central_alignment),
                                         satellites_orientation = sats_orientation2(satellite_alignment_strength=1, halocat=halocat),
                                         model_feature_calling_sequence = (
                                         'centrals_occupation',
                                         'centrals_profile',
                                         'satellites_occupation',
                                         'satellites_profile',
                                         'satellites_radial_alignment_strength',
                                         'centrals_orientation',
                                         'satellites_orientation')
                                        )

        model_instance.populate_mock(halocat, seed=132358712)
        model_instance._input_model_dictionary["satellites_orientation"].inherit_halocat_properties( Lbox = halocat.Lbox )
        models[i] = model_instance

    ndim, nwalkers = 2, 5
    #ndim, nwalkers = 1, 4

    p0 = 2*((np.random.rand(nwalkers, ndim)) - 0.5)

    if jackknife_cov is None:
        omega_df = pd.read_csv(values_f_name, index_col=False)
        omega_df = np.array(omega_df).T
        cov = np.cov( omega_df )
        p, n = omega_df.shape
        p = len(omega_df[:split])
    else:
        # Actually do the jackknife here instead?
        cov = np.load(jackknife_cov)
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
        args = [inv_cov, rbin_centers, truth_mean, halocat, rbins, split, front, correlation_group]
        moves = [emcee.moves.StretchMove(a=2),emcee.moves.StretchMove(a=1.1),emcee.moves.StretchMove(a=1.5),emcee.moves.StretchMove(a=1.3)]

        #with Pool() as pool:
        #    print("Starting")
        #    start = time.time()
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=args, backend=backend, moves=moves)
        sampler.run_mcmc(p0, 10000, store=True, progress=True)
        #    print(time.time()-start)
    
    except Exception as e:
        print(e)
