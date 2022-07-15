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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from halotools.mock_observables import tpcf

from halotools.sim_manager import HaloTableCache
from halotools.sim_manager import CachedHaloCatalog
from halotools_ia.correlation_functions import ed_3d, ee_3d

from halotools.utils import crossmatch

import time

import emcee

import sys
import os

import warnings
warnings.filterwarnings("ignore")

##### FUNCTIONS

def get_specific_galaxy_coords(model, gal_type):
    x = model.mock.galaxy_table[model.mock.galaxy_table["gal_type"]==gal_type]["x"]
    y = model.mock.galaxy_table[model.mock.galaxy_table["gal_type"]==gal_type]["y"]
    z = model.mock.galaxy_table[model.mock.galaxy_table["gal_type"]==gal_type]["z"]
    return np.vstack( (x,y,z) ).T

def get_specific_galaxy_orientations(model, gal_type):
    axis_x = model.mock.galaxy_table[model.mock.galaxy_table["gal_type"]==gal_type]["galaxy_axisA_x"]
    axis_y = model.mock.galaxy_table[model.mock.galaxy_table["gal_type"]==gal_type]["galaxy_axisA_y"]
    axis_z = model.mock.galaxy_table[model.mock.galaxy_table["gal_type"]==gal_type]["galaxy_axisA_z"]
    return np.vstack( (axis_x,axis_y,axis_z) ).T

# Create mock catalogue instance
def create_mock(cens_occ_model, cens_prof_model, cens_orientation, sats_occ_model, 
                sats_prof_model, sats_prof_args, sats_orientation, central_alignment_strength, satellite_alignment_strength):
    # Set up the models
    mod = {}
    mod["cens_occ_model"] = cens_occ_model()
    mod["cens_prof_model"] = cens_prof_model()
    mod["cens_orientation"] = cens_orientation(central_alignment_strength=central_alignment_strength)
    mod["sats_occ_model"] = sats_occ_model()
    mod["sats_prof_model"] = sats_prof_model( *sats_prof_args )
    mod["sats_orientation"] = sats_orientation(satellite_alignment_strength=satellite_alignment_strength)
    
    model_instance = HodModelFactory(centrals_occupation = mod["cens_occ_model"],
                                     centrals_profile = mod["cens_prof_model"],
                                     satellites_occupation = mod["sats_occ_model"],
                                     satellites_profile = mod["sats_prof_model"],
                                     centrals_orientation = mod["cens_orientation"],
                                     satellites_orientation = mod["sats_orientation"],
                                     model_feature_calling_sequence = (
                                     'centrals_occupation',
                                     'centrals_profile',
                                     'satellites_occupation',
                                     'satellites_profile',
                                     'centrals_orientation',
                                     'satellites_orientation')
                                    )
    
    # Populate the two mocks
    model_instance.populate_mock(halocat)
    print("number of galaxies: ", len(model_instance.mock.galaxy_table))
    
    return model_instance

def get_galaxy_coordinates_and_orientations(model_instance, halocat):
    galaxy_coords = np.vstack((model_instance.mock.galaxy_table['x'],
                                model_instance.mock.galaxy_table['y'],
                                model_instance.mock.galaxy_table['z'])).T
    galaxy_orientations = np.vstack((model_instance.mock.galaxy_table['galaxy_axisA_x'],
                                    model_instance.mock.galaxy_table['galaxy_axisA_y'],
                                    model_instance.mock.galaxy_table['galaxy_axisA_z'])).T
    
    # Account for nan values
    coord_nans = np.array( [ np.isnan( np.sum(val)) for val in galaxy_coords ] )
    orientation_nans = np.array( [ np.isnan( np.sum(val)) for val in galaxy_orientations ] )
    
    if np.sum(coord_nans) > 0:
        print("NaN rows in galaxy_coords: ", np.sum(coord_nans))
    if np.sum(orientation_nans) > 0:
        print("NaN rows in galaxy_orientations: ", np.sum(orientation_nans))
    
    all_nans = coord_nans | orientation_nans
    inds1, inds2 = crossmatch(all_nans, [True])
    mask = np.ones(len(all_nans), dtype=bool)
    mask[inds1] = False
    
    return galaxy_coords[mask], galaxy_orientations[mask]

def get_halo_coordinates_and_orientations(halocat):
    mask = halocat.halo_table['halo_mpeak']>10**11.9
    halo_coords = np.vstack((halocat.halo_table['halo_x'],
                                halocat.halo_table['halo_y'],
                                halocat.halo_table['halo_z'])).T
    halo_orientations = np.vstack((halocat.halo_table['halo_axisA_x'],
                                    halocat.halo_table['halo_axisA_y'],
                                    halocat.halo_table['halo_axisA_z'])).T
    
    return halo_coords, halo_orientations, mask

def galaxy_alignment_correlations(galaxy_coords, galaxy_orientations, rbins):
    galaxy_omega = ed_3d( galaxy_coords, galaxy_orientations, galaxy_coords, rbins, period=halocat.Lbox )
    galaxy_eta = ee_3d(galaxy_coords, galaxy_orientations, galaxy_coords, galaxy_orientations, rbins, period=halocat.Lbox)
    galaxy_xi = tpcf(galaxy_coords, rbins, galaxy_coords, period=halocat.Lbox)
    
    return galaxy_omega, galaxy_eta, galaxy_xi

def halo_alignment_correlations(halo_coords, halo_orientations, mask, rbins):
    halo_omega = ed_3d(halo_coords[mask], halo_orientations[mask], halo_coords[mask],rbins, period=halocat.Lbox)
    halo_eta = ee_3d(halo_coords[mask], halo_orientations[mask], halo_coords[mask], halo_orientations[mask], rbins, period=halocat.Lbox)
    halo_xi = tpcf(halo_coords[mask], rbins, halo_coords[mask], period=halocat.Lbox)
    
    return halo_omega, halo_eta, halo_xi

def galaxy_cross_correlations(model_instance):
    central_coords = get_specific_galaxy_coords(model_instance, "centrals")
    central_orientations = get_specific_galaxy_orientations(model_instance, "centrals")
    satellite_coords = get_specific_galaxy_coords(model_instance, "satellites")
    satellite_orientations = get_specific_galaxy_orientations(model_instance, "satellites")
    
    # Account for nan values
    cen_coord_nans = np.array( [ np.isnan( np.sum(val)) for val in central_coords ] )
    cen_orientation_nans = np.array( [ np.isnan( np.sum(val)) for val in central_orientations ] )
    sat_coord_nans = np.array( [ np.isnan( np.sum(val)) for val in satellite_coords ] )
    sat_orientation_nans = np.array( [ np.isnan( np.sum(val)) for val in satellite_orientations ] )
    
    # Eliminate NaN values
    cen_nans = cen_coord_nans | cen_orientation_nans
    sat_nans = sat_coord_nans | sat_orientation_nans
    # For centrals
    inds1, inds2 = crossmatch(cen_nans, [True])
    mask = np.ones(len(cen_nans), dtype=bool)
    mask[inds1] = False
    central_coords = central_coords[mask]
    central_orientations = central_orientations[mask]
    # Now for satellites
    inds1, inds2 = crossmatch(sat_nans, [True])
    mask = np.ones(len(sat_nans), dtype=bool)
    mask[inds1] = False
    satellite_coords = satellite_coords[mask]
    satellite_orientations = satellite_orientations[mask]
    
    correlations = {}
    
    # Position-Shape
    correlations["cencen_omega"] = ed_3d(central_coords, central_orientations, central_coords, rbins, period=halocat.Lbox)
    correlations["censat_omega"] = ed_3d(central_coords, central_orientations, satellite_coords, rbins, period=halocat.Lbox)
    correlations["satsat_omega"] = ed_3d(satellite_coords, satellite_orientations, satellite_coords, rbins, period=halocat.Lbox)
    
    # Shape-Shape
    correlations["cencen_eta"] = ee_3d(central_coords, central_orientations, central_coords, central_orientations, rbins, period=halocat.Lbox)
    correlations["censat_eta"] = ee_3d(central_coords, central_orientations, satellite_coords, satellite_orientations, rbins, period=halocat.Lbox)
    correlations["satsat_eta"] = ee_3d(satellite_coords, satellite_orientations, satellite_coords, satellite_orientations, rbins, period=halocat.Lbox)
    
    # Position-Position
    cencen_xi, censat_xi, satsat_xi = tpcf(central_coords, rbins, satellite_coords, period=halocat.Lbox)
    correlations["cencen_xi"] = cencen_xi
    correlations["censat_xi"] = censat_xi
    correlations["satsat_xi"] = satsat_xi
    
    return correlations

def plot_eta_cross_correlations(correlations1, alignment_strength1, correlations2=None, alignment_strength2=None):
    if not (correlations2 is None and alignment_strength2 is None):
        fig = plt.figure(figsize=(16,8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111)
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    
    p1, = ax1.plot(rbin_centers, abs( correlations1["cencen_eta"] ), 'o', color='blue', mec='none', label="cenXcen")
    p2, = ax1.plot(rbin_centers, abs( correlations1["censat_eta"] ), 'o', color='red', mec='none', label="cenXsat")
    p3, = ax1.plot(rbin_centers, abs( correlations1["satsat_eta"] ), 'o', color='green', mec='none', label="satXsat")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$r~[h^{-1}\rm Mpc]$')
    ax1.set_ylabel(r'$\eta(r)$')
    ax1.set_title("Alignment Strength = " + str(alignment_strength1) + " Shape-Shape Correlations")
    ax1.legend()
    
    if not (correlations2 is None and alignment_strength2 is None):
        p1, = ax2.plot(rbin_centers, abs( correlations2["cencen_eta"] ), 'o', color='blue', mec='none', label="cenXcen")
        p2, = ax2.plot(rbin_centers, abs( correlations2["censat_eta"] ), 'o', color='red', mec='none', label="cenXsat")
        p3, = ax2.plot(rbin_centers, abs( correlations2["satsat_eta"] ), 'o', color='green', mec='none', label="satXsat")
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'$r~[h^{-1}\rm Mpc]$')
        ax2.set_ylabel(r'$\eta(r)$')
        ax2.set_title("Alignment Strength = " + str(alignment_strength2) + " Shape-Shape Correlations")
        ax2.legend() 
    
    plt.show()
    
def plot_omega_cross_correlations(correlations1, alignment_strength1, correlations2=None, alignment_strength2=None):
    
    if not (correlations2 is None and alignment_strength2 is None):
        fig = plt.figure(figsize=(16,8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111)
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    
    p1, = ax1.plot(rbin_centers, abs( correlations1["cencen_omega"] ), 'o', color='blue', mec='none', label="cenXcen")
    p2, = ax1.plot(rbin_centers, abs( correlations1["censat_omega"] ), 'o', color='red', mec='none', label="cenXsat")
    p3, = ax1.plot(rbin_centers, abs( correlations1["satsat_omega"] ), 'o', color='green', mec='none', label="satXsat")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$r~[h^{-1}\rm Mpc]$')
    ax1.set_ylabel(r'$\omega(r)$')
    ax1.set_title("Alignment Strength = " + str(alignment_strength1) + " Shape-Position Correlations")
    ax1.legend()
    
    if not (correlations2 is None and alignment_strength2 is None):
        p1, = ax2.plot(rbin_centers, abs( correlations2["cencen_omega"] ), 'o', color='blue', mec='none', label="cenXcen")
        p2, = ax2.plot(rbin_centers, abs( correlations2["censat_omega"] ), 'o', color='red', mec='none', label="cenXsat")
        p3, = ax2.plot(rbin_centers, abs( correlations2["satsat_omega"] ), 'o', color='green', mec='none', label="satXsat")
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'$r~[h^{-1}\rm Mpc]$')
        ax2.set_ylabel(r'$\omega(r)$')
        ax2.set_title("Alignment Strength = " + str(alignment_strength2) + " Shape-Position Correlations")
        ax2.legend() 
    
    plt.show()

def plot_xi_cross_correlations(correlations1, alignment_strength1, correlations2=None, alignment_strength2=None):
    
    if not (correlations2 is None and alignment_strength2 is None):
        fig = plt.figure(figsize=(16,8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111)
    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    
    p1, = ax1.plot(rbin_centers, abs( correlations1["cencen_xi"] ), 'o', color='blue', mec='none', label="cenXcen")
    p2, = ax1.plot(rbin_centers, abs( correlations1["censat_xi"] ), 'o', color='red', mec='none', label="cenXsat")
    p3, = ax1.plot(rbin_centers, abs( correlations1["satsat_xi"] ), 'o', color='green', mec='none', label="satXsat")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$r~[h^{-1}\rm Mpc]$')
    ax1.set_ylabel(r'$\xi(r)$')
    ax1.set_title("Alignment Strength = " + str(alignment_strength1) + " Position-Position Correlations")
    ax1.legend()
    
    if not (correlations2 is None and alignment_strength2 is None):
        p1, = ax2.plot(rbin_centers, abs( correlations2["cencen_xi"] ), 'o', color='blue', mec='none', label="cenXcen")
        p2, = ax2.plot(rbin_centers, abs( correlations2["censat_xi"] ), 'o', color='red', mec='none', label="cenXsat")
        p3, = ax2.plot(rbin_centers, abs( correlations2["satsat_xi"] ), 'o', color='green', mec='none', label="satXsat")
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'$r~[h^{-1}\rm Mpc]$')
        ax2.set_ylabel(r'$\xi(r)$')
        ax2.set_title("Alignment Strength = " + str(alignment_strength2) + " Position-Position Correlations")
        ax2.legend() 
    
    plt.show()
    
def plot_galaxy_correlations(first_plot, second_plot=None, halo=None, halo2=None):
    
    values1, alignments1, title1, y_axis1 = first_plot
    pts1 = []
    
    if not second_plot is None:
        values2, alignments2, title2, y_axis2 = second_plot
        pts2 = []
        fig = plt.figure(figsize=(16,8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111)

    fig.subplots_adjust(left=0.2, right=0.9, bottom=0.2, top=0.9)
    
    if not halo is None:
        pts1.append( ax1.plot(rbin_centers, halo, '-', color="blue", label="Halo") )
        if not second_plot is None:
            if not halo2 is None:
                pts2.append( ax2.plot(rbin_centers, halo2, '-', color="blue", label="Halo") )
            else:
                pts2.append( ax2.plot(rbin_centers, halo, '-', color="blue", label="Halo") )
    
    for i in range(len(alignments1)):
        pts1.append( ax1.plot(rbin_centers, abs( values1[i] ), 'o', mec='none', label="A.S. = "+str(alignments1[i])) )
    
    if not second_plot is None:
        for i in range(len(alignments2)):
            pts2.append( ax2.plot(rbin_centers, abs( values2[i] ), 'o', mec='none', label="A.S. = "+str(alignments2[i])) )
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$r~[h^{-1}\rm Mpc]$')
    ax1.set_ylabel(y_axis1)
    ax1.set_title(title1)
    #ax1.legend()
    
    if not second_plot is None:
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'$r~[h^{-1}\rm Mpc]$')
        ax2.set_ylabel(y_axis2)
        ax2.set_title(title2)
        #ax2.legend()
        
    plt.show()
    
def vary_satellite_alignment(central_alignment, satellite_alignments, halocat, halo_info, plot_cross=True):
    galaxy_etas = []
    galaxy_omegas = []
    galaxy_xis = []
    neg_galaxy_etas = []
    neg_galaxy_omegas = []
    neg_galaxy_xis = []
    halo_coords, halo_orientations, halo_omega, halo_eta, halo_xi, mask = halo_info
    
    for alignment in satellite_alignments:
        print("A.S. = ", alignment)
        
        print("Getting Model Instance")
        model_instance = create_mock(cens_occ_model, cens_prof_model, cens_orientation,
                                     sats_occ_model, sats_prof_model, sats_orientation,
                                     central_alignment_strength=central_alignment,
                                     satellite_alignment_strength=alignment)
        print("Getting Coordinates and Orientations")
        galaxy_coords, galaxy_orientations = get_galaxy_coordinates_and_orientations(model_instance, halocat)
        print("Getting Galaxy Correlations")
        galaxy_omega, galaxy_eta, galaxy_xi = galaxy_alignment_correlations(galaxy_coords, galaxy_orientations)
        
        galaxy_omegas.append( galaxy_omega )
        galaxy_etas.append( galaxy_eta )
        galaxy_xis.append( galaxy_xi )
        
        print("Getting Cross-Correlations")
        correlations = galaxy_cross_correlations(model_instance)
        
        print("*****Repeat For Negative Alignment")
        print("Getting Model Instance")
        model_instance = create_mock(cens_occ_model, cens_prof_model, cens_orientation,
                                     sats_occ_model, sats_prof_model, sats_orientation,
                                     central_alignment_strength=central_alignment,
                                     satellite_alignment_strength=alignment*-1)
        print("Getting Coordinates and Orientations")
        galaxy_coords, galaxy_orientations = get_galaxy_coordinates_and_orientations(model_instance, halocat)
        print("Getting Galaxy Correlations")
        galaxy_omega, galaxy_eta, galaxy_xi = galaxy_alignment_correlations(galaxy_coords, galaxy_orientations)
        
        neg_galaxy_omegas.append( galaxy_omega )
        neg_galaxy_etas.append( galaxy_eta )
        neg_galaxy_xis.append( galaxy_xi )
        
        print("Getting Cross-Correlations")
        neg_correlations = galaxy_cross_correlations(model_instance)
        
        if plot_cross:
            print("Plotting")
            plot_omega_cross_correlations(correlations, alignment, neg_correlations, alignment*-1)
            plot_eta_cross_correlations(correlations, alignment, neg_correlations, alignment*-1)
            plot_xi_cross_correlations(correlations, alignment, neg_correlations, alignment*-1)
        print("\n\n")
    
    plot_galaxy_correlations( (galaxy_omegas, satellite_alignments, "Galaxy Shape-Position Correlations", r'$\omega(r)$'), 
                             (neg_galaxy_omegas,satellite_alignments*-1, "Galaxy Shape-Position Correlations", r'$\omega(r)$'),
                             halo_omega)
    plot_galaxy_correlations( (galaxy_etas, satellite_alignments, "Galaxy Shape-Shape Correlations", r'$\eta(r)$'),
                             (neg_galaxy_etas, satellite_alignments*-1, "Galaxy Shape-Shape Correlations", r'$\eta(r)$'),
                             halo_eta)
    plot_galaxy_correlations( (galaxy_xis, satellite_alignments, "Galaxy Position-Position Correlations", r'$\xi(r)$'), 
                             (neg_galaxy_xis, satellite_alignments*-1, "Galaxy Position-Position Correlations", r'$\xi(r)$'),
                             halo_xi)

def vary_central_alignment(central_alignments, satellite_alignment, halocat, halo_info, plot_cross=True):
    galaxy_etas = []
    galaxy_omegas = []
    galaxy_xis = []
    neg_galaxy_etas = []
    neg_galaxy_omegas = []
    neg_galaxy_xis = []
    halo_coords, halo_orientations, halo_omega, halo_eta, halo_xi, mask = halo_info
    
    for alignment in central_alignments:
        print("A.S. = ", alignment)
        
        print("Getting Model Instance")
        model_instance = create_mock(cens_occ_model, cens_prof_model, cens_orientation,
                                     sats_occ_model, sats_prof_model, sats_orientation,
                                     central_alignment_strength=alignment,
                                     satellite_alignment_strength=satellite_alignment)
        print("Getting Coordinates and Orientations")
        galaxy_coords, galaxy_orientations = get_galaxy_coordinates_and_orientations(model_instance, halocat)
        print("Getting Galaxy Correlations")
        galaxy_omega, galaxy_eta, galaxy_xi = galaxy_alignment_correlations(galaxy_coords, galaxy_orientations)
        
        galaxy_omegas.append( galaxy_omega )
        galaxy_etas.append( galaxy_eta )
        galaxy_xis.append( galaxy_xi )
        
        print("Getting Cross-Correlations")
        correlations = galaxy_cross_correlations(model_instance)
        
        print("*****Repeat For Negative Alignment")
        print("Getting Model Instance")
        model_instance = create_mock(cens_occ_model, cens_prof_model, cens_orientation,
                                     sats_occ_model, sats_prof_model, sats_orientation,
                                     central_alignment_strength=central_alignment,
                                     satellite_alignment_strength=alignment*-1)
        print("Getting Coordinates and Orientations")
        galaxy_coords, galaxy_orientations = get_galaxy_coordinates_and_orientations(model_instance, halocat)
        print("Getting Galaxy Correlations")
        galaxy_omega, galaxy_eta, galaxy_xi = galaxy_alignment_correlations(galaxy_coords, galaxy_orientations)
        
        neg_galaxy_omegas.append( galaxy_omega )
        neg_galaxy_etas.append( galaxy_eta )
        neg_galaxy_xis.append( galaxy_xi )
        
        print("Getting Cross-Correlations")
        neg_correlations = galaxy_cross_correlations(model_instance)
        
        if plot_cross:
            print("Plotting")
            plot_omega_cross_correlations(correlations, alignment, neg_correlations, alignment*-1)
            plot_eta_cross_correlations(correlations, alignment, neg_correlations, alignment*-1)
            plot_xi_cross_correlations(correlations, alignment, neg_correlations, alignment*-1)
        print("\n\n")
    
    plot_galaxy_correlations( (galaxy_omegas, satellite_alignments, "Galaxy Shape-Position Correlations", r'$\omega(r)$'), 
                             (neg_galaxy_omegas,satellite_alignments*-1, "Galaxy Shape-Position Correlations", r'$\omega(r)$'),
                             halo_omega)
    plot_galaxy_correlations( (galaxy_etas, satellite_alignments, "Galaxy Shape-Shape Correlations", r'$\eta(r)$'),
                             (neg_galaxy_etas, satellite_alignments*-1, "Galaxy Shape-Shape Correlations", r'$\eta(r)$'),
                             halo_eta)
    plot_galaxy_correlations( (galaxy_xis, satellite_alignments, "Galaxy Position-Position Correlations", r'$\xi(r)$'), 
                             (neg_galaxy_xis, satellite_alignments*-1, "Galaxy Position-Position Correlations", r'$\xi(r)$'),
                             halo_xi)

# Eliminate halos with 0 for halo_axisA_x(,y,z)
def mask_bad_halocat(halocat):
    bad_mask = (halocat.halo_table["halo_axisA_x"] == 0) & (halocat.halo_table["halo_axisA_y"] == 0) & (halocat.halo_table["halo_axisA_z"] == 0)
    halocat._halo_table = halocat.halo_table[ ~bad_mask ]
    
def log_prob(theta, inv_cov, x, y, model_instance, halocat, rbins, split):
    a, gamma = theta
    #a = theta

    if a < -5.0 or a > 5.0:
        return -np.inf

    avg_runs = 2
    omegas = []
    
    for i in range(avg_runs):
        # Reassign a and gamma for RadialSatellitesAlignmentStrength
        model_instance.model_dictionary["satellites_radial_alignment_strength"].param_dict["a"] = a
        model_instance.model_dictionary["satellites_radial_alignment_strength"].param_dict["gamma"] = gamma

        model_instance.model_dictionary["satellites_radial_alignment_strength"].assign_satellite_alignment_strength( table=model_instance.mock.galaxy_table )
        model_instance._input_model_dictionary["satellites_orientation"].assign_satellite_orientation( table=model_instance.mock.galaxy_table )
        model_instance._input_model_dictionary["centrals_orientation"].assign_central_orientation( table=model_instance.mock.galaxy_table )
    
        # Perform correlation functions on galaxies
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
        #galaxy_coords, galaxy_orientations = get_galaxy_coordinates_and_orientations(model_instance, halocat)
        #galaxy_omega, galaxy_eta, galaxy_xi = galaxy_alignment_correlations(galaxy_coords, galaxy_orientations, rbins)
        omega = ed_3d( sat_coords, sat_orientations, cen_coords, rbins, period=halocat.Lbox )

        omegas.append( omega )
    
    omegas = np.array( omegas )
    omega = np.mean( omegas, axis=0 )
    diff = omega[:split] - y[:split]

    return -0.5 * np.dot( diff, np.dot( inv_cov, diff ) )

global_nums = []
    
if __name__ == "__main__":
    job = sys.argv[1]
    jackknife_cov = None
    split = 10
    if len(sys.argv) > 2:
        jackknife_cov = sys.argv[2]
        if jackknife_cov.lower() == "none":
            jackknife_cov = None
    if len(sys.argv) > 3:
        split = int(sys.argv[3])

    omega_df = pd.read_csv("static_halos_galaxy_sat_shape_cen_pos_omegas_a=0.8036_gamma=-0.0385_seed=132358712.csv", index_col=False)
    subhalo_df = pd.read_csv("static_halos_halo_sat_shape_cen_pos_omegas_radial_AS_seed=132358712.csv", index_col=False)
    #eta_df = pd.read_csv("Halo_eta_w_false_subhalo.csv", index_col=False)
    #xi_df = pd.read_csv("Halo_xi_w_false_subhalo.csv", index_col=False)

    omega_mean = np.array( subhalo_df.mean() )
    
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
    cens_prof_model = TrivialPhaseSpace
    cens_orientation = CentralAlignment
    sats_occ_model = Leauthaud11Sats
    sats_prof_model1 = SubhaloPhaseSpace
    prof_args1 = ("satellites", np.logspace(10.5, 15.2, 15))
    #sats_orientation1 = SubhaloAlignment(satellite_alignment_strength=0.5, halocat=halocat)
    sats_orientation2 = RadialSatelliteAlignment
    #sats_orientation1 = SubhaloAlignment
    sats_strength = RadialSatelliteAlignmentStrengthAlternate()
    Lbox = halocat.Lbox
    sats_strength.inherit_halocat_properties(Lbox=Lbox)


    title1 = "Radially Dependent Radial Alignment Strength"
    title2 = "Constant Radial Alignment Strength"

    central_alignment = 1

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
    #model_instance._input_model_dictionary["satellites_orientation"] = sats_orientation2(satellite_alignment_strength=0.8, halocat=halocat)
    model_instance._input_model_dictionary["satellites_orientation"].inherit_halocat_properties( Lbox = halocat.Lbox )
    #model_instance.model_dictionary["satellites_radial_alignment_strength"].param_dict["gamma"] = 0
    #print("number of galaxies: ", len(model_instance.mock.galaxy_table))
    ndim, nwalkers = 2, 5
    #ndim, nwalkers = 1, 4

    p0 = 2*((np.random.rand(nwalkers, ndim)) - 0.5)

    if jackknife_cov is None:
        omega_df = np.array(omega_df).T
        cov = np.cov( omega_df )
        p, n = omega_df.shape
        p = len(omega_df[:split])
        factor = (n-p-2)/(n-1)
    else:
        cov = np.load(jackknife_cov)
        factor = 1

    cov = cov[:split,:split]
    inv_cov = np.linalg.inv(cov)
    # Include the factor from the paper
    inv_cov *= factor

    try:
        start = time.time()
        f_name = os.path.join("front_results","MCMC_"+job+".h5")
        backend = emcee.backends.HDFBackend(f_name)
        moves = [emcee.moves.StretchMove(a=2),emcee.moves.StretchMove(a=1.1),emcee.moves.StretchMove(a=1.5),emcee.moves.StretchMove(a=1.3)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[inv_cov, rbin_centers, omega_mean, model_instance, halocat, rbins, split], backend=backend, moves=moves)
        sampler.run_mcmc(p0, 5000, store=True, progress=True)
        chain = sampler.get_chain(flat=True, discard=500)
        autocorr = sampler.get_autocorr_time(quiet=True)
    
        #np.save( os.path.join("results","MCMC_"+job+".npy"), chain)
        #np.save( os.path.join("results","autocorr_"+job+".npy"), autocorr )
    except Exception as e:
        print(e)
        #file = open( os.path.join( "logs", "Run_"+job+".txt"), "w"  )
        #file.write(str(e))
        #file.close()
    finally:
        #timing = time.time() - start
        #file = open( os.path.join("timings","timing_"+job+".txt"),"w")
        #file.write(str(timing))
        #file.close()
        #f = open( os.path.join("paths","Path_"+job+".txt"), "w" )
        #for el in global_nums:
        #    f.write(str(el))
        #    f.write("\n")
        #f.close()
        pass
