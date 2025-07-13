#!/usr/bin/env python
# coding: utf-8
# %reset -f
# Import a bunch of libraries
import gc
gc.collect()

from ipywidgets import *

## OPTIONAL
from pprint import pprint

#toggle these two lines (use top for standalone python, bottom two for jupyter notebook)
# from IPython.display import display ##
get_ipython().magic(u'matplotlib inline')

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import os
import time # For creating folders with unique names, in chronological order.
import datetime
import re #RegEx used to find file names for curve fitting

from scipy.signal import argrelextrema # Used for making plots versus AOI of the peak-to-peak absorbance
import scipy.fftpack as ft # Used for the Hilbert (Kramers-Kronig) transform
import sympy as sp # used for the solveDrude() function; could be eliminated by providing the exact analytical expression
import csv # used for exporting data as .csv files
from ipywidgets import interact # used for fitting part of program
import ipywidgets as widgets # used for fitting part of program

import itertools # Used for generating n-phase expressions for Bruggeman EMT and Wiener Limits.

# Libraries used for plotting ellipsoids to visualize the coordinates when setting up a shape-anisotropic system
# import numpy as np #
# from matplotlib import pyplot as plt #
# import itertools #
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Arc
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.ticker import StrMethodFormatter # for formatting strings on plots when creating GIFs

# style = """
#     <style>
#        .jupyter-widgets-output-area .output_scroll {
#             height: unset !important;
#             border-radius: unset !important;
#             -webkit-box-shadow: unset !important;
#             box-shadow: unset !important;
#         }
#         .jupyter-widgets-output-area  {
#             height: auto !important;
#         }
#     </style>
#     """
# display(HTML(style))


# Global variables:
UNIT_m = 1
UNIT_s = 1

UNIT_cm = UNIT_m*1E-2
UNIT_mm = UNIT_m*1E-3
UNIT_um = UNIT_m*1E-6
UNIT_nm = UNIT_m*1E-9
UNIT_Angstrom = UNIT_m*1E-10

UNIT_icm = 1/UNIT_cm

UNIT_Hz = 1/UNIT_s
UNIT_MHz = 1E-9 /UNIT_s
UNIT_GHz = 1E-12/UNIT_s
UNIT_THz = 1E-15/UNIT_s

CONST_c = 299792458*UNIT_m/UNIT_s
CONST_e = 1.602176634E-19 # coulombs
CONST_h = 6.62607004E-34 # Joules * seconds
CONST_eps = 8.8541878128E-12 # Farad / metre
CONST_NA = 6.02214076E23 # mol**-1
CONST_me = 9.1093837015E-31 # kg
CONST_macheps = 6E-16 # approx 3x machine epsilon, used to deal with float errors, e.g. selecting correct branch of arcsin function with signed zeros

UNIT_eV = UNIT_cm*(CONST_e / (CONST_h*CONST_c))

def testImport():
    print("Import successful.")
    return

# A function for exporting data, maybe it should be part of a class, not too sure.
def exportDat(dataToExport, pathFileName):
    with open(pathFileName, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(zip(*dataToExport))
    print("Export complete")

def importColumnsCSV(fileName, path):
    ## Check if the provided material name occurs at the begining of any of the file names in materials_n_k directory
    files = []
    for i in os.listdir(path):
        if(i.startswith(fileName)):
#                 if(materialName in i): ## Use instead if you want the given name to be able to appear anywhere in filename
            files.append(i)

    ## If multiple files begin with the provided fileName, warn the user, but continue on using the first matching file
    if(len(files) == 0):
        unfoundError = "No files begining with " + fileName + " found."
        print(unfoundError)
    if(len(files) > 1):
        underspecifiedError = "Multiple files begining with " + fileName + " found. Using <" + files[0] + ">."
#         self.softError.append(underspecifiedError)
        print(underspecifiedError)
    if(len(files) >= 1):
#         filePath = os.path.relpath(path + files[0]) # CHANGED 2022-03-11 to the line below:
        filePath = os.path.join(path, files[0])
        dat = np.genfromtxt((filePath), delimiter = ',')
        columns = []
        for i in range(dat.shape[1]): # For each column
            columns.append(dat[:,][:,i])
    return columns, files[0]

##############################################################################
## FUNCTIONS FOR PLOTTING ELLIPSOIDS #########################################
## Should probably be moved to their own class or as a separate .py program ##
##############################################################################

#NOTMYFUNCTION from https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

# Used for drawing a 2d patch along any plane in a 3d plot. The 2d patch is projected onto a specified plane, then rotated by the rotation matrix.
# Heavily modified code from https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
def patch_3d(pathpatch, plane, rotMat, d = 0, plotBool=False):
    path = pathpatch.get_path() # Get the 2d path and make an array of the vertices
    trans = pathpatch.get_patch_transform()
    path = trans.transform_path(path)
    verts = path.vertices

    pathpatch.__class__ = art3d.PathPatch3D # Make a 3d pathpatch and copy over a bunch of properties to the 3d pathpatch
    pathpatch._code3d = path.codes
    pathpatch._facecolor3d = pathpatch.get_facecolor

    if (plotBool):
        ax2d = fig.add_subplot(1, 2, 1)
        ax2d.plot(verts[:,0], verts[:,1])
        ax2d.set_xlim(-1, 1)
        ax2d.set_ylim(-1, 1)

    # w and h stand for width and height, respectively
    # Define chirality: Right-hand rule -- if index points along the normal of the plane,
    # the thumb points along the width and the middle finger points along the height
    if(plane == "xy" or plane == "yx" or plane == "z"):
#         print("working along x-y plane")
        pathpatch._segment3d = np.array([np.dot(rotMat, (w, h, 0)) + np.dot(rotMat, (0, 0, d)) for w, h in verts])
    elif(plane == "xz" or plane == "zx" or plane == "y"):
#         print("working along x-z plane")
        pathpatch._segment3d = np.array([np.dot(rotMat, (h, 0, w)) + np.dot(rotMat, (0, d, 0)) for w, h in verts])
    elif(plane == "zy" or plane == "yz" or plane == "x"):
#         print("working along z-y plane")
        pathpatch._segment3d = np.array([np.dot(rotMat, (0, w, h)) + np.dot(rotMat, (d, 0, 0)) for w, h in verts])

# Returns the generalized rotation matrix for clockwise angles about the x, y, and z axes (i.e. yaw, pitch, and roll). Parameters are in degrees.
def general_rotation(theta_xDeg, theta_yDeg, theta_zDeg):
    theta_x = theta_xDeg*np.pi/180
    theta_y = theta_yDeg*np.pi/180
    theta_z = theta_zDeg*np.pi/180
    R_row1 = np.array([np.cos(theta_z)*np.cos(theta_y), np.cos(theta_z)*np.sin(theta_y)*np.sin(theta_x) - np.sin(theta_z)*np.cos(theta_x), np.cos(theta_z)*np.sin(theta_y)*np.cos(theta_x) + np.sin(theta_z)*np.sin(theta_x)])
    R_row2 = np.array([np.sin(theta_z)*np.cos(theta_y), np.sin(theta_z)*np.sin(theta_y)*np.sin(theta_x) + np.cos(theta_z)*np.cos(theta_x), np.sin(theta_z)*np.sin(theta_y)*np.cos(theta_x) - np.cos(theta_z)*np.sin(theta_x)])
    R_row3 = np.array([-np.sin(theta_y), np.cos(theta_y)*np.sin(theta_x), np.cos(theta_y)*np.cos(theta_x)])
    R = np.array([R_row1, R_row2, R_row3])
    return(R)

# See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
# Calculates the rotation matrix for counterclockwise rotations about the basis unit vector provided by the angle provided.
# To improve performance (maybe), consider implementing  http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf
# (See this for details: https://math.stackexchange.com/a/3219491)
def rodrigues_rotation(basisUnitVector, counterClockwiseAngleDeg):
    k = basisUnitVector/np.linalg.norm(basisUnitVector) # Normalize to a unit vector
    theta = counterClockwiseAngleDeg*np.pi/180
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = I + np.sin(theta)*K + (1 - np.cos(theta))*np.matmul(K,K)
#     print("Rodrigues R: ", R, "\n")
    return (R)

def drawEllipsoid(mpl_subplot, a, b, c, rotMatParams, rotMatType = "general", plotUnitVec = True):
    if (rotMatType == "general" or rotMatType == "General" or rotMatType == "YawPitchRoll"):
        theta_x, theta_y, theta_z = rotMatParams
        R = general_rotation(theta_x, theta_y, theta_z)
    elif(rotMatType == "rodrigues" or rotMatType == "Rodrigues" or rotMatType == "AxisAngle"):
        basisUnitVector, counterClockwiseAngle = rotMatParams
        R = rodrigues_rotation(basisUnitVector, counterClockwiseAngle)
    else:
        print(rotMatType)
        raise ValueError("ERROR: The final parameter passed to drawEllipsoid() must be a string, either 'general' or 'rodrigues', depending upon the type of rotation matrix to apply")
    plotEllipsoid(mpl_subplot, a, b, c, R, plotUnitVec)

# Draws an ellipsoid with semi-axes radii a, b, c along x, y, z respectively, and then rotated by rotation matrix, rotMat
def plotEllipsoid(mpl_subplot, a, b, c, rotMat, plotUnitVec = True):
    # Draw arrows indicating the diameters of the semi-axes
    arrows = np.array([[a, 0, 0], [0, b, 0], [0, 0, c], [-a, 0, 0], [0, -b, 0], [0, 0, -c]])
    origin = [0, 0, 0]
    for arrow in arrows:
        rotatedOrigin = np.dot(rotMat, origin)
        rotatedArrow = np.dot(rotMat, arrow)
        arrowElement = Arrow3D([rotatedOrigin[0], rotatedArrow[0]], [rotatedOrigin[1], rotatedArrow[1]], [rotatedOrigin[2], rotatedArrow[2]], mutation_scale=20, lw=2, arrowstyle="-|>", color='grey')
        mpl_subplot.add_artist(arrowElement)

    # Define the widths and heights. Take note of the right-hand rule definition in the patch_3d function.
    depths = [a, b, c]
    widths = [b, c, a]
    heights = [c, a, b]
    charAxes = ["x", "y", "z"] # The angles [theta_x, theta_y, theta_z] by which to rotate the 2d ellipse
    rgbColor = ["r", "g", "b"]
    RGB = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for coord in range(3):
#         curColour = RGB[coord] # Plot the x, y, and z cross sections in red, green, and blue respectively
        curColour = [0,0,0]
        alpha = 0.1
        curColour.append(alpha)
        curColour = tuple(curColour)

        ellipse = Arc((0, 0), width=2*widths[coord],height=2*heights[coord], color=rgbColor[coord])
        mpl_subplot.add_patch(ellipse)
        patch_3d(ellipse, charAxes[coord], rotMat, d=0)

        halfNumSegs = 30 # Half the number of elliptical cross sections constituting the wireframe

        for i in range(1, halfNumSegs, 1):
            dist = depths[coord]*i/halfNumSegs # The distance along the ellipsoid that the cross section will be calculated and drawn
            ellipseLim = lambda A, B, X: np.sqrt(B**2 - X**2*B**2/A**2) # Returns the Y value of an ellipse with semi-radii A and B at a distance of X along A. The formula for an ellipse is (x/a)**2 + (y/b)**2 = 1

            ww = (ellipseLim(depths[coord], widths[coord], dist))
            hh = ellipseLim(depths[coord], heights[coord], dist)
#             print("ww: ", ww, "hh: ", hh, "dist: ", dist)

            ellipse_section = Arc((0, 0), width=2*ww, height=2*hh, color=curColour)
            mpl_subplot.add_patch(ellipse_section)
            patch_3d(ellipse_section, charAxes[coord], rotMat, d=dist)
            ellipse_section = Arc((0, 0), width=2*ww, height=2*hh, color=curColour)
            mpl_subplot.add_patch(ellipse_section)
            patch_3d(ellipse_section, charAxes[coord], rotMat, d=-dist)
    if (plotUnitVec == True):
        unit_x = np.array([1,0,0])
        unit_y = np.array([0,1,0])
        unit_z = np.array([0,0,1])
        arrows = [unit_x, unit_y, unit_z]
        colour = [(.6, 0, 0, 0.6), (0, .6, 0, 0.6), (0, 0, .6, 0.6)]
        for axis in range(len(arrows)):
            arrows.append(np.dot(rotMat, arrows[axis]))
            curColour = [0,0,0]
            curColour[axis] = 1
            alpha = 1
            curColour.append(alpha)
            curColour = tuple(curColour)
            colour.append(curColour)

        for i, arrow in enumerate(arrows):
            a = Arrow3D([0, arrow[0]], [0, arrow[1]], [0, arrow[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color=colour[i])
            mpl_subplot.add_artist(a)

#################
#################
#################
class phaseSys(object):
    def __init__(self, phaseNum, materialsObj):
        # The following 3 lines is probably terrible practice, but I don't know how to do differently
        self.nu = materialsObj.nu
        self.aoi = materialsObj.aoi
        self.size = materialsObj.size
        self.materialsCopy = materialsObj
        self.thicknesses = np.zeros(phaseNum-2, dtype=np.float_)
        self.phaseNum = phaseNum
        self.layerNames = []
        self.layerPermittivityFunction = []
        self.layerParameters = []
        # self.etas = np.zeros(phaseNum)

        # Used by the propMatrix function and the function that calculates E as a function of z.
        self.theta = np.zeros([self.phaseNum, self.size], dtype=np.complex_)

    def getParameters(self):
        params = [
            ["Layer_material", *self.layerNames],
            ["Layer_thickness", "semi-infinite", *self.thicknesses, "semi-infinite"],
            ["Permittivity_function", *self.layerPermittivityFunction],
            ["Parameters_of_permittivity_function", *self.layerParameters],
            ["AOI__min_max_len", self.aoi[0], self.aoi[-1], len(self.aoi)],
            ["wavenumber__min_max_len", self.nu[0], self.nu[-1], len(self.nu)]
        ]
        return params


    def setThicknesses(self, vals):
        if len(vals) != self.phaseNum-2:
            raise ValueError("Error: The number of phases passed to setThicknesses() does not match the number of phases passed to PhaseSys()")
        for i in range(len(vals)):
            self.thicknesses[i] = vals[i]

    def setLayers(self, vals):
        if len(vals) != self.phaseNum:
            raise ValueError("Error: The number of phases passed to setLayers() does not match the number of phases passed to PhaseSys()")
        self.etas = []
        penultimate = vals[len(vals)-2]
        self.penultimatePermittivityFunction = self.materialsCopy.matDict[penultimate]["permittivityFunction"] # This variable records how the second-to last layer's permittivity values were computed
#         print("penultimate material:", penultimate)
        for i in vals:
            self.layerNames.append(i)
            self.layerPermittivityFunction.append(self.materialsCopy.matDict[i]["permittivityFunction"])
            self.layerParameters.append(self.materialsCopy.matDict[i]["parameters"])
            self.etas.append(self.materialsCopy.matDict[i]["eta"])
        if self.penultimatePermittivityFunction == "setBruggeman":
            # The reference parameters for the setBruggeman() method. Look carefully at the function call on the next
            # line. All the arguments originally input to setBruggeman() are the same, except the Lorentz-oscillator
            # layer, which is replaced with the terminal phase.
            # (i.e. the method call creates the Bruggeman permittivity for the reference case)
            self.materialsCopy.setBruggeman("BR_ref", self.materialsCopy.BReps[0], self.materialsCopy.BReps[2], self.materialsCopy.BReps[2], self.materialsCopy.BRthick, self.materialsCopy.BRmolec, self.materialsCopy.BRratio1, self.materialsCopy.BRF)
            etaWithoutLO = self.materialsCopy.matDict["BR_ref"]["eta"]
        if self.penultimatePermittivityFunction == "setOsawaBruggeman":
            self.materialsCopy.setOsawaBruggeman("BR_ref", self.materialsCopy.BReps[0], self.materialsCopy.BReps[2], self.materialsCopy.BReps[2], self.materialsCopy.BRthick, self.materialsCopy.BRmolec, self.materialsCopy.BRratio1, self.materialsCopy.BRF)
            etaWithoutLO = self.materialsCopy.matDict["BR_ref"]["eta"]
        else:
            etaWithoutLO = self.etas[len(self.etas) - 1] #If no Bruggeman was used, the penultimate is filled with the terminal layer
        self._etasRef = np.array(self.etas) # The reference system is the same as the "sample" system, with the exception that the organic layer is removed.
        self._etasRef[len(self._etasRef) - 2] = etaWithoutLO # Remove the organic layer by replacing it with one of the above cases.
#         for i in self._etas:
#             print(i)

        ##MAYBE I NEED TO ADD A LINE SETTING BRused to zero?? I think somewhere it is needed, I'm not sure where tho
    @property
    def etasRef(self):
        return self._etasRef
#     @etasRef.setter # I don't think I need this ??
#     def etasRef(self, vals):

    def calcA(self, pol, mode="ratioR"):
        absorbance = np.zeros(len(self.nu))
        if mode == "ratioR":
            reference = self.propMatrix(pol, True)["R"]
            sample = self.propMatrix(pol, False)["R"]
            for i in range(len(self.nu)):
                absorbance[i] = -1000*np.log10(sample[i]/reference[i])
        elif mode == "totalRT":
            reflection = self.propMatrix(pol, False)["R"]
            transmission = self.propMatrix(pol, False)["T"]
            for i in range(len(self.nu)):
                absorbance[i] = 1 - reflection[i] - transmission[i]
        return absorbance

    def calcR(self, pol, refEta=False):
        return self.propMatrix(pol, refEta)["R"]

    def calcT(self, pol, refEta=False):
        return self.propMatrix(pol, refEta)["T"]

    def peakHeights(self, yvals):
        maxVal = [np.take(yvals, argrelextrema(yvals, np.greater)[0]), "max"]
        minVal = [np.take(yvals, argrelextrema(yvals, np.less)[0]), "min"]

        all_extrema = [minVal, maxVal]
        proper_lims = []
        midpoint = np.zeros(2)
        left_bound = yvals[0]
        right_bound = yvals[-1]
        midpoint = np.abs(left_bound - right_bound)/2 + np.amin([left_bound, right_bound])
#         plt.axhline(midpoint[trace_ID], color=colours[trace_ID], linestyle='dotted')

        for i, ext in enumerate(all_extrema):
            num_ext = len(ext[0])
            cur_ext = ext[0]
            type_extrema = ext[1]
            if(num_ext == 0):
#                 ax.plot([erick.nu[0], erick.nu[-1]], [left_bound, right_bound], color=colours[trace_ID], linestyle="--")
                proper_lims.append(midpoint)
            elif(num_ext==1):
                proper_lims.append(cur_ext[0])
#                 plt.axhline(proper_lims[i], color=colours[trace_ID], linestyle='--')
            elif(num_ext > 1): # Take absolute max or min
                if(type_extrema == "max"):
                    proper_lims.append(np.max(cur_ext))
                elif(type_extrema == "min"):
                    proper_lims.append(np.min(cur_ext))
#                 plt.axhline(proper_lims[i], color=colours[trace_ID], linestyle='--')
#         ax.legend(["Abs s-pol", "Abs p-pol"], loc='lower right')
#         ax.text(0.05, 0.95, curStr, transform=ax.transAxes, fontsize=12, verticalalignment='top')
#         fig.suptitle(subtitle)
    #     plt.ylim(0., 35.)
#         ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
#         fig.savefig(file_title + ".png")
    #     plt.show()
#         plt.close()

        peak_height = np.abs(proper_lims[1] - proper_lims[0])
        upward_peak = proper_lims[1] - midpoint
        bimodality = upward_peak/peak_height ## A number which goes from 0 (fully inverted) to 1 (fully upward) with 0.5 being equally-lobed bimodal
    #     print("max: ", proper_lims[2], "mid: ", midpoint[0])
    #     print("total: ", peak_height_s, "+ve: ", upward_peak_s, "bimodality param: ", bimodality_s)
        return(proper_lims, peak_height, bimodality)

#         startInd = 0
#         endInd = -1
#         midInd = self.size//2
#         if (printNuBool):
#             print(self.nu[startInd], self.nu[midInd], self.nu[endInd])

#         def getPH:


#         if self.materialsCopy.domain == "aoi":
#             #
#         elif self.materialsCopy.domain == "frequency":

#         Hp.append(Ap[midInd] - (Ap[startInd] + Ap[endInd]) / 2)
#         Hs.append(As[midInd] - (As[startInd] + As[endInd]) / 2)

#         fig_ph = plt.figure(figsize=(10,5))
#         ax_ph = fig_ph.add_subplot(111)
#         phx = list(aoi)
#         phy1 = Hp
#         phy2 = Hs
#         phy3 = list(np.array(Hp)/np.array(Hs))
#         ax_ph.plot(phx, phy1, phx, phy2)
#         plt.legend(["Height p-pol", "Height s-pol"], loc="lower right")
#         plt.xlabel("wavenumber (cm^-1)")
#         plt.ylabel("peak height")
#         phx.insert(0, "AOI (deg)")
#         phy1.insert(0, "Height p-pol")
#         phy2.insert(0, "Height s-pol")
#         phy3.insert(0, "Ratio Height p-/s-")
#         aoiStr = ("{:2.1f}".format(aoi_i)).zfill(4)
#         title = "peak_height_vs_aoi_f=" + str(ff)
#         fig_ph.suptitle(title)
#         ph_folder = folder + "peak_height/"
#         if not os.path.exists(ph_folder):
#             os.mkdir(ph_folder)
#         fig.savefig(ph_folder + title +".png")
#         exportDat([phx,phy1,phy2,phy3], ph_folder + title + ".csv")
#         plt.show()
#         plt.close()

    # An explanation on nomenclature: I tried to follow the nomenclature of Ohta.
    # The letter j is included in a variable name for arrays which hold values for a single phase.
    # For example the D matrix is named D, since it holds values for all phases.
    # Kzj has "j" in its name because it only holds values for a particular user-specified phase (the j-th phase).
    # fwd refers to forward propagating waves and ref refers to backward propagating (reflected) waves

    def calcE(self, z, incidentFieldP = 1, incidentFieldS = 1):
        delZ = z
        sumOfThick = 0

        # Note: l is a pseudo-j because l is the incrementer for thicknesses, which do not need to be defined
        # for the incident and terminal phases, so to convert to j we need to subtract 1 to make up for the
        # incident phase, for which there is no thickness defined. But in the code, as written, we overshoot by 1, so it comes out in the wash
        for l in range(len(self.thicknesses)+1):
            if delZ < 0:
                if (l == 0): # z is in the incident semi-infinite medium
                    j = 0
                else: # z is in one of the middle layers
                    j = l
                    delZ += self.thicknesses[l-1] # we overshot (z is in the previous layer), so take a step back
                break
            elif l == len(self.thicknesses): # Since delZ is positive, z is in the terminal semi-infinite medium
                j=l+1
                break
            else: # We haven't determined which layer z is in yet.
        #         print("layer #", l+1, ": ", delZ, "(remainder) +", sumOfThick, "(sum of prev layers) =", z, "(desired z)")
                delZ -= self.thicknesses[l]
                sumOfThick += self.thicknesses[l]

        EjfwdP, EjrefP = self.calcEAmp('p', j, incidentFieldP)
        EjfwdS, EjrefS = self.calcEAmp('s', j, incidentFieldS)
#         print("The distance z = ", z, "specified lies within layer #", j, "theta=", self.theta[j])
        delZ_m = delZ*1E-9 # convert to metres
#         for i in range(self.size):
        Kzj = 2 * np.pi * self.nu * self.etas[j] * np.cos(self.theta[j]) *100 # The factor of 100 converts from inverse centimetres to inverse metres
        EfwdZP = EjfwdP * np.exp( 1j * Kzj * delZ_m)
        ErefZP = EjrefP * np.exp(-1j * Kzj * delZ_m)
        EfwdZS = EjfwdS * np.exp( 1j * Kzj * delZ_m)
        ErefZS = EjrefS * np.exp(-1j * Kzj * delZ_m)
        Ex = (EfwdZP - ErefZP) * np.cos(self.theta[j])
        Ey = EfwdZS + ErefZS
        Ez = (EfwdZP + ErefZP) * np.sin(self.theta[j])

        Fx = np.real(np.conj(Ex)*Ex / np.conj(incidentFieldP)*incidentFieldP)
        Fy = np.real(np.conj(Ey)*Ey / np.conj(incidentFieldS)*incidentFieldS)
        Fz = np.real(np.conj(Ez)*Ez / np.conj(incidentFieldP)*incidentFieldP)
        # print("delZ: ", delZ)
        # print("j: ", j)
#         print(type(Kzj))
        Fs = Fy
        Fp = Fx + Fz
        F = (Fz + Fy + Fz) / 2
        returnedDictionary = {'K': Kzj, 'Ex':Ex, 'Ey':Ey, 'Ez':Ez, 'Fx':Fx, 'Fy':Fy, 'Fz':Fz, 'Fp':Fp, 'Fs':Fs, 'F':F}
        return returnedDictionary

    def calcEAmp(self, pol, jthBoundary, incidentField = 1): #jthBoundary, incidentFieldP = 1, incidentFieldS = 1): # The incident electric fields are normalized to 1.
        Efwd = np.zeros([self.size, self.phaseNum], dtype=np.complex_)
        Eref = np.zeros([self.size, self.phaseNum], dtype=np.complex_)
#
        Edict = self.propMatrix(pol)["Edict"]
        D = Edict["D"]
        tprod = Edict["tprod"]

        for i in range(self.size):
            # for p in range(self.phaseNum):
                # print("D[",p,"][",i,"]: \n", D[p][i])
            # print("\n")
            a = D[0][i][0][0] # This is the (1, 1) element of the Cprod matrix.
            # print(a)
            for o in range(self.phaseNum):
                aj = D[o][i][0][0] # The (1, 1) element of the jth D-matrix. Remember, j counts down from terminal (m+1) to incident (0th) phase.
                cj = D[o][i][1][0] # The (2, 1) element of the jth D-matrix
                Efwd[i][o] = tprod[i][o]* aj * incidentField / a
                Eref[i][o] = tprod[i][o]* cj * incidentField / a
                # print("i, o, tprod: ", i, o, tprod)
                # if(o==0 and i==0):
                #     print("tprod[i][o]: ", tprod[i][o])

                # print(tprod[i][o])
        # print("Efwd: ", Efwd)
        return Efwd[:,jthBoundary], Eref[:,jthBoundary]

    # This function calculates the C propagation matrices
    def propMatrix(self, pol, refEta=False):
#         print("calling propMatrix()")
        n = np.zeros([self.phaseNum], dtype=np.complex_)
        eps = np.zeros([self.phaseNum], dtype=np.complex_)
        h = np.zeros([self.phaseNum])
        # This "h" array has two more elements than necessary (the initial and terminal phases are taken to be infinite),
        # but the first and final elements remain as the zeroes they were initialized to in the assignment, and are not
        # ever used

        delta = np.zeros([self.phaseNum], dtype=np.complex_)
        Xi = np.zeros([self.phaseNum], dtype=np.complex_)
        xi = np.zeros([self.phaseNum], dtype=np.complex_)
        r = np.zeros([self.phaseNum, self.size], dtype=np.complex_)
        t = np.ones([self.phaseNum], dtype=np.complex_) # Changed from "zeros" to "ones" on 2023-11-07. So that in the incident semi-infinite phase, t = 1, and the electric fields are calculated properly.
        tprod = np.zeros([self.size, self.phaseNum], dtype=np.complex_)
        C = np.zeros([self.phaseNum, self.size, 2, 2], dtype=np.complex_)
        D = np.ones([self.phaseNum, self.size, 2, 2], dtype=np.complex_)

        R = np.zeros([len(self.nu)])
        T = np.zeros([len(self.nu)])

        for i in range(self.size): # Iterate over each frequency / wavelength
            for o in range(self.phaseNum): # Iterate over each layer
                if refEta == True:
                    n[o] = self.etasRef[o][i]
                    eps[o] = (self.etasRef[o][i])**2
                elif refEta == False:
                    n[o] = self.etas[o][i]
                    eps[o] = (self.etas[o][i])**2
                if o == 0: # Incident semi-infinite layer
                    self.theta[o][i] = np.pi*self.aoi[i]/180
                    delta[o] = 0
                elif o == self.phaseNum - 1: # Terminal semi-infinite layer
                    argument = n[o-1]*np.sin(self.theta[o-1][i])/n[o]
                    if np.abs(np.imag(argument)) < CONST_macheps: # This condition is used to select the correct branch of the arcsin function. If the complex argument has an tiny imaginary component fluctuating on either side zero (e.g. a signed zero), the arcsin function takes on a large value which fluctuates between negative and positive, which is not the expected behaviour.
                        argument = np.real(argument) + np.imag(argument)*1J*0
                    self.theta[o][i] = np.arcsin(argument)
                else: # Intermediate layer with defined thickness
                    h[o] = self.thicknesses[o - 1]*1E-7 #Changed 2021-04-05 from h[o] = self.thicknesses[o - 1]*10**-7
                    self.theta[o][i] = np.arcsin(n[o-1]*np.sin(self.theta[o-1][i])/n[o])
                    delta[o] = 2*np.pi*self.nu[i]*n[o]*np.cos(self.theta[o][i])*h[o]
                Xi[o] = n[o]*np.cos(self.theta[o][i])
                xi[o] = np.abs(np.real(Xi[o]))+1j*np.abs(np.imag(Xi[o]))
                if o > 0:
#                     if pol == 'p':
#                         r[o][i] = (eps[o]*xi[o-1] - eps[o-1]*xi[o])/(eps[o]*xi[o-1] + eps[o-1]*xi[o])
#                         t[o] = (2*xi[o-1]*n[o-1]*n[o])/(eps[o]*xi[o-1] + eps[o-1]*xi[o])
#                     elif pol == 's':
#                         r[o][i] = (xi[o-1] - xi[o])/(xi[o-1] + xi[o])
#                         t[o] = (2*xi[o-1])/(xi[o-1] + xi[o])

                    ###########################
                    ### MODIFIED 2021-08-24 ###
                    if pol == 'p':
                        r[o] = (n[o]*np.cos(self.theta[o-1]) - n[o-1]*np.cos(self.theta[o]))/(n[o]*np.cos(self.theta[o-1]) + n[o-1]*np.cos(self.theta[o]))
                        t[o] = (2*xi[o-1]*n[o-1]*n[o])/(eps[o]*xi[o-1] + eps[o-1]*xi[o])
                    elif pol == 's':
                        r[o] = (n[o-1]*np.cos(self.theta[o-1]) - n[o]*np.cos(self.theta[o]))/(n[o-1]*np.cos(self.theta[o-1]) + n[o]*np.cos(self.theta[o]))
                        t[o] = (2*xi[o-1])/(xi[o-1] + xi[o])
                    ###########################
                    else:
                        print(r"Error: the 2nd argument passed to propMatrix(), pol, must be a char, either 's' or 'p', indicating the polarization of light")
                        return None
                    C[o][i] = np.array([[np.exp(-delta[o-1]*1j), r[o][i]*np.exp(-delta[o-1]*1j)],[r[o][i]*np.exp(delta[o-1]*1j), np.exp(delta[o-1]*1j)]])
#                 print(self.theta[o][i])
#             print(" ")

            # Set D[max] to the identity matrix.
            D[self.phaseNum - 1][i] = np.array([[1, 0], [0, 1]])
            # Count down from D[max-1] to D[0] since the result of D[j+1] is used to compute the result of D[j]
            for o in range(2, self.phaseNum+1):
                # Uncomment to see what is being assigned
                # print("D[",self.phaseNum-o,"][", i, "] assignment; multiplying C[", self.phaseNum - o + 1, "][", i, "],  with D[", self.phaseNum - o + 1, "][", i, "].")
                # print(C[i][self.phaseNum - o + 1], "\n\n")
                D[self.phaseNum-o][i] = np.matmul(C[self.phaseNum - o + 1][i], D[self.phaseNum - o + 1][i]) # multiply on the left with the previous result

            # This loop could probably become slightly more optimal (but slightly less readable) by including
            # the operations in the above loop. However, I think it's clearer to count upwards when calculating t.
            for o in range(self.phaseNum):
                if o <= 1:
                    tprod[i][o] = t[o]
                else:
                    tprod[i][o] = tprod[i][o-1]*t[o]
            # This is redundant. The matrix D[0][i] = Cprod for a given i (the product of all C matrices)
            Cprod = D[0][i]
            rfinal = Cprod[1][0]/Cprod[0][0]
            R[i] = np.real(np.conj(rfinal)*rfinal)
            tfinal = tprod[i][self.phaseNum-1]/Cprod[0][0]
            T[i] = np.real(np.real(xi[self.phaseNum - 1]/xi[0])*tfinal*np.conj(tfinal))
        Edict = {"D":D, "tprod":tprod}
        returnedDictionary = {'R': R, 'T':T, 'Edict':Edict, 'rfinal':rfinal, 'r':r}
        return returnedDictionary
        # if mode == 'R' or mode == 'r':
                # return R
        # elif mode == 'T' or mode == 't':
            # return T
        # else:
            # print("Error: the final argument, 'mode', must be a string, either 'T' (transmissivity), 'R' (reflectivity), or 'RT' (both) indicating the quantity to be returned by the function.")
            # return None


# Used for manual curve fitting of experimental Reflectivity curves to the Drude model.
# In order to use, put the reflectivity curves (.dpt files exported from OPUS) in a folder.
# The folder should be in the same directory as where this notebook is running.
# The name of this folder is passed to the function create fit as the first parameter, dataFolderName.
# The files in this folder should be names with a string including:
# the aoi as "##deg", the thickness as "###A", and the polarization as "ppol" or "spol".
# NOTE: this function assumes that the experimental data is a function of wavenumber and is sorted in either ascending or descending wavenumber space
class createFit(object):
    def __init__(self, dataFolderName, pol, aoi, thickness, material, spectrum, min_wavenum=1000, max_wavenum=8000):
        self.pol = pol
        self.aoi = aoi
        self.thickness = thickness
        self. material = material
        self.spectrum = spectrum
        self.title = ('External reflectivity of ' + str(self.thickness) + ' angstrom ' + str(self.material) +' on Si at ' + str(self.aoi) + ' deg for ' + self.pol + '-pol')
        self.min_wavenum = min_wavenum
        self.max_wavenum = max_wavenum

        fnames = os.listdir(os.path.relpath(dataFolderName))
        matches = []
        for f in fnames:
            if re.search(str(self.thickness) + "A", f) and re.search(str(self.aoi) + "deg", f) and re.search(str(self.pol) + "p", f):
                matches.append(f)
#         if len(matches) > 1:
#             print("SOFT ERROR: Multiple matches found, loading file: " + matches[0])
        filePath = os.path.abspath(os.path.join(dataFolderName, matches[0]))
        print(matches)
        print("Loading file: " + filePath)
        typ
        dat = np.genfromtxt(filePath, delimiter = ',')
        expt_wavenumbers = dat[:][:,0]
        min_wavenum_index = np.abs(np.asarray(expt_wavenumbers) - min_wavenum).argmin()
        max_wavenum_index = np.abs(np.asarray(expt_wavenumbers) - max_wavenum).argmin()
        minIndex = min(min_wavenum_index, max_wavenum_index)
        maxIndex = max(min_wavenum_index, max_wavenum_index)

        #The single beam part below hasn't been tested after I overhauled it quite significantly
        if self.spectrum == 'sb':
            dat = np.genfromtxt(filePath, delimiter = ',')
            matches = []
            for f in fnames:
                if re.search(str(self.thickness) + "A", f) and re.search(str(self.aoi) + "deg", f) and re.search(self.pol + "p", f):
                    matches.append(f)
            if len(matches) > 1:
                print("SOFT ERROR: Multiple matches found, loading file: " + matches[0])
            refFilePath = os.path.abspath(os.path.join(dataFolderName, matches[0]))
            dat2 = np.genfromtxt(refFilePath, delimiter = ',') # I've used the file naming convention "auAA.csv" where AA is the angle in degrees read from the Pike Box.
            self.exptWnum = dat[minIndex:maxIndex][:,0]
            izo = dat[minIndex:maxIndex][:,1]
            ref = dat2[minIndex:maxIndex][:,1]
            self.exptRefl = np.divide(izo, ref)
        elif self.spectrum == 'refl':
            self.exptWnum = dat[minIndex:maxIndex][:,0]
            self.exptRefl = dat[minIndex:maxIndex][:,1]
        else:
            print("Error: the variable spectrum must be provided with a string; either 'sb' or 'refl'.")

    def drude1(self, vP, vD, e_inf, thick, ang, plot=True):
        mat = materials("frequency", ang, self.min_wavenum, self.max_wavenum, 50)
        mat.setDrude("ITO", vP, vD, e_inf)
        mat.importMat(["Si"])
        mat.setFixed("Air", 1)

#         plt.plot(mat.nu, np.real(mat.matDict["ITO"]["eps"]), mat.nu, np.imag(mat.matDict["ITO"]["eps"]))
#         plt.legend(["Re", "Im"])
#         plt.show()

        system = phaseSys(3, mat)
        system.setLayers(["Air", "ITO", "Si"])
        system.setThicknesses([thick])
        simulatedY = system.calcR(self.pol)
        simulatedX = mat.nu

        if(plot):
            plt.plot(simulatedX, simulatedY, "b-", self.exptWnum, self.exptRefl, "r-")
            plt.ylabel('Reflectivity')
            plt.xlabel('wavenumber (cm^-1)')
            plt.title(self.title)
            plt.legend(('Simulated spectrum', 'Experimental spectrum'), loc=(0.573,0.86), fontsize=10)
            plt.tight_layout()
            plt.show()
            return

        return simulatedX, simulatedY


    def drude2(self, vP_surface, vD_surface, vP_bulk, vD_bulk, e_inf, frac, thick, ang):
        mat = materials("frequency", ang, self.min_wavenum, self.max_wavenum, 50)
        mat.setDrude("ITOs", vP_surface, vD_surface, e_inf)
        mat.setDrude("ITOb", vP_bulk, vD_bulk, e_inf)
        mat.setFixed("Air", 1)
        mat.importMat(["Si"])

        system = phaseSys(4, mat)
        system.setLayers(["Air", "ITOs", "ITOb", "Si"])
        system.setThicknesses([frac*thick, (1-frac)*thick])
        simulatedY = system.calcR(self.pol)
        simulatedX = mat.nu

        plt.plot(simulatedX, simulatedY, "b-", self.exptWnum, self.exptRefl, "r-")
        plt.ylabel('Reflectivity')
        plt.xlabel('wavenumber (cm^-1)')
        plt.title(self.title)
        plt.legend(('Simulated spectrum', 'Experimental spectrum'), loc=(0.573,0.86), fontsize=10)
        plt.tight_layout()
        plt.show()

#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
#################
class materials(object):
    def __init__(self, domainInput, fixed_aoi_or_wavenumber, loInput=1500, hiInput=1700, numVals=101, wavenumARR=None):
        self.softError = [] # An error log that can be forwarded onto the JS webpage
        self.lo = loInput
        self.hi = hiInput
        self.size = numVals
        self.fixedVal = fixed_aoi_or_wavenumber
        if (numVals == 1):
            self.incr = 0;
        else:
            self.incr = (hiInput - loInput)/(numVals-1)
        self.matDict = dict()
        self.domain = domainInput
        if (self.domain == "frequency"):
            self.nu = np.linspace(loInput, hiInput, numVals)
            self.aoi = np.full(numVals, fixed_aoi_or_wavenumber)
        elif (self.domain == "aoi"):
            self.nu = np.full(numVals, fixed_aoi_or_wavenumber)
            self.aoi = np.linspace(loInput, hiInput, numVals)
        if(not wavenumARR is None):
            self.nu = wavenumARR
            self.lo = np.amin(wavenumARR)
            self.hi = np.amax(wavenumARR)
            self.size = len(wavenumARR)
        self.addedMat = dict() # This stores the fixed values passed to setFixed(). Not sure the purpose; may be deperecated.
        self.BRused = None
        self.statDict = dict()

    ############################################
    #####~~Various helpful math functions~~#####
    ############################################

    # Helper function to call interpolateCubic() using at a single value of x.
    def interpolateCubicVal(self, xVals, yVals, xDesired, allowNegative=False, plotResult = False):
        xOut, yOut = self.interpolateCubic(xVals, yVals, 0, xDesired, 1, allowNegative, plotResult)
        return yOut[0]

    # Helper function to call interpolateCubic() with start, end and increment parameters for the output data.
    def interpolateCubicRange(self, xVals, yVals, xIncr, xStart = "first", xEnd = "last", allowNegative=False, plotResult = False):
        # Error checking
        if(xStart == "first"):
            xStart = xVals[0]
        if(xEnd == "last"):
            xEnd = xVals[len(xVals)-1]
        if (xStart < xVals[0] or xEnd > xVals[len(xVals)-1]):
            print("xStart: ", xStart, "\nxVals[0]:", xVals[0], "\nxEnd: ", xEnd, "\nxVals[len(xVals)-1]: ", xVals[len(xVals)-1])
            raise ValueError("The requested xStart and xEnd values are outside the ranges of the data provided. The interpolateCubic() function does not work for extrapolation.")
            return

        if(xIncr == 0):
            raise valueError("The third parameter passed to interpolateCubicRange(), <xIncr>, must be non-zero.")
            return
        numPts = int((xEnd - xStart) / xIncr)
        return self.interpolateCubic(xVals, yVals, xIncr, xStart, numPts, allowNegativeplotResult)

    # This function interpolates from the value of xStart, and increments by xIncr until numPts have been calculated.
    def interpolateCubic(self, xVals, yVals, xIncr, xStart = "first", numPts = 10, allowNegative=True, plotResult = False):
        # Error checking
        xVals = np.array(xVals) # make sure all data is in a numpy array and not a python list
        yVals = np.array(yVals)
        if(xStart == "first"):
            xStart = xVals[0]
        xEnd = xStart + xIncr*(numPts-1)
        if (xStart < xVals[0] or xEnd > xVals[-1]):
            print("xStart: ", xStart, "must be greater than xVals[0]: ", xVals[0], "\nxEnd: ", xEnd, "must be less than xVals[len(xVals)-1]: ", xVals[len(xVals)-1])
            raise ValueError("The requested xStart and xEnd values are outside the ranges of the data provided. The interpolateCubic() function does not work for extrapolation.")
            return

        ## This first part calculates the coefficients for a cubic function that is valid between the x1 and x2
        a = []
        b = []
        c = []
        d = []

        for i in range((xVals.size)-1):
            x1 = xVals[i]
            x2 = xVals[i+1]

            y0 = yVals[i]# If at the begining of the series, make up some data by duplicating the initial value
            if (i!=0):
                y0 = y1
            y1 = yVals[i]
            y2 = yVals[i+1]
            y3 = yVals[i+1] # If at the end of the series, make up some data by duplicating the final value
            if (i != (yVals.size)-2):
                y3 = yVals[i+2]

            a.append((-y0/2) + (3*y1/2) - (3*y2/2) + (y3/2))
            b.append(y0 - (5*y1/2) + (2*y2) - (y3/2))
            c.append((-y0/2) + y2/2)
            d.append(y1)

        ## This part calculates values, using the cubic spline segments calculated above
        xOut = np.zeros(numPts)
        yOut = np.zeros(numPts)

        negative_n_k_error = False
        for i in range(numPts):
            xOut[i] = xIncr*i + xStart
            index = 0
            for j in range((xVals.size)-1): # CHANGED FROM <<for j in range(len(xVals)): >> on June 3, 2021 ## changed to numpy array 2023-11-07
                if (xVals[j] <= xOut[i]):
                    index = j
                else:
                    break
            deltaX = xOut[i] - xVals[index] # This gives the x-value to input to the cubic function to get the desired y-value
            delX = deltaX / (xVals[index + 1] - xVals[index]) # Normalize such that the input is on the interval [0, 1] between xVal[index] and xVal[index + 1].
            yOut[i] = a[index]*delX**3 + b[index]*delX**2 + c[index]*delX + d[index]

            # The Bruggeman equation generates n-number of roots for n constituent materials. There is then an
            # algorithm to determine which root is physically valid. The algorithm as implemented rejects all roots
            # with negative imaginary epsilon our of hand.
            # If importing n and kappa values which have a steep minimum near zero, it is possible that a cubic
            # interpolation algorithm will cause the interpolated values to be negative. At certain fill factors,
            # it is also possible that the Bruggeman effective permittivity will likewise have a negative imaginary
            # permittivity. The algorithm will reject those roots and

            # If you want to work with metamaterials which have negative n or kappa, comment out this if-statement or
            # call the function with the allowNegative kwarg set to True. Note that you will also have to implement
            # another more advanced root-selection algorithm to use the Bruggeman EMT functions.
            if (yOut[i] < 0 and not allowNegative):
                yOut[i] = 0
                negative_n_k_error = True
        if (negative_n_k_error):
            print("During the interpolation of some data, a negative value was produced. It has been set to 0 to prevent failure with the root-selection algorithm for the bruggeman EMT. If a negative value was expected, set the kwarg <allowNegative> to True when calling the interpolateCubic() function.")

        ## Optionally, plot the result
        if(plotResult):
            plt.scatter(xVals, yVals, c="red", s=50)
            plt.scatter(xOut, yOut, c="blue", s=2)
            plt.legend(["Input Data", "Cubic Spline Interpolation"])
            plt.ylabel("yVals")
            plt.xlabel("xVals")
            plt.title(plotResult)
            plt.show()
            print("Number of interpolated values: ", len(xOut))
        return xOut, yOut

    # Calculates the roots of a cubic function.
    def cubicRoots(self, a, b, c, d):
        var = sympy.symbols('var')
        solns = sympy.solve(a*var**3 + b*var**2 + c*var + d, var)
    #     for i in range(len(solns)):
    #         print("solution #", i, ": ", complex(solns[i]))
        return(solns)

    ## Consider a product of several binomials of the form: (a1 + a2*x)*(b1 + b2*x)*(c1 + c2*x)*...(n1 + n2*x)
    ## When you expand out the product of n-number of binomials, you will get a polynomial of degree n.
    ## The purpose of the code below is to calculate the coefficients of the polynomial.
    ## This code takes two arrays containing the coefficients of the binomial terms
    ## coef1 =[a1, b1, c1, d1, ..., n1];  coef2 =[a2, b2, c2, d2, ..., n2]
    ## If the coefficients of the resulting polynomial are expressed as z0*x^0 + z1*x^1 + z2*x^2 + ... + zn*x^n
    ## Then the output is an array <polyCoef> containing the coefficients: [z1, z2, z3, z4, z5]
    def binomialProdCoefs(self, coef1, coef2):
        polyCoef = []
        indices = np.arange(0, len(coef1))
        for i in range(0, len(coef1)+1):
            polyCoef.append(0)
            toReplace = list(itertools.combinations(indices, i))

            if(len(toReplace)):
                for item in toReplace:
                    if(len(item)):
                        element = coef1.copy()
                        for jtem in item:
                            element[jtem] = coef2[jtem]
                        polyCoef[i] += np.prod(element)
                    else: #for first one
                        polyCoef[i] = np.prod(coef1)
    #         print("i: ", i, "toReplace: ", toReplace)
        return np.array(polyCoef)


    ###########################################################
    #####~~Permittivity functions (Non-Effective medium)~~#####
    ###########################################################

    def importMat(self, passedVals, plotResult = False):
        for i in passedVals:
            self.matDict[i] = dict(permittivityFunction="importMat")
            self.matDict[i]["eta"], self.matDict[i]["eps"], filename = self.import_n_k(i, plotResult)
            self.matDict[i]["parameters"] = [filename]

    # This function imports a .csv file where the first column is wavenumber, v, the second is refractive index, n,
    # and the third (optional) is damping factor, k. The data is imported and then cubic-spline interpolated to
    # give data of the needed resolution. If you set the final parameter to some string, it will plot the original
    # and interpolated data points and title the plot whatever string is passed.
    def import_n_k(self, materialName, plotResult = False):
        lo = self.lo
        hi = self.hi
        incr = self.incr

        vnk, filename = importColumnsCSV(materialName, 'materials_n_k/')
        v = vnk[0]
        n = vnk[1]
        if (len(vnk) > 2):
            k = vnk[2]
        else: # Assume kappa is equal to zero if no data is provided
            k = np.zeros(len(n))

        if self.domain == "aoi":
            desired_n = self.interpolateCubicVal(v, n, self.fixedVal)
            desired_k = self.interpolateCubicVal(v, k, self.fixedVal)
            complex_n = np.complex_(desired_n + 1j*desired_k)
            eta = np.full(self.size, complex_n, dtype = np.complex_)
            eps = np.full(self.size, complex_n**2, dtype = np.complex_)

        elif self.domain == "frequency":
            vInterp, nInterp = self.interpolateCubic(v, n, incr, lo, self.size, plotResult)
            vInterp, kInterp = self.interpolateCubic(v, k, incr, lo, self.size, plotResult)
            eta = np.zeros(self.size, dtype = np.complex_)
            eps = np.zeros(self.size, dtype = np.complex_)
            for i in range(self.size):
                complex_n = np.complex_(nInterp[i] + 1j*kInterp[i])
                eta[i] = complex_n
                eps[i] = complex_n**2

        else:
            error_msg = "self.domain is supposed to be either 'frequency' or 'aoi', but it was set to " + str(self.domain) + "."
            raise ValueError(error_msg)
        return eta, eps, filename

    def import_eps(self, materialName, plotResult = False):
        lo = self.lo
        hi = self.hi
        incr = self.incr

        ve1e2, filename = importColumnsCSV(materialName, 'materials_n_k/')
        v = ve1e2[0]
        e1 = ve1e2[1]
        if (len(ve1e2) > 2):
            e2 = ve1e2[2]
        else: # Assume kappa is equal to zero if no data is provided
            e2 = np.zeros(len(v))

        if self.domain == "aoi":
            desired_e1 = self.interpolateCubicVal(v, e1, self.fixedVal)
            desired_e2 = self.interpolateCubicVal(v, e2, self.fixedVal)
            complex_e = np.complex_(desired_e1 + 1j*desired_e2)
            eps = np.full(self.size, complex_e, dtype = np.complex_)
            eta = self.etaFromEps(eps)

        elif self.domain == "frequency":
            vInterp, e1Interp = self.interpolateCubic(v, e1, incr, lo, self.size, plotResult)
            vInterp, e2Interp = self.interpolateCubic(v, e2, incr, lo, self.size, plotResult)
            eps = np.complex_(e1Interp + 1j*e2Interp)
            eta = self.etaFromEps(eps)
        else:
            error_msg = "self.domain is supposed to be either 'frequency' or 'aoi', but it was set to " + str(self.domain) + "."
            raise ValueError(error_msg)

        self.matDict[materialName] = dict(permittivityFunction="importMat")
        self.matDict[materialName]["eta"] = eta
        self.matDict[materialName]["eps"] = eps
        self.matDict[materialName]["parameters"] = filename
        return eta, eps

    # A function used for converting permittivity to complex refractive index.
    def etaFromEps(self, eps):
#         eps1 = np.real(eps)
#         eps2 = np.imag(eps)
#         kappa = np.sqrt(0.5)*np.sqrt(np.sqrt(eps1**2 + eps2**2)-eps1)
#         n = np.sqrt(0.5)*np.sqrt(np.sqrt(eps1**2 + eps2**2)+eps1)
#         etaEff = np.complex_(n + 1j*kappa)
#         return etaEff
        return np.sqrt(eps)

# The following 5 functions setFixed(), set_n_k(), setLorentz(), setDrude(), and setBruggeman() are 4 functions for modelling complex refractive indices and permittivities.
    def setFixed(self, matName, nx, kx = 0, ny = 0, ky = 0, nz = 0, kz = 0, rotMat = None):
#         self.addedMat[matName] = [independent_n, independent_k] #I dunno what this line was used for...?
        if(ny != 0 or ky != 0 or nz != 0 or kz != 0): # then anisotropic
            eta_fixed = np.array([[nx + 1j*kx, 0, 0], [0, ny + 1j*ky, 0], [0, 0, nz + 1j*kz]])
            eps_fixed = eta_fixed**2
            new_eta = np.full((self.size, 3, 3), eta_fixed)
            new_eps = np.full((self.size, 3, 3), eps_fixed)
        else:
            new_eta = np.full(self.size, nx + 1j*kx)
            new_eps = np.full(self.size, (nx + 1j*kx)**2)
        self.matDict[matName] = dict(permittivityFunction="setFixed")
        self.matDict[matName]["eta"] = new_eta
        self.matDict[matName]["eps"] = new_eps
        self.matDict[matName]["parameters"] = [nx, kx, ny, ky, nz, kz, rotMat]
        return(new_eps)

    def setArray(self, matName, nARR, kARR):
        self.matDict[matName] = dict(permittivityFunction="setArray")
        self.matDict[matName]["eta"] = nARR + 1J*kARR
        self.matDict[matName]["eps"] = (nARR + 1j*kARR)**2
        self.matDict[matName]["parameters"] = [nARR, kARR]
        return(self.matDict[matName]["eps"])

    def setEps(self, matName, ReEps, ImEps):
        new_eps = np.full(self.size, ReEps + 1j*ImEps)
        new_eta = self.etaFromEps(new_eps)
        self.matDict[matName] = dict(permittivityFunction="setEps")
        self.matDict[matName]["eta"] = new_eta
        self.matDict[matName]["eps"] = new_eps
        self.matDict[matName]["parameters"] = [ReEps, ImEps]
        return(new_eps)

    def set_n_k(self, matName, vARR, nARR, kARR = False, plotResult=False):
        eta = np.zeros(self.size, dtype = np.complex_)
        eps = np.zeros(self.size, dtype = np.complex_)
        if(not kARR):
            kARR = np.zeros(len(nARR))
        n_interp = self.interpolateCubic(vARR, nARR, self.incr, self.lo, self.size, plotResult)[1]
        k_interp = self.interpolateCubic(vARR, kARR, self.incr, self.lo, self.size, plotResult)[1]
        eta = np.complex_(n_interp + 1j*k_interp)
        eps = np.complex_(eta**2)
        self.matDict[matName] = dict(permittivityFunction="set_n_k")
        self.matDict[matName]["eta"] = eta
        self.matDict[matName]["eps"] = eps
        self.matDict[matName]["parameters"] = [vARR, nARR, kARR]
        return(eps)

    def set_n_k_3x3(self, matName, permittivityTensor):
        self.matDict[matName] = dict(permittivityFunction="set_n_k_3x3")
        self.matDict[matName]["eta"] = np.full((self.size, 3, 3), permittivityTensor)
        self.matDict[matName]["eps"] = np.full((self.size, 3, 3), permittivityTensor**2)
        self.matDict[matName]["parameters"] = [matName, permittivityTensor]

    # Calculation of complex refractive index of the Lorentz oscillator -- only for use with dielectrics, when the plasma frequency is not provided.
    # Note: plasma frequency no longer appears here, there is only the oscillator strength (update 2022-03-15)
    # As per Osawa et al (Journal of Electron Spectroscopy and Related Phenomena, 64/65 (1993) 371-379)
    # The parameter B relates to the oscillator strength and *MUST HAVE* units of cm^-2.
    def setLorentz(self, matName, B, vD, epsInf, vR):
        epsInf = float(epsInf)
        epsLO = np.full(self.size, epsInf, dtype=np.complex_)
        w = 2 * np.pi * CONST_c * (UNIT_cm/UNIT_m) * self.nu
        B_LO = (2 * np.pi * CONST_c * (UNIT_cm/UNIT_m))**2 * np.array(B)
        wD = 2 * np.pi * CONST_c * (UNIT_cm/UNIT_m) * np.array(vD)
        w_LO = 2 * np.pi * CONST_c * (UNIT_cm/UNIT_m) * np.array(vR)
        for j in range(len(vR)):
            epsLO += (B_LO[j]) / (w_LO[j]**2 - w**2 - 1J*w*wD[j])
        etaLO = self.etaFromEps(epsLO)
        self.matDict[matName] = dict(permittivityFunction="setLorentz")
        self.matDict[matName]["eta"] = etaLO
        self.matDict[matName]["eps"] = epsLO
        self.matDict[matName]["parameters"] = [B, vD, epsInf, vR]
        return(epsLO)

    # A function to calculate the permittivity of simple conductors using the Drude model.
    # Unbound electrons only -- no interband transitions
    def setDrude(self, matName, vP, vD, epsInf, method="Franzen"):
        epsInf = float(epsInf) # make sure it is a float to avoid type errors
        eps1 = np.zeros(self.size)
        eps2 = np.zeros(self.size)
        etaDrude = np.zeros(self.size, dtype = np.complex_)
        epsDrude = np.zeros(self.size, dtype = np.complex_)
        wp = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vP
        Gamma = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vD
        w = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*self.nu
        if method == "Franzen" :
            eps1 = (epsInf) - (wp**2/(w**2 + Gamma**2))
            eps2 = (Gamma*wp**2)/(w*(w**2 + Gamma**2))
        elif method == "LeRu":
            eps1 = epsInf*(1-wp**2/(w**2 + Gamma**2))
            eps2 = (epsInf*Gamma*wp**2)/(w*(w**2 + Gamma**2))
        else:
            error_method = "The final parameter, method, must be either 'Franzen' or 'LeRu', indicating which method to use to calculate Drude"
            raise ValueError(error_method)
        epsDrude = np.complex_(eps1 + 1j*eps2)
        etaDrude = self.etaFromEps(epsDrude)
        self.matDict[matName] = dict(permittivityFunction="setDrude")
        self.matDict[matName]["eta"] = etaDrude
        self.matDict[matName]["eps"] = epsDrude
        self.matDict[matName]["parameters"] = [vP, vD, epsInf, method]
        return(epsDrude)

    # Updated 2022-03-15 to align with conventions in thesis and supporting documentation.
    # This uses the parameterization provided by the equation under the "Multiple Resonanaces" section of my thesis.
    # Same as: https://youtu.be/h4XBfAISAJs?t=389
    def setLorentzDrude(self, matName, vP, f, vD, vj):
        # DRUDE part: Free electron contribution; intraband
        w = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*self.nu
        wp =  2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*np.array(vP)
        f = np.array(f)
        wD = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*np.array(vD)
        wj = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*np.array(vj)
        eps = 1
        # LORENTZ part: Bound electron contribution; interband
        for j in range(0, len(f), 1):
            eps += f[j]*wp**2/((wj[j]**2 - w**2) - 1J*w*wD[j])
        eta = self.etaFromEps(eps)
        self.matDict[matName] = dict(permittivityFunction="setLorentzDrude")
        self.matDict[matName]["eta"] = eta
        self.matDict[matName]["eps"] = eps

    def setAnisotropic(self, matName, eps11, eps22, eps33, mu="not provided; assuming not magnetically dispersive, defaulting to mu_rel = 1"):
        self.matDict[matName] = dict(permittivityFunction="setAnisotropic")
        if (mu == "not provided; assuming not magnetically dispersive, defaulting to mu_rel = 1"):
            self.matDict[matName]["mu"] = np.ones(self.size, dtype = np.complex_)
        else:
            self.matDict[matName]["mu"] = mu
        n = len(eps11)
        out = np.zeros((n,3,3), dtype = np.complex_)
        for i in range(n):
            out[i] = [[eps11[i], 0, 0], [0, eps22[i], 0], [0, 0, eps33[i]]]
        self.matDict[matName]["eps"] = out
        self.matDict[matName]["eta"] = np.sqrt(out)
        self.matDict[matName]["parameters"] = [eps11, eps22, eps33, mu]

    def setPhonon(self, matName, vLO, vTO, gamma, eps_inf): #Berreman-Unterwald-Lowndes  Infrared dielectric functions, phonon modes, and free-charge carrier properties of high-Al-content AlxGa1−xN alloys determined by mid infrared spectroscopic ellipsometry and optical Hall effect # https://aip.scitation.org/doi/10.1063/1.4983765
        # All parameters are arrays except eps_inf
        w = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*self.nu
        epsPhonon = np.ones(self.size, dtype = np.complex_)
        for i in range(len(vLO)):
            wLO = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vLO[i]
            wTO = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vTO[i]
            y = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*gamma[i]
            epsPhonon *= (w**2 - wLO**2 + 1J*y*w)/(w**2 - wTO**2 + 1J*y*w)
        epsPhonon = eps_inf*(1 + epsPhonon) # Dependinig on how eps_inf is defined, sometimes the "1 + " shoud be omitted.
        etaPhonon = self.etaFromEps(epsPhonon)
        self.matDict[matName] = dict(permittivityFunction="setPhonon")
        self.matDict[matName]["eta"] = etaPhonon
        self.matDict[matName]["eps"] = epsPhonon
        self.matDict[matName]["parameters"] = [vLO, vTO, gamma, eps_inf]
        return epsPhonon

    def setPhononPassler(self, matName, vLO, vTO, gamma, eps_inf): #
        # All parameters are arrays except eps_inf
        w = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*self.nu
        epsPhonon = np.zeros(self.size, dtype = np.complex_)
        wL = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vLO[0]
        wT = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vTO[0]
        y = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*gamma[0]
        epsPhonon = (w**2 - wL**2)/(w**2 - wT**2 + 1J*y*w)
        for i in range(1, len(vLO)):
            wLO = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vLO[i]
            wTO = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vTO[i]
            y = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*gamma[i]
            epsPhonon += (wTO**2 - wLO**2)/(w**2 - wTO**2 + 1J*y*w)
        epsPhonon = eps_inf*(1 + epsPhonon) # Dependinig on how eps_inf is defined, sometimes the "1 + " shoud be omitted.
        etaPhonon = self.etaFromEps(epsPhonon)
        self.matDict[matName] = dict(permittivityFunction="setPhonon")
        self.matDict[matName]["eta"] = etaPhonon
        self.matDict[matName]["eps"] = epsPhonon
        self.matDict[matName]["parameters"] = [vLO, vTO, gamma, eps_inf]
        return epsPhonon

    def setPhononPasslerStupid(self, matName, vLO, vTO, gamma, eps_inf): #
        # All parameters are arrays except eps_inf
        w = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*self.nu
        epsPhonon = np.zeros(self.size, dtype = np.complex_)
        wL = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vLO[0]
        wT = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vTO[0]
        y = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*gamma[0]
        epsPhonon = (w**2 - wL**2+ 1J*y*w)/(w**2 - wT**2 + 1J*y*w)
        for i in range(1, len(vLO)):
            wLO = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vLO[i]
            wTO = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vTO[i]
            y = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*gamma[i]
            epsPhonon += (wTO**2 - wLO**2)/(w**2 - wTO**2 + 1J*y*w)
        epsPhonon = eps_inf*(epsPhonon) # Dependinig on how eps_inf is defined, sometimes the "1 + " shoud be omitted.
        etaPhonon = self.etaFromEps(epsPhonon)
        self.matDict[matName] = dict(permittivityFunction="setPhonon")
        self.matDict[matName]["eta"] = etaPhonon
        self.matDict[matName]["eps"] = epsPhonon
        self.matDict[matName]["parameters"] = [vLO, vTO, gamma, eps_inf]
        return epsPhonon
    ##############################
    # A function to fudge the value of kappa, and hence the imaginary part of epsilon
    # This function calculates the permittivity, epsilon, based on the parameters, converts to
    # complex refractive index, eta, (alternatively, n and kappa), multiplies the value of kappa by the
    # fudge factor input, and then re-converts to complex refractive index, kappa.
    def setDrudeKappa(self, matName, vP, vD, epsInf, fudgeKappa, method="Franzen"):
        eps1 = np.zeros(self.size)
        eps2 = np.zeros(self.size)
        etaDrude = np.zeros(self.size, dtype = np.complex_)
        epsDrude = np.zeros(self.size, dtype = np.complex_)
        wp = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vP
        Gamma = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*vD
        w = 2*np.pi*CONST_c*(UNIT_cm/UNIT_m)*self.nu
        if method == "Franzen" :
            eps1 = (epsInf) - (wp**2/(w**2 + Gamma**2))
            eps2 = (Gamma*wp**2)/(w*(w**2 + Gamma**2))
        elif method == "LeRu":
            eps1 = epsInf*(1-wp**2/(w**2 + Gamma**2))
            eps2 = (epsInf*Gamma*wp**2)/(w*(w**2 + Gamma**2))
        else:
            error_method = "The final parameter, method, must be either 'Franzen' or 'LeRu', indicating which method to use to calculate Drude"
            raise ValueError(error_method)
        kappa = fudgeKappa*np.sqrt((np.sqrt(eps1**2 + eps2**2)-eps1)/2)
        n = eps2/(2*kappa)
        etaDrude = np.complex_(n + 1j*kappa)
        eps1 = n**2 + kappa**2
        eps2 = 2*n*kappa
        epsDrude = np.complex_(eps1 + 1j*eps2)
        self.matDict[matName] = dict(permittivityFunction="setDrudeKappa")
        self.matDict[matName]["eta"] = etaDrude
        self.matDict[matName]["eps"] = epsDrude
        self.matDict[matName]["parameters"] = [vP, vD, epsInf, fudgeKappa, method]

    def setSymbolic(self, matName): # The material name is the symbolic string
        epsSym = []
        etaSym = [np.zeros(self.size)]
        syms = matName + " eta_" + matName
        print(syms)
        epsSymbol, etaSymbol = sp.symbols(syms)
        for i in range(0, self.size, 1):
            epsSym.append(epsSymbol)
            etaSym.append(etaSymbol)
        self.matDict[matName] = dict(permittivityFunction="setSymbolic")
        self.matDict[matName]["eta"] = np.array(etaSym)
        self.matDict[matName]["eps"] = np.array(epsSym)
        self.matDict[matName]["parameters"] = []

    def setAbsLorentzian(self, matName, peakPosition, peakHeight, FWHM, pathLength_um, n_inf):
        if (len(peakPosition) != len(peakHeight) or len(peakHeight) != len(FWHM)):
            raise ValueError("Error: setAbsLorentzian() takes three parameters which must be of the same length.")
        Abs = 0
        for i in range(len(peakPosition)):
            Abs += peakHeight[i] / (1 + ( (peakPosition[i] - self.nu ) / (FWHM[i]/2) )**2)
        kappa = Abs*np.log10(np.exp(1)) / (4*np.pi*self.nu*(10**-4)*pathLength_um)
        n = n_inf + ft.hilbert(kappa)
        eps1 = n**2 - kappa**2
        eps2 = 2*n*kappa
        self.matDict[matName] = dict(permittivityFunction="setDrudeKappa")
        self.matDict[matName]["eta"] = np.complex_(n + 1j*kappa)
        self.matDict[matName]["eps"] = np.complex_(eps1 + 1j*eps2)
        self.matDict[matName]["parameters"] = [peakPosition, peakHeight, FWHM, pathLength_um, n_inf]

#     def (gamma, v0, n_inf):

#     def getAbsCoef(molarAbsorptivity, molarConcentration):
#         return np.log10(np.eps(1))*molarAbsorptivity*molarConcentration


    ##############################

    # This function calculates the plasma frequency and relaxation rate (damping factor) as a function of frequency.
    # This allows for the analysis shown in:
    # Fahsold, G., Bartel, A., Krauth, O., Magg, N., Pucci, A.  Phys. Rev. B. vol 61 (21). 2000. pg. 14108 - 14113.
    def solveDrude(self, matName):
        eps1Arr = np.real(self.matDict[matName]["eps"])
        eps2Arr = np.imag(self.matDict[matName]["eps"])
        vp = np.zeros(self.size, dtype=np.complex_)
        vt = np.zeros(self.size, dtype=np.complex_)
        for i in range(self.size):
            w = 2*np.pi*3*10**10*self.nu[i]
            wt = eps2Arr[i]*w/(1-eps1Arr[i])
            wp = np.sqrt(complex((1-eps1Arr[i])*(w**2 + wt**2)))
            vt[i] = wt / (2*np.pi*CONST_c*(UNIT_cm/UNIT_m))
            vp[i] = wp / (2*np.pi*CONST_c*(UNIT_cm/UNIT_m))
        return vp, vt

    # Although they are written with omegas, all the values in equation (5) in Fahsold (Ref 23 in Priebe) are in fact nus (in wavenumbers)
    def fahsoldModel(self, wt0, gamma):
        vt = np.zeros(self.size, dtype=np.complex_)
        for i in range(self.size):
            vt[i] = wt0 + gamma*self.nu[i]**2
        return vt

    # vF is the velocity at the Fermi level in nm/s; d is also in nm.
    def fahsoldTF(self, vp, vt, vF, d):
        vp_TF = np.zeros(self.size, dtype=np.complex_)
        vt_TF = np.zeros(self.size, dtype=np.complex_)

        # Calculate vp for thin film (eqn. 6)
        cols, filename = importColumnsCSV("beta", 'empirical_data/')
        db, beta = cols
        b = self.interpolateCubicVal(db, beta, d)
        for i in range(self.size):
            vp_TF[i] = b * vp[i]

        # Calculate vt for thin film (eqns. 3 & 4)
        cols, filename = importColumnsCSV("alpha", 'empirical_data/')
        da, alpha = cols
        a = self.interpolateCubicVal(da, alpha, d)
        ws = a * vF / (2*d) # presumably in radians per second
        vs = ws / (2*np.pi*CONST_c*(UNIT_cm/UNIT_m))
        for i in range(self.size):
            vt_TF[i] = vt[i] + vs

        return vp_TF, vt_TF

    ########################################################
    #####~~Effective media approximation models below~~#####
    ########################################################

    # This function uses the Wiener bounds to select the physically correct solution of the n-phase Bruggeman effective medium approximation.
    # See Jansson & Arwin 1994. Optics communication 106. p 133-138.

    # wienerZ_ARR is an array of the complex permittivities of the components
    # roots_ARR is an array of the the possible roots of which you want to determine the correct physically meaningful root
    # The length of wienerZ_ARR and roots_ARR should be the same.
    def getPhysicalRoot(self, wienerZ_ARR, roots_ARR, rejectGain = True):
        def wienerZeta(z, z1, z2): # This function assumes that the point 0 + 0*1J maps below the line joining xi_z1 and xi_z2 when s = 1. Need to check that this is valid when calling the function.
            z0 = z1*z2*(np.conj(z1) - np.conj(z2)) / (np.conj(z1)*z2 - z1*np.conj(z2))
            return ((z - z0)*np.conj(z2 - z1)) / (np.absolute(z0) * np.absolute(z2 - z1))

        def wienerLims(z, z1, z2): # Returns the distance of the point to the circle segment.
            # THESE NEVER OCCUR, SINCE THEY ARE HANDLED ONE LEVEL UP.
#             if(z1 == z2 and np.abs(z - z1) <= delta):
# #                 print("case1")
#                 return True
#             elif (z1 == z2 and np.abs(z - z1) > delta):
# #                 print("case2")
#                 return False
#             print("case3")

            def dist_to_nearest_chord_endpt(chord_endpt1, chord_endpt2, test_pt):
                dist_to_endpt1 = np.absolute(test_pt - chord_endpt1)
                dist_to_endpt2 = np.absolute(test_pt - chord_endpt2)
                nearest_dist = np.amin([dist_to_endpt1, dist_to_endpt2])
                return nearest_dist

            zeta_z = wienerZeta(z, z1, z2)
            zeta_z1 = wienerZeta(z1, z1, z2)
            zeta_z2 = wienerZeta(z2, z1, z2)
            zeta_0 = wienerZeta(0+0*1J, z1, z2)
#             print("Zeta_0: ", zeta_0)
            if (np.imag(zeta_0) <= np.imag((zeta_z1 + zeta_z2)/2)): # CHanged from: if (np.imag(zeta_0) <= np.imag(zeta_z1)):
                s = 1
            else:
                s = -1
#             print("Im Z(z0):", np.imag(zeta_0), "Im Z(z):", np.imag(zeta_z), "Im Z(z1):", np.imag(zeta_z1), "Im Z(z2)", np.imag(zeta_z2))
#             print(s)
            zeta_z *= s
            zeta_z1 *= s
            zeta_z2 *= s
            zeta_0 *= s

            arg_zeta_z = np.angle(zeta_z)
            arg_zeta_z1 = np.angle(zeta_z1)
            arg_zeta_z2 = np.angle(zeta_z2)
            dist_to_circle = np.absolute(zeta_z) - 1
            dist_to_chord = np.imag((zeta_z1 + zeta_z2)/2) - np.imag(zeta_z)
            chord_is_in_upper_semicircle = np.imag((zeta_z1 + zeta_z2)/2) > 0
            if(not chord_is_in_upper_semicircle):
                dist_to_chord *= -1

            # PLOT JUNK1
#             fig, ax = plt.subplots()
#             fig.set_size_inches(4,4)
#             ax.axis([-1.5, 1.5, -1.5, 1.5])
#             cir = Arc((0,0), width=2, height=2)
#             ax.add_patch(cir)
#             ax.scatter(np.real(zeta_z), np.imag(zeta_z), c="r")
#             # plot region lines
#             ax.plot([0, np.real(zeta_z1)], [0, np.imag(zeta_z1)], "b--")
#             ax.plot([0, np.real(zeta_z2)], [0, np.imag(zeta_z2)], "b--")
#         #     ax.set_aspect('equal')
#             plt.show()

            if (dist_to_circle <= 0):
                dist_to_circle = 0
            if (dist_to_chord <= 0):
                dist_to_chord = 0

            pt_is_within_circle = dist_to_circle == 0 # bool
            pt_is_beyond_line = dist_to_chord == 0 # bool
            pt_is_within_x_projection_of_chord = (np.real(zeta_z) > np.real(zeta_z1)) and (np.real(zeta_z) < np.real(zeta_z2))
            ang_is_between_upper_ang12_limits = (arg_zeta_z2 < arg_zeta_z) and (arg_zeta_z < arg_zeta_z1)# bool
            ang_is_between_lower_ang12_limits = (arg_zeta_z1 < arg_zeta_z) and (arg_zeta_z < arg_zeta_z2)# bool
            pt_is_in_upper_wedge = chord_is_in_upper_semicircle and ang_is_between_upper_ang12_limits
            pt_is_in_lower_wedge = (not chord_is_in_upper_semicircle) and ang_is_between_lower_ang12_limits

            pt_is_in_region1 = not pt_is_beyond_line and pt_is_within_x_projection_of_chord
            pt_is_in_region2 = (not pt_is_within_circle) and (pt_is_in_upper_wedge or pt_is_in_lower_wedge)

            if (pt_is_within_circle and pt_is_beyond_line): # within the segment
                final_dist =  0
            elif (pt_is_in_region1):
                final_dist = dist_to_chord  # In region #1, the point is closest to the chord.
            elif (pt_is_in_region2):
                final_dist = dist_to_circle # In region #2, the point is closest to the arc.
            else: # within region #3
                dist_to_endpt1 = np.absolute(zeta_z - zeta_z1)
                dist_to_endpt2 = np.absolute(zeta_z - zeta_z2)
                nearest_dist = np.amin([dist_to_endpt1, dist_to_endpt2])
                final_dist = nearest_dist   # In region #3, the point is closest to the endpoint (intersection of chord and arc).

#             # UNCOMMENT for error diagnostic information.
#             print("Testing root: ", z)
# #             print("np.absolute(zeta_z):", np.absolute(zeta_z))
#             print("dist_to_chord: ", dist_to_chord)
#             print("dist_to_circle: ", dist_to_circle)
#             print("final_dist: ", final_dist)
#             fig, ax = plt.subplots()
#             ax.scatter(np.real(zeta_z1), np.imag(zeta_z1), c="b")
#             ax.scatter(np.real(zeta_z2), np.imag(zeta_z2), c="b")
#             ax.scatter(np.real(zeta_0), np.imag(zeta_0), c="g")
#             ax.scatter(np.real(zeta_z), np.imag(zeta_z), c="r")
#             cir = plt.Circle((0,0), 1, color='b', fill=False)
#             ax.add_artist(cir)
#             ax.set_ylim(np.imag(.98*zeta_z), np.imag(1.02*zeta_z))
#             ax.set_xlim(np.real(.5*zeta_z), np.real(1.2*zeta_z))
#             plt.plot([np.real(zeta_z1), np.real(zeta_z2)], [np.imag(zeta_z1), np.imag(zeta_z2)])
#             plt.show()
#             plt.close()
#             print("\n\n\n")
            return final_dist

            # AS FAR AS I CAN TELL : JUNK!
#             delta1 = 0 # A threshold value to prevent float errors from rejecting valid solutions
#             delta2 = 0 # A threshold value to prevent float errors from rejecting valid solutions
#             cond1 = testVal1 <= 1 + delta1
#             cond2 = testVal2 + delta2 >= 0

        def wienerPlot(z, z1, z2, title="title"):
            fig, ax = plt.subplots()
            ax.scatter(np.real(z1), np.imag(z1), c="g")
            ax.scatter(np.real(z2), np.imag(z2), c="g")
            ax.scatter(0, 0, c="g")
            ax.scatter(np.real(z), np.imag(z), c="r")
            z0 = z1*z2*(np.conj(z1) - np.conj(z2)) / (z2*np.conj(z1) - z1*np.conj(z2))
            cir = plt.Circle((np.real(z0), np.imag(z0)), np.abs(z0), color='b', fill=False)
            ax.add_artist(cir)
            plt.plot([np.real(z1), np.real(z2)], [np.imag(z1), np.imag(z2)])
            plt.title(title)
            plt.show()
            plt.close()
            print("z", z, "\nz1", z1, "\nz2", z2)

        def wienerW(z, z1, z2):
            return(z*(np.conj(z2-z1))/(np.abs(z2-z1)))

        def distToCircularSeg(wienerZToTest, rootsToTest):
            threshold =  0 # Not fine-tuned
            validRoot = False # return value, if not updated by the code below, the function will return false.

            zPairs = np.array(list(itertools.combinations(wienerZToTest, 2)))
            zPairs = np.unique(zPairs, axis=0)
#             print(zPairs)

            validRoot = None
            roots_dist = np.full(len(rootsToTest), np.Inf, dtype=np.float_)
            for r in range(len(rootsToTest)):
#                 print("working on root:", rootsToTest[r])
                for p in range(len(zPairs)):
                    wienerZ1, wienerZ2 = zPairs[p]
                    # Special case #1: the two components are the same, so both roots give the same answer. Arbitrarily select root1. The root is physically valid if it is the same as well.
                    if (wienerZ1 == wienerZ2):
                        if(np.abs(rootsToTest[r] - wienerZ1) <= threshold): #i.e. if z = z1 (= z2)
                            validRoot = rootsToTest[r]
                            roots_dist[r] = 0
                    # Special case #2: When the permittivities of the two components and the origin (0 + 0i) lie on a line, need an alternate test.
                    elif (np.real(wienerZ1)*np.imag(wienerZ2) - np.real(wienerZ2)*np.imag(wienerZ1) == 0):# On 2021-05-06, changed from:(np.abs(np.real(wienerZ1)*(np.imag(wienerZ2)) + np.real(wienerZ2)*(-np.imag(wienerZ1))) <= threshold): # z1, z2 and origin are all on a line (aka the area enclosed by the 3 points is zero)
                        w_z1 = wienerW(wienerZ1, wienerZ1, wienerZ2)
                        w_z2 = wienerW(wienerZ2, wienerZ1, wienerZ2)
                        if (np.real(w_z2) >= np.real(w_z1)):
                            s = 1
                        else:
                            s = -1
                        w_testRoot = s*wienerW(rootsToTest[r], wienerZ1, wienerZ2)
                        if (np.imag(w_testRoot) <= threshold and np.real(w_z1) <= np.real(w_testRoot) and np.real(w_testRoot) <= np.real(w_z2)):
                            validRoot = rootsToTest[r]
                            roots_dist[r] = 0

                    # Normal Wiener limits, no special case.
                    else:
#                         print("Checking: ", rootsToTest[r])
                        roots_dist[r] = np.amin([wienerLims(rootsToTest[r], wienerZ1, wienerZ2), roots_dist[r]])
            return (roots_dist)

        def triPlot(tupPts, roots):
            pts = np.array([[np.real(tupPts[0]), np.imag(tupPts[0])] , [np.real(tupPts[1]), np.imag(tupPts[1])], [np.real(tupPts[2]), np.imag(tupPts[2])]])
            for r in roots:
                plt.scatter(pts[:,0], pts[:,1], color=(0,0,1))
                triangle = plt.Polygon(pts, color=(0,0,1,0.1))
                plt.gca().add_patch(triangle)
                plt.scatter(np.real(r), np.imag(r), color="red")
                plt.xlim(np.real(r)-5, np.real(r)+5)
                plt.ylim(np.imag(r)-5, np.imag(r)+5)
                plt.show()

        # Returns an array of length rootsToTest containing the shortest distance of that root to the nearest triangle
        def distToTriangles(wienerZToTest, rootsToTest):
            def distToSeg(a,b,t):
                AB = b - a
                AT = t - a
                dot = np.real(np.vdot(AB, AT))
                square_len = np.real(AB*np.conj(AB))
                if square_len != 0:
                    param = dot / square_len
                else:
                    param = -1
                if param < 0: # point A is closest
                    closest = a
                elif param > 1: # point B is closest
                    closest = b
                else: # some other point on the line segment is closest
                    closest = a + param*AB
                dist = np.abs(t - closest)
                return (dist)
            roots_dist = np.full(len(rootsToTest), np.Inf, dtype=np.float_)
            zTriangles = list(itertools.combinations(wienerZToTest, 3))
            for tri_verts in range(len(zTriangles)):
                (a, b, c) = zTriangles[tri_verts]
                crossComplex = lambda z1, z2, z3: (np.real(z2) - np.real(z1))*(np.imag(z3) - np.imag(z1)) - (np.imag(z2) - np.imag(z1))*(np.real(z3) - np.real(z1))  # See: https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
                AB_cross_AC = crossComplex(a, b, c)
                BC_cross_BA = crossComplex(b, c, a)
                CA_cross_CB = crossComplex(c, a, b)
                for ri, t in enumerate(rootsToTest): # loop over all roots, ri is the root index
                    AB_cross_AT = crossComplex(a, b, t)
                    BC_cross_BT = crossComplex(b, c, t)
                    CA_cross_CT = crossComplex(c, a, t)

                    is_outside_AB = AB_cross_AC*AB_cross_AT < 0
                    is_outside_BC = BC_cross_BA*BC_cross_BT < 0
                    is_outside_CA = CA_cross_CB*CA_cross_CT < 0

        #             # UNCOMMENT TO PLOT
        #             fig = plt.figure()
        #             ax = fig.add_subplot(111)
        #             ax.scatter([x1],[y1], c='r')
        #             ax.scatter([x2],[y2], c='g')
        #             ax.scatter([x3],[y3], c='b')
        #             triPlot([a,b,c], [t])

                    if(is_outside_AB): # point is nearest to segment AB
                        roots_dist[ri] =  np.amin([distToSeg(a,b,t), roots_dist[ri]])
                    elif(is_outside_BC): # point is nearest to segment BC
                        roots_dist[ri] =  np.amin([distToSeg(b,c,t), roots_dist[ri]])
                    elif(is_outside_CA): # point is nearest to segment CA
                        roots_dist[ri] =  np.amin([distToSeg(c,a,t), roots_dist[ri]])
                    else: # is firmly inside the triangle
                        roots_dist[ri] = 0
            return(roots_dist)

        # Actual block to run
        wienerZ_ARR = np.array(wienerZ_ARR) # The slicing used requires a numpy array, although I expect most users will simply use a python list in that call, so here I cast the list to a numpy array.
        epsEff = np.linspace(-42 + 0*1J, -42 + 0*1J, self.size, dtype=np.complex_)
        assigned = np.zeros(self.size, dtype=np.int_)
        for i in range(self.size): # Loop over each wavenumber
            curRootsToTest = np.array(roots_ARR)[:, i]
            if (rejectGain):
#                 print("before culling:", curRootsToTest)
                curRootsToTest = curRootsToTest[np.imag(curRootsToTest) > -1E-15] # Reject all roots with negative imaginary permittivities (i.e. materials exhibiting gain)
#                 print("after culling:", curRootsToTest)

            # Check all semicircles
            dist_to_semi_circles = distToCircularSeg(wienerZ_ARR[:, i], curRootsToTest)
            # Check all triangles
            dist_to_triangles = distToTriangles(wienerZ_ARR[:, i], curRootsToTest)
            dist_to_region = np.amin(np.vstack((dist_to_semi_circles, dist_to_triangles)), axis = 0) # The minimum distance of each root to the entire region of validity (the union of the semicircles and the triangle)
            epsEff[i] = curRootsToTest[np.argmin(dist_to_region)] # The "correct" root is selected as the root which is closest to the region of validity.
        return epsEff


    def setBasicMaxwellGarnett(self, matName, inclusionMat, matrixMat, f):
        ep = self.matDict[inclusionMat]["eps"]
        em = self.matDict[matrixMat]["eps"]
#         epsMG = em*(-2*em*f + 2*em + 2*ep*f + ep)/(em*f + 2*em - ep*f + ep)
        gam = (ep - em)/(ep + 2*em)
        epsMG = em*(1 + 2*f*gam)/(1 - f*gam)
        self.matDict[matName] = dict(permittivityFunction="setBasicMaxwellGarnett")
        self.matDict[matName]["eps"] = epsMG
        self.matDict[matName]["eta"] = self.etaFromEps(epsMG)
        self.matDict[matName]["parameters"] = [inclusionMat, matrixMat, f]

# Unused; solving MG and Bruggeman symbolically
# eMG, eBR, ep, em, f = sympy.symbols('eMG eBR ep em f')
# BRterm1 = (f*(ep-eBR))/(ep+2*eBR)
# BRterm2 = ((1-f)*(em-eBR))/(em+2*eBR)
# BReqn = BRterm1 + BRterm2
# MGterm1 = (eMG - em)/(eMG + 2*em)
# MGterm2 = -f*(ep-em)/(ep+2*em)
# MGeqn = MGterm1 + MGterm2

# solved = sympy.solve((BReqn), eBR)
# print(solved)

# Basic 2-phase symmetric bruggeman model using Arwin's method to determine the physical root (getPhysicalRoot)
# Although symmetric, the parameters are named "inclusionMat" and "matrixMat" to indicate that the volume fill
# fraction, f, refers to the "inclusionMat" and the "matrixMat" occupies the balance of the volume (1-f)
    def setBasicBruggeman(self, matName, inclusionMat, matrixMat, f, d = 3):
        ep = self.matDict[inclusionMat]["eps"]
        em = self.matDict[matrixMat]["eps"]
        coeffC = ep*em
        coeffB = em*(d - 1 - f*d) + ep*(f*d - 1)
        coeffA = np.full(self.size, 1-d)
        roots1 = np.zeros(self.size, dtype = np.complex_)
        roots2 = np.zeros(self.size, dtype = np.complex_)
        for i in range(self.size):
            roots = np.roots([coeffA[i], coeffB[i], coeffC[i]])
            roots1[i] = roots[0]
            roots2[i] = roots[1]
#         # Derived symbolically from solving the roots with sympy; not currently used
#         sqroot = np.sqrt(9*em**2*f**2 - 12*em**2*f + 4*em**2 - 18*em*ep*f**2 + 18*em*ep*f + 4*em*ep + 9*ep**2*f**2 - 6*ep**2*f + ep**2)
#         root1 = -3*em*f/4 + em/2 + 3*ep*f/4 - ep/4 + sqroot/4
#         root2 = -3*em*f/4 + em/2 + 3*ep*f/4 - ep/4 - sqroot/4

#         # Plots comparing symbolic and direct numerical roots
#         plt.plot(self.nu, np.real(root1) - np.real(roots1), 'r', self.nu, np.imag(root1) - np.imag(roots1), 'r--')
#         plt.legend(["Real DELTA root1", "Imag DELTA root1"])
#         plt.show()
#         plt.plot(self.nu, np.real(root2) - np.real(roots2), 'b', self.nu, np.imag(root2) - np.imag(roots2), 'b--')
#         plt.legend(["Real DELTA root2", "Imag DELTA root2"])
#         plt.show()

        # See Jansson & Arwin 1994. Optics communication 106. p 133-138.
#         wienerZ1 = f*ep + (1-f)*em
#         wienerZ2 = 1 / ( (f/ep) + ((1-f) / em) )
        wienerZ1 = em
        wienerZ2 = ep

        epsEff = self.getPhysicalRoot([wienerZ1, wienerZ2], [roots1, roots2])
        self.matDict[matName] = dict(permittivityFunction="setBasicBruggeman")
        self.matDict[matName]["eps"] = epsEff
        self.matDict[matName]["eta"] = self.etaFromEps(epsEff)
        self.matDict[matName]["parameters"] = [inclusionMat, matrixMat, f, d]


#PLAN for setEMA1(): first implement this as a two phase model for isotropic materials,
# then generalize to the n-phase case with anisotropic permittivity tensors and specify an axis along which to calculate

# Follows the syntactical form as equation 15 (referred to as EMA1) in: Noh, T.; Song, P.; Sievers, A. Self-Consistency Conditions for the Effective-Medium Approximation in Composite Materials. Phys. Rev. B Condens. Matter 1991, 44, 5459–5464. https://doi.org/10.1103/PhysRevB.44.5459.
# Uses the Jansson and Arwin method getPhysicalRoots
#     def setSimpleEMA1(self, matName, constituentNameARR, fARR, LiARR):
        # TO DO: Make helper function for materials that have isotropic permittivites
#         for i in range(len(materialNameARR)):


    def setMultiBruggeman(self, matName, constituentMat_ARR, fill_ARR):
        # Calls binomialProdCoefs() to determine the roots of an n-phase system
        def multiPhaseRoots(permittivities, fillFactors, symbolicBool):
            roots = []
            for i in range(self.size):
                coef2 = np.full(len(fillFactors), 2)
                coef2[0] = -1
                coef1 = np.array(permittivities).copy()
                coef1 = coef1[:, i]
                finalCoefs = np.zeros(len(fillFactors)+1)
                for m in range(len(fillFactors)):
                    coef1 = np.delete(coef1, [m])
                    coef1 = np.insert(coef1, [0], permittivities[m][i])
                    polyCoefs = self.binomialProdCoefs(coef1, coef2)
                    polyCoefs = np.multiply(polyCoefs, fillFactors[m])
                    finalCoefs = np.add(finalCoefs, polyCoefs)
                finalCoefs = np.flip(finalCoefs, 0)
                if(symbolicBool):
                    epsilon = sp.Symbol("epsilon")
                    expr = 0
                    for c in range(len(finalCoefs)):
                        power = len(finalCoefs) - 1 - c # The 0th term is the highest power; the final term is the 0th power
                        expr += finalCoefs[c]*epsilon**(power)
                    print("The symbolic expression being solved is:\n", expr)
                    roots.append(sp.solve(expr, epsilon))
                else:
                    roots.append(np.roots(finalCoefs))
            return np.array(roots).transpose()


        permittivities = []
        roots_ARR = []
        for m in range(len(constituentMat_ARR)):
            permittivities.append(self.matDict[constituentMat_ARR[m]]["eps"])
            roots_ARR.append([])
        if (isinstance(np.any(permittivities), sp.Symbol)): # Changed from sp.symbol.Symbol to sp.Symbol
            symbolicBool = True
        else:
            symbolicBool = False
        roots_ARR = multiPhaseRoots(permittivities, fill_ARR, symbolicBool)
        if(not symbolicBool):
            epsEff = self.getPhysicalRoot(permittivities, roots_ARR)
            self.matDict[matName] = dict(permittivityFunction="setMultiBruggeman")
            self.matDict[matName]["eps"] = epsEff
            self.matDict[matName]["eta"] = self.etaFromEps(epsEff)
            self.matDict[matName]["parameters"] = [matName, constituentMat_ARR, fill_ARR]

    def setCoatedInMatrix(self, matName, constituentMat_ARR, L_ARR, q, f):
        # c=core ; s=shell; m=medium
        ec = self.matDict[constituentMat_ARR[0]]["eps"]
        es = self.matDict[constituentMat_ARR[1]]["eps"]
        em = self.matDict[constituentMat_ARR[2]]["eps"]
        Lc, Ls, Lm = L_ARR
        # f is the fraction of coated particles
        # q is the ratio of core to (core + coated)
        X = es +(ec - es)*(Lc - q*Ls)
        Y = q*es*(ec - es)
        coef0 = Lm*X*em*es*f + Lm*Y*em*f - Ls*X*em*es*f + Ls*X*em*es - Ls*Y*em*f + Ls*Y*em - X*em**2*f + X*em**2
        coef1 = -Lm*X*em*f - Lm*X*es*f - Lm*Y*f + Ls*X*em*f - Ls*X*em + Ls*X*es*f - Ls*X*es + Ls*Y*f - Ls*Y + X*em*f - X*em + X*es*f + Y*f
        coef2 = Lm*X*f - Ls*X*f + Ls*X - X*f

#         coef0 = (1-f)*(xi*L_sh*e_med*e_sh + L_sh*e_p*B) + f*(L_med*xi*e_sh*e_med + L_med*B*e_med)
#         coef1 = (1-f)*(xi*e_med*(1-L_sh) - xi*L_sh*e_sh + L_sh*B) + f*(xi*e_sh - L_med*xi*(e_med - e_sh) + B*(1-L_med))
#         coef2 = xi*(1-L_sh + f*L_sh - f*L_med)

        roots_ARR = np.zeros([2, self.size], dtype=np.complex_)
        for i in range(self.size):
            coefs = [coef0[i], coef1[i], coef2[i]]
            roots_ARR[:,i] = np.roots(coefs)
        permittivities_ARR = [ec, es, em]
        epsEff = self.getPhysicalRoot(permittivities_ARR, roots_ARR, rejectGain=False)

#         epsEff = np.zeros(self.size, dtype=np.complex_)
#         for i in range(self.size):
#             coefs = [coef0[i], coef1[i], coef2[i]]
#             epsEff[i] = np.roots(coefs)[0]

        self.matDict[matName] = dict(permittivityFunction="setCoatedInMatrix")
        self.matDict[matName]["eps"] = epsEff
        self.matDict[matName]["eta"] = self.etaFromEps(epsEff)
        self.matDict[matName]["parameters"] = [matName, constituentMat_ARR, L_ARR, q, f]

    # deprecated; use getGeometric() instead
    def getYama_f(self, aspect):
        return (aspect**2/(2*np.sqrt((aspect**2-1)**3)))*((np.pi/2) - (np.sqrt(aspect**2 - 1)/aspect**2) - np.arctan(1/(np.sqrt(aspect**2-1))))


    def setYamaguchi(self, matName, substrateMat, ellipsoidMat, hostMat, thickness, aspect, fillFraction, modified = "no"):
        epsSubstrate = self.matDict[substrateMat]["eps"]
        epsEllipsoid = self.matDict[ellipsoidMat]["eps"]
        epsHost = self.matDict[hostMat]["eps"]
        epsEff = np.zeros(self.size, dtype=np.complex_)
        etaEff = np.zeros(self.size, dtype=np.complex_)

        f = self.getYama_f(aspect)
        F = np.zeros(self.size, dtype=np.complex_)
        ## For classic Yamaguchi
        if (modified == "unmodified" or modified == "classic" or modified == "no"):
            for i in range(self.size):
                term2F = (aspect**2)*(epsSubstrate[i] - epsHost[i])/(24*(epsSubstrate[i] + epsHost[i]))
                term3F = 0.719*np.sqrt(6/np.pi)*(epsHost[i]*np.sqrt(fillFraction**3)/(aspect*(epsSubstrate[i] + epsHost[i])))
                F[i] = f - term2F - term3F

        ## For modified Yamaguchi (conducive to more significant deviations from spheres)
        ## Note: xi, x, y, z below correspond to xi', x', y', z' in Fedotov paper. The prime symbol is omitted below for clarity
        elif (modified == "modified" or modified == "improved" or modified == "yes"):
            def getA(x, y, z):
                xi = 1/2*(np.sqrt((x**2 + y**2 + z**2 - aspect**-2 - 1)**2 - 4*(aspect**-2 * (1 - x**2 - y**2) - z**2)) + x**2 + y**2 + z**2 - aspect**-2 - 1)
                deriv_xi = x*(1+((x**2 + y**2 + z**2 + aspect**-2 - 1)*np.sqrt((x**2 + y**2 + z**2 - aspect**2 - 1)**2 - 4*(aspect**-2*(1 - x**2 - y**2) - z**2))))
                print(xi)
                Afactor1 = -1/(2*(aspect**2 - 1))
                Aterm1 = np.pi*aspect**2/(2*np.sqrt(aspect**2 - 1))
                Aterm2 = -np.sqrt(aspect**2*xi + 1) / (xi + 1)
                Aterm3 = -aspect**2/np.sqrt(aspect**2 - 1)*np.arctan(np.sqrt((aspect**2*xi + 1)/(aspect**2 - 1)))
                Aterm4 = x*deriv_xi*(np.sqrt(aspect**2 + 1)/(xi**2 + 1)**2 - aspect**2/(np.sqrt(aspect**2*xi + 1)*(xi + 1)))
                A = Afactor1*(Aterm1 + Aterm2 + Aterm3 + Aterm4)
                return A
            x = np.sqrt(2*np.pi/(3*fillFraction))
            y = np.sqrt(2*np.pi/(3*fillFraction))

            Fterm1 = f
            Fterm2 = 0
            Fterm3 = 0
            ## Set the values of A
            for i in range(-2, 3):
                for j in range (-2, 3):
                    ## A[i][j]  = getA(i*x, j*y, 0)
                    ## Add A[i][j] to the sum
                    Fterm2 += getA(i*x, j*y, 0)
                    Fterm3 += getA(i*x, j*y, 2/aspect)
            ## The above part was generally true (frequency independent), but the expressions below are a function of wavenumber.
            for i in range(self.size):
                Fterm4 = -2*epsHost[i] * 0.177 * np.sqrt(3*fillFraction**3/(2*np.pi)) / ((epsHost[i] + epsSubstrate[i]) * aspect)
                F[i] = Fterm1 - Fterm2 - ((epsSubstrate[i] - epsHost[i])/(epsSubstrate[i] + epsHost[i]))*Fterm3 + Fterm4

        else:
            raise ValueError("The final parameter, <modified>, passed to setYamaguchi() must be a string indicating whether to use the modified version described by Fedotov et al.")

        epsEff = epsHost*(fillFraction*(epsEllipsoid - epsHost)/(epsHost + F*(epsEllipsoid - epsHost)) + 1)
        etaEff = self.etaFromEps(epsEff)

        self.matDict[matName] = dict(permittivityFunction="setYamaguchi")
        self.matDict[matName]["eps"] = epsEff
        self.matDict[matName]["eta"] = etaEff
        self.matDict[matName]["parameters"] = [substrateMat, ellipsoidMat, hostMat, thickness, aspect, fillFraction, modified]

        return

    def setCoreShellYamaguchi(self, matName, substrateMat, ellipsoidMat, adsorbateMat, hostMat, thickness, molec, aspect, fillFraction, modified = "no"):
        # Note on terminology: the suffix '1' indicates the core ellipsoidal particle and the suffix '2' indicates the coating on the particle
        eps1 = self.matDict[ellipsoidMat]["eps"]
        eps2 = self.matDict[adsorbateMat]["eps"]
        epsExt = self.matDict[hostMat]["eps"]

        f1 = self.getYama_f(aspect)
        # 'molec' refers to a uniform molecular shell that completly coats the particle; i.e. molec is a constant
        # amount added to the RADIUS of each semi-axis. Since this implementation of yamaguchi uses oblate spheroids
        # laying on their flat side (i.e. the short dimension pointing up from the surface), the minor semi-axis
        # of the core particle is given by:
        c1 = thickness
        # And the major semi-axis of the core particle is given by:
        a1 = thickness*aspect

        # Since we are calculating aspect ratio in terms of diameters, we must add two times the thickness of 'molec'.
        # Thus, the minor and major semi-axes of the COATED particle are given by:
        c2 = c1 + 2*molec
        a2 = a1 + 2*molec

        # Finally, we can calculate the aspect ratio of the coated particle:
        aspect2 = a2 / c2

        f2 = self.getYama_f(aspect2)
        print(f1, f2)
        coatedEllipsoidMat = matName.join("coatedEllipsoid")

        self.matDict[coatedEllipsoidMat] = dict(permittivityFunction="setCoreShellYamaguchi")
        epsInt = np.zeros(self.size, dtype=np.complex_)

        # Fraction of volume occupied by the core
        coreFrac = (a1**2)*c1 / ((a2**2)*c2)

        for i in range(self.size):
            alphaPrimeNumerator = (eps2[i] - epsExt[i])*(eps2[i] + (eps1[i] - eps2[i])*(f1 - coreFrac*f2)) + coreFrac*eps2[i]*(eps1[i] - eps2[i])
            alphaPrimeDenominator = (eps2[i] + (eps1[i] - eps2[i])*(f1 - coreFrac*f2))*(epsExt[i] + (eps2[i] - epsExt[i])*f2) + coreFrac*f2*eps2[i]*(eps1[i] - eps2[i])
            alphaPrime = alphaPrimeNumerator / alphaPrimeDenominator
            epsInt[i] = epsExt[i]*(alphaPrime*(f2 - 1) - 1) / (alphaPrime*f2 - 1)

        self.matDict[coatedEllipsoidMat]["eps"] = epsInt
        self.setYamaguchi(matName, substrateMat, coatedEllipsoidMat, hostMat, thickness + 2*molec, aspect2, fillFraction, modified)
        return

    #################################
    # Bruggeman formula for 3 phase, coated core-shell ellipsoids, NO self consistency condition
    def setBruggeman(self, matName, coreMat, shellMat, matrixMat, thickness, molec, ratio1, F, plotResult=False, iterations=1):
        # print("matName", matName, "\ncoreMat", coreMat, "\nshellMat", shellMat, "\nmatrixMat", matrixMat, "\nthickness", thickness, "\nmolec", molec, "\nratio1", ratio1,"\nF", F)
        epsCore = self.matDict[coreMat]["eps"]
        epsShell = self.matDict[shellMat]["eps"]
        epsMatrix = self.matDict[matrixMat]["eps"]
        self.BReps = np.array([coreMat, shellMat, matrixMat])
        self.BRthick = thickness
        self.BRmolec = molec
        self.BRratio1 = ratio1
        self.BRF = F
        etaBR_ARR = np.zeros(len(epsCore), dtype=np.complex_)
        epsBR_ARR = np.zeros(len(epsCore), dtype=np.complex_)
        a1 = float(ratio1*thickness/2)
        c1 = float(thickness/2)
        # NOTE: This is not valid!! Cannot simply add a uniformly thick molecular shell since the equations only work
        # for confocal ellipsoids, that is: a1^2 - a2^2 = c1^2 - c2^2 = xi1. When xi = 0, this describes the surface
        # of the inner ellipsoid, and when xi = xi1, this describes the surface of the outer ellipsoid. a1, c1 refer
        # to the inner ellipsoid, and a2, c2 refer to the outer ellipsoid.
        # Below I used a1-a2 = c1-c2 which is NOT true for confocal ellipsoids. But it might be a reasonable
        # approximation for a thin enough shell.
        a2 = float(molec + a1)
        c2 = float(molec + c1)
        Q = (a1*c1**2)/(a2*c2**2)

#         print("a1", a1, "\nc1", c1, "\na2", a2, "\nc2", c2)
        Lcore = self.getGeometricFactor(a1, c1, c1, True)
        Lshell = self.getGeometricFactor(a2, c2, c2, True)
#         print("Q", Q, "\nLcore", Lcore, "\nLshell", Lshell)
        # I don't think there is rationale for this, but I want to try it to replicate my previous result.
        LcoreEff = (Lcore[0] + Lcore[2])/2
        LshellEff = (Lshell[0] + Lshell[2])/2

        # Expression for alpha as it appears in Bohren and Huffman, page 149
        def calcAlpha(ec, es, em, Q, Lc, Ls=0):
            num = (es - em)*(es + (ec - es)*(Lc - Q*Ls)) + Q*es*(ec - es)
            denom = (es + (ec - es)*(Lc - Q*Ls))*(em + Ls*(es - em)) + Q*Ls*es*(ec - es)
            if (denom.any() == 0):
                raise ValueError("The denominator of <alpha> worked out to be zero. Please verify that your inputs are correct.")
                return
#             alpha = num / denom
#             plt.plot(self.nu, np.real(alpha))
#             plt.legend(["Lc = " + str(round(Lc, 2)) + " Ls = " + str(round(Ls, 2))])
#             plt.show()
            return num / denom

        def calcRandomAlpha(ec, es, em, Q, LcoreARR, LshellARR=[0,0,0]):
            alphas = calcMultipleAlpha(ec, es, em, Q, LcoreARR, LshellARR)
            averageAlpha = np.mean(alphas, axis = 0)
            return averageAlpha

        def calcMultipleAlpha(ec, es, em, Q, LcoreARR, LshellARR=[0,0,0]):
            if len(LcoreARR) > 3:
                raise ValueError("The third argument, LcARR passed to calcRandomAlpha should have at most 3 parameters corresponding to the 3 semiaxes of an ellipsoid.")
            alphas = []
            for semiax in range(3):
                alphas.append(calcAlpha(ec, es, em, Q, LcoreARR[semiax], LshellARR[semiax]))
            return alphas

        for j in range(iterations):
            if j == 0:
                selfConsistent = epsMatrix
            alpha = calcRandomAlpha(epsCore, epsShell, selfConsistent, Q, Lcore, Lshell)
            # Expression from Granqvist and Hunderi 1978 equation 6, page 2898
            epsBRnum = epsMatrix*(3 - 3*F + F*alpha)
            epsBRdenom = (3 - 3*F - 2*F*alpha)
            if (epsBRdenom.any() == 0):
                raise ValueError("The denominator of <epsBR> worked out to be zero. Please verify that your inputs are correct.")
            epsBR_ARR = epsBRnum / epsBRdenom
            etaBR_ARR = self.etaFromEps(epsBR_ARR)
            selfConsistent = epsBR_ARR
            self.matDict[matName] = dict(permittivityFunction="setBruggeman")
            self.matDict[matName]["eta"] = etaBR_ARR
            self.matDict[matName]["eps"] = epsBR_ARR
            self.matDict[matName]["parameters"] = [coreMat, shellMat, matrixMat, thickness, molec, ratio1, F, plotResult, iterations]
            # Uncomment to get a record of all the iterations
#             matName = "BR" + str(j)
#             self.matDict[matName] = dict(permittivityFunction="setBruggeman")
#             self.matDict[matName]["eta"] = etaBR_ARR
#             self.matDict[matName]["eps"] = epsBR_ARR
        if(plotResult):
            if (self.domain == "aoi"):
                x = self.aoi
            elif (self.domain == "frequency"):
                x = self.nu
            plt.plot(x, np.real(epsBR_ARR), x, np.imag(epsBR_ARR))
            plt.legend(["n", "k"])
            plt.show()

#############################

    def setOsawaBruggeman(self, matName, ellipsoidMat, adsorbateMat, hostMat, thickness, molec, ratio1, F):
        epsM = self.matDict[ellipsoidMat]["eps"]
        epsLO = self.matDict[adsorbateMat]["eps"]
        epsH = self.matDict[hostMat]["eps"]
        self.BReps = np.array([ellipsoidMat, adsorbateMat, hostMat])
        self.BRthick = thickness
        self.BRmolec = molec
        self.BRratio1 = ratio1
        self.BRF = F
        etaBR_ARR = np.zeros(len(epsLO), dtype=np.complex_)
        epsBR_ARR = np.zeros(len(epsLO), dtype=np.complex_)
        a1 = float(ratio1*thickness/2)
        c1 = float(thickness/2)
        a2 = float(molec + a1)
        c2 = float(molec + c1)
        # Use the following line instead to create a confocal ellipsoid coating
        # c2 = float(np.sqrt(c1**2 + a2**2 - a1**2))
        Q = (a1*c1**2)/(a2*c2**2)
        Da1, Db1, Dc1 = self.getGeometricFactor(a1, c1, c1)
        Da2, Db2, Dc2 = self.getGeometricFactor(a2, c2, c2)


        L1 = (Da1 + Dc1)/2
        L2 = (Da2 + Dc2)/2

#         ratio2 = float(a2/c2)
#         Da1 = (1/(float(ratio1)**2 - 1))*((float(ratio1)/np.sqrt(float(ratio1)**2-1))*np.log(float(ratio1) + np.sqrt(float(ratio1)**2 - 1))-1)
#         Da2 = (1/(ratio2**2 - 1))*((ratio2/np.sqrt(ratio2**2-1))*np.log(ratio2 + np.sqrt(ratio2**2 - 1))-1)
#         Dc1 = (1/(2*(float(ratio1)**2 - 1)))*(float(ratio1)**2 - (float(ratio1)/np.sqrt(float(ratio1)**2-1))*np.log(float(ratio1) + np.sqrt(float(ratio1)**2-1)))
#         Dc2 = (1/(2*(ratio2**2 - 1)))*(ratio2**2 - (ratio2/np.sqrt(ratio2**2-1))*np.log(ratio2 + np.sqrt(ratio2**2-1)))
#         L1 = (Da1 + Dc1)/2
#         L2 = (Da2 + Dc2)/2

    #     An alternate function to define obtain the roots, which demonstrates much better where the below expression
    #     comes from. You need to import sympy to be able to use this function.
    #
    #     def roots(ed, eh, em, L1, L2, Q, F):
    #         eBR = Symbol('eBR')
    #         LHS = (3*(eh - F*eh + F*eBR - eBR)/(-F*(2*eBR + eh)))
    #         RHSnum = (ed - eBR)*(em*L1 + ed*(1 - L1)) + Q*(em - ed)*(ed*(1 - L2) + eBR*L2)
    #         RHSdenom = (ed*L2 + eBR*(1 - L2))*(em*L1 + ed*(1 - L1)) + Q*(em - ed)*(ed - eBR)*L2*(1 - L2)
    #         eqn = -1*LHS + RHSnum/RHSdenom
    #         solved = np.array(solve((eqn), eBR), dtype=np.complex_)
    #         return solved

    #  A long, inelegant expression, but it's much faster than solving it symbolically every time!
        def roots(ed, eh, em, L1, L2, Q, F):
            return [(3*F*L1*L2*ed**2 + 3*F*L1*L2*ed*eh - 3*F*L1*L2*ed*em - 3*F*L1*L2*eh*em + 2*F*L1*ed**2 - 4*F*L1*ed*eh - 2*F*L1*ed*em + 4*F*L1*eh*em - 3*F*L2**2*Q*ed**2 - 3*F*L2**2*Q*ed*eh + 3*F*L2**2*Q*ed*em + 3*F*L2**2*Q*eh*em + F*L2*Q*ed**2 + 4*F*L2*Q*ed*eh - F*L2*Q*ed*em - 4*F*L2*Q*eh*em - 3*F*L2*ed**2 - 3*F*L2*ed*eh + 2*F*Q*ed**2 - 2*F*Q*ed*em - 2*F*ed**2 + 4*F*ed*eh - 3*L1*L2*ed**2 - 3*L1*L2*ed*eh + 3*L1*L2*ed*em + 3*L1*L2*eh*em + 3*L1*ed*eh - 3*L1*eh*em + 3*L2**2*Q*ed**2 + 3*L2**2*Q*ed*eh - 3*L2**2*Q*ed*em - 3*L2**2*Q*eh*em - 3*L2*Q*ed**2 - 3*L2*Q*ed*eh + 3*L2*Q*ed*em + 3*L2*Q*eh*em + 3*L2*ed**2 + 3*L2*ed*eh - 3*ed*eh - np.sqrt(9*F**2*L1**2*L2**2*ed**4 - 18*F**2*L1**2*L2**2*ed**3*eh - 18*F**2*L1**2*L2**2*ed**3*em + 9*F**2*L1**2*L2**2*ed**2*eh**2 + 36*F**2*L1**2*L2**2*ed**2*eh*em + 9*F**2*L1**2*L2**2*ed**2*em**2 - 18*F**2*L1**2*L2**2*ed*eh**2*em - 18*F**2*L1**2*L2**2*ed*eh*em**2 + 9*F**2*L1**2*L2**2*eh**2*em**2 + 12*F**2*L1**2*L2*ed**4 + 12*F**2*L1**2*L2*ed**3*eh - 24*F**2*L1**2*L2*ed**3*em - 24*F**2*L1**2*L2*ed**2*eh**2 - 24*F**2*L1**2*L2*ed**2*eh*em + 12*F**2*L1**2*L2*ed**2*em**2 + 48*F**2*L1**2*L2*ed*eh**2*em + 12*F**2*L1**2*L2*ed*eh*em**2 - 24*F**2*L1**2*L2*eh**2*em**2 + 4*F**2*L1**2*ed**4 - 20*F**2*L1**2*ed**3*eh - 8*F**2*L1**2*ed**3*em + 16*F**2*L1**2*ed**2*eh**2 + 40*F**2*L1**2*ed**2*eh*em + 4*F**2*L1**2*ed**2*em**2 - 32*F**2*L1**2*ed*eh**2*em - 20*F**2*L1**2*ed*eh*em**2 + 16*F**2*L1**2*eh**2*em**2 - 18*F**2*L1*L2**3*Q*ed**4 + 36*F**2*L1*L2**3*Q*ed**3*eh + 36*F**2*L1*L2**3*Q*ed**3*em - 18*F**2*L1*L2**3*Q*ed**2*eh**2 - 72*F**2*L1*L2**3*Q*ed**2*eh*em - 18*F**2*L1*L2**3*Q*ed**2*em**2 + 36*F**2*L1*L2**3*Q*ed*eh**2*em + 36*F**2*L1*L2**3*Q*ed*eh*em**2 - 18*F**2*L1*L2**3*Q*eh**2*em**2 - 6*F**2*L1*L2**2*Q*ed**4 - 42*F**2*L1*L2**2*Q*ed**3*eh + 12*F**2*L1*L2**2*Q*ed**3*em + 48*F**2*L1*L2**2*Q*ed**2*eh**2 + 84*F**2*L1*L2**2*Q*ed**2*eh*em - 6*F**2*L1*L2**2*Q*ed**2*em**2 - 96*F**2*L1*L2**2*Q*ed*eh**2*em - 42*F**2*L1*L2**2*Q*ed*eh*em**2 + 48*F**2*L1*L2**2*Q*eh**2*em**2 - 18*F**2*L1*L2**2*ed**4 + 36*F**2*L1*L2**2*ed**3*eh + 18*F**2*L1*L2**2*ed**3*em - 18*F**2*L1*L2**2*ed**2*eh**2 - 36*F**2*L1*L2**2*ed**2*eh*em + 18*F**2*L1*L2**2*ed*eh**2*em + 16*F**2*L1*L2*Q*ed**4 + 52*F**2*L1*L2*Q*ed**3*eh - 32*F**2*L1*L2*Q*ed**3*em - 32*F**2*L1*L2*Q*ed**2*eh**2 - 104*F**2*L1*L2*Q*ed**2*eh*em + 16*F**2*L1*L2*Q*ed**2*em**2 + 64*F**2*L1*L2*Q*ed*eh**2*em + 52*F**2*L1*L2*Q*ed*eh*em**2 - 32*F**2*L1*L2*Q*eh**2*em**2 - 24*F**2*L1*L2*ed**4 - 24*F**2*L1*L2*ed**3*eh + 24*F**2*L1*L2*ed**3*em + 48*F**2*L1*L2*ed**2*eh**2 + 24*F**2*L1*L2*ed**2*eh*em - 48*F**2*L1*L2*ed*eh**2*em + 8*F**2*L1*Q*ed**4 - 20*F**2*L1*Q*ed**3*eh - 16*F**2*L1*Q*ed**3*em + 40*F**2*L1*Q*ed**2*eh*em + 8*F**2*L1*Q*ed**2*em**2 - 20*F**2*L1*Q*ed*eh*em**2 - 8*F**2*L1*ed**4 + 40*F**2*L1*ed**3*eh + 8*F**2*L1*ed**3*em - 32*F**2*L1*ed**2*eh**2 - 40*F**2*L1*ed**2*eh*em + 32*F**2*L1*ed*eh**2*em + 9*F**2*L2**4*Q**2*ed**4 - 18*F**2*L2**4*Q**2*ed**3*eh - 18*F**2*L2**4*Q**2*ed**3*em + 9*F**2*L2**4*Q**2*ed**2*eh**2 + 36*F**2*L2**4*Q**2*ed**2*eh*em + 9*F**2*L2**4*Q**2*ed**2*em**2 - 18*F**2*L2**4*Q**2*ed*eh**2*em - 18*F**2*L2**4*Q**2*ed*eh*em**2 + 9*F**2*L2**4*Q**2*eh**2*em**2 - 6*F**2*L2**3*Q**2*ed**4 + 30*F**2*L2**3*Q**2*ed**3*eh + 12*F**2*L2**3*Q**2*ed**3*em - 24*F**2*L2**3*Q**2*ed**2*eh**2 - 60*F**2*L2**3*Q**2*ed**2*eh*em - 6*F**2*L2**3*Q**2*ed**2*em**2 + 48*F**2*L2**3*Q**2*ed*eh**2*em + 30*F**2*L2**3*Q**2*ed*eh*em**2 - 24*F**2*L2**3*Q**2*eh**2*em**2 + 18*F**2*L2**3*Q*ed**4 - 36*F**2*L2**3*Q*ed**3*eh - 18*F**2*L2**3*Q*ed**3*em + 18*F**2*L2**3*Q*ed**2*eh**2 + 36*F**2*L2**3*Q*ed**2*eh*em - 18*F**2*L2**3*Q*ed*eh**2*em - 11*F**2*L2**2*Q**2*ed**4 - 32*F**2*L2**2*Q**2*ed**3*eh + 22*F**2*L2**2*Q**2*ed**3*em + 16*F**2*L2**2*Q**2*ed**2*eh**2 + 64*F**2*L2**2*Q**2*ed**2*eh*em - 11*F**2*L2**2*Q**2*ed**2*em**2 - 32*F**2*L2**2*Q**2*ed*eh**2*em - 32*F**2*L2**2*Q**2*ed*eh*em**2 + 16*F**2*L2**2*Q**2*eh**2*em**2 + 6*F**2*L2**2*Q*ed**4 + 42*F**2*L2**2*Q*ed**3*eh - 6*F**2*L2**2*Q*ed**3*em - 48*F**2*L2**2*Q*ed**2*eh**2 - 42*F**2*L2**2*Q*ed**2*eh*em + 48*F**2*L2**2*Q*ed*eh**2*em + 9*F**2*L2**2*ed**4 - 18*F**2*L2**2*ed**3*eh + 9*F**2*L2**2*ed**2*eh**2 + 4*F**2*L2*Q**2*ed**4 + 20*F**2*L2*Q**2*ed**3*eh - 8*F**2*L2*Q**2*ed**3*em - 40*F**2*L2*Q**2*ed**2*eh*em + 4*F**2*L2*Q**2*ed**2*em**2 + 20*F**2*L2*Q**2*ed*eh*em**2 - 16*F**2*L2*Q*ed**4 - 52*F**2*L2*Q*ed**3*eh + 16*F**2*L2*Q*ed**3*em + 32*F**2*L2*Q*ed**2*eh**2 + 52*F**2*L2*Q*ed**2*eh*em - 32*F**2*L2*Q*ed*eh**2*em + 12*F**2*L2*ed**4 + 12*F**2*L2*ed**3*eh - 24*F**2*L2*ed**2*eh**2 + 4*F**2*Q**2*ed**4 - 8*F**2*Q**2*ed**3*em + 4*F**2*Q**2*ed**2*em**2 - 8*F**2*Q*ed**4 + 20*F**2*Q*ed**3*eh + 8*F**2*Q*ed**3*em - 20*F**2*Q*ed**2*eh*em + 4*F**2*ed**4 - 20*F**2*ed**3*eh + 16*F**2*ed**2*eh**2 - 18*F*L1**2*L2**2*ed**4 + 36*F*L1**2*L2**2*ed**3*eh + 36*F*L1**2*L2**2*ed**3*em - 18*F*L1**2*L2**2*ed**2*eh**2 - 72*F*L1**2*L2**2*ed**2*eh*em - 18*F*L1**2*L2**2*ed**2*em**2 + 36*F*L1**2*L2**2*ed*eh**2*em + 36*F*L1**2*L2**2*ed*eh*em**2 - 18*F*L1**2*L2**2*eh**2*em**2 - 12*F*L1**2*L2*ed**4 - 30*F*L1**2*L2*ed**3*eh + 24*F*L1**2*L2*ed**3*em + 42*F*L1**2*L2*ed**2*eh**2 + 60*F*L1**2*L2*ed**2*eh*em - 12*F*L1**2*L2*ed**2*em**2 - 84*F*L1**2*L2*ed*eh**2*em - 30*F*L1**2*L2*ed*eh*em**2 + 42*F*L1**2*L2*eh**2*em**2 + 24*F*L1**2*ed**3*eh - 24*F*L1**2*ed**2*eh**2 - 48*F*L1**2*ed**2*eh*em + 48*F*L1**2*ed*eh**2*em + 24*F*L1**2*ed*eh*em**2 - 24*F*L1**2*eh**2*em**2 + 36*F*L1*L2**3*Q*ed**4 - 72*F*L1*L2**3*Q*ed**3*eh - 72*F*L1*L2**3*Q*ed**3*em + 36*F*L1*L2**3*Q*ed**2*eh**2 + 144*F*L1*L2**3*Q*ed**2*eh*em + 36*F*L1*L2**3*Q*ed**2*em**2 - 72*F*L1*L2**3*Q*ed*eh**2*em - 72*F*L1*L2**3*Q*ed*eh*em**2 + 36*F*L1*L2**3*Q*eh**2*em**2 - 12*F*L1*L2**2*Q*ed**4 + 96*F*L1*L2**2*Q*ed**3*eh + 24*F*L1*L2**2*Q*ed**3*em - 84*F*L1*L2**2*Q*ed**2*eh**2 - 192*F*L1*L2**2*Q*ed**2*eh*em - 12*F*L1*L2**2*Q*ed**2*em**2 + 168*F*L1*L2**2*Q*ed*eh**2*em + 96*F*L1*L2**2*Q*ed*eh*em**2 - 84*F*L1*L2**2*Q*eh**2*em**2 + 36*F*L1*L2**2*ed**4 - 72*F*L1*L2**2*ed**3*eh - 36*F*L1*L2**2*ed**3*em + 36*F*L1*L2**2*ed**2*eh**2 + 72*F*L1*L2**2*ed**2*eh*em - 36*F*L1*L2**2*ed*eh**2*em - 24*F*L1*L2*Q*ed**4 - 78*F*L1*L2*Q*ed**3*eh + 48*F*L1*L2*Q*ed**3*em + 48*F*L1*L2*Q*ed**2*eh**2 + 156*F*L1*L2*Q*ed**2*eh*em - 24*F*L1*L2*Q*ed**2*em**2 - 96*F*L1*L2*Q*ed*eh**2*em - 78*F*L1*L2*Q*ed*eh*em**2 + 48*F*L1*L2*Q*eh**2*em**2 + 24*F*L1*L2*ed**4 + 60*F*L1*L2*ed**3*eh - 24*F*L1*L2*ed**3*em - 84*F*L1*L2*ed**2*eh**2 - 60*F*L1*L2*ed**2*eh*em + 84*F*L1*L2*ed*eh**2*em + 24*F*L1*Q*ed**3*eh - 48*F*L1*Q*ed**2*eh*em + 24*F*L1*Q*ed*eh*em**2 - 48*F*L1*ed**3*eh + 48*F*L1*ed**2*eh**2 + 48*F*L1*ed**2*eh*em - 48*F*L1*ed*eh**2*em - 18*F*L2**4*Q**2*ed**4 + 36*F*L2**4*Q**2*ed**3*eh + 36*F*L2**4*Q**2*ed**3*em - 18*F*L2**4*Q**2*ed**2*eh**2 - 72*F*L2**4*Q**2*ed**2*eh*em - 18*F*L2**4*Q**2*ed**2*em**2 + 36*F*L2**4*Q**2*ed*eh**2*em + 36*F*L2**4*Q**2*ed*eh*em**2 - 18*F*L2**4*Q**2*eh**2*em**2 + 24*F*L2**3*Q**2*ed**4 - 66*F*L2**3*Q**2*ed**3*eh - 48*F*L2**3*Q**2*ed**3*em + 42*F*L2**3*Q**2*ed**2*eh**2 + 132*F*L2**3*Q**2*ed**2*eh*em + 24*F*L2**3*Q**2*ed**2*em**2 - 84*F*L2**3*Q**2*ed*eh**2*em - 66*F*L2**3*Q**2*ed*eh*em**2 + 42*F*L2**3*Q**2*eh**2*em**2 - 36*F*L2**3*Q*ed**4 + 72*F*L2**3*Q*ed**3*eh + 36*F*L2**3*Q*ed**3*em - 36*F*L2**3*Q*ed**2*eh**2 - 72*F*L2**3*Q*ed**2*eh*em + 36*F*L2**3*Q*ed*eh**2*em + 6*F*L2**2*Q**2*ed**4 + 54*F*L2**2*Q**2*ed**3*eh - 12*F*L2**2*Q**2*ed**3*em - 24*F*L2**2*Q**2*ed**2*eh**2 - 108*F*L2**2*Q**2*ed**2*eh*em + 6*F*L2**2*Q**2*ed**2*em**2 + 48*F*L2**2*Q**2*ed*eh**2*em + 54*F*L2**2*Q**2*ed*eh*em**2 - 24*F*L2**2*Q**2*eh**2*em**2 + 12*F*L2**2*Q*ed**4 - 96*F*L2**2*Q*ed**3*eh - 12*F*L2**2*Q*ed**3*em + 84*F*L2**2*Q*ed**2*eh**2 + 96*F*L2**2*Q*ed**2*eh*em - 84*F*L2**2*Q*ed*eh**2*em - 18*F*L2**2*ed**4 + 36*F*L2**2*ed**3*eh - 18*F*L2**2*ed**2*eh**2 - 12*F*L2*Q**2*ed**4 - 24*F*L2*Q**2*ed**3*eh + 24*F*L2*Q**2*ed**3*em + 48*F*L2*Q**2*ed**2*eh*em - 12*F*L2*Q**2*ed**2*em**2 - 24*F*L2*Q**2*ed*eh*em**2 + 24*F*L2*Q*ed**4 + 78*F*L2*Q*ed**3*eh - 24*F*L2*Q*ed**3*em - 48*F*L2*Q*ed**2*eh**2 - 78*F*L2*Q*ed**2*eh*em + 48*F*L2*Q*ed*eh**2*em - 12*F*L2*ed**4 - 30*F*L2*ed**3*eh + 42*F*L2*ed**2*eh**2 - 24*F*Q*ed**3*eh + 24*F*Q*ed**2*eh*em + 24*F*ed**3*eh - 24*F*ed**2*eh**2 + 9*L1**2*L2**2*ed**4 - 18*L1**2*L2**2*ed**3*eh - 18*L1**2*L2**2*ed**3*em + 9*L1**2*L2**2*ed**2*eh**2 + 36*L1**2*L2**2*ed**2*eh*em + 9*L1**2*L2**2*ed**2*em**2 - 18*L1**2*L2**2*ed*eh**2*em - 18*L1**2*L2**2*ed*eh*em**2 + 9*L1**2*L2**2*eh**2*em**2 + 18*L1**2*L2*ed**3*eh - 18*L1**2*L2*ed**2*eh**2 - 36*L1**2*L2*ed**2*eh*em + 36*L1**2*L2*ed*eh**2*em + 18*L1**2*L2*ed*eh*em**2 - 18*L1**2*L2*eh**2*em**2 + 9*L1**2*ed**2*eh**2 - 18*L1**2*ed*eh**2*em + 9*L1**2*eh**2*em**2 - 18*L1*L2**3*Q*ed**4 + 36*L1*L2**3*Q*ed**3*eh + 36*L1*L2**3*Q*ed**3*em - 18*L1*L2**3*Q*ed**2*eh**2 - 72*L1*L2**3*Q*ed**2*eh*em - 18*L1*L2**3*Q*ed**2*em**2 + 36*L1*L2**3*Q*ed*eh**2*em + 36*L1*L2**3*Q*ed*eh*em**2 - 18*L1*L2**3*Q*eh**2*em**2 + 18*L1*L2**2*Q*ed**4 - 54*L1*L2**2*Q*ed**3*eh - 36*L1*L2**2*Q*ed**3*em + 36*L1*L2**2*Q*ed**2*eh**2 + 108*L1*L2**2*Q*ed**2*eh*em + 18*L1*L2**2*Q*ed**2*em**2 - 72*L1*L2**2*Q*ed*eh**2*em - 54*L1*L2**2*Q*ed*eh*em**2 + 36*L1*L2**2*Q*eh**2*em**2 - 18*L1*L2**2*ed**4 + 36*L1*L2**2*ed**3*eh + 18*L1*L2**2*ed**3*em - 18*L1*L2**2*ed**2*eh**2 - 36*L1*L2**2*ed**2*eh*em + 18*L1*L2**2*ed*eh**2*em + 18*L1*L2*Q*ed**3*eh - 18*L1*L2*Q*ed**2*eh**2 - 36*L1*L2*Q*ed**2*eh*em + 36*L1*L2*Q*ed*eh**2*em + 18*L1*L2*Q*ed*eh*em**2 - 18*L1*L2*Q*eh**2*em**2 - 36*L1*L2*ed**3*eh + 36*L1*L2*ed**2*eh**2 + 36*L1*L2*ed**2*eh*em - 36*L1*L2*ed*eh**2*em - 18*L1*ed**2*eh**2 + 18*L1*ed*eh**2*em + 9*L2**4*Q**2*ed**4 - 18*L2**4*Q**2*ed**3*eh - 18*L2**4*Q**2*ed**3*em + 9*L2**4*Q**2*ed**2*eh**2 + 36*L2**4*Q**2*ed**2*eh*em + 9*L2**4*Q**2*ed**2*em**2 - 18*L2**4*Q**2*ed*eh**2*em - 18*L2**4*Q**2*ed*eh*em**2 + 9*L2**4*Q**2*eh**2*em**2 - 18*L2**3*Q**2*ed**4 + 36*L2**3*Q**2*ed**3*eh + 36*L2**3*Q**2*ed**3*em - 18*L2**3*Q**2*ed**2*eh**2 - 72*L2**3*Q**2*ed**2*eh*em - 18*L2**3*Q**2*ed**2*em**2 + 36*L2**3*Q**2*ed*eh**2*em + 36*L2**3*Q**2*ed*eh*em**2 - 18*L2**3*Q**2*eh**2*em**2 + 18*L2**3*Q*ed**4 - 36*L2**3*Q*ed**3*eh - 18*L2**3*Q*ed**3*em + 18*L2**3*Q*ed**2*eh**2 + 36*L2**3*Q*ed**2*eh*em - 18*L2**3*Q*ed*eh**2*em + 9*L2**2*Q**2*ed**4 - 18*L2**2*Q**2*ed**3*eh - 18*L2**2*Q**2*ed**3*em + 9*L2**2*Q**2*ed**2*eh**2 + 36*L2**2*Q**2*ed**2*eh*em + 9*L2**2*Q**2*ed**2*em**2 - 18*L2**2*Q**2*ed*eh**2*em - 18*L2**2*Q**2*ed*eh*em**2 + 9*L2**2*Q**2*eh**2*em**2 - 18*L2**2*Q*ed**4 + 54*L2**2*Q*ed**3*eh + 18*L2**2*Q*ed**3*em - 36*L2**2*Q*ed**2*eh**2 - 54*L2**2*Q*ed**2*eh*em + 36*L2**2*Q*ed*eh**2*em + 9*L2**2*ed**4 - 18*L2**2*ed**3*eh + 9*L2**2*ed**2*eh**2 - 18*L2*Q*ed**3*eh + 18*L2*Q*ed**2*eh**2 + 18*L2*Q*ed**2*eh*em - 18*L2*Q*ed*eh**2*em + 18*L2*ed**3*eh - 18*L2*ed**2*eh**2 + 9*ed**2*eh**2))/(2*(3*F*L1*L2*ed - 3*F*L1*L2*em - F*L1*ed + F*L1*em - 3*F*L2**2*Q*ed + 3*F*L2**2*Q*em + F*L2*Q*ed - F*L2*Q*em - 3*F*L2*ed + F*ed - 3*L1*L2*ed + 3*L1*L2*em + 3*L1*ed - 3*L1*em + 3*L2**2*Q*ed - 3*L2**2*Q*em - 3*L2*Q*ed + 3*L2*Q*em + 3*L2*ed - 3*ed)), (3*F*L1*L2*ed**2 + 3*F*L1*L2*ed*eh - 3*F*L1*L2*ed*em - 3*F*L1*L2*eh*em + 2*F*L1*ed**2 - 4*F*L1*ed*eh - 2*F*L1*ed*em + 4*F*L1*eh*em - 3*F*L2**2*Q*ed**2 - 3*F*L2**2*Q*ed*eh + 3*F*L2**2*Q*ed*em + 3*F*L2**2*Q*eh*em + F*L2*Q*ed**2 + 4*F*L2*Q*ed*eh - F*L2*Q*ed*em - 4*F*L2*Q*eh*em - 3*F*L2*ed**2 - 3*F*L2*ed*eh + 2*F*Q*ed**2 - 2*F*Q*ed*em - 2*F*ed**2 + 4*F*ed*eh - 3*L1*L2*ed**2 - 3*L1*L2*ed*eh + 3*L1*L2*ed*em + 3*L1*L2*eh*em + 3*L1*ed*eh - 3*L1*eh*em + 3*L2**2*Q*ed**2 + 3*L2**2*Q*ed*eh - 3*L2**2*Q*ed*em - 3*L2**2*Q*eh*em - 3*L2*Q*ed**2 - 3*L2*Q*ed*eh + 3*L2*Q*ed*em + 3*L2*Q*eh*em + 3*L2*ed**2 + 3*L2*ed*eh - 3*ed*eh + np.sqrt(9*F**2*L1**2*L2**2*ed**4 - 18*F**2*L1**2*L2**2*ed**3*eh - 18*F**2*L1**2*L2**2*ed**3*em + 9*F**2*L1**2*L2**2*ed**2*eh**2 + 36*F**2*L1**2*L2**2*ed**2*eh*em + 9*F**2*L1**2*L2**2*ed**2*em**2 - 18*F**2*L1**2*L2**2*ed*eh**2*em - 18*F**2*L1**2*L2**2*ed*eh*em**2 + 9*F**2*L1**2*L2**2*eh**2*em**2 + 12*F**2*L1**2*L2*ed**4 + 12*F**2*L1**2*L2*ed**3*eh - 24*F**2*L1**2*L2*ed**3*em - 24*F**2*L1**2*L2*ed**2*eh**2 - 24*F**2*L1**2*L2*ed**2*eh*em + 12*F**2*L1**2*L2*ed**2*em**2 + 48*F**2*L1**2*L2*ed*eh**2*em + 12*F**2*L1**2*L2*ed*eh*em**2 - 24*F**2*L1**2*L2*eh**2*em**2 + 4*F**2*L1**2*ed**4 - 20*F**2*L1**2*ed**3*eh - 8*F**2*L1**2*ed**3*em + 16*F**2*L1**2*ed**2*eh**2 + 40*F**2*L1**2*ed**2*eh*em + 4*F**2*L1**2*ed**2*em**2 - 32*F**2*L1**2*ed*eh**2*em - 20*F**2*L1**2*ed*eh*em**2 + 16*F**2*L1**2*eh**2*em**2 - 18*F**2*L1*L2**3*Q*ed**4 + 36*F**2*L1*L2**3*Q*ed**3*eh + 36*F**2*L1*L2**3*Q*ed**3*em - 18*F**2*L1*L2**3*Q*ed**2*eh**2 - 72*F**2*L1*L2**3*Q*ed**2*eh*em - 18*F**2*L1*L2**3*Q*ed**2*em**2 + 36*F**2*L1*L2**3*Q*ed*eh**2*em + 36*F**2*L1*L2**3*Q*ed*eh*em**2 - 18*F**2*L1*L2**3*Q*eh**2*em**2 - 6*F**2*L1*L2**2*Q*ed**4 - 42*F**2*L1*L2**2*Q*ed**3*eh + 12*F**2*L1*L2**2*Q*ed**3*em + 48*F**2*L1*L2**2*Q*ed**2*eh**2 + 84*F**2*L1*L2**2*Q*ed**2*eh*em - 6*F**2*L1*L2**2*Q*ed**2*em**2 - 96*F**2*L1*L2**2*Q*ed*eh**2*em - 42*F**2*L1*L2**2*Q*ed*eh*em**2 + 48*F**2*L1*L2**2*Q*eh**2*em**2 - 18*F**2*L1*L2**2*ed**4 + 36*F**2*L1*L2**2*ed**3*eh + 18*F**2*L1*L2**2*ed**3*em - 18*F**2*L1*L2**2*ed**2*eh**2 - 36*F**2*L1*L2**2*ed**2*eh*em + 18*F**2*L1*L2**2*ed*eh**2*em + 16*F**2*L1*L2*Q*ed**4 + 52*F**2*L1*L2*Q*ed**3*eh - 32*F**2*L1*L2*Q*ed**3*em - 32*F**2*L1*L2*Q*ed**2*eh**2 - 104*F**2*L1*L2*Q*ed**2*eh*em + 16*F**2*L1*L2*Q*ed**2*em**2 + 64*F**2*L1*L2*Q*ed*eh**2*em + 52*F**2*L1*L2*Q*ed*eh*em**2 - 32*F**2*L1*L2*Q*eh**2*em**2 - 24*F**2*L1*L2*ed**4 - 24*F**2*L1*L2*ed**3*eh + 24*F**2*L1*L2*ed**3*em + 48*F**2*L1*L2*ed**2*eh**2 + 24*F**2*L1*L2*ed**2*eh*em - 48*F**2*L1*L2*ed*eh**2*em + 8*F**2*L1*Q*ed**4 - 20*F**2*L1*Q*ed**3*eh - 16*F**2*L1*Q*ed**3*em + 40*F**2*L1*Q*ed**2*eh*em + 8*F**2*L1*Q*ed**2*em**2 - 20*F**2*L1*Q*ed*eh*em**2 - 8*F**2*L1*ed**4 + 40*F**2*L1*ed**3*eh + 8*F**2*L1*ed**3*em - 32*F**2*L1*ed**2*eh**2 - 40*F**2*L1*ed**2*eh*em + 32*F**2*L1*ed*eh**2*em + 9*F**2*L2**4*Q**2*ed**4 - 18*F**2*L2**4*Q**2*ed**3*eh - 18*F**2*L2**4*Q**2*ed**3*em + 9*F**2*L2**4*Q**2*ed**2*eh**2 + 36*F**2*L2**4*Q**2*ed**2*eh*em + 9*F**2*L2**4*Q**2*ed**2*em**2 - 18*F**2*L2**4*Q**2*ed*eh**2*em - 18*F**2*L2**4*Q**2*ed*eh*em**2 + 9*F**2*L2**4*Q**2*eh**2*em**2 - 6*F**2*L2**3*Q**2*ed**4 + 30*F**2*L2**3*Q**2*ed**3*eh + 12*F**2*L2**3*Q**2*ed**3*em - 24*F**2*L2**3*Q**2*ed**2*eh**2 - 60*F**2*L2**3*Q**2*ed**2*eh*em - 6*F**2*L2**3*Q**2*ed**2*em**2 + 48*F**2*L2**3*Q**2*ed*eh**2*em + 30*F**2*L2**3*Q**2*ed*eh*em**2 - 24*F**2*L2**3*Q**2*eh**2*em**2 + 18*F**2*L2**3*Q*ed**4 - 36*F**2*L2**3*Q*ed**3*eh - 18*F**2*L2**3*Q*ed**3*em + 18*F**2*L2**3*Q*ed**2*eh**2 + 36*F**2*L2**3*Q*ed**2*eh*em - 18*F**2*L2**3*Q*ed*eh**2*em - 11*F**2*L2**2*Q**2*ed**4 - 32*F**2*L2**2*Q**2*ed**3*eh + 22*F**2*L2**2*Q**2*ed**3*em + 16*F**2*L2**2*Q**2*ed**2*eh**2 + 64*F**2*L2**2*Q**2*ed**2*eh*em - 11*F**2*L2**2*Q**2*ed**2*em**2 - 32*F**2*L2**2*Q**2*ed*eh**2*em - 32*F**2*L2**2*Q**2*ed*eh*em**2 + 16*F**2*L2**2*Q**2*eh**2*em**2 + 6*F**2*L2**2*Q*ed**4 + 42*F**2*L2**2*Q*ed**3*eh - 6*F**2*L2**2*Q*ed**3*em - 48*F**2*L2**2*Q*ed**2*eh**2 - 42*F**2*L2**2*Q*ed**2*eh*em + 48*F**2*L2**2*Q*ed*eh**2*em + 9*F**2*L2**2*ed**4 - 18*F**2*L2**2*ed**3*eh + 9*F**2*L2**2*ed**2*eh**2 + 4*F**2*L2*Q**2*ed**4 + 20*F**2*L2*Q**2*ed**3*eh - 8*F**2*L2*Q**2*ed**3*em - 40*F**2*L2*Q**2*ed**2*eh*em + 4*F**2*L2*Q**2*ed**2*em**2 + 20*F**2*L2*Q**2*ed*eh*em**2 - 16*F**2*L2*Q*ed**4 - 52*F**2*L2*Q*ed**3*eh + 16*F**2*L2*Q*ed**3*em + 32*F**2*L2*Q*ed**2*eh**2 + 52*F**2*L2*Q*ed**2*eh*em - 32*F**2*L2*Q*ed*eh**2*em + 12*F**2*L2*ed**4 + 12*F**2*L2*ed**3*eh - 24*F**2*L2*ed**2*eh**2 + 4*F**2*Q**2*ed**4 - 8*F**2*Q**2*ed**3*em + 4*F**2*Q**2*ed**2*em**2 - 8*F**2*Q*ed**4 + 20*F**2*Q*ed**3*eh + 8*F**2*Q*ed**3*em - 20*F**2*Q*ed**2*eh*em + 4*F**2*ed**4 - 20*F**2*ed**3*eh + 16*F**2*ed**2*eh**2 - 18*F*L1**2*L2**2*ed**4 + 36*F*L1**2*L2**2*ed**3*eh + 36*F*L1**2*L2**2*ed**3*em - 18*F*L1**2*L2**2*ed**2*eh**2 - 72*F*L1**2*L2**2*ed**2*eh*em - 18*F*L1**2*L2**2*ed**2*em**2 + 36*F*L1**2*L2**2*ed*eh**2*em + 36*F*L1**2*L2**2*ed*eh*em**2 - 18*F*L1**2*L2**2*eh**2*em**2 - 12*F*L1**2*L2*ed**4 - 30*F*L1**2*L2*ed**3*eh + 24*F*L1**2*L2*ed**3*em + 42*F*L1**2*L2*ed**2*eh**2 + 60*F*L1**2*L2*ed**2*eh*em - 12*F*L1**2*L2*ed**2*em**2 - 84*F*L1**2*L2*ed*eh**2*em - 30*F*L1**2*L2*ed*eh*em**2 + 42*F*L1**2*L2*eh**2*em**2 + 24*F*L1**2*ed**3*eh - 24*F*L1**2*ed**2*eh**2 - 48*F*L1**2*ed**2*eh*em + 48*F*L1**2*ed*eh**2*em + 24*F*L1**2*ed*eh*em**2 - 24*F*L1**2*eh**2*em**2 + 36*F*L1*L2**3*Q*ed**4 - 72*F*L1*L2**3*Q*ed**3*eh - 72*F*L1*L2**3*Q*ed**3*em + 36*F*L1*L2**3*Q*ed**2*eh**2 + 144*F*L1*L2**3*Q*ed**2*eh*em + 36*F*L1*L2**3*Q*ed**2*em**2 - 72*F*L1*L2**3*Q*ed*eh**2*em - 72*F*L1*L2**3*Q*ed*eh*em**2 + 36*F*L1*L2**3*Q*eh**2*em**2 - 12*F*L1*L2**2*Q*ed**4 + 96*F*L1*L2**2*Q*ed**3*eh + 24*F*L1*L2**2*Q*ed**3*em - 84*F*L1*L2**2*Q*ed**2*eh**2 - 192*F*L1*L2**2*Q*ed**2*eh*em - 12*F*L1*L2**2*Q*ed**2*em**2 + 168*F*L1*L2**2*Q*ed*eh**2*em + 96*F*L1*L2**2*Q*ed*eh*em**2 - 84*F*L1*L2**2*Q*eh**2*em**2 + 36*F*L1*L2**2*ed**4 - 72*F*L1*L2**2*ed**3*eh - 36*F*L1*L2**2*ed**3*em + 36*F*L1*L2**2*ed**2*eh**2 + 72*F*L1*L2**2*ed**2*eh*em - 36*F*L1*L2**2*ed*eh**2*em - 24*F*L1*L2*Q*ed**4 - 78*F*L1*L2*Q*ed**3*eh + 48*F*L1*L2*Q*ed**3*em + 48*F*L1*L2*Q*ed**2*eh**2 + 156*F*L1*L2*Q*ed**2*eh*em - 24*F*L1*L2*Q*ed**2*em**2 - 96*F*L1*L2*Q*ed*eh**2*em - 78*F*L1*L2*Q*ed*eh*em**2 + 48*F*L1*L2*Q*eh**2*em**2 + 24*F*L1*L2*ed**4 + 60*F*L1*L2*ed**3*eh - 24*F*L1*L2*ed**3*em - 84*F*L1*L2*ed**2*eh**2 - 60*F*L1*L2*ed**2*eh*em + 84*F*L1*L2*ed*eh**2*em + 24*F*L1*Q*ed**3*eh - 48*F*L1*Q*ed**2*eh*em + 24*F*L1*Q*ed*eh*em**2 - 48*F*L1*ed**3*eh + 48*F*L1*ed**2*eh**2 + 48*F*L1*ed**2*eh*em - 48*F*L1*ed*eh**2*em - 18*F*L2**4*Q**2*ed**4 + 36*F*L2**4*Q**2*ed**3*eh + 36*F*L2**4*Q**2*ed**3*em - 18*F*L2**4*Q**2*ed**2*eh**2 - 72*F*L2**4*Q**2*ed**2*eh*em - 18*F*L2**4*Q**2*ed**2*em**2 + 36*F*L2**4*Q**2*ed*eh**2*em + 36*F*L2**4*Q**2*ed*eh*em**2 - 18*F*L2**4*Q**2*eh**2*em**2 + 24*F*L2**3*Q**2*ed**4 - 66*F*L2**3*Q**2*ed**3*eh - 48*F*L2**3*Q**2*ed**3*em + 42*F*L2**3*Q**2*ed**2*eh**2 + 132*F*L2**3*Q**2*ed**2*eh*em + 24*F*L2**3*Q**2*ed**2*em**2 - 84*F*L2**3*Q**2*ed*eh**2*em - 66*F*L2**3*Q**2*ed*eh*em**2 + 42*F*L2**3*Q**2*eh**2*em**2 - 36*F*L2**3*Q*ed**4 + 72*F*L2**3*Q*ed**3*eh + 36*F*L2**3*Q*ed**3*em - 36*F*L2**3*Q*ed**2*eh**2 - 72*F*L2**3*Q*ed**2*eh*em + 36*F*L2**3*Q*ed*eh**2*em + 6*F*L2**2*Q**2*ed**4 + 54*F*L2**2*Q**2*ed**3*eh - 12*F*L2**2*Q**2*ed**3*em - 24*F*L2**2*Q**2*ed**2*eh**2 - 108*F*L2**2*Q**2*ed**2*eh*em + 6*F*L2**2*Q**2*ed**2*em**2 + 48*F*L2**2*Q**2*ed*eh**2*em + 54*F*L2**2*Q**2*ed*eh*em**2 - 24*F*L2**2*Q**2*eh**2*em**2 + 12*F*L2**2*Q*ed**4 - 96*F*L2**2*Q*ed**3*eh - 12*F*L2**2*Q*ed**3*em + 84*F*L2**2*Q*ed**2*eh**2 + 96*F*L2**2*Q*ed**2*eh*em - 84*F*L2**2*Q*ed*eh**2*em - 18*F*L2**2*ed**4 + 36*F*L2**2*ed**3*eh - 18*F*L2**2*ed**2*eh**2 - 12*F*L2*Q**2*ed**4 - 24*F*L2*Q**2*ed**3*eh + 24*F*L2*Q**2*ed**3*em + 48*F*L2*Q**2*ed**2*eh*em - 12*F*L2*Q**2*ed**2*em**2 - 24*F*L2*Q**2*ed*eh*em**2 + 24*F*L2*Q*ed**4 + 78*F*L2*Q*ed**3*eh - 24*F*L2*Q*ed**3*em - 48*F*L2*Q*ed**2*eh**2 - 78*F*L2*Q*ed**2*eh*em + 48*F*L2*Q*ed*eh**2*em - 12*F*L2*ed**4 - 30*F*L2*ed**3*eh + 42*F*L2*ed**2*eh**2 - 24*F*Q*ed**3*eh + 24*F*Q*ed**2*eh*em + 24*F*ed**3*eh - 24*F*ed**2*eh**2 + 9*L1**2*L2**2*ed**4 - 18*L1**2*L2**2*ed**3*eh - 18*L1**2*L2**2*ed**3*em + 9*L1**2*L2**2*ed**2*eh**2 + 36*L1**2*L2**2*ed**2*eh*em + 9*L1**2*L2**2*ed**2*em**2 - 18*L1**2*L2**2*ed*eh**2*em - 18*L1**2*L2**2*ed*eh*em**2 + 9*L1**2*L2**2*eh**2*em**2 + 18*L1**2*L2*ed**3*eh - 18*L1**2*L2*ed**2*eh**2 - 36*L1**2*L2*ed**2*eh*em + 36*L1**2*L2*ed*eh**2*em + 18*L1**2*L2*ed*eh*em**2 - 18*L1**2*L2*eh**2*em**2 + 9*L1**2*ed**2*eh**2 - 18*L1**2*ed*eh**2*em + 9*L1**2*eh**2*em**2 - 18*L1*L2**3*Q*ed**4 + 36*L1*L2**3*Q*ed**3*eh + 36*L1*L2**3*Q*ed**3*em - 18*L1*L2**3*Q*ed**2*eh**2 - 72*L1*L2**3*Q*ed**2*eh*em - 18*L1*L2**3*Q*ed**2*em**2 + 36*L1*L2**3*Q*ed*eh**2*em + 36*L1*L2**3*Q*ed*eh*em**2 - 18*L1*L2**3*Q*eh**2*em**2 + 18*L1*L2**2*Q*ed**4 - 54*L1*L2**2*Q*ed**3*eh - 36*L1*L2**2*Q*ed**3*em + 36*L1*L2**2*Q*ed**2*eh**2 + 108*L1*L2**2*Q*ed**2*eh*em + 18*L1*L2**2*Q*ed**2*em**2 - 72*L1*L2**2*Q*ed*eh**2*em - 54*L1*L2**2*Q*ed*eh*em**2 + 36*L1*L2**2*Q*eh**2*em**2 - 18*L1*L2**2*ed**4 + 36*L1*L2**2*ed**3*eh + 18*L1*L2**2*ed**3*em - 18*L1*L2**2*ed**2*eh**2 - 36*L1*L2**2*ed**2*eh*em + 18*L1*L2**2*ed*eh**2*em + 18*L1*L2*Q*ed**3*eh - 18*L1*L2*Q*ed**2*eh**2 - 36*L1*L2*Q*ed**2*eh*em + 36*L1*L2*Q*ed*eh**2*em + 18*L1*L2*Q*ed*eh*em**2 - 18*L1*L2*Q*eh**2*em**2 - 36*L1*L2*ed**3*eh + 36*L1*L2*ed**2*eh**2 + 36*L1*L2*ed**2*eh*em - 36*L1*L2*ed*eh**2*em - 18*L1*ed**2*eh**2 + 18*L1*ed*eh**2*em + 9*L2**4*Q**2*ed**4 - 18*L2**4*Q**2*ed**3*eh - 18*L2**4*Q**2*ed**3*em + 9*L2**4*Q**2*ed**2*eh**2 + 36*L2**4*Q**2*ed**2*eh*em + 9*L2**4*Q**2*ed**2*em**2 - 18*L2**4*Q**2*ed*eh**2*em - 18*L2**4*Q**2*ed*eh*em**2 + 9*L2**4*Q**2*eh**2*em**2 - 18*L2**3*Q**2*ed**4 + 36*L2**3*Q**2*ed**3*eh + 36*L2**3*Q**2*ed**3*em - 18*L2**3*Q**2*ed**2*eh**2 - 72*L2**3*Q**2*ed**2*eh*em - 18*L2**3*Q**2*ed**2*em**2 + 36*L2**3*Q**2*ed*eh**2*em + 36*L2**3*Q**2*ed*eh*em**2 - 18*L2**3*Q**2*eh**2*em**2 + 18*L2**3*Q*ed**4 - 36*L2**3*Q*ed**3*eh - 18*L2**3*Q*ed**3*em + 18*L2**3*Q*ed**2*eh**2 + 36*L2**3*Q*ed**2*eh*em - 18*L2**3*Q*ed*eh**2*em + 9*L2**2*Q**2*ed**4 - 18*L2**2*Q**2*ed**3*eh - 18*L2**2*Q**2*ed**3*em + 9*L2**2*Q**2*ed**2*eh**2 + 36*L2**2*Q**2*ed**2*eh*em + 9*L2**2*Q**2*ed**2*em**2 - 18*L2**2*Q**2*ed*eh**2*em - 18*L2**2*Q**2*ed*eh*em**2 + 9*L2**2*Q**2*eh**2*em**2 - 18*L2**2*Q*ed**4 + 54*L2**2*Q*ed**3*eh + 18*L2**2*Q*ed**3*em - 36*L2**2*Q*ed**2*eh**2 - 54*L2**2*Q*ed**2*eh*em + 36*L2**2*Q*ed*eh**2*em + 9*L2**2*ed**4 - 18*L2**2*ed**3*eh + 9*L2**2*ed**2*eh**2 - 18*L2*Q*ed**3*eh + 18*L2*Q*ed**2*eh**2 + 18*L2*Q*ed**2*eh*em - 18*L2*Q*ed*eh**2*em + 18*L2*ed**3*eh - 18*L2*ed**2*eh**2 + 9*ed**2*eh**2))/(2*(3*F*L1*L2*ed - 3*F*L1*L2*em - F*L1*ed + F*L1*em - 3*F*L2**2*Q*ed + 3*F*L2**2*Q*em + F*L2*Q*ed - F*L2*Q*em - 3*F*L2*ed + F*ed - 3*L1*L2*ed + 3*L1*L2*em + 3*L1*ed - 3*L1*em + 3*L2**2*Q*ed - 3*L2**2*Q*em - 3*L2*Q*ed + 3*L2*Q*em + 3*L2*ed - 3*ed))]

        root = np.array(roots(epsLO, epsH, epsM, L1, L2, Q, F), dtype=np.complex_)


        epsBR_ARR = root[1]
        etaBR_ARR = self.etaFromEps(epsBR_ARR)
        self.matDict[matName] = dict(permittivityFunction="setOsawaBruggeman")
        self.matDict[matName]["eta"] = etaBR_ARR
        self.matDict[matName]["eps"] = epsBR_ARR
        self.matDict[matName]["parameters"] = [ellipsoidMat, adsorbateMat, hostMat, thickness, molec, ratio1, F]

    # This function gives the geometric factor for any ellipsoid.
    def getGeometricFactor(self, a, b, c, forceGeneral = False):
        if(a < 0):
            raise ValueError("Error when calculating geometric factor of an ellipsoid: a, b, c must be positive numbers representing the radii of the ellipsoid.")
        if(a < b or b < c):
            print("a = ", a, "b = ", b, "c = ", c)
            print("a < b?", a<b, "b < c?", b<c)
            raise ValueError("Error when calculating geometric factor of an ellipsoid. <a> must be greater or equal to <b> which must be greater or equal to <c>.")
        # When eccentricity = 0 (or aspect ratio = 1, i.e. a = b = c, for a sphere), the analytical expressions
        # for prolate and oblate ellipsoids are not defined, so need to have a separate if statement to deal with
        # this case.
        if (a == b and b == c and not forceGeneral):## Sphere.
            return(1/3, 1/3, 1/3)

        if (a == b and not forceGeneral): ## Oblate spheroid.
            # Defined in terms of eccentricity (Bohren and Huffman)
            e = np.sqrt(1 - (c**2 / a**2))
            g_e = np.sqrt((1 - e**2) / e**2)
            L1 = (g_e/(2*e**2))*(np.pi/2 - np.arctan(g_e)) - g_e**2/2

            # Defined in terms of aspct ratio (Osborne, Fedotov)
#             m = a / c
#             L1 = (m**2 / (2*np.sqrt((m**2 - 1)**3))) *(np.pi/2 - np.arctan(1/np.sqrt(m**2 - 1)) - np.sqrt(m**2 - 1)/m**2)
            L3 = 1 - 2*L1
            return (L1, L1, L3)

        if (b == c and not forceGeneral): ## Prolate spheroid.
            # Defined in terms of eccentricity (Bohren and Huffman)
            e = np.sqrt(1 - (c**2 / a**2))
            L1 = ((1 - e**2)/e**2)*(-1 + (1/(2*e))*np.log((1 + e)/(1 - e)))

            # Defined in terms of aspect ratio (Osborne)
#             m = a / c
#             L1 = (1/(m**2 - 1))*((m/(2*np.sqrt(m**2 - 1)))*np.log((m + np.sqrt(m**2 - 1)) / (m - np.sqrt(m**2 - 1))) - 1)
            L3 = 1/2 - L1/2
            return (L1, L3, L3)

        else: ## General expression, given in Bohren and Huffman, page 14
            # Seems to work much better if you normalize to 1, at lest for small aspect ratios. Haven't tried for really long aspect ratios where you might get floating point errors.
            b = b/a
            c = c/a
            a = 1
            funcL1 = lambda q: 1 / ((((a**2 + q)*(b**2 + q)*(c**2 + q))**(1/2))*(a**2 + q))
            funcL2 = lambda q: 1 / ((((a**2 + q)*(b**2 + q)*(c**2 + q))**(1/2))*(b**2 + q))
            funcL3 = lambda q: 1 / ((((a**2 + q)*(b**2 + q)*(c**2 + q))**(1/2))*(c**2 + q))
            L1 = integrate.quad(funcL1, 0, np.inf)[0]*a*b*c/2
            L2 = integrate.quad(funcL2, 0, np.inf)[0]*a*b*c/2
            L3 = integrate.quad(funcL3, 0, np.inf)[0]*a*b*c/2
            return [L1, L2, L3]

    def bohrenLambda(self, particleMat, hostMat, Li, wavenumIndex):
        epsP = self.matDict[particleMat]["eps"][wavenumIndex]
        epsH = self.matDict[hostMat]["eps"][wavenumIndex]
        return epsH / (epsH + Li*(epsP - epsH))

###################
###################
###################
    def addDat(self, datName, dataset):
        self.statDict[datName] = dict()
        self.statDict[datName]["data"] = dataset
        self.statDict[datName]["n"] = len(dataset)
        self.statDict[datName]["u"] = np.mean(dataset)
        self.statDict[datName]["s"] = np.sqrt(np.sum((dataset - self.statDict[datName]["u"])**2) / (self.statDict[datName]["n"] - 1))
        self.statDict[datName]["u_dev"] = self.statDict[datName]["data"] - self.statDict[datName]["u"]

    def calcL(self, aName, bName, cName):
        self.abcValidity(aName, bName, cName)
        La, Lb, Lc = [], [], []
        for i in range(self.statDict[aName]["n"]):
            Lvals = self.getGeometricFactor(self.statDict[aName]["data"][i], self.statDict[bName]["data"][i], self.statDict[cName]["data"][i])
            La.append(Lvals[0])
            Lb.append(Lvals[1])
            Lc.append(Lvals[2])
        self.addDat("La", La)
        self.addDat("Lb", Lb)
        self.addDat("Lc", Lc)

    def abcValidity(self, aName, bName, cName):
        if (self.statDict[aName]["n"] != self.statDict[bName]["n"]) or (self.statDict[aName]["n"] != self.statDict[cName]["n"]):
            raise valueError("Error: the size of the three data sets passed to the calcL() function must be the same.")
        a = self.statDict[aName]["data"]
        b = self.statDict[bName]["data"]
        c = self.statDict[cName]["data"]
        for i in range(self.statDict[aName]["n"]):
            sortedVals = np.sort([a[i], b[i], c[i]])
            if (a[i] != sortedVals[2] or c[i] != sortedVals[0]):
                print("There was an error in the paired data. The data should be in order from largest semi-axis to smallest semi-axis. The program has auto-corrected the data, however, you should verify that a value was not accidentally omitted. The input values were: (a = " + str(a[i]) + "); (b = ", str(b[i]) + "); (c = " + str(c[i]) + "). The corrected order is: (a = " + str(sortedVals[2]) + "); (b = ", str(sortedVals[1]) + "); (c = " + str(sortedVals[0]) + ").")
#                 raise valueError("There was an error in the paired data. The data should be in order from largest semi-axis to smallest semi-axis. The program has auto-corrected the data, however, you should verify that a value was not accidentally omitted. The input values were: (a = " + str(a[i]) + "); (b = ", str(b[i]) + "); (c = " + str(c[i]) + "). The corrected order is: (a = " + str(sortedVals[2]) + "); (b = ", str(sortedVals[1]) + "); (c = " + str(sortedVals[0]) + ").")
                self.statDict[aName]["data"][i] = sortedVals[2]
                self.statDict[bName]["data"][i] = sortedVals[1]
                self.statDict[cName]["data"][i] = sortedVals[0]

    ## Automatic calculates a domain 3standard deviations on each side of the mean
    def gaussian(self, datName):
        domain = np.linspace(mean - 3*sd, mean + 3*sd, 100)
        return self.gaussianCore(datName, domain)

    def gaussianCore(self, datName, domain):
        return (1 / (np.sqrt(2*np.pi)*self.statDict[datName]["s"])) * np.exp(-(domain-self.statDict[datName]["u"])**2/(2*self.statDict[datName]["s"]**2))

    # Used for most intuitive syntax when calculating a bivariate Gaussian distribution
    def bivariateGaussian(self, datName1, datName2, domain1=False, domain2=False):
        # If no domain is provided, calculate the Gaussian up to 3 std devs from the mean with 100 data points
        domain1Given = isinstance(domain1, list) or isinstance(domain1, np.ndarray)
        domain2Given = isinstance(domain1, list) or isinstance(domain1, np.ndarray)
        if(not domain1Given):
            print("domain1 not provided as a list or numpy.ndarray, defaulting to a domain 3 standard deviations on either side of the mean.")
            domain1 = np.linspace(self.statDict[datName1]["u"] - 3*self.statDict[datName1]["s"], self.statDict[datName1]["u"] + 3*self.statDict[datName1]["s"], 10)
        if(not domain2Given):
            print("domain2 not provided as a list or numpy.ndarray, defaulting to a domain 3 standard deviations on either side of the mean.")
            domain2 = np.linspace(self.statDict[datName2]["u"] - 3*self.statDict[datName2]["s"], self.statDict[datName2]["u"] + 3*self.statDict[datName2]["s"], 10)
        x, y = np.meshgrid(domain1, domain2)
        return (x, y, self.bivariateGaussianCore(x, y, datName1, datName2))

    # Used for integrating with sympy and scipy
    def bivariateGaussianCore(self, domain1, domain2, datName1, datName2):
        rho = self.correlation(datName1, datName2)
        u1 = self.statDict[datName1]["u"]
        u2 = self.statDict[datName2]["u"]
        s1 = self.statDict[datName1]["s"]
        s2 = self.statDict[datName2]["s"]
#         print("s1", s1, "s2", s2, "rho", rho)
        preExp = 1 / (2*np.pi*np.sqrt(1-rho**2)*s1*s2)
        numTerm1 = (domain1 - u1)**2 / s1**2
        numTerm2 = rho*(domain1 - u1)*(domain2 - u2) / s1*s2
        numTerm3 = (domain2 - u2)**2 / s2**2
        num = numTerm1 - numTerm2 + numTerm3
        expFact = np.exp(-num/(2*(1 - rho**2)))
        G = preExp*expFact
#         print("numTerm1: ", numTerm1, "numTerm2: ", numTerm2, "numTerm3: ", numTerm3)
#         print("rho: ", rho)#, "preExp: ", preExp)
#         print("num: ", num)
#         print("expFact: ", expFact, "G: ", G)
#         print(-num/(2*(1 - rho**2)))
        return G

    def covariance(self, datName1, datName2):
        ## DEAL WITH: should
        if(self.statDict[datName1]["n"] != self.statDict[datName2]["n"]):
            raise valueError("Error: the size of the two data sets passed to the covariance() function must be the same.")
            return
        prod = np.multiply(self.statDict[datName1]["u_dev"], self.statDict[datName2]["u_dev"])
        return np.sum(prod)/(self.statDict[datName1]["n"]-1)

    def variance(self, datName):
        return np.sum((self.statDict["datName"]["data"] - self.statDict["datName"]["u"])**2)/(self.statDict["datName"]["n"] - 1)

    def correlation(self, datName1, datName2):
        correl = self.covariance(datName1, datName2)/ (self.statDict[datName1]["s"]*self.statDict[datName2]["s"])
        # This series off if statements is required to prevent floating point errors from returning a value outside the range (-1 < x < 1)
        if (correl < -1):
            return -1
        elif (correl > 1):
            return 1
        else:
            return correl

    def unitStep(self, x):
        if (x < 0):
            return 0
        else:
            return 1

    def unitStepARR(self, xARR):
        U = np.zeros(len(xARR))
        for i in range(len(xARR)):
            U[i] = self.unitStep(xARR[i])
        return U

#     def getU2(self, L1, L2):
#         return(self.unitStep(L1)*self.unitStep(L2-L1)*self.unitStep(1-2*L2-L1))

    def GU1_int(self, L1, datName):
        G = self.gaussianCore(datName, L1)
        U =  self.unitStep(L1)*self.unitStep(1/3 - L1)
        return G*U

    def GU2_int(self, L1, L2, datName1, datName2):
        G = self.bivariateGaussianCore(L1, L2, datName1, datName2)
        U =  (self.unitStep(L1)*self.unitStep(L2-L1)*self.unitStep(1-2*L2-L1))
        return G*U

    def P1(self, L1, lowLim, upLim, datName):
        GU = self.gaussianCore(datName, L1)*self.unitStepARR(L1)*self.unitStepARR(1/3 - L1)
        K1D_reciprocal, K1Derror = integrate.quad(self.GU1_int, lowLim, upLim, args=(datName))
        P1 = GU/K1D_reciprocal
        return P1

    def beta1(self, L1, argDict):
        datName = argDict["datName"]
        particleMat = argDict["particleMat"]
        hostMat = argDict["hostMat"]
        wavenumIndex = argDict["wavenumIndex"]
        G = self.gaussianCore(datName, L1)
        U =  self.unitStep(L1)*self.unitStep(1/3 - L1)
        K1D_reciprocal, K1Derror = integrate.quad(self.GU1_int, 0, 1/3, args=(datName))
        lambda1 = self.bohrenLambda(particleMat, hostMat, L1, wavenumIndex)
        lambda2 = self.bohrenLambda(particleMat, hostMat, (1-L1)/2, wavenumIndex)
        return G*U*lambda1*lambda2**2/(3*K1D_reciprocal)

    def beta2(self, L2, L1, argDict):
        datName1 = argDict["datName1"]
        datName2 = argDict["datName2"]
        particleMat = argDict["particleMat"]
        hostMat = argDict["hostMat"]
        wavenumIndex = argDict["wavenumIndex"]
        G = self.bivariateGaussianCore(L1, L2, datName1, datName2)
        U = self.unitStep(L1)*self.unitStep(L2-L1)*self.unitStep(1-2*L2-L1)
        K1D_reciprocal, K1Derror = integrate.dblquad(self.GU2_int, 0, 1/3, lambda x: 0, lambda x: 1/2, args=(datName1, datName2))
        lambda1 = self.bohrenLambda(particleMat, hostMat, L1, wavenumIndex)
        lambda2 = self.bohrenLambda(particleMat, hostMat, L2, wavenumIndex)
        lambda3 = self.bohrenLambda(particleMat, hostMat, (1-L1-L2), wavenumIndex)
        return G*U*lambda1*lambda2**2/(3*K1D_reciprocal)

    def gmg(self, name, particleName, hostName, fillFactor, ):
        return

    # Pecharroman & Cuesta 2004 J Electroanalytical Chem 563; 91-109
    def setPecharroman(self, matName, r_core, r_shell, e_core, e_shell, e_matrix, f):
        h = (r_shell**3 - r_core**3) / r_core**3
        e_p = e_shell[0][0]
    #     print(e_p, "\n\n", e_core, "\n\n", e_matrix)
        # e_eff = e_eff0 + h*e_eff1

        # Anisotropic shell
        c20 = np.full(len(e_core), 6)
        c21 = 2*h*(e_core/e_p - 1)
        c10 = 3*((1-3*f)*e_core - (2 - 3*f)*e_matrix)
        c11 = -(e_core/e_p - 1)*(2 - 3*f)*e_matrix - 2*(1 - 3*f)*e_core + 2*e_p*(1-f)
        c00 = -3*e_core*e_matrix
        c01 = 2*e_matrix*(e_core - (1 - f)*e_p)
        print(c20, "\n\n", c10, "\n\n", c00)
        coefs = np.array([c20, c10, c00])
    #     print(coefs)
        allRoots = [np.full(len(e_core), 0+0*1j)]
        for i in range(len(coefs[0])):
            allRoots.append(np.roots(coefs[i]))
        print(allRoots)
    # wienerZ_ARR is an array of the complex permittivities of the components
    # roots_ARR is an array of the the possible roots of which you want to determine the correct physically meaningful root
    # The length of wienerZ_ARR and roots_ARR should be the same.
        self.getPhysicalRoot(np.array([e_core, e_shell, e_p]), allRoots)
        eta = self.etaFromEps(eps)
        self.matDict[matName] = dict(permittivityFunction="setPecharroman")
        self.matDict[matName]["eta"] = eta
        self.matDict[matName]["eps"] = eps

    # 2017_zhao_mei_J._Phys._D _Appl._Phys._50_505001
    def setCoatedEllipsoid(self, matName, coreMat, shellMat, a_core, b_core, c_core, lamda, rotMat=None):
        # lamda is the volume fraction of the outer ellipsoid to the inner ellipsoid =(Vcore + Vshell)/ Vshell
        # physically valid lamdas: 0 <= lambda < 1
        epsCore = self.matDict[coreMat]["eps"]
        epsShell = self.matDict[shellMat]["eps"]
        b1 = float(b_core)/float(a_core)
        c1 = float(c_core)/float(a_core)
        a1 = 1.
        if (lamda == 0):
            a2 = a1
            b2 = b1
            c2 = c1
        else:
            # Calculate a2, b2, c2 given a1, b1, c1, and lamda under the confocal requirement.
    #         a1, b1, c1 = sp.symbols(["a1", "b1", "c1"])
            ARR1 = [a1**2, b1**2, c1**2]
            ARR2 = [1, 1, 1]
            polyCoef = self.binomialProdCoefs(ARR1, ARR2)
            polyCoef[0] *= (1 - lamda)
    #         for i in range(len(polyCoef)):
    #             print("coef of x^" + str(i) + " : " + str(polyCoef[i]))
            rutz = np.roots(polyCoef[::-1])
            # print(rutz)
            xi2 = np.real(rutz[2]) # TODO: Verify that this is the case. -> # Select the root which yields purely real a2, b2, c2. The second root always seems to be correct.
            a2 = np.sqrt(a1**2 + xi2)
            b2 = np.sqrt(b1**2 + xi2)
            c2 = np.sqrt(c1**2 + xi2)
    #         print(rutz)
#             for r in rutz:
#                 a2 = np.sqrt(a1**2 + r)
#                 b2 = np.sqrt(b1**2 + r)
#                 c2 = np.sqrt(c1**2 + r)
#                 print("xi2: ", "{:.2f}".format(r))
#                 print("a2: ", "{:.2f}".format(a2), "b2: ", "{:.2f}".format(b2), "c2: ", "{:.2f}".format(c2), "\n\n")
    #         def R(s, a, b, c):
    #             return(np.sqrt((s+a**2)*(s+b**2)*(s+c**2)))
        N1a, N1b, N1c = self.getGeometricFactor(a1,b1,c1)
        N2a, N2b, N2c = self.getGeometricFactor(a2,b2,c2)
        epsEff_x = epsShell*(1 - (epsShell - epsCore)*lamda / (epsShell + (epsShell - epsCore)*(lamda*N1a - N2a)))
        epsEff_y = epsShell*(1 - (epsShell - epsCore)*lamda / (epsShell + (epsShell - epsCore)*(lamda*N1b - N2b)))
        epsEff_z = epsShell*(1 - (epsShell - epsCore)*lamda / (epsShell + (epsShell - epsCore)*(lamda*N1c - N2c)))

        n = len(epsEff_x)
        out = np.zeros((n,3,3), dtype = np.complex_)
        for i in range(n):
            out[i] = [[epsEff_x[i], 0, 0], [0, epsEff_y[i], 0], [0, 0, epsEff_z[i]]]

#         self.matDict[matName] = dict(permittivityFunction="setCoatedEllipsoid")
#         self.matDict[matName]["eta"] = self.etaFromEps(out)
#         self.matDict[matName]["eps"] = out
        self.matDict[matName] = dict(permittivityFunction="setCoatedEllipsoid")
        self.matDict[matName]["eta"] = self.etaFromEps(epsEff_z)
        self.matDict[matName]["eps"] = epsEff_z
        return out

    # 2017_zhao_mei_J._Phys._D _Appl._Phys._50_505001
    def setCoatedEllipsoidAniso(self, matName, coreMat, shellMat, a_core, b_core, c_core, lamda, rotMat=None):
        # lamda is the volume fraction of the outer ellipsoid to the inner ellipsoid =(Vcore + Vshell)/ Vshell
        # physically valid lamdas: 0 <= lambda < 1
        epsCore = self.matDict[coreMat]["eps"]
        epsShell = self.matDict[shellMat]["eps"]
        b1 = float(b_core)/float(a_core)
        c1 = float(c_core)/float(a_core)
        a1 = 1.
        if (lamda == 0):
            a2 = a1
            b2 = b1
            c2 = c1
        else:
            # Calculate a2, b2, c2 given a1, b1, c1, and lamda under the confocal requirement.
    #         a1, b1, c1 = sp.symbols(["a1", "b1", "c1"])
            ARR1 = [a1**2, b1**2, c1**2]
            ARR2 = [1, 1, 1]
            polyCoef = self.binomialProdCoefs(ARR1, ARR2)
            polyCoef[0] *= (1 - lamda)
    #         for i in range(len(polyCoef)):
    #             print("coef of x^" + str(i) + " : " + str(polyCoef[i]))
            rutz = np.roots(polyCoef[::-1])
            # print(rutz)
            xi2 = np.real(rutz[2]) # TODO: Verify that this is the case. -> # Select the root which yields purely real a2, b2, c2. The second root always seems to be correct.
            a2 = np.sqrt(a1**2 + xi2)
            b2 = np.sqrt(b1**2 + xi2)
            c2 = np.sqrt(c1**2 + xi2)
    #         print(rutz)
#             for r in rutz:
#                 a2 = np.sqrt(a1**2 + r)
#                 b2 = np.sqrt(b1**2 + r)
#                 c2 = np.sqrt(c1**2 + r)
#                 print("xi2: ", "{:.2f}".format(r))
#                 print("a2: ", "{:.2f}".format(a2), "b2: ", "{:.2f}".format(b2), "c2: ", "{:.2f}".format(c2), "\n\n")
    #         def R(s, a, b, c):
    #             return(np.sqrt((s+a**2)*(s+b**2)*(s+c**2)))
        N1a, N1b, N1c = self.getGeometricFactor(a1,b1,c1)
        N2a, N2b, N2c = self.getGeometricFactor(a2,b2,c2)
        epsEff_x = epsShell*(1 - (epsShell - epsCore)*lamda / (epsShell + (epsShell - epsCore)*(lamda*N1a - N2a)))
        epsEff_y = epsShell*(1 - (epsShell - epsCore)*lamda / (epsShell + (epsShell - epsCore)*(lamda*N1b - N2b)))
        epsEff_z = epsShell*(1 - (epsShell - epsCore)*lamda / (epsShell + (epsShell - epsCore)*(lamda*N1c - N2c)))

#         n = len(epsEff_x)
#         out = np.zeros((n,3,3), dtype = np.complex_)
#         for i in range(n):
#             out[i] = [[epsEff_x[i], 0, 0], [0, epsEff_y[i], 0], [0, 0, epsEff_z[i]]]

#         self.matDict[matName] = dict(permittivityFunction="setCoatedEllipsoid")
#         self.matDict[matName]["eta"] = self.etaFromEps(out)
#         self.matDict[matName]["eps"] = out
        self.matDict[matName] = dict(permittivityFunction="setCoatedEllipsoidAniso")


        n = len(epsEff_x)
        out = np.zeros((n,3,3), dtype = np.complex_)
        for i in range(n):
            out[i] = [[epsEff_x[i], 0, 0], [0, epsEff_y[i], 0], [0, 0, epsEff_z[i]]]
        self.matDict[matName]["eps"] = out
        self.matDict[matName]["eta"] = np.sqrt(out)
        self.matDict[matName]["parameters"] = [coreMat, shellMat, a_core, b_core, c_core, lamda, rotMat]
        return out
