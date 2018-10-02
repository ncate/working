from osgeo import gdal
import numpy as np
import polTable
from snappy import ProductIO

filePath = r"E:\School\Graduate_MS\0_Thesis\0_Data\Imagery\SLC_interfere3\tiffs\S1_2016_06_10_22_Orb_Stack_ifgFED8FEP1001OI5_deb.tif"

img = gdal.Open(filePath)
img = img.ReadAsArray()


orbitH = 693000 # Orbit height in meters. Does this need to be more precise?
wavelengthCM = 5.5465763 # Wavelength in cm
Baseline = 30.57856580367306 # rough horizontal offset based on the first pixel of 2 images
Lwindow = 11 # For coherence and ground phase - Based on Touzi 1999
Wvs = polTable.create_weight_vector_table()


def flat_earth_removal(baseImg, h=orbitH, wavelength=wavelengthCM, B=Baseline):
    flatEarthPhase = np.zeros_like(baseImg)
    for y in range(baseImg.shape[1]):
        r1 = np.sqrt(h**2 + y**2)  # Where y is the ground range coordinate
        r2 = np.sqrt((h**2) + ((y+B)**2))
        flatEarthPhase[:, y] = np.exp(i * ((4 * np.pi) / wavelength) * (r2 - r1))
    return flatEarthPhase

def vert_wave_est(theta, B, h, wavelength):
    delta_theta = np.arctan(np.tan(theta) + (B/h)) - theta
    kz = (4 * np.pi * delta_theta) / (wavelength * np.sin(theta))
    return kz

def ground_phase_est(gammaV, gammaS):
    A = (np.abs(gammaS)**2)-1
    B = 2 * np.real((gammaV - gammaS) * np.conj(gammaS))  # Check that the conj is correct here
    C = np.abs(gammaV-gammaS)**2

    LWs = (-B - np.sqrt((B**2) - 4 * A * C)) / (2 * A)

    phi = np.angle(gammaV - gammaS * (1 - LWs))
    return phi

def veg_height_est(gammaV, phi, kz, epsilon):
    hv = ((np.angle(gammaV) - phi) / (kz)) + epsilon * ((2 * np.sinc(np.abs(gammaV))) / (kz)) # Make sure this in inverse sinc
    return hv



#
# test_theta = np.zeros([100,100])
# t = 20
# for i in range(test_theta.shape[1]):
#     test_theta[:, i] = t
#     t += 0.3
#
# testimg = np.zeros([100,100])
# t = 1
# for i in range(testimg.shape[1]):
#     testimg[:, i] = t
#     t += 1
#

# nearInc = prod.getMetadataRoot().getElement('Abstracted_Metadata').getAttributeDouble('incidence_near')
# farInc = prod.getMetadataRoot().getElement('Abstracted_Metadata').getAttributeDouble('incidence_far')
# diff = farInc - nearInc
# step = diff / width
# theta = np.zeros((height, width))
# currentInc = nearInc
# for i in range(width):
#     theta[:,i] = currentInc
#     currentInc += step