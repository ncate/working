import numpy as np
from snappy import ProductIO
from scipy import constants


# filepath = r"E:\School\Graduate_MS\0_Thesis\0_Data\Imagery\SLC_interfere3\outputs\subsetTHIS_0_of_S1_2016_06_10_22_Orb_Stack_ifgFED8FEP1001OI5_deb.dim"

def get_bands_in_np (filepath):
    prod = ProductIO.readProduct(filepath)

    # Get some Metadata
    width = prod.getSceneRasterWidth()
    height = prod.getSceneRasterHeight()

    # Extract bands and tie points from the product
    bandnames = list(prod.getBandNames())
    tpnames = list(prod.getTiePointGridNames())
    bandStack = list(prod.getBands())
    tiepointStack = list(prod.getTiePointGrids())

    prod.dispose()

    img = np.zeros((len(bandnames), width, height), dtype=np.float32)

    try:
        i = 0
        for currentBand in bandStack:
            img[i, :, :] = currentBand.readPixels(0, 0, width, height, img[i, :, :])
            i += 1
    except:
        for currentBand in bandStack:
            for y in range(height):
                # print("processing line ", y, " of ", height)
                currentBand.readPixels(0, y, width, 1, img)

    return img, bandnames

def get_prod_metadata(first_image_location, second_image_location):
    prod1 = ProductIO.readProduct(first_image_location)
    prod2 = ProductIO.readProduct(second_image_location)

    # Get some Metadata
    freqMHz = prod1.getMetadataRoot().getElement('Abstracted_Metadata').getAttributeDouble('radar_frequency')
    wavelengthM = constants.c / (freqMHz * 10 ** 6)

    long1 = prod1.getMetadataRoot().getElement('Abstracted_Metadata').getAttributeDouble('first_near_long')
    long2 = prod2.getMetadataRoot().getElement('Abstracted_Metadata').getAttributeDouble('first_near_long')
    #lat = prod

    prod1.dispose()
    prod2.dispose()

    return wavelengthM

# import jpy
# Runtime = jpy.get_type('java.lang.Runtime')
# max_memory = Runtime.getRuntime().maxMemory()
# total_memory = Runtime.getRuntime().totalMemory()
# free_memory = Runtime.getRuntime().freeMemory()
# gb = 1e+9
# print('max memory:', max_memory / gb, "GB")
# print('total memory:', total_memory / gb, "GB")
# print('free memory:', free_memory / gb, "GB")