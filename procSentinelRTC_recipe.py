#!/usr/bin/python
###################################################################
#
#  Alaska Satellite Facility DAAC
#
#  Sentinel Radiometric Terrain Correction using Sentinel Toolbox (SNAP)
#
###################################################################
#
# Copyright (C) 2016 Alaska Satellite Facility
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#
###################################################################
import sys,os,re
import numpy
import datetime
import time
from optparse import OptionParser

global baseSNAP
global infile 
global extDEM
global pixsiz

###################################################################
#  SET THIS VARIABLE
#
#  In order to run this program, you must set the variable baseSNAP 
#  to point to the gpt executable on your system.
#
###################################################################
baseSNAP = '/Applications/snap/bin/gpt '
###################################################################

###################################################################
#  Code setup section
###################################################################
usage = "usage: %prog [options] <S1A Zip File>"
parser = OptionParser(usage=usage)
parser.add_option("-r",dest="pixsiz",help="Pixel resolution - default = 10m",metavar="PS")
parser.add_option("-d","--dem",dest="extDEM",help="External DEM file name",metavar="DEM")
parser.add_option("-c","--clean",action="store_true",dest="cleanTemp",default=False,
                  help="Clean intermediate files")
(options, args) = parser.parse_args()

if (len(args)==0):
    parser.print_help()
    print ""
    print "ERROR: No S1A zip file specified"
    print ""
    quit()

extDEM = options.extDEM
cleanTemp = options.cleanTemp
pixsiz = options.pixsiz

infile = args[0]
baseGran = re.split('/',infile.replace('.zip',''))[-1]

tdir = os.getcwd()
if (extDEM == None):
    print "No external DEM file specified"
if (extDEM != None):
    print "Using External DEM file %s" % extDEM
    if "/" not in extDEM:
        extDEM = tdir + "/" +  extDEM 
if "/" not in infile:
    infile = tdir + "/" + infile
if (pixsiz == None):
    pixsiz = 10.0

print "INFILE = %s" % infile
print "BASENAME = %s" % baseGran
print "TDIR = %s" % tdir
print "DEM = %s" % extDEM
print "CLEANUP = %s" % cleanTemp
print "PIXEL SIZE = %s" % pixsiz

###################################################################
#  Subroutine definitions
###################################################################
def timestamp(date):
    return time.mktime(date.timetuple())

# Create output directory
def tempDir(tdir,ifile):
    td2 = '%s/%s' % (tdir,ifile)
    if not os.path.exists(tdir):
        os.system('mkdir %s' % tdir)
    if not os.path.exists(td2):
        os.system('mkdir %s' % td2)
    return td2

# Apply precise orbit file
def applyOrbit(granule,td2,baseGran):
    aoFlag = 'Apply-Orbit-File '
    oType = '-PcontinueOnFail=\"true\" -PorbitType=\'Sentinel Precise (Auto Download)\' '
    out = '-t %s/%s ' % (td2,baseGran+'_OB')
    cmd = baseSNAP + aoFlag + out + oType + granule
    print 'Applying Precise Orbit file'
    print cmd
    os.system(cmd)
    return '%s' % baseGran+'_OB.dim'

# Apply calibration
def applyCal(inData,td2):
    calFlag = 'Calibration -PoutputBetaBand=true -PoutputSigmaBand=false '
    out = '-t %s/%s ' % (td2,inData.replace('.dim','_CAL'))
    inD = '-Ssource=%s/%s' % (td2,inData)
    cmd = baseSNAP + calFlag + out + inD
    print 'Applying Calibration'
    os.system(cmd)
    return '%s' % inData.replace('.dim','_CAL.dim')

#Apply terrain flattening
def applyTF(inData,td2):
    tfFlag = 'Terrain-Flattening '
    out = '-t %s/%s ' % (td2,inData.replace('.dim','_TF'))
    inD = '-Ssource=%s/%s' % (td2,inData)
    if extDEM != None:
        inD = inD + ' -PdemName=\"External DEM\" -PexternalDEMFile=%s -PexternalDEMNoDataValue=0 ' % extDEM
    else:
        inD = inD + ' -PdemName=\"SRTM 1Sec HGT\" '
    cmd = baseSNAP + tfFlag + out + inD
    print 'Applying Terrain Flattening -- This will take some time'
    print cmd
    os.system(cmd)
    return '%s' % inData.replace('.dim','_TF.dim')

# Apply terrain correction
def applyTC(inData,td2,zone,cm,hemi):
    tcFlag = 'Terrain-Correction '
    out = '-t %s/%s ' % (td2,inData.replace('.dim','_TC'))
    inD = '-Ssource=%s/%s ' % (td2,inData)
    inD = inD + '-PsaveDEM=true '
    inD = inD + '-PsaveProjectedLocalIncidenceAngle=true ' 
    inD = inD + '-PpixelSpacingInMeter=%s ' % pixsiz

# This is the alternate form of specifying the output projection:
#    inD = inD + '-PmapProjection=\'PROJCS[\"UTM Zone %s / World Geodetic System 1984\", GEOGCS[\"World Geodetic System 1984\", DATUM[\"World Geodetic System 1984\", SPHEROID[\"WGS 84\", 6378137.0, 298.257223563, AUTHORITY[\"EPSG\",\"7030\"]], AUTHORITY[\"EPSG\",\"6326\"]], PRIMEM[\"Greenwich\", 0.0, AUTHORITY[\"EPSG\",\"8901\"]], UNIT[\"degree\", 0.017453292519943295], AXIS[\"Geodetic longitude\", EAST], AXIS[\"Geodetic latitude\", NORTH]], PROJECTION[\"Transverse_Mercator\"], PARAMETER[\"central_meridian\", %s], PARAMETER[\"latitude_of_origin\", 0.0], PARAMETER[\"scale_factor\", 0.9996], PARAMETER[\"false_easting\", 500000.0], PARAMETER[\"false_northing\", 0.0], UNIT[\"m\", 1.0], AXIS[\"Easting\", EAST], AXIS[\"Northing\", NORTH]]\" \' ' % (zone,cm)

    if hemi == "S":
        inD = inD + '-PmapProjection=EPSG:327%02d ' % zone
    else:
        inD = inD + '-PmapProjection=EPSG:326%02d ' % zone
    
    if extDEM != None:
        inD = inD + ' -PdemName=\"External DEM\" -PexternalDEMFile=%s -PexternalDEMNoDataValue=0 ' % extDEM
    else:
        inD = inD + ' -PdemName=\"SRTM 1Sec HGT\" '
    cmd = baseSNAP + tcFlag + out + inD
    print 'Applying Terrain Correction -- This will take some time'
    print cmd
    os.system(cmd)
    return '%s' % inData.replace('.dim','_TC.dim')

# Get the UTM zone, central meridian, and hemisphere 
def getZone(inData):
    temp = inData.replace('.zip','.SAFE')
    if not os.path.isdir(temp):
        cmd = "unzip %s" % inData
        print cmd
        os.system(cmd)
    back = os.getcwd()
    os.chdir(temp)
    os.chdir('annotation')

    paths = os.listdir(os.getcwd())
    for temp in paths:
        if os.path.isfile(temp):
            toread = temp
            break
    f = open(toread,'r')

    min_lon = 180
    max_lon = -180    
    for line in f:
        m = re.search('<longitude>(.+?)</longitude>', line)
        if m:
            lon = float(m.group(1))
            if lon > max_lon:
                max_lon = lon
	    if lon < min_lon:
                min_lon = lon
    f.close
    print "Found max_lon of %s" % max_lon
    print "Found min_lon of %s" % min_lon
    center_lon = (float(min_lon) + float(max_lon)) / 2.0
    print "Found center_lon of %s" % center_lon
    zone = int(float(lon)+180)/6 + 1
    print "Found UTM zone of %s" % zone
    central_meridian = (zone-1)*6-180+3
    print "Found central meridian of %s" % central_meridian

    f = open(toread,'r')

    min_lat = 180
    max_lat = -180
    for line in f:
        m = re.search('<latitude>(.+?)</latitude>', line)
        if m:
            lat = float(m.group(1))
            if lat > max_lat:
                max_lat = lat
            if lat < min_lat:
                min_lat = lat
    f.close
    print "Found max_lat of %s" % max_lat
    print "Found min_lat of %s" % min_lat
    center_lat = (float(min_lat) + float(max_lat)) / 2.0
    print "Found center_lat of %s" % center_lat
    if (center_lat < 0):
        hemi = "S";
    else:
        hemi = "N";
    print "Found hemisphere of %s" % hemi

    os.chdir(back)
#    cmd = "rm -r %s" % inData.replace('.zip','.SAFE')
#    os.system(cmd)
    return (zone, central_meridian, hemi)

###################################################################
#  Start of main executable code
###################################################################
start = datetime.datetime.now()
td2 = tempDir(tdir,baseGran)
print  "Processing in directory %s" % td2

tempFile = infile
(zone, cm, hemi) = getZone(infile)

#
# Apply precise orbit 
#
obOut = applyOrbit(infile,td2,baseGran)
print obOut
print 'Time to fix orbit: ',
print(timestamp(datetime.datetime.now())-timestamp(start))
tempFile = obOut

#
# Apply calibration
#
lasttime = datetime.datetime.now()
calOut = applyCal(tempFile,td2)
print calOut
print 'Time to calibrate: ',
print(timestamp(datetime.datetime.now())-timestamp(lasttime))

#
# Apply terrain flattening
#
lasttime = datetime.datetime.now()
tfOut = applyTF(calOut,td2)
print tfOut
print 'Time to terrain flatten: ',
print(timestamp(datetime.datetime.now())-timestamp(lasttime))

#
# Apply terrain correction
#
lasttime = datetime.datetime.now()
tcOut = applyTC(tfOut,td2,zone,cm,hemi)
print tcOut
print 'Time to terrain correct: ',
print(timestamp(datetime.datetime.now())-timestamp(lasttime))

#
# Reformat files into GeoTIFFs
#
os.chdir('%s/%s' % (td2,tcOut.replace('.dim','.data')))
lasttime = datetime.datetime.now()

if os.path.isfile('Gamma0_VV.img'):
    print 'Writing output file GVV'
    cmd = 'gdal_translate -of GTiff Gamma0_VV.img Gamma0_VV_utm.tif'
    print cmd
    os.system(cmd)
    cmd = 'mv Gamma0_VV_utm.tif ../%s' % tcOut.replace('.dim','_GVV.tif')
    print cmd
    os.system(cmd)

if os.path.isfile('Gamma0_VH.img'):
    print 'Writing output file GVH'
    cmd = 'gdal_translate -of GTiff Gamma0_VH.img Gamma0_VH_utm.tif'
    print cmd
    os.system(cmd)
    cmd = 'mv Gamma0_VH_utm.tif ../%s' % tcOut.replace('.dim','_GVH.tif')
    print cmd
    os.system(cmd)

if os.path.isfile('Gamma0_HH.img'):
    print 'Writing output file GHH'
    cmd = 'gdal_translate -of GTiff Gamma0_HH.img Gamma0_HH_utm.tif'
    print cmd
    os.system(cmd)
    cmd = 'mv Gamma0_HH_utm.tif ../%s' % tcOut.replace('.dim','_GHH.tif')
    print cmd
    os.system(cmd)

if os.path.isfile('Gamma0_HV.img'):
    print 'Writing output file GHV'
    cmd = 'gdal_translate -of GTiff Gamma0_HV.img Gamma0_HV_utm.tif'
    print cmd
    os.system(cmd)
    cmd = 'mv Gamma0_HV_utm.tif ../%s' % tcOut.replace('.dim','_GHV.tif')
    print cmd
    os.system(cmd)

if os.path.isfile('projectedLocalIncidenceAngle.img'):
    print 'Writing output file INC'
    cmd = 'gdal_translate -of GTiff projectedLocalIncidenceAngle.img projectedLocalIncidenceAngle_utm.tif'
    print cmd
    os.system(cmd)
    cmd = 'mv projectedLocalIncidenceAngle_utm.tif ../%s' % tcOut.replace('.dim','_INC.tif')
    print cmd
    os.system(cmd)

if os.path.isfile('elevation.img'):
    print 'Writing output file DEM'
    cmd = 'gdal_translate -of GTiff elevation.img elevation_utm.tif'
    print cmd
    os.system(cmd)
    cmd = 'mv elevation_utm.tif ../%s' % tcOut.replace('.dim','_DEM.tif')
    print cmd
    os.system(cmd)

print 'Time to export: ',
print(timestamp(datetime.datetime.now())-timestamp(lasttime))

print 'Total processing time: ',
print(timestamp(datetime.datetime.now())-timestamp(start))

if cleanTemp == True:
    cmd = 'cd %s; rm -r *OB.* *CAL.* *TF.*' % td2
    print cmd
    os.system(cmd)
