from osgeo import gdal, ogr, osr
import os
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

if not os.path.exists('years'):
    os.mkdir('years')

src_filename = r"C:\Users\ncate\Desktop\FNF_LCMS_sieved_UTM.tif"
src_ds = gdal.Open(src_filename)
prj = src_ds.GetProjection()
srs = osr.SpatialReference(wkt=prj)
drv = ogr.GetDriverByName("ESRI Shapefile")

year = 1985
num_bands = src_ds.RasterCount

for band in range(num_bands):
    srcband = src_ds.GetRasterBand(band+1)
    dst_layername = "POLYGONIZED_LCMS_"+str(year)
    dst_ds = drv.CreateDataSource('years//'+dst_layername + ".shp" )
    dst_layer = dst_ds.CreateLayer(dst_layername, srs=srs)
    gdal.Polygonize(srcband, None, dst_layer, -1)
    
    year += 1
    dst_ds = None
    dst_layer = None
    
shps = os.listdir('years')
shps = [i for i in shps if i.endswith('shp')]

for shp in shps:
    dataSource = drv.Open('years//'+shp, 1)
    layer = dataSource.GetLayer()
    new_field = ogr.FieldDefn("Area", ogr.OFTReal)
    new_field.SetWidth(32)
    new_field.SetPrecision(2) #added line to set precision
    layer.CreateField(new_field)
    layer.CreateField("loss_year")
    
    for feature in layer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea()
        if (area>3320677170) or (area==900):
            layer.DeleteFeature(feature.GetFID())
        else:
            feature.SetField("Area", area)
            feature.SetField("loss_year", year)
            layer.SetFeature(feature)
        year+=1
    dataSource = None 
