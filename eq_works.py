import os
os.environ["PYSPARK_PYTHON"]="/home/sahmadov/anaconda3/bin/python"
from pyspark import SparkContext, SQLContext
from scipy.spatial import distance
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
import math

def load_csv(fname="data/DataSample.csv"):
    sqlContext = SQLContext(sc)
    df = sqlContext.read.csv(header="true", path = fname)
    return df

def remove_outliers(data, poi_stats):

    print("Before: ", data.count())
    mean, stdev = poi_stats
    r = data.filter(data.distance.cast(DoubleType()) < mean +2*stdev)
    r = r.filter(r.distance.cast(DoubleType()) > mean -2*stdev)
    print("After: ", r.count())
    return r

def calc_score(data):
    data = data.withColumn('dist', data['distance'].cast(DoubleType()))

    sm, cnt, mx,mn = data.agg(F.sum('dist'), F.count('dist'), F.max('dist'),F.min('dist')).collect()[0]

    score = (sm-cnt*mn)/(cnt * (mx-mn))
    return 20*score-10

if __name__=='__main__':
    # Start spark app
    sc = SparkContext(appName="Eqwork")
    # Load csv file to dataframe

    data = load_csv()
    poi = load_csv('data/POIList.csv')

    # Task 0: Cleanup, remove duplicates from dataframe (extra add show duplicates)
    print("Number of total rows before dropping duplicates: {}".format(data.count()))
    data_count = data.groupBy(' TimeSt','Latitude','Longitude').count()
    data = data.join(data_count,[' TimeSt','Latitude','Longitude']).filter('count=1').drop('count')
    # data = data.drop_duplicates([' TimeSt','Latitude', 'Longitude'])
    print("Number of total rows after dropping duplicates: {}".format(data.count()))

    # Task 1: Label
    # Cross-join requests with poi
    requests_poi = data.crossJoin(poi.withColumnRenamed(' Latitude', 'Lat').withColumnRenamed('Longitude', 'Long'))

    # UDF distance function
    distance_udf = F.udf(lambda x, y: float(distance.euclidean(x, y)))
    # Calculate distances between requests and poi
    distances = requests_poi.withColumn('distance',
                                        distance_udf(F.array(F.col('Latitude').cast(DoubleType()), F.col('Longitude').cast(DoubleType())),
                                   F.array(F.col('Lat').cast(DoubleType()), F.col('Long').cast(DoubleType()))))

    min_idx = distances.groupBy('_ID').agg(F.min('distance').alias('min_d')).withColumnRenamed('_ID', 'id').alias('min_label')
    labeled_data = distances.join(min_idx, (F.col('min_label.id') == F.col('_ID')) & (F.col('min_label.min_d') == F.col('distance')))
    labeled_data = labeled_data.drop_duplicates(['_ID']).drop('id')

    # Task 2 Analysis
    # Avg and stdev of each poi
    poi_stats = labeled_data.groupBy('POIID').agg(F.stddev('distance').alias('avg'), F.avg('distance').alias('stdev'))
    # Radius of circle will be the furthest point
    radius = labeled_data.groupBy('POIID').agg(F.max('distance').alias('radius'))
    cnt = labeled_data.groupBy('POIID').count()
    # Calculate area of circle
    density_udf = F.udf(lambda x,y: float(x/(math.pi*y**2)))
    joined = radius.join(cnt, 'POIID')
    # Density, radius of each POI
    den_rad = joined.withColumn('density', density_udf(F.col('count').cast(DoubleType()), F.col('radius').cast(DoubleType())))

    # Bonus
    # Remove outliers
    sorted_data = labeled_data.orderBy('POIID','distance', ascending=False).select('POIID', 'distance')
    po1 = sorted_data.filter(sorted_data.POIID=='POI1')
    po1_stat = poi_stats.filter(po1.POIID=='POI1').select('avg','stdev').collect()[0]
    po3 = sorted_data.filter(sorted_data.POIID=='POI3')
    po3_stat = poi_stats.filter(po1.POIID=='POI3').select('avg','stdev').collect()[0]
    po4 = sorted_data.filter(sorted_data.POIID=='POI4')
    po4_stat = poi_stats.filter(po1.POIID=='POI4').select('avg','stdev').collect()[0]

    po1_clean = remove_outliers(po1, po1_stat)
    po3_clean = remove_outliers(po3, po3_stat)
    po4_clean = remove_outliers(po4, po4_stat)
    # Print scores for each POIs
    print("PO1 score: {}\nPO3 score: {}\nPO4 score: {}\n".format(calc_score(po1_clean), calc_score(po3_clean), calc_score(po4_clean)))
