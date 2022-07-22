import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import TimestampType, DateType, IntegerType
from pyspark.sql.functions import (year, month, dayofmonth, hour,
                                   weekofyear, dayofweek, date_format,
                                   monotonically_increasing_id)


config = configparser.ConfigParser()
config.read('dl.cfg')

# uncomment the next two lines if you would like to use config files
#os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
#os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']

#uncomment the next two lines if you would like to add your aws key id and access key directly
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''

def create_spark_session():
    '''
    Creates a Spark Session.
    '''
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    '''
    Process song data to create songs and artist table

    Input:
        spark - spark session object
        input_data - S3 bucket with data to input
        output_data - S3 bucket with data to output to
    '''
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/A/A/*/*.json') # this is a subset of the data
    #uncomment line below if you would like to use full dataset and add comment to line above
    #song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')

    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(["song_id", "title", "artist_id", "year", "duration"]).dropDuplicates()

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(os.path.join(output_data, 'songs'), partitionBy=['year', 'artist_id'], mode = "overwrite")

    # extract columns to create artists table
    artists_table = df.select('artist_id', 'artist_name', 'artist_location',
                              'artist_latitude', 'artist_longitude') \
                      .withColumnRenamed('artist_name', 'name') \
                      .withColumnRenamed('artist_location', 'location') \
                      .withColumnRenamed('artist_latitude', 'latitude') \
                      .withColumnRenamed('artist_longitude', 'longitude') \
                      .dropDuplicates()

    # write artists table to parquet files
    artists_table.write.parquet(os.path.join(output_data, 'artist'),
                                mode = "overwrite")


def process_log_data(spark, input_data, output_data):
    '''
    Process log data to create users, time and songplays table

    Input:
        spark - spark session object
        input_data - S3 bucket with data to input
        output_data - S3 bucket with data to output to
    '''
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log_data/*/*/*events.json')

    # read log data file
    df = spark.read.json(log_data)

    # filter by actions for song plays
    df = df.where(df.page == 'NextSong')

    # extract columns for users table
    users_table =df.selectExpr(['userId as user_id', 'firstName as first_name',
                                'lastName as last_name', 'gender', 'level']).dropDuplicates()

    # write users table to parquet files
    users_table.write.parquet(os.path.join(output_data, 'users'),
                              mode = "overwrite")

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: datetime.fromtimestamp(x / 1000),
                        TimestampType())
    df = df.withColumn('timestamp', get_timestamp(df.ts))

    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x/1000),
                       TimestampType())
    df = df.withColumn('datetime', get_datetime(df.ts))

    # extract columns to create time table
    time_table = df.select('datetime') \
                   .withColumn('start_time', df.datetime) \
                   .withColumn('hour', hour('datetime')) \
                   .withColumn('day', dayofmonth('datetime')) \
                   .withColumn('week', weekofyear('datetime')) \
                   .withColumn('month', month('datetime')) \
                   .withColumn('year', year('datetime')) \
                   .withColumn('weekday', dayofweek('datetime')) \
                   .dropDuplicates()

    # write time table to parquet files partitioned by year and month
    time_table.write.parquet(os.path.join(output_data, 'time'),
                             partitionBy=['year', 'month'],
                             mode = "overwrite")

    # read in song data to use for songplays table
    song_df = spark.read.json(os.path.join(input_data, 'song_data/A/A/*/*.json')) # this is a subset of the data
    #uncomment line below if you would like to use full dataset and add comment to line above
    #song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')

    # extract columns from joined song and log datasets to create songplays table
    songplays_table = df.join(
    song_df,
    song_df.artist_name == df.artist, 'inner') \
    .distinct() \
    .select(
        col('timestamp'),
        col('userId'),
        col('level'),
        col('sessionId'),
        col('location'),
        col('userAgent'),
        col('song_id'),
        col('artist_id')) \
    .withColumn('songplay_id', monotonically_increasing_id()) \
    .withColumn('month', month('timestamp')) \
    .withColumn('year', year('timestamp')) \
    .dropDuplicates()


    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.parquet(os.path.join(output_data, 'songplays'),
                                  partitionBy=['year', 'month'],
                                  mode = "overwrite")


def main():
    """
    Perform the following roles:
    1.) Get or create a spark session.
    1.) Read the song and log data from s3.
    2.) Take the data and transform them to tables
    3.) Write tables to parquet files.
    4.) Load the parquet files on s3.
    """
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://lake-bucket-sparkify/" #make sure to replace this with bucket created in aws S3

    process_song_data(spark, input_data, output_data)
    process_log_data(spark, input_data, output_data)

if __name__ == "__main__":
    main()
