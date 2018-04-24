
## Using Spark's ML library for Machine Learning

Through this project, I will guide you around the use of Apache Spark's Machine Learning library. In this exercise, we'll be working with the 'flights' database, which contains information on flights in the US including departure and arrival airport, delays information, airline, among others.  

Our objective will be to predict a flight's departure delay with available data. To do this, we will use different models, tuned through a Parameter map with which we can try different parameter values and finally run through a 'Magic Loop' to try both of them out. Feature engineering, if required, will be done with a Pipeline object.

Dataset used may be downloaded from here: https://www.kaggle.com/usdot/flight-delays  
A glossary for the variables can be found here: https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time

### Creating a Spark Session
In this project will be working with Spark in 'local mode', this means we will only be using the resources in our computer for data processing, instead of using a cluster of machines, as Spark is more commonly used.


```python
# Added libraries.
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml import Transformer
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Bucketizer, RFormula
import pandas as pd
import numpy as np
import time
pd.set_option('display.max_columns', 40)

spark = SparkSession.builder.master("local[7]")\
.config("spark.executor.memory", "10G")\
.config("spark.driver.memory", "10G")\
.config("spark.memory.fraction", ".8")\
.getOrCreate() 
# Using * within the box in local means we grant Spark access to all threads on our computer, this may be changed.
```

### Exploratory Analysis
After this, we'll load the database into Spark and we'll take a first look at what we have:


```python
flights = spark.read.csv('/home/jovyan/work/flights.csv', header =True) 
# read.csv has an argument 'inferSchema' which will try to parse columns to their 'appropiate' type. However,
# here we avoid it as it produces unwanted results, such as parsing time "0005" to 5, which means we lose information
# As it is, it will parse all columns to type 'string'
```

First, we will take a look at what data we have:


```python
num_rows = flights.count()
num_rows
```




    5819079



So we have information from 5,819,079 flights.

Next, we will look at our data and check if we need to make any changes:


```python
flights.limit(10).toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>DAY</th>
      <th>DAY_OF_WEEK</th>
      <th>AIRLINE</th>
      <th>FLIGHT_NUMBER</th>
      <th>TAIL_NUMBER</th>
      <th>ORIGIN_AIRPORT</th>
      <th>DESTINATION_AIRPORT</th>
      <th>SCHEDULED_DEPARTURE</th>
      <th>DEPARTURE_TIME</th>
      <th>DEPARTURE_DELAY</th>
      <th>TAXI_OUT</th>
      <th>WHEELS_OFF</th>
      <th>SCHEDULED_TIME</th>
      <th>ELAPSED_TIME</th>
      <th>AIR_TIME</th>
      <th>DISTANCE</th>
      <th>WHEELS_ON</th>
      <th>TAXI_IN</th>
      <th>SCHEDULED_ARRIVAL</th>
      <th>ARRIVAL_TIME</th>
      <th>ARRIVAL_DELAY</th>
      <th>DIVERTED</th>
      <th>CANCELLED</th>
      <th>CANCELLATION_REASON</th>
      <th>AIR_SYSTEM_DELAY</th>
      <th>SECURITY_DELAY</th>
      <th>AIRLINE_DELAY</th>
      <th>LATE_AIRCRAFT_DELAY</th>
      <th>WEATHER_DELAY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>AS</td>
      <td>98</td>
      <td>N407AS</td>
      <td>ANC</td>
      <td>SEA</td>
      <td>0005</td>
      <td>2354</td>
      <td>-11</td>
      <td>21</td>
      <td>0015</td>
      <td>205</td>
      <td>194</td>
      <td>169</td>
      <td>1448</td>
      <td>0404</td>
      <td>4</td>
      <td>0430</td>
      <td>0408</td>
      <td>-22</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>AA</td>
      <td>2336</td>
      <td>N3KUAA</td>
      <td>LAX</td>
      <td>PBI</td>
      <td>0010</td>
      <td>0002</td>
      <td>-8</td>
      <td>12</td>
      <td>0014</td>
      <td>280</td>
      <td>279</td>
      <td>263</td>
      <td>2330</td>
      <td>0737</td>
      <td>4</td>
      <td>0750</td>
      <td>0741</td>
      <td>-9</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>US</td>
      <td>840</td>
      <td>N171US</td>
      <td>SFO</td>
      <td>CLT</td>
      <td>0020</td>
      <td>0018</td>
      <td>-2</td>
      <td>16</td>
      <td>0034</td>
      <td>286</td>
      <td>293</td>
      <td>266</td>
      <td>2296</td>
      <td>0800</td>
      <td>11</td>
      <td>0806</td>
      <td>0811</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>AA</td>
      <td>258</td>
      <td>N3HYAA</td>
      <td>LAX</td>
      <td>MIA</td>
      <td>0020</td>
      <td>0015</td>
      <td>-5</td>
      <td>15</td>
      <td>0030</td>
      <td>285</td>
      <td>281</td>
      <td>258</td>
      <td>2342</td>
      <td>0748</td>
      <td>8</td>
      <td>0805</td>
      <td>0756</td>
      <td>-9</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>AS</td>
      <td>135</td>
      <td>N527AS</td>
      <td>SEA</td>
      <td>ANC</td>
      <td>0025</td>
      <td>0024</td>
      <td>-1</td>
      <td>11</td>
      <td>0035</td>
      <td>235</td>
      <td>215</td>
      <td>199</td>
      <td>1448</td>
      <td>0254</td>
      <td>5</td>
      <td>0320</td>
      <td>0259</td>
      <td>-21</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>DL</td>
      <td>806</td>
      <td>N3730B</td>
      <td>SFO</td>
      <td>MSP</td>
      <td>0025</td>
      <td>0020</td>
      <td>-5</td>
      <td>18</td>
      <td>0038</td>
      <td>217</td>
      <td>230</td>
      <td>206</td>
      <td>1589</td>
      <td>0604</td>
      <td>6</td>
      <td>0602</td>
      <td>0610</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>NK</td>
      <td>612</td>
      <td>N635NK</td>
      <td>LAS</td>
      <td>MSP</td>
      <td>0025</td>
      <td>0019</td>
      <td>-6</td>
      <td>11</td>
      <td>0030</td>
      <td>181</td>
      <td>170</td>
      <td>154</td>
      <td>1299</td>
      <td>0504</td>
      <td>5</td>
      <td>0526</td>
      <td>0509</td>
      <td>-17</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>US</td>
      <td>2013</td>
      <td>N584UW</td>
      <td>LAX</td>
      <td>CLT</td>
      <td>0030</td>
      <td>0044</td>
      <td>14</td>
      <td>13</td>
      <td>0057</td>
      <td>273</td>
      <td>249</td>
      <td>228</td>
      <td>2125</td>
      <td>0745</td>
      <td>8</td>
      <td>0803</td>
      <td>0753</td>
      <td>-10</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>AA</td>
      <td>1112</td>
      <td>N3LAAA</td>
      <td>SFO</td>
      <td>DFW</td>
      <td>0030</td>
      <td>0019</td>
      <td>-11</td>
      <td>17</td>
      <td>0036</td>
      <td>195</td>
      <td>193</td>
      <td>173</td>
      <td>1464</td>
      <td>0529</td>
      <td>3</td>
      <td>0545</td>
      <td>0532</td>
      <td>-13</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>DL</td>
      <td>1173</td>
      <td>N826DN</td>
      <td>LAS</td>
      <td>ATL</td>
      <td>0030</td>
      <td>0033</td>
      <td>3</td>
      <td>12</td>
      <td>0045</td>
      <td>221</td>
      <td>203</td>
      <td>186</td>
      <td>1747</td>
      <td>0651</td>
      <td>5</td>
      <td>0711</td>
      <td>0656</td>
      <td>-15</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for null values, with guidance from:
# https://stackoverflow.com/questions/44627386/how-to-find-count-of-null-and-nan-values-for-each-column-in-a-pyspark-dataframe
flights.select([(count(when(isnan(c) | col(c).isNull(), c))/num_rows).alias(c) for c in flights.columns]).toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>MONTH</th>
      <th>DAY</th>
      <th>DAY_OF_WEEK</th>
      <th>AIRLINE</th>
      <th>FLIGHT_NUMBER</th>
      <th>TAIL_NUMBER</th>
      <th>ORIGIN_AIRPORT</th>
      <th>DESTINATION_AIRPORT</th>
      <th>SCHEDULED_DEPARTURE</th>
      <th>DEPARTURE_TIME</th>
      <th>DEPARTURE_DELAY</th>
      <th>TAXI_OUT</th>
      <th>WHEELS_OFF</th>
      <th>SCHEDULED_TIME</th>
      <th>ELAPSED_TIME</th>
      <th>AIR_TIME</th>
      <th>DISTANCE</th>
      <th>WHEELS_ON</th>
      <th>TAXI_IN</th>
      <th>SCHEDULED_ARRIVAL</th>
      <th>ARRIVAL_TIME</th>
      <th>ARRIVAL_DELAY</th>
      <th>DIVERTED</th>
      <th>CANCELLED</th>
      <th>CANCELLATION_REASON</th>
      <th>AIR_SYSTEM_DELAY</th>
      <th>SECURITY_DELAY</th>
      <th>AIRLINE_DELAY</th>
      <th>LATE_AIRCRAFT_DELAY</th>
      <th>WEATHER_DELAY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00253</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014805</td>
      <td>0.014805</td>
      <td>0.015303</td>
      <td>0.015303</td>
      <td>0.000001</td>
      <td>0.018056</td>
      <td>0.018056</td>
      <td>0.0</td>
      <td>0.015898</td>
      <td>0.015898</td>
      <td>0.0</td>
      <td>0.015898</td>
      <td>0.018056</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.984554</td>
      <td>0.81725</td>
      <td>0.81725</td>
      <td>0.81725</td>
      <td>0.81725</td>
      <td>0.81725</td>
    </tr>
  </tbody>
</table>
</div>




```python
flights.withColumn("delay", flights["DEPARTURE_DELAY"].cast(DoubleType())).describe("delay").show()
```

    +-------+------------------+
    |summary|             delay|
    +-------+------------------+
    |  count|           5732926|
    |   mean| 9.370158275198389|
    | stddev|37.080942496787124|
    |    min|             -82.0|
    |    max|            1988.0|
    +-------+------------------+
    


### Data Cleaning and Feature Engineering

From the latter, we know the following changes can be made:
- Parse to integer:
    + Departure Delay (our response variable)
    + Distance
    + Arrival delay
    + Taxi Out
    + Scheduled time
    + Elapsed time
    + Air time
    + Taxi in
    + Diverted
    + Cancelled  
    
 
- Parse to time - will separate into two columns (hour and minute) which will probably be more useful when modeling -
   + Scheduled Departure
   + Departure time
   + Wheels-off (When the plane actually left the ground)
   + Wheels-on
   + Scheduled arrival
   + Arrival time
   + Scheduled Arrival  
   
  
   
- Columns to delete - too many missing values -
    + Cancellation Reason
    + Air System Delay
    + Security Delay
    + Airline Delay
    + Late Aircraft Delay
    + Weather Delay
    + Year (we know all our data is from 2015, so this column has no variance and thus is useless)  
    

- Doubtful variables - may be redundant to other information (high correlation) & data leakage - 
    + Taxi in
    + Taxi out
    + Air time
    + Wheels on
    + Wheels off   
    
I will also remove all remaining rows with Null values, which will represent 1.8% of the data at the most, which is insignificant.  

It is also important to consider that some of these variables may produce **'data leakage'** as they could only be known after or along with the fact, for example: DEPARTURE_TIME, WHEELS_OFF, among others. We need to remove these variables for the same flight. 

** Since we want to fit all these steps in a pipeline, we need to creates a series of classes that have a transform method within them. These are outlined next **



```python
# To parse strings into integers
class ParserDouble(Transformer):  

    def __init__(self, columns=[None]):
        self.columns = columns

    def _transform(self, df):
        for col in self.columns:
            df = df.withColumn(col, df[col].cast(DoubleType()))
        self = df
        return self
```


```python
# To separate time into hours + minutes
class getHour(Transformer):  

    def __init__(self, columns=[None]):
        self.columns = columns

    def _transform(self, df):
        for col in self.columns:
            df = df.withColumn(col+"_HOUR", substring(col,0,2))
        self = df
        return self
```


```python
# To remove unwanted columns
class rm_cols(Transformer):  

    def __init__(self, columns=[None]):
        self.columns = columns

    def _transform(self, df):
        for col in self.columns:
            df = df.drop(col)
        self = df
        return self
```


```python
# To remove rows with Null values
class rm_nulls(Transformer):  
    def _transform(self, df):
        self = df.na.drop(how="any")
        return self
```

It's important to note some variables aren't relevant for out prediction, such as those that happened after flight departure, eg. Arrival time at destination for flight of interest. However, this same variable might be relevant for our same flight's arrival previous to its departure time. We might be even interested in generating a 'time at airport' variable, which considers the time from landing to departure. 

We can do this by lagging rows for every flight tail number. This way we can 'track' the specific airplanes.

We work on this next: 


```python
# Lag relevant columns from previous flights
class lagger(Transformer):  

    def __init__(self, columns=[None], grouper = "TAIL_NUMBER"):
        self.columns = columns
        self.grouper = grouper
        
    def _transform(self, df):
        for col in self.columns:
            df = df.withColumn(col+"_PREV",lag(flights[col])
                                 .over(Window.partitionBy(self.grouper).orderBy(self.grouper)))
        self = df
        return self
```


```python
# Create column with time difference between events
class t_diff(Transformer):  
    def __init__(self, col_name = None, columns=[None]):
        self.columns = columns
        self.col_name = col_name
    def _transform(self, df):
        self = df.withColumn(self.col_name, when(df["DAY_OF_WEEK"] > df["DAY_OF_WEEK_PREV"], 
                                                2400 + df[self.columns[0]] - df[self.columns[1]]).
                             otherwise(df[self.columns[0]] - df[self.columns[1]]))
        #self = df.withColumn(self.col_name, df[self.columns[0]] - df[self.columns[1]])
        return self
```

Afterwards, we configure all the steps that will be used in out pipeline, which include removing columns with high number of null values and those that represent data leakage, lagging variables to get information on the airplanes previous flight and creating new variables, such as the time at airport variable. Finally, for use in our models, we have to 'One hot encode' our categorical variables.


```python
# Remove columns with significantly high NAs
rm_cols_nas = rm_cols(["CANCELLATION_REASON","AIR_SYSTEM_DELAY","SECURITY_DELAY","AIRLINE_DELAY","LATE_AIRCRAFT_DELAY",
                    "WEATHER_DELAY"])

# Remove columns not useful to analysis or that present data leakage risk
rm_cols_unnecessary = rm_cols(["WHEELS_OFF", "TAXI_OUT", "CANCELLED", "DIVERTED", "WHEELS_ON",
                              "AIR_TIME", "TAXI_IN", "YEAR", "ELAPSED_TIME", "SCHEDULED_TIME", "AIR_TIME"])

# Remove rows with null values
rm_nulls1 = rm_nulls()

# Lag columns from the flights previous trip (its grouped by TAIL_NUMBER)
lagger1 = lagger(columns = ["SCHEDULED_ARRIVAL", "DEPARTURE_TIME", "DEPARTURE_DELAY", "ARRIVAL_DELAY",
                           "DAY_OF_WEEK"], grouper = "TAIL_NUMBER")

# Remove columns not useful to analysis or that present data leakage risk
rm_cols_leakage= rm_cols(["SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY", "DEPARTURE_TIME"])

# Convert to integer
double_maker = ParserDouble(["DEPARTURE_DELAY", "DISTANCE", "ARRIVAL_DELAY_PREV"])

# Create time at airport variable (it doesn't really produce a time value as such, but represents a time 'proxy')
t_at_airport = t_diff(col_name = "SCHEDULED_TIME_AT_AIRPORT", columns = ["SCHEDULED_DEPARTURE","SCHEDULED_ARRIVAL_PREV"])

# Separate time columns into hour-minute columns
h_m_sep = getHour(["SCHEDULED_DEPARTURE"])

# Final parse of strings to ints for relevant variables
double_maker2 = ParserDouble(["DEPARTURE_DELAY_PREV","SCHEDULED_DEPARTURE_HOUR", "DAY"])

bucketizer_day = Bucketizer(splits=[-float("inf"), 7, 14, 21, float("inf")], inputCol="DAY", outputCol="WEEK")
bucketizer_time = Bucketizer(splits=[-float("inf"), 4, 8, 12, 16, 20, float("inf")], inputCol="SCHEDULED_DEPARTURE_HOUR",
                             outputCol="SCHED_DEPARTURE_PERIOD")

# Finally, we need to 'One-hot encode' our categorical variables for use in the models
cat_vars = ["AIRLINE", "ORIGIN_AIRPORT", "MONTH", "DAY_OF_WEEK", "WEEK", "SCHED_DEPARTURE_PERIOD"]

indexers = [StringIndexer(inputCol=column, outputCol=column+"_INDEX") for column in cat_vars ]
hot_encoders = [OneHotEncoder(inputCol=column+"_INDEX", outputCol=column+"_HOT") for column in cat_vars ]

# Remove all unnecessary variables remaining
rm_rem= rm_cols(["SCHEDULED_ARRIVAL_PREV", "SCHEDULED_DEPARTURE", "ARRIVAL_TIME_PREV", "DEPARTURE_TIME_PREV", 
                 "AIRLINE_INDEX", 'TAIL_NUMBER','ORIGIN_AIRPORT_INDEX',"AIRLINE",'FLIGHT_NUMBER', 'TAIL_NUMBER',
                 'ORIGIN_AIRPORT','DESTINATION_AIRPORT', "DAY_OF_WEEK_PREV", "MONTH", "DAY", "DAY_OF_WEEK",
                 "MONTH_INDEX", "WEEK_INDEX","DAY_OF_WEEK_INDEX", "WEEK", "SCHEDULED_ARRIVAL_PREV",
                 "SCHEDULED_DEPARTURE_HOUR", "SCHED_DEPARTURE_PERIOD_INDEX", "SCHED_DEPARTURE_PERIOD"])
```


```python
data_prepare = Pipeline(stages = [rm_cols_nas, rm_cols_unnecessary, rm_nulls1, lagger1, rm_cols_leakage, double_maker, 
                                  rm_nulls1, t_at_airport, h_m_sep, double_maker2, bucketizer_day, bucketizer_time] + 
                        indexers + hot_encoders + [rm_rem])
```


```python
clean_flights = data_prepare.fit(flights).transform(flights)
```

And now, our data is ready to be run through models!


```python
clean_flights.limit(10).show()
```

    +---------------+--------+--------------------+------------------+-------------------------+--------------+------------------+---------------+---------------+---------+--------------------------+
    |DEPARTURE_DELAY|DISTANCE|DEPARTURE_DELAY_PREV|ARRIVAL_DELAY_PREV|SCHEDULED_TIME_AT_AIRPORT|   AIRLINE_HOT|ORIGIN_AIRPORT_HOT|      MONTH_HOT|DAY_OF_WEEK_HOT| WEEK_HOT|SCHED_DEPARTURE_PERIOD_HOT|
    +---------------+--------+--------------------+------------------+-------------------------+--------------+------------------+---------------+---------------+---------+--------------------------+
    |           82.0|   930.0|                -8.0|             -20.0|                    -86.0|(13,[8],[1.0])|  (627,[14],[1.0])|(11,[10],[1.0])|  (6,[0],[1.0])|(3,[],[])|             (5,[0],[1.0])|
    |           81.0|   930.0|                82.0|              89.0|                     46.0|(13,[8],[1.0])|   (627,[9],[1.0])|(11,[10],[1.0])|  (6,[0],[1.0])|(3,[],[])|             (5,[1],[1.0])|
    |           20.0|   650.0|                81.0|              70.0|                    296.0|(13,[8],[1.0])|  (627,[14],[1.0])|(11,[10],[1.0])|  (6,[0],[1.0])|(3,[],[])|             (5,[4],[1.0])|
    |           -2.0|   650.0|                20.0|               1.0|                   1017.0|(13,[8],[1.0])|  (627,[24],[1.0])|(11,[10],[1.0])|  (6,[1],[1.0])|(3,[],[])|             (5,[0],[1.0])|
    |           -4.0|   328.0|                -2.0|             -13.0|                    109.0|(13,[8],[1.0])|  (627,[14],[1.0])|(11,[10],[1.0])|  (6,[1],[1.0])|(3,[],[])|             (5,[0],[1.0])|
    |            0.0|   328.0|                -4.0|              -7.0|                     45.0|(13,[8],[1.0])|  (627,[60],[1.0])|(11,[10],[1.0])|  (6,[1],[1.0])|(3,[],[])|             (5,[1],[1.0])|
    |           13.0|   130.0|                 0.0|              -3.0|                    100.0|(13,[8],[1.0])|  (627,[14],[1.0])|(11,[10],[1.0])|  (6,[1],[1.0])|(3,[],[])|             (5,[2],[1.0])|
    |           -7.0|   130.0|                13.0|              14.0|                     53.0|(13,[8],[1.0])|  (627,[43],[1.0])|(11,[10],[1.0])|  (6,[1],[1.0])|(3,[],[])|             (5,[2],[1.0])|
    |            1.0|   650.0|                -7.0|             -16.0|                    108.0|(13,[8],[1.0])|  (627,[14],[1.0])|(11,[10],[1.0])|  (6,[1],[1.0])|(3,[],[])|             (5,[4],[1.0])|
    |          148.0|  1013.0|                 1.0|              -8.0|                    -78.0|(13,[8],[1.0])|  (627,[24],[1.0])|(11,[10],[1.0])|  (6,[1],[1.0])|(3,[],[])|             (5,[4],[1.0])|
    +---------------+--------+--------------------+------------------+-------------------------+--------------+------------------+---------------+---------------+---------+--------------------------+
    



```python
from tabulate import tabulate
```

And just to verify how much data we lost from Null values:


```python
(1 - clean_flights.count()/flights.count()) * 100
```




    1.8897664046148899



We only lost 1.9% of our data to null values.  

### Running models and tuning  

As stated at the beggining, our objetive is to find the best model to predict a flight's departure delay. This means we will try to predict a number and, thus, will need a regression model. In this case, we will work with two models: the standard Linear Regression Model and the GBT (Gradient Boosted Trees) Regressor Model.  

To start with, we have to divide our data in a training and a test sample, which we'll do 70% and 30% respectively. We can do a random split as we don't have sequential data to worry about


```python
(train_flights, test_flights) = clean_flights.randomSplit([0.005, 0.995])
```


```python
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.param import Params
```

Now we'll set up a 'magic loop' (so called by [Rayid Ghani](https://github.com/rayidghani) from Chicago's Data Science, Public Policy, Social Good Center for Data Science and Public Policy). The Magic Loop will iterate through our both models, running each one with different parameters, which we'll define with the ParamGridBuilder. In each model run we will use 10 fold cross validation and finally will evaluate the 'best model's' performance on the test sample.

Instead of using 'for' loops for our Magic Loop, I adapted code from Bryan Cutler's [CV Pipelines](https://bryancutler.github.io/cv-pipelines/), which allows us to try diffferent models on a single CrossValidator object, which acts out as our Magic Loop. The benefit from doing this, is that our CrossValidator will run all the different models we give it and keep the best one.


```python
# Define the base characteristics of the models we will use:
lr = LinearRegression(maxIter=5, featuresCol='features', labelCol='label')
rf = RandomForestRegressor(subsamplingRate=0.2, featuresCol='features', labelCol='label')

# Spark ML models require you input a 'label' column and a 'features' column.
# To do this, we use vector assembler, which produces a 'features' column where each record is a vector of our covariates
formula = RFormula(formula = "DEPARTURE_DELAY ~ .")
#rm_extra = rm_cols(f_vars)
# The following structure was adapted from https://bryancutler.github.io/cv-pipelines/

pipeline = Pipeline(stages=[formula])

paramGrid_lr = ParamGridBuilder() \
    .baseOn({pipeline.stages: [formula, lr]}) \
    .addGrid(lr.elasticNetParam, [0.5, 1.0, 0.0]) \
    .addGrid(lr.regParam, [0.01, 0.001, 0.1])\
    .build()
    
paramGrid_rf = ParamGridBuilder() \
    .baseOn({pipeline.stages : [formula, rf]}) \
    .addGrid(rf.numTrees, [5, 10, 15]) \
    .addGrid(rf.featureSubsetStrategy, ['onethird', '0.5', 'sqrt'])\
    .build()
    
evaluator = RegressionEvaluator(metricName='rmse')

grids = paramGrid_rf + paramGrid_lr
```


```python
crossval = CrossValidator(estimatorParamMaps=grids,
                          estimator=pipeline,
                          evaluator=evaluator,
                          numFolds=2,
                         parallelism = 4)
```

Here we fit out traning data to all our models. cvModel contains our best model along with other information from the Cross Validation runs:


```python
start_time = time.time()
cvModel = crossval.fit(train_flights)
print("--- Magic Loop execution: %s seconds ---" % (time.time() - start_time))
```

    --- Magic Loop execution: 1252.3645656108856 seconds ---


Spark is more designed for production than for model exploration, which is why identifying model's performace isn't exactly straightforward, howevery, we can gather some details from our model runs:  

Following, we can see the average performance (RMSE) for each model during its k-fold runs, the metrics are shown in order of exectution, which means that the first half of values belong to the Linear Regression and the second half belong to our RandomForest runs:


```python
cvModel.avgMetrics
```




    [32.286392243926905,
     32.095573435644056,
     35.17196743515325,
     32.33656064911139,
     31.729165734444404,
     35.34249287932781,
     32.162478460159065,
     31.521971344142354,
     35.313644878669805,
     34.485730474097906,
     34.504544685636986,
     34.39395018453051,
     34.472228959937425,
     34.50278869812698,
     34.32380894820564,
     34.49988623447918,
     34.50053189994395,
     34.49431642009493]



To obtain the best model we can use find the index of the element with the lowest value (in this case because its RMSE, but in others we might want to find the maximum), and extract its corresponding parameters. This only shows the best parameters as run by the CrossValidors, the rest of the model's parameters are either at default values, or as defined when the models are declared.

#### Best Model


```python
cvModel.getEstimatorParamMaps()[ np.argmin(cvModel.avgMetrics) ]
```




    {Param(parent='Pipeline_4c42b13972143ec26f1b', name='stages', doc='a list of pipeline stages'): [RFormula_44d8bafe83a948dbe85c,
      RandomForestRegressor_48b0b951e11bb2477066],
     Param(parent='RandomForestRegressor_48b0b951e11bb2477066', name='numTrees', doc='Number of trees to train (>= 1).'): 15,
     Param(parent='RandomForestRegressor_48b0b951e11bb2477066', name='featureSubsetStrategy', doc='The number of features to consider for splits at each tree node. Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n].'): '0.5'}



#### Worst performer


```python
cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ]
```




    {Param(parent='Pipeline_4c42b13972143ec26f1b', name='stages', doc='a list of pipeline stages'): [RFormula_44d8bafe83a948dbe85c,
      RandomForestRegressor_48b0b951e11bb2477066],
     Param(parent='RandomForestRegressor_48b0b951e11bb2477066', name='numTrees', doc='Number of trees to train (>= 1).'): 10,
     Param(parent='RandomForestRegressor_48b0b951e11bb2477066', name='featureSubsetStrategy', doc='The number of features to consider for splits at each tree node. Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n].'): 'sqrt'}



We can also manually give the index to find out the parameters for a specific model. (Remember, the index corresponds to the avgMetrics array.

#### Mid-performer


```python
cvModel.getEstimatorParamMaps()[1]
```




    {Param(parent='Pipeline_4c42b13972143ec26f1b', name='stages', doc='a list of pipeline stages'): [RFormula_44d8bafe83a948dbe85c,
      RandomForestRegressor_48b0b951e11bb2477066],
     Param(parent='RandomForestRegressor_48b0b951e11bb2477066', name='numTrees', doc='Number of trees to train (>= 1).'): 5,
     Param(parent='RandomForestRegressor_48b0b951e11bb2477066', name='featureSubsetStrategy', doc='The number of features to consider for splits at each tree node. Supported options: auto, all, onethird, sqrt, log2, (0.0-1.0], [1-n].'): '0.5'}



Finally, we can test our 'best' model from the training phase on new, unseen data:


```python
evaluator.evaluate(cvModel.transform(test_flights))
```




    31.529744731140976




```python
cvModel.transform(test_flights).limit(10).select("label","prediction").show()
```

    +-----+------------------+
    |label|        prediction|
    +-----+------------------+
    |-36.0| 2.137113818601739|
    |-21.0| 2.137113818601739|
    |-20.0| 2.137113818601739|
    |-19.0| 21.48559570562177|
    |-19.0| 2.137113818601739|
    |-19.0| 35.30579919526348|
    |-19.0| 2.137113818601739|
    |-19.0|1.7944772412172914|
    |-19.0| 2.137113818601739|
    |-19.0| 2.137113818601739|
    +-----+------------------+
    



```python
start_time = time.time()
evaluator.evaluate(cvModel.transform(test_flights))
end_time = time.time()
print("--- Magic Loop execution: %s seconds ---" % (end_time - start_time))
```

    --- Magic Loop execution: 31.72217559814453 seconds ---



```python
print("--- Magic Loop execution: %s seconds ---" % (end_time - start_time))
```

    --- Magic Loop execution: 31.72217559814453 seconds ---

