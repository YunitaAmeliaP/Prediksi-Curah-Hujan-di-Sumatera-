pip install pyspark

pip install seaborn

pip install matplotlib

# Import SparkSession from pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.sql.functions import col, isnan, when, count, udf, year, month, to_date, mean
import pyspark.sql.functions as F
import seaborn as sns
import matplotlib.pyplot as plt

# Create my_spark
spark = SparkSession.builder.getOrCreate()
print(spark)

## Membaca File

# Read File
data = spark.read \
    .option("header", False) \
    .option("sep", ",") \
    .option("inferSchema", True) \
    .csv(path=f'/content/idn-rainfall-adm2-full.csv')

data.printSchema()
data.show()

## Cleaning Data

def quick_overview(df):
    # Tampilkan beberapa baris pertama
    print("FIRST RECORDS")
    print(df.limit(2).sort("_c0").toPandas())  # Mengurutkan berdasarkan kolom "Tanggal"

    # Hitung nilai null
    print("COUNT NULL VALUES")
    print(df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns if c != "_c0"]).toPandas())

    # Periksa duplikasi berdasarkan kolom "Tanggal"
    duplicates = df.groupBy("_c0").count().where('count > 1').limit(5).toPandas()
    print("DUPLICATE RECORDS")
    print(duplicates)

    # Tampilkan skema DataFrame
    print("PRINT SCHEMA")
    print(df.printSchema())

# Menjalankan fungsi quick_overview
quick_overview(data)

## Menghapus null data

# Mengganti nama kolom
data = data.withColumnRenamed("_c0", "date") \
           .withColumnRenamed("_c1", "adm2_id") \
           .withColumnRenamed("_c2", "ADM2_PCODE") \
           .withColumnRenamed("_c3", "n_pixels") \
           .withColumnRenamed("_c4", "rfh") \
           .withColumnRenamed("_c5", "rfh_avg") \
           .withColumnRenamed("_c6", "r1h") \
           .withColumnRenamed("_c7", "r1h_avg") \
           .withColumnRenamed("_c8", "r3h") \
           .withColumnRenamed("_c9", "r3h_avg") \
           .withColumnRenamed("_c10", "rfq") \
           .withColumnRenamed("_c11", "r1q") \
           .withColumnRenamed("_c12", "r3q") \
           .withColumnRenamed("_c13", "status")

data.show()

from pyspark.sql.functions import monotonically_increasing_id
def remove_first_row(data):
    data_with_index = data.withColumn("index", monotonically_increasing_id())

    first_index = data_with_index.first()["index"]

    data_without_first_row = data_with_index.filter(data_with_index["index"] != first_index).drop("index")

    return data_without_first_row

data = remove_first_row(data)

data.show()

Menghapus kolom yang tidak digunakan diawal

# List nama kolom yang akan dihapus
columns_to_drop = ["_c14", "_c15", "status", "ADM2_PCODE"]

# Menghapus kolom yang tidak diinginkan
data = data.drop(*columns_to_drop)

# Menampilkan DataFrame yang sudah dimodifikasi
data.show()

Mengisi nilai null

# List kolom yang berisi nilai NULL yang akan diganti dengan rata-rata
cols_to_replace_null = ["r1h", "r1h_avg", "r3h", "r3h_avg", "rfq", "r1q", "r3q"]

# Menghitung rata-rata nilai untuk setiap kolom
avg_values = data.agg(*(mean(col).alias(col) for col in cols_to_replace_null)).collect()[0]

# Mengganti nilai NULL dengan rata-rata yang sesuai
for col in cols_to_replace_null:
    data = data.withColumn(col, when(data[col].isNull(), avg_values[col]).otherwise(data[col]))

# Menampilkan DataFrame setelah nilai NULL diganti dengan rata-rata
data.show()

from pyspark.sql.functions import col, to_date

# Konversi kolom 'date' ke format tanggal
data = data.withColumn('date', to_date(col('date'), 'M/d/yyyy'))

# Mengisi nilai yang hilang (jika ada)
data = data.na.fill(0)

# Memeriksa skema data
data.printSchema()

menampilkan data yang telah di cleaning

data.show()

data_terakhir = spark.createDataFrame(data.tail(5))
data_terakhir.show()

columns_to_convert = ["n_pixels", "rfh_avg", "r1h_avg", "r3h_avg", "rfq", "r1q", "r3q", "rfh"]
for column in columns_to_convert:
    data = data.withColumn(column, col(column).cast("float"))

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
feature_columns = ["n_pixels", "rfh_avg", "r1h_avg", "r3h_avg", "rfq", "r1q", "r3q"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

feature_columns = ["n_pixels", "rfh_avg", "r1h_avg", "r3h_avg", "rfq", "r1q", "r3q"]: Ini adalah daftar kolom fitur yang akan digunakan dalam model regresi.

data.printSchema()

# Pembagian data menjadi set pelatihan dan pengujian
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Melatih model Linear Regression
lr = LinearRegression(labelCol="rfh", featuresCol="features")
lr_model = lr.fit(train_data)

# Evaluasi model menggunakan RMSE
evaluator = RegressionEvaluator(labelCol="rfh", predictionCol="prediction", metricName="rmse")

# Evaluasi model Linear Regression pada test data
predictions_lr = lr_model.transform(test_data)
rmse_lr = evaluator.evaluate(predictions_lr)
print(f"Root Mean Squared Error (RMSE) for Linear Regression on test data = {rmse_lr}")

# Evaluasi model menggunakan MAE dan R-squared
mae_evaluator = RegressionEvaluator(labelCol="rfh", predictionCol="prediction", metricName="mae")
r2_evaluator = RegressionEvaluator(labelCol="rfh", predictionCol="prediction", metricName="r2")

mae = mae_evaluator.evaluate(predictions_lr)
r2 = r2_evaluator.evaluate(predictions_lr)

print(f"Mean Absolute Error (MAE) for Linear Regression on test data = {mae}")
print(f"R-squared (R2) for Linear Regression on test data = {r2}")

# Mengakses nilai koefisien
coefficients = lr_model.coefficients

# Mengakses nilai intercept
intercept = lr_model.intercept

# Mencetak nilai koefisien
print("Nilai koefisien:")
for i, coef in enumerate(coefficients):
    print(f"Koefisien untuk fitur {feature_columns[i]}: {coef}")

# Mencetak nilai intercept
print(f"\nNilai intercept: {intercept}")

**Perhitungan Manual dengan rumus regresi linier berganda**

Y = B0 + B1 * n_pixels + B2 * rfh_avg + B3 * r1h_avg + B4 * r3h_avg + B5 * rfq + B6 * r1q + B7 * r3q + ε

Dimana:

*   Y: Nilai yang ingin diprediksi
*   B0: Nilai konstan (intercept)
*   B1, B2, ..., B7: Koefisien regresi

*  n_pixels: Nilai piksel dalam gambar
*  rfh_avg: Rata-rata nilai fitur wajah
* r1h_avg: Rata-rata nilai fitur mata kanan
* r3h_avg: Rata-rata nilai fitur mata kiri
* rfq: Kualitas gambar yang diminta
* r1q: Kualitas gambar mata kanan yang diminta
*r3q: Kualitas gambar mata kiri yang diminta
*ε: Term error yang mewakili ketidakpastian atau kesalahan dalam model


**Perhitungan Matematis**

Y = -58.59417310658215 + 0.0004976932282559812 * 82 + 0.9959743162620077 * 88.4516 - 0.00031750641550863505 * 246.5321 + 0.00126508660381937 * 667.5024 + 0.6528024132826401 *  123.7499 - 0.07099954314428193 * 124.6361 - 0.004205064984077347 * 118.8386 + ε = **101.74373555987746**

from pyspark.sql.functions import date_add
from pyspark.sql.functions import lit

#tanggal sebelum prediksi
tanggal_sebelumnya = "2024-04-21"

# Data terakhir yang tersedia sebelum tanggal prediksi
data_terakhir = train_data.filter(data["date"] < tanggal_prediksi).orderBy("date", ascending=False).limit(1)

# Buat DataFrame baru untuk tanggal prediksi
data_prediksi = data_terakhir.withColumn("date", date_add(lit(tanggal_prediksi), 1))

# Melakukan prediksi pada data prediksi
prediksi = lr_model.transform(data_prediksi)

# Menampilkan hasil prediksi
prediksi.show()

Berdasarkan   penghitungan   yang   dilakukan pada tanggal 22 April 2024, diperoleh nilai prediksi curah hujan harian menggunakan prediktor yang telah ditentukan sebesar 104.80255678245767.    Sedangkan    pada    hasil penghitungan manual telah menunjukkan  hasil prediksi curah hujan harian sebesar 101.74373555987746

# Ambil tanggal dan nilai aktual dari DataFrame
tanggal_actual = train_data.select("date").collect()
actual_value = train_data.select("rfh").collect()

# Konversi tanggal aktual menjadi string
tanggal_actual = [row.date for row in tanggal_actual]

# Konversi nilai aktual menjadi float
actual_value = [row.rfh for row in actual_value]

# Plot hasil aktual dan prediksi
plt.figure(figsize=(10, 6))
plt.plot(tanggal_actual, actual_value, label='Aktual', marker='o', linestyle='-', color='blue')
plt.title('Prediksi Curah Hujan')
plt.xlabel('Tanggal')
plt.ylabel('Curah Hujan (mm)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Ambil tanggal dan nilai prediksi dari DataFrame
tanggal = prediksi.select("date").collect()
prediksi_value = prediksi.select("prediction").collect()

# Konversi tanggal menjadi string
tanggal = [row.date for row in tanggal]

# Konversi nilai prediksi menjadi float
prediksi_value = [row.prediction for row in prediksi_value]

# Plot hasil prediksi
plt.figure(figsize=(10, 6))
plt.plot(tanggal, prediksi_value, marker='o', linestyle='-')
plt.title('Prediksi Curah Hujan')
plt.xlabel('Tanggal')
plt.ylabel('Prediksi Curah Hujan / mm')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
