

# Install and import dependencies
!apt-get install -qq openjdk-17-jdk-headless > /dev/null
!pip install -q pyspark==3.5.1 pandas matplotlib seaborn openpyxl

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Create Spark session
spark = (
    SparkSession.builder
    .appName("Heart_Stroke_RF_Colab")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

print("Spark session started!")

# Load dataset from Colab path
DATA_PATH = "/home/sat3812/healthcare-dataset-stroke-data.csv"

# Try reading Excel first, fallback to CSV
try:
    df_pd = pd.read_excel(DATA_PATH)
except Exception:
    df_pd = pd.read_csv(DATA_PATH)

# Normalize column names
df_pd.columns = [c.strip().lower().replace(" ", "_") for c in df_pd.columns]
print("Dataset shape:", df_pd.shape)
display(df_pd.head())

# Convert to Spark DataFrame
df = spark.createDataFrame(df_pd)
print(f"Rows: {df.count()}, Columns: {len(df.columns)}")
df.printSchema()

# Drop ID and cast numeric columns
if "id" in df.columns:
    df = df.drop("id")

for c in ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease", "stroke"]:
    if c in df.columns:
        df = df.withColumn(c, F.col(c).cast("double"))

# Define numeric and categorical features
num_cols = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
cat_cols = [c for c in ["gender","ever_married","work_type","residence_type","smoking_status"] if c in df.columns]

# Build preprocessing pipeline
imputer = Imputer(strategy="median", inputCols=num_cols, outputCols=[f"{c}_imp" for c in num_cols])
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in cat_cols],
    outputCols=[f"{c}_oh" for c in cat_cols],
    dropLast=False
)
assembler = VectorAssembler(
    inputCols=[f"{c}_imp" for c in num_cols] + [f"{c}_oh" for c in cat_cols],
    outputCol="features"
)

# Train/test split
train, test = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train size: {train.count()} | Test size: {test.count()}")

# Random Forest classifier
rf = RandomForestClassifier(
    labelCol="stroke",
    featuresCol="features",
    numTrees=150,
    maxDepth=8,
    seed=42
)
pipeline = Pipeline(stages=[imputer] + indexers + [encoder, assembler, rf])

# Train model
t0 = time.time()
model = pipeline.fit(train)
train_time = time.time() - t0
print(f"Training completed in {train_time:.2f}s")

# Evaluate model
preds = model.transform(test).select("stroke", "prediction", "probability")

tp = preds.filter((F.col("stroke")==1.0) & (F.col("prediction")==1.0)).count()
tn = preds.filter((F.col("stroke")==0.0) & (F.col("prediction")==0.0)).count()
fp = preds.filter((F.col("stroke")==0.0) & (F.col("prediction")==1.0)).count()
fn = preds.filter((F.col("stroke")==1.0) & (F.col("prediction")==0.0)).count()
total = tp + tn + fp + fn

accuracy = (tp + tn) / total
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0
auc = BinaryClassificationEvaluator(
    labelCol="stroke",
    rawPredictionCol="probability",
    metricName="areaUnderROC"
).evaluate(preds)

print("\n=== 🩺 Heart Stroke Detection Report ===")
print(f"Training time: {train_time:.2f} s")
print(f"Accuracy:     {accuracy:.4f}")
print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"Specificity:  {specificity:.4f}")
print(f"AUC (ROC):    {auc:.4f}")
print(f"Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print("=====================================\n")

# Feature importances (top 10)
rf_stage = model.stages[-1]
imp = pd.DataFrame(
    list(zip(assembler.getInputCols(), rf_stage.featureImportances.toArray())),
    columns=["feature","importance"]
).sort_values("importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(7,5))
sns.barplot(x="importance", y="feature", data=imp.head(10), color="#6C8AE4")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
