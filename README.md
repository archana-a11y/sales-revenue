# Customer Segmentation Project

## Thiranex Task

This project performs customer segmentation based on demographics and customer behavior using Python and scikit-learn.

### Features
- Customer demographic analysis
- Purchase behavior analysis
- K-Means clustering
- Elbow method
- Silhouette score
- PCA visualization
- Segment analysis
- CSV result export

## Project Structure

```text
Customer_Segmentation_Project/
├── customer_segmentation.py
├── customer_data.csv
├── requirements.txt
├── README.md
└── outputs/
```

## How to Run

### 1. Open the folder in VS Code

### 2. Install libraries

```bash
pip install -r requirements.txt
```

### 3. Run

```bash
python customer_segmentation.py
```

The program will create charts and result files inside the `outputs` folder.

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Algorithm

K-Means clustering is used to group customers with similar demographic and purchasing behavior.

## Output

The project generates:
- Age distribution
- Income distribution
- Elbow method graph
- Segment count chart
- Income vs purchase chart
- Purchase frequency vs value chart
- PCA cluster visualization
- Segment heatmap
- Customer segmentation CSV
- Segment analysis CSV
