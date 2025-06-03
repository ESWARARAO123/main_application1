# Frontend AI Conversation Guide

## Route Prediction System - User & AI Conversation Flow

### 1. **Activation**
**User:** "To obtain the results, I need to train and then make a prediction using the 3 csvs"

**AI Response:**
```
🤖 Route Prediction System Activated

I'm ready to help you train machine learning models and generate route predictions!

📊 Available Data Tables:
• Place table: ariane_place_sorted_csv
• CTS table: ariane_cts_sorted_csv  
• Route table: ariane_route_sorted_csv

🔄 Workflow:
1. Train → Use Place + CTS + Route tables to train the neural network model
2. Predict → Generate Route table predictions from Place + CTS data
3. Download → Export prediction results as CSV

💬 Quick Start Commands:
• Type "train" to start model training
• Type "predict" to generate predictions (after training)
• Type "download" to export results

🎯 What would you like to do?
```

### 2. **Training Phase**
**User:** "train"

**AI Response 1 (Start):**
```
🔧 Starting Model Training

📊 Training Configuration:
• Place table: ariane_place_sorted_csv
• CTS table: ariane_cts_sorted_csv  
• Route table: ariane_route_sorted_csv
• Model type: Neural Network (Route Slack Prediction)

⏳ Training in progress... This may take a few moments.
```

**AI Response 2 (Completion):**
```
✅ Training Completed Successfully!

📊 Model Performance Metrics:
• R² Score: 0.9973 (99.73% accuracy)
• Mean Absolute Error: 0.1595
• Mean Squared Error: 0.0324
• RMSE: 0.1801

🎯 Next Steps:
The model is now ready for predictions! Type "predict" to generate route table predictions.
```

### 3. **Prediction Phase**
**User:** "predict"

**AI Response 1 (Start):**
```
🔮 Generating Route Predictions

📊 Input Data:
• Place table: ariane_place_sorted_csv
• CTS table: ariane_cts_sorted_csv

⚡ Processing data and generating route table predictions...
```

**AI Response 2 (Success with Table):**
```
✅ Route Prediction Completed Successfully!

🎯 Generated Route Table
📊 Input Sources:
• Place table: ariane_place_sorted_csv
• CTS table: ariane_cts_sorted_csv

📈 Results:
• Total predicted routes: 3969
• Preview: First 10 routes shown below
• Full table: Available for download

📊 Model Performance:
• R² Score: 0.9985 (99.85% accuracy)
• Mean Absolute Error: 0.1006
• Mean Squared Error: 0.0180

📋 Route Table Preview (showing first 10 of 3969 routes):

[Interactive Table Display]
- Endpoint | Place Slack | CTS Slack | Predicted Route Slack | Fanout | Net Count
- [Table rows with actual prediction data]

[Download Button: "Download Complete Route Table (3969 routes) as CSV"]
```

### 4. **Download Phase**
**User:** "download"

**AI Response:**
```
📥 Prediction results downloaded as CSV file.
```

### 5. **Error Handling Example**
**User:** "predict" (when model not trained)

**AI Response:**
```
❌ Prediction Error

API Error (400): Models not trained yet. Please train first.

🔧 Troubleshooting:
- You need to train the model before making predictions
- Type "train" to start the training process
- Training typically takes 1-2 minutes to complete
```

## Common User Questions & AI Responses

### Q: "What can you do?"
**AI:** 
```
I'm a Route Prediction System that can:

🔧 Train machine learning models using your Place, CTS, and Route data
🔮 Generate route table predictions from Place and CTS inputs  
📊 Provide detailed performance metrics and accuracy scores
📥 Export prediction results as downloadable CSV files

Commands: "train", "predict", "download"
```

### Q: "How accurate is the model?"
**AI:**
```
📊 Current Model Performance:
• R² Score: 0.9973 (99.73% accuracy)
• This means the model explains 99.73% of the variance in route slack predictions
• Mean Absolute Error: 0.1595 (average prediction error)
• The model is highly accurate for route slack prediction tasks
```

### Q: "What data do you use?"
**AI:**
```
📊 Training Data Sources:
• Place table (ariane_place_sorted_csv): Contains placement timing data
• CTS table (ariane_cts_sorted_csv): Contains clock tree synthesis data  
• Route table (ariane_route_sorted_csv): Contains actual routing results for training

🎯 Prediction Process:
Input: Place + CTS data → Output: Predicted Route table with slack values
```

### Q: "Can I see the prediction results?"
**AI:**
```
📋 Prediction results are displayed in two ways:

1. Interactive Table Preview (first 10 rows)
2. Full CSV Download (all predictions)

The table shows:
• Endpoint names
• Place slack values
• CTS slack values  
• Predicted route slack values
• Fanout and net count data

Type "predict" to generate new predictions or "download" to get the CSV file.
```

## Tips for Users

1. **Always train before predicting** - The model needs to be trained first
2. **Use simple commands** - "train", "predict", "download" work best
3. **Wait for completion** - Training takes 1-2 minutes, predictions take 10-30 seconds
4. **Check the table preview** - Verify results before downloading
5. **Download for analysis** - Use CSV export for detailed analysis in Excel/Python

## Error Messages & Solutions

| Error | Solution |
|-------|----------|
| "Models not trained yet" | Type "train" first |
| "Shape mismatch" | Backend data issue - contact admin |
| "No common endpoints" | Data tables don't have matching endpoints |
| "Connection error" | Check if backend server is running |
| "No prediction data" | Model didn't generate results - try training again |