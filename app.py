from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)


def train_ai():
    if not os.path.exists('flights.csv'):
        print("❌ Error: flights.csv not found!")
        return None, None, None
    
    
    df = pd.read_csv('flights.csv')
    
    
    df.columns = df.columns.str.strip()

    
    df['Departure_Delay'] = pd.to_numeric(df['Departure_Delay'], errors='coerce').fillna(0)
    
    
    df['is_delayed'] = (df['Departure_Delay'] > 15).astype(int)
    
    
    df['Scheduled_Departure'] = pd.to_datetime(df['Scheduled_Departure'], errors='coerce')
    df['hour'] = df['Scheduled_Departure'].dt.hour.fillna(12)
    
    
    le_air = LabelEncoder().fit(df['Airline'].unique())
    le_wea = LabelEncoder().fit(df['Weather_Condition'].unique())
    
    df['air_enc'] = le_air.transform(df['Airline'])
    df['wea_enc'] = le_wea.transform(df['Weather_Condition'])
    
   
    features = ['air_enc', 'hour', 'wea_enc', 'Temperature', 'Wind_Speed', 'Visibility', 'Congestion_Level']
    
   
    for col in ['Temperature', 'Wind_Speed', 'Visibility', 'Congestion_Level']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    X = df[features]
    y = df['is_delayed']
    
   
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("🚀 AI Model Trained Successfully!")
    return model, le_air, le_wea


model, le_air, le_wea = train_ai()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "AI Model not trained. Check flights.csv"}), 500
        
    data = request.json
    try:
        
        input_df = pd.DataFrame([[
            le_air.transform([data['airline']])[0],
            int(data['hour']),
            le_wea.transform([data['weather']])[0],
            float(data['temp']),
            float(data['wind']),
            float(data['vis']),
            float(data['cong'])
        ]], columns=['air_enc', 'hour', 'wea_enc', 'Temperature', 'Wind_Speed', 'Visibility', 'Congestion_Level'])
        
        prob = model.predict_proba(input_df)[0][1]
        
        if prob > 0.6: status = "High Risk"
        elif prob > 0.3: status = "Moderate Risk"
        else: status = "Low Risk"
        
        return jsonify({"probability": round(prob * 100, 1), "status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('flights.csv')
df.columns = df.columns.str.strip()
df['Departure_Delay'] = pd.to_numeric(df['Departure_Delay'], errors='coerce').fillna(0)


plt.style.use('ggplot')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Chart 1: Average Delay by Airline
sns.barplot(x='Airline', y='Departure_Delay', data=df, ax=axes[0], palette='viridis')
axes[0].set_title('Average Delay (Minutes) per Airline')
axes[0].tick_params(axis='x', rotation=45)


sns.boxplot(x='Weather_Condition', y='Departure_Delay', data=df, ax=axes[1])
axes[1].set_title('Delay Distribution by Weather')

plt.tight_layout()
plt.savefig('analysis.png')
print(" Visualization saved as 'analysis.png'")
plt.show()
