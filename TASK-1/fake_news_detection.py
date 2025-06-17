import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load dataset using pandas
df = pd.read_csv('fake_news.csv')  # Make sure your CSV file has 'text' and 'label' columns
# Check the first few rows
print(df.head())
# Separate features and labels
X = df['text']         # News content
y = df['label']        # 0 = Real, 1 = Fake
# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Function to predict new news
def predict_news(news_text):
    news_vector = vectorizer.transform([news_text])
    prediction = model.predict(news_vector)
    return "Fake" if prediction[0] == 1 else "Real"
# Example usage
print(predict_news("Breaking: Government announces new job policies for youth."))
# Allow user to enter custom news input
while True:
    user_input = input("\nEnter a news headline (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result = predict_news(user_input)
    print(f"Prediction: {result}")
