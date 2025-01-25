
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import seaborn as sns
from nltk.stem import WordNetLemmatizer
import spacy
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc

path = './Dataset/'
real_news_path= path + 'True.csv'
fake_news_path=path+ 'Fake.csv'

#Exception Handling just in case file is not present
try:
    real_news=pd.read_csv(real_news_path)
    fake_news=pd.read_csv(fake_news_path)
except FileNotFoundError as e:
    print(f"File Not Found Error: {e}")

real_news["category"] = 0
fake_news["category"] = 1

print("Real news records:", real_news.shape[0])
print("Fake news records:", fake_news.shape[0])

dataset = pd.concat([real_news,fake_news]).reset_index(drop = True)
dataset.head()

dataframe = dataset[["text","category"]]
dataframe.head()
dataframe.tail()

#check for empty strings in dataset
blanks = []
for index, text in dataframe["text"].items():
  if text.isspace():
    blanks.append(index)
#blanks list is storing index of empty cells, we need to reomve them
len(blanks)
dataframe.drop(blanks, inplace = True)
dataframe.shape

nltk.download('punkt_tab')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
lemma = WordNetLemmatizer()
stop_words_1 = nlp.Defaults.stop_words
stop_words_2 = stopwords.words('english')
stop_words = set(set(stop_words_1)|set(stop_words_2))
print("Number of total stop words: ", len(stop_words))
def clean_text(text):
  text = text.lower()
  text = re.sub('[^a-z0-9]',' ',text)
  return text
dataframe['cleaned_text'] = dataframe['text'].apply(clean_text)
dataframe.head()
dataframe.tail()

#Tokenization
nltk.download('punkt')
def tokenize_text(text):
  return nltk.word_tokenize(text)
dataframe['tokenized_text'] = dataframe['cleaned_text'].apply(tokenize_text)
dataframe.head()
dataframe.tail()
#Stop word removal
def remove_stopwords(tokens):
  return [word for word in tokens if word not in stop_words]
dataframe['tokenized_text'] = dataframe['tokenized_text'].apply(remove_stopwords)
dataframe.head()
dataframe.tail()

# Lemmatization

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def lemmatize_text(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]
dataframe['lemmatized_text'] = dataframe['tokenized_text'].apply(lemmatize_text)
dataframe['lemmatized_text'] = dataframe['lemmatized_text'].apply(lambda x: ' '.join(x))

dataframe.tail()


#Creating the Bag of Words(with unigram and bigram)
countVectorizer = CountVectorizer(ngram_range=(1,2), max_features=200)
bow = countVectorizer.fit_transform(dataframe['lemmatized_text'])
#applying TF-IDF on the BoW result
tfidfTransformer = TfidfTransformer()
tfidf_matrix = tfidfTransformer.fit_transform(bow)
#Convert the TF-IDF matrix to a DataFrame and include the labels
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=countVectorizer.get_feature_names_out())
#Appedning the category column to the tfidf matrix
tfidf_df['category'] = dataframe['category'].values
#printing the tfidf dataframe
tfidf_df.head()

df_real = tfidf_df[tfidf_df['category'] == 0]  # Real news
df_fake = tfidf_df[tfidf_df['category'] == 1]  # Fake news
#printing the data to check
print(f"Number of real news articles: {len(df_real)}")
print(f"Number of fake news articles: {len(df_fake)}")

#visulizing the top 50 tf-idf words
import matplotlib.pyplot as plt
import seaborn as sns
top_n =50
tfidf_sum = tfidf_df.sum(axis=0).sort_values(ascending=False).head(top_n)
plt.figure(figsize=(10, 6))
sns.barplot(x=tfidf_sum.values, y=tfidf_sum.index)
plt.title(f"Top {top_n} TF-IDF Features")
plt.xlabel("TF-IDF Score")
plt.ylabel("Features")
plt.show()

X = tfidf_df.drop(columns=['category']).values #all columns except the last column is x
y = tfidf_df['category'].values   #Last column is y(target)
#Checking unique values in the category column
unique_values = tfidf_df['category'].unique()
print(f"Unique values in the category column: {unique_values}")
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

df_real =tfidf_df[tfidf_df.iloc[:,-1] == 0]  #Real news - 0
df_fake =tfidf_df[tfidf_df.iloc[:,-1] == 1]  #Fake news - 1

#printing the fake and real to make sure datset is correct
print(f"Number of real news articles: {len(df_real)}")
print(f"Number of fake news articles: {len(df_fake)}")

#Randomly sampling the majority real news class
df_fake_undersampled = resample(df_fake, replace=False, n_samples=len(df_real),random_state=42)
df_balanced = pd.concat([df_real, df_fake_undersampled])

#shuffling dataframe
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

#balanced x,y
x = df_balanced.iloc[:, :-1].values  # All columns except the last one (features)
y = df_balanced.iloc[:, -1].values   # The last column (labels)

#splittign the balanced x,y dataset
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=0, stratify=y) #60% is training data and 40% is test data
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp) #out of the test data, half is for validation other is for testing

#defining the model
num_features = x.shape[1]  # Number of features
model =tf.keras.Sequential([
  layers.Dense(64, activation='relu', input_shape=(num_features,)),
  layers.Dropout(0.5),  #Dropout layer for regularization
    layers.Dense(32, activation='relu'),    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') ]) #final output layer for binary classification


#Compiling the model using ;Adam' optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

#early stopping(stop when model converges based on validation loss)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,            #stops after 5 epochs of no improvement in val loss
    restore_best_weights=True #Restoring weights from the best epoch
)

#Initially fitting the model with 10 epochs
initial_epochs = 10
history_initial = model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=initial_epochs,
    validation_data=(x_valid, y_valid),
    verbose=1 )


#Fitting model with early stopping
history_final = model.fit(
    x_train, y_train,
    batch_size=16,
    epochs=50,  #higher number fo epochs for better accuracy
    initial_epoch=initial_epochs,  #startign from last epoch
    validation_data=(x_valid, y_valid),
    callbacks=[early_stopping],
    verbose=1
)

#Predicting for the test set
y_test_prob =model.predict(x_test)
y_test_pred = (y_test_prob >= 0.5).astype(int).flatten()

print("Test Set Classification Report:\n", classification_report(y_test,y_test_pred))

#generating confusion matrix
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Test Set')
plt.show()

fpr, tpr,thresholds=roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr,tpr)

#Plotting the  ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

tf_model = model
#Cleaning the input text
def clean_text(text):
    text=re.sub(r'\W+', ' ', text)
    text=text.lower()
    return text

def predict_news(news_text):
    cleaned_text=clean_text(news_text)
    text_vectorized = countVectorizer.transform([cleaned_text])
    text_tfidf=tfidfTransformer.transform(text_vectorized)
    prediction_prob= tf_model.predict(text_tfidf)
    prediction=(prediction_prob >= 0.5).astype(int)
    return "Real News" if prediction[0][0] == 0 else "Fake News"

# user_input = input("Please enter the news article text: ")
# result = predict_news(user_input)
# print(f"The prediction is: {result}")


#initialize and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(x_train, y_train)

# Evaluation on the validation set
y_valid_pred = nb_model.predict(x_valid)

print("Validation Accuracy:", accuracy_score(y_valid, y_valid_pred))
print("\nClassification Report:\n", classification_report(y_valid, y_valid_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_valid, y_valid_pred))

#Evaluation on the test set
y_test_pred = nb_model.predict(x_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))

#Confusion matrix for NB model
y_valid_pred = nb_model.predict(x_valid)
cm = confusion_matrix(y_valid, y_valid_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Validation Set')
plt.show()

#ROC displaying
y_valid_prob = nb_model.predict_proba(x_valid)[:, 1]
fpr, tpr, _ = roc_curve(y_valid, y_valid_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Validation Set')
plt.legend(loc='lower right')
plt.show()

#Saving the model and TF-IDF vectorizer
model_path = './Dataset/nb_model.pkl'
tfidf_path = './Dataset/tfidf_vectorizer.pkl'
joblib.dump(nb_model, model_path)
joblib.dump(countVectorizer, tfidf_path)
print("Model and vectorizer saved successfully!")

nb_model = joblib.load(model_path)
countVectorizer = joblib.load(tfidf_path)

joblib.dump(countVectorizer, 'count_vectorizer.pkl')
joblib.dump(tfidfTransformer, 'tfidf_transformer.pkl')

nb_model = joblib.load(path + 'nb_model.pkl')
countVectorizer = joblib.load(path + 'tfidf_vectorizer.pkl')


def predict_news(news_text):
    cleaned_text = clean_text(news_text)
    tokenized_text = tokenize_text(cleaned_text)
    lemmatized_text = lemmatize_text(tokenized_text)

    text_for_vectorization = ' '.join(lemmatized_text)
    text_vectorized = countVectorizer.transform([text_for_vectorization])

    #Making prediction
    prediction = nb_model.predict(text_vectorized)
    return "Real News" if prediction[0] == 0 else "Fake News"

def plot_confusion_matrix(x_valid, y_valid):
    y_valid_pred = nb_model.predict(x_valid)  # Predict on validation data
    cm = confusion_matrix(y_valid, y_valid_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Validation Set')
    return plt.gcf()

def plot_roc_curve(x_valid, y_valid):
    y_valid_prob = nb_model.predict_proba(x_valid)[:, 1]
    fpr, tpr, _ = roc_curve(y_valid, y_valid_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Validation Set')
    plt.legend(loc='lower right')
    return plt.gcf()

news_input = gr.Textbox(lines=5, placeholder="Enter the news article text...")
prediction_output = gr.Textbox(label="Prediction")

iface = gr.Interface(
    fn=predict_news,
    inputs=news_input,
    outputs=prediction_output,
    title="News Article Classifier",
    description="Classify news articles as Real or Fake using a Naive Bayes model.",
)

iface_conf_matrix = gr.Interface(fn=lambda: plot_confusion_matrix(x_valid, y_valid), inputs=None, outputs=gr.Plot(),
                                title="Confusion Matrix")
iface_roc_curve = gr.Interface(fn=lambda: plot_roc_curve(x_valid, y_valid), inputs=None, outputs=gr.Plot(),
                                title="ROC Curve")

dashboard = gr.TabbedInterface([iface, iface_conf_matrix, iface_roc_curve],
tab_names=["Predict", "Confusion Matrix", "ROC Curve"])
dashboard.launch()
