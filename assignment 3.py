import pandas as pd 
import seaborn as sns
from sklearn.model_selection  import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


# Load data and preprocessing data
data = pd.read_csv('https://raw.githubusercontent.com/annwanginnt/AI2/main/reviews.csv', delimiter ='\t')

# define a function
def classify_rating(rating):
    if rating <= 2:
        return '0'
    elif rating == 3:
        return '1'
    else: 
        return '2'
    
data['Sentiment'] = data['RatingValue'].apply(classify_rating)


# undersampling
sample_size = min(data['Sentiment'].value_counts())
print(sample_size)

df_balanced = pd.concat([
    data[data['Sentiment'] == '0'].sample(sample_size, random_state=42),
    data[data['Sentiment'] == '1'].sample(sample_size, random_state =42),
    data[data['Sentiment'] =='2'].sample(sample_size, random_state=42)
])


df = df_balanced[['Sentiment', 'Review']]

df = df.reset_index(drop=True)

df = df.reset_index()


print('Table 1')
print()

print(df)


# Split data as train.csv and valid.csv
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv('train.csv')
valid_df.to_csv('valid.csv')



train_data = pd.read_csv('train.csv')
valid_data = pd.read_csv('valid.csv')

X = train_data['Review']
y = train_data['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_val = valid_data['Review']
y_val = valid_data['Sentiment']

# Model Building
model =make_pipeline(TfidfVectorizer(ngram_range=(1,2)), MultinomialNB(alpha=0.1))

# find the best parameters
parameters = {
    'tfidfvectorizer__max_df': (0.5, 0.75, 1.0),
    'tfidfvectorizer__min_df': (1, 2, 3),
    'tfidfvectorizer__ngram_range': ((1, 1), (1, 2), (2, 2)),
    'multinomialnb__alpha': (0.1, 1, 10),
}

grid_search = GridSearchCV(model, parameters, cv=5)

grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_test)



# Evaluation 

print(f'Best Parameters: {grid_search.best_params_}')

accuracy = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average ='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test,y_pred)

print()
print(f'accuracy on the train set: {accuracy}')
print()
print(f'f1-score on the train set: {f1}')
print()
print(class_report)
print()

print('Confusion_matrix on the train set')
lables=['0', '1', '2']
conf_matrix_df = pd.DataFrame(conf_matrix, index= lables, columns=lables)

print(conf_matrix_df)


# On validation data 

y_pred_val = grid_search.predict(X_val)


accuracy_val = accuracy_score(y_pred_val, y_val)
f1_val = f1_score(y_pred_val, y_val, average ='weighted')
conf_matrix_val = confusion_matrix(y_val, y_pred_val)
class_report_val = classification_report(y_val,y_pred_val)

print()
print(f'accuracy on the validation set: {accuracy_val}')
print()
print(f'f1-score on the validation set: {f1_val}')
print()
print(class_report_val)
print()

print('Confusion_matrix on the validation set')
lables=['0', '1', '2']
conf_matrix_df_val = pd.DataFrame(conf_matrix_val, index= lables, columns=lables)

print(conf_matrix_df_val)
