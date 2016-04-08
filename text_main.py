# Text-mining on essays submitted by applicants.
# Author: Cao Yankun, ycaoae@ust.hk

import pandas
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Get the essays and left join on "projectid" with outcomes.
# Only save "essay" and class label "is_exciting".
essays = pandas.read_csv("essays.csv")
columns = essays.columns.values.tolist()
columns.pop(-1) # Pop "essay".
columns.pop(0) # Pop "projectid".
essays = essays.drop(columns, axis=1)
outcomes = pandas.read_csv("outcomes.csv")
columns = outcomes.columns.values.tolist()
columns.pop(0) # Pop "projectid".
columns.pop(0) # Pop "is_exciting".
outcomes = outcomes.drop(columns, axis=1)
essays = essays.merge(outcomes, how="left", on="projectid")

# Cleaning the text.
# The lists to save cleaned data.
processed_training_essays = []
class_labels = []
processed_unknown_essays = []
unknown_essay_id = []

# Load stopwords and stemmer
stops = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

size = len(essays.index)
for i in range(size):
	# Cleaning.
	text = essays.iloc[i]["essay"]
	if not pandas.isnull(text): # Non-empty texts.
		# Remove non letters, remove stopwords and stem the words.
		text = text.replace("\\n", " ").replace("\\r", " ")
		text = re.sub("[^a-zA-Z]", " ", text).lower()
		words = text.split()
		meaningful_words = [w for w in words if not w in stops]
		root_words = [stemmer.stem(w) for w in meaningful_words]
		text = " ".join(root_words)
	else: # Empty texts.
		text = ""
	
	# Read the label and decide where to put the text.
	label = essays.iloc[i]["is_exciting"]
	if not pandas.isnull(label): # Training data.
		processed_training_essays.append(text)
		class_labels.append(label)
	else: # Data to predict.
		processed_unknown_essays.append(text)
		unknown_essay_id.append(essays.iloc[i]["projectid"])

# Vectorization of processed text.
vectorizer = CountVectorizer(max_features=5000)
essay_vectors = vectorizer.fit_transform(processed_training_essays)
unknown_essay_vectors = vectorizer.transform(processed_unknown_essays)

# Naive Bayes classification.
bayes_classifier = MultinomialNB()
bayes_classifier = bayes_classifier.fit(essay_vectors, class_labels)
prediction = bayes_classifier.predict_proba(unknown_essay_vectors)

# Extract probability("is_exciting" == "t").
exciting_proba = []
size = len(prediction)
for i in range(size):
	exciting_proba.append(prediction[i][1])

# Write to csv file.
output = pandas.DataFrame(data={"projectid":unknown_essay_id,"is_exciting":exciting_proba})
output = output[["projectid","is_exciting"]]
output.to_csv("submission.csv",index=False)