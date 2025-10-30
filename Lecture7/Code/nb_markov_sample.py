import pandas as pd
from matplotlib.pyplot import *
from pydtmc import *
from sklearn.naive_bayes import *
from sklearn.model_selection import train_test_split


#Naive Bayes
#Input Dateset
org_df = pd.read_csv("diabetes.csv")

#Labels and Features
label_df = org_df.loc[:,org_df.columns == 'Outcome']
feat_df = org_df.loc[:,org_df.columns != 'Outcome']

#Split Train and Test Data
x_train, x_test, y_train, y_test = train_test_split(feat_df, label_df, test_size = 0.3, random_state=1)

#Create Naive Bayes Model
nb_model = GaussianNB()  # CategoricalNB() for categorical data
nb_model.fit(x_train, y_train)

#Accuracy of Model
print("Test accuracy:  ", nb_model.score(x_test,y_test))


# Markov Chain
# The states
states = ["Tea","Coffee","Water"]
#
# # Transition matrix
transition_matrix = [[0.2,0.6,0.2],
                     [0.3,0,0.7],
                     [0.5,0,0.5]]

# Create Markov Chain
mc = MarkovChain(transition_matrix, states)
print(mc)

# Show stationary state
print(mc.steady_states)

# Visualize results
matplotlib.pyplot.ion()
plot_graph(mc)
plot_redistributions(mc, 10, plot_type='projection', initial_status='Coffee')


# Hidden Markov
hidden_states = ['Low', 'High']
observation_symbols = ['Rain', 'Dry']
transition_matrix = [[0.5, 0.5],
                     [0.3, 0.7]]
observation_matrix = [[0.8, 0.2],
                      [0.4, 0.6]]

# Create Hidden Markov Model
hmm = HiddenMarkovModel(transition_matrix, observation_matrix, hidden_states, observation_symbols)

# Visualize HMM
matplotlib.pyplot.ion()
plot_graph(hmm)

# Predict hidden states of observations
_, most_probable_states = hmm.predict('viterbi', ['Dry','Dry','Rain'])
print(most_probable_states)
