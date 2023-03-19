import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# assigning the title of webpage
st.title("Sentiment Analyzer")

# gathering all the resources
stopwords_english = pickle.load(open('stopwords_english.pkl', 'rb'))
word2ind = pickle.load(open('word2ind.pkl', 'rb'))

# defining the model
class SentimentAnalysis(nn.Module):
    def __init__(self, num_layers=2, vocab_size=1001, embedding_dim=64, hidden_dim=256, drop_prob=0.3):
        super(SentimentAnalysis, self).__init__()
        # lstm will calculate the output for all type of seq_len so we don't need to give the argument for seq_len
        self.num_layers = num_layers  # this means how many LSTM's to be stacked on top of each other
        self.vocab_size = vocab_size  # this is pretty obvious 
        self.embedding_dim = embedding_dim  
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        
    def forward(self, x):
        h0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim))
        c0 = torch.zeros((self.num_layers, x.shape[0], self.hidden_dim))
        hidden = (h0,c0)
        
        batch_size, seq_len = x.shape
        embeds = self.embedding(x) 
        lstm_out, hidden = self.lstm(embeds, hidden)  
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)  
        lstm_out = self.dropout(lstm_out) 
        
        op = self.fc(lstm_out)  
        op = op.view(batch_size, seq_len)  
        op = op[:, -1]  
        return op 
    
# let us first define the model and load it
model = SentimentAnalysis()
checkpoint = torch.load('my_checkpoint.pth', map_location='cpu')
model.load_state_dict(checkpoint["state_dict"])


# adding a textbar through which user can enter a review
text = st.text_input('Your can enter your review here', '')

# now the required preprocessing function

def preprocessing(string, stopwords):
    string  = string.lower() # converting to the lowercase letters
    string = re.sub(r'[^a-zA-Z]', ' ', string)
    tokens = re.split(' ', string) # splitting the sentence into with delimeter being white space
    
    clean_tokens = []
    for word in tokens:
        if word!='' and word not in stopwords:
            clean_tokens.append(word)
          
    return clean_tokens

# now the predictions 
MAX_LEN = 400
prep = preprocessing(text, stopwords_english)
prep = prep[:MAX_LEN] 
prep = [word2ind[word] for word in prep if word in word2ind.keys()]
#print(prep)
prep = [0]*(MAX_LEN-len(prep)) + prep
prep = torch.tensor(prep).type(torch.LongTensor)
prep = prep.reshape(1, prep.shape[0])
op = model(prep)
op = torch.sigmoid(op)
op[0] = (op[0]*1e4)//1e2
if text:
    if op[0]>0.5:
        st.write("This review seems poistive to us with a probability of {}%".format(op[0].item()))
    else:
        st.write("This review seems negative to us with a probability of {}%".format(100-op[0].item()))