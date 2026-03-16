import numpy as np
import tensorflow as tf
from tensorflow.keras.model import sequential
from tensorflow.keras.layers import simpleRNN, Dense, Embedding
from tensorflow.keras.utils import to_categorical

# load and preprocess dataset
file = open('story.txt', 'r',encoding="utf-8").read()
text = file.lower()

chars = sorted(list(set(text)))
char_to_index = {char : idx for idx, char in enumerate(chars)}
index_to_char = {idx : char for idx, char in enumerate(chars)}

encoded_text = [char_to_index[c] for c in text]
seq_length = 40
x = []
y = []

for i in range(0, len(encoded_text) - seq_length):
  x.append(encoded_text[i:i + seq_length])
  y.append(encoded_text[i + seq_length])

x = np.array(x)
y = to_categorical(y, num_classes=len(chars))

# build the model RNN
model = sequential([
  Embedding(len(chars), 50, input_length=seq_length),
  simpleRNN(128, return_sequences=False), 
  Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# train the model
history = model.fit(x, y, epochs=100, batch_size=500)

# text generation function
def generate_text(seed,length = 200):
    result = seed
    for _ in range(length):
      encoded = np.array([[char_to_index[c] for c in result[-seq_length:]]])
      preds = model.predict(encoded, verbose=0)[0]
      next_index = np.argmax(preds)
      next_char = index_to_char[next_index]
      result += next_char
      seed += next_char
    return result

# example usage
seed_text = "machine learning in branch"
generated = generate_text(seed_text)
print("\nGenerated Text:\n")
print(generated)

# save output
with open('generated_story.txt', 'w', encoding="utf-8") as f:
    f.write(generated)