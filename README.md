# Building LLM from Scratch - Project under Summer of CODE
Mentor - YASHWANT VVS
Reference Text - Build a LLM (Sebastian Raschka)

## Timeline so far:
Week 1: Learning the basics of PyTorch and setting up Python Notebook

Week 2: Tokenising Data, Generating Vocabularies, Creating Python Classes and functions for formatting raw data into test data, intitialising token embeddings

Week 3: Implementing Attention mechanisms to help the model understand contextually, applied causal attention and dropout, created a MuiltHeadAttention object.

Week 4: Normalising Layer Outputs, Activation, Shortcut Connections. Created Transformer Class. Compiled multiple Transformer Blocks into the unified LLM architecture. Calculated storage requirements for parameter data.

## So far, we have created a untrained LLM, that is capable of next word prediction after appropriate training.

Week 5: Training the Model on Text
We used the GPT model we built and trained it on a small text dataset. Since we don’t have powerful GPUs, we reduced the input size (context length). We trained it to predict the next word and used loss to measure how well it was doing.

Week 6: Making the Model Classify Text (Spam Detection)
We changed the model to do text classification, like checking if an SMS is spam or not. Instead of predicting the next word, it now outputs a class label. We used pre-trained GPT-2 weights to help with training and save time.

Week 7: Teaching the Model to Follow Instructions
We trained the model to understand and follow user instructions, like answering questions or completing tasks. This is similar to how a chatbot like ChatGPT works. The model was trained to give helpful responses based on prompts.

PROJECT COMPLETE :))
