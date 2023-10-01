![](2023-10-01-08-40-59.png)
![](2023-10-01-08-41-15.png)
- word embeddings papers like 
    - google word to vec paper
        - https://code.google.com/archive/p/word2vec/
    - glove paper
    - Sanjeev Arora's paper

![](2023-10-01-08-43-30.png)
![](2023-10-01-08-45-53.png)    
- Bag of words model, it wont pay attention to the next word, order or position, it doesnt matter
- the probability estimate would be the same
- even this model is quite good to learn quite a bit of properties of the model
![](2023-10-01-08-49-17.png)
![](2023-10-01-08-49-43.png)
    - tuesday, thursady,sunday are close
    - nokia , samsung are close
- how do we learn good word vectors
![](2023-10-01-08-52-08.png)
- NN in general is not convex
![](2023-10-01-08-54-25.png)
![](2023-10-01-08-55-10.png)
- in stochastic gradient descent , we take smal batch and perform gradient computation 

- cs25 Transformers https://www.youtube.com/watch?v=P127jhj-8-Y&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM

![](2023-10-01-09-00-40.png)
![](2023-10-01-09-01-35.png)
- word vectors are represented as rows
![](2023-10-01-09-03-01.png)
- two vwctors for each word type, 
    - centre vector
    - outside vector
- SKip gram model   
     - predict context word from given centre word
- Continuous Bag of Words
    - predict centre word from context words
- both give similar results
- SGNS -> skip gram negative Sampling
![](2023-10-01-09-07-43.png)
- naive softmax will be more expensive to calculate the denominator
- if we have 100,000 words, we need to do 100,000 dot products
![](2023-10-01-09-09-36.png)
- overall we want to optimize is   still an average of the loss for each particular center word, and each particular window, we going to take the dot product as before the center and the outside word
- now we are not using softmax, instead we are putting it into a logistic function, often called Sigmoid function
- What logistic does is , it converts any real number to a probability of 0 and 1 open interval, 
- so if the dot product is large, the logistic of the dot product will virtually be 1
![](2023-10-01-09-14-03.png)
https://youtu.be/gqaHkPEZAew?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&t=1279
- unigram distribution of words
    - how often does each word occur in the corpus
- 3/4th power is used to make the distribution less skewed, so that the most frequent words are not too much more frequent than the least frequent words
![](2023-10-01-09-28-17.png)
![](2023-10-01-09-28-28.png)
- this is a symmetric matrix
- What is negative sampling?
    - we are going to take the center word and the context word, and we are going to say that we want to maximize the probability of the context word given the center word, and we are going to minimize the probability of the context word given the center word, for all the words that dont appear in the context
- why do we take only one negative sample?
    - because we are going to do this for every center word, and every context word, so we are going to have a lot of negative samples
    - so we dont need to take a lot of negative samples
- in softmax , we have to calculate the probability of the context word given the center word, for all the words in the vocabulary, we do that by taking the dot product of the center word and the context word, and then we exponentiate that, and then we normalize it by the sum of the exponentiated dot products of the center word and all the context words
- negative sampling is trying to do the same thing, but instead of doing it for all the words in the vocabulary, we are going to do it for the context word, and one negative sample
- co occurence matrix will give , how often does each word occur with each other word
- Are these kind of count word vectors good to use?
    - they are not good to use, because they are very sparse, and they are very high dimensional
    - they are very high dimensional, because the number of words in the vocabulary is very large
    - they are very sparse, because most words dont occur with most other words
![](2023-10-01-09-40-31.png)
![](2023-10-01-09-41-35.png)
- SVD is a way of taking a matrix and decomposing it into a product of three matrices
- SVD has various applications in machine learning like 
    - dimensionality reduction
    - matrix completion
    - collaborative filtering
    - latent semantic analysis
    - word embeddings
    - topic modeling
    - etc
- U , diagonal matrix and V transpose are the three matrices
- this works for any shape
![](2023-10-01-09-46-02.png)
![](2023-10-01-09-47-43.png)
![](2023-10-01-09-48-51.png)
![](2023-10-01-09-50-40.png)
![](2023-10-01-09-53-06.png)
![](2023-10-01-09-53-16.png)
![](2023-10-01-09-54-54.png)
![](2023-10-01-09-55-55.png)
- what is glove model?
    - it is a model that is trained to predict the co occurence of words
    - it is trained to predict the probability of word j appearing in the context of word i , given the vector representation of word i and word j
![](2023-10-01-09-58-54.png)
![](2023-10-01-10-02-07.png)
![](2023-10-01-10-03-22.png)
![](2023-10-01-10-11-48.png)
![](2023-10-01-10-04-14.png)
![](2023-10-01-10-05-35.png)
![](2023-10-01-10-06-39.png)
![](2023-10-01-10-09-24.png)
![](2023-10-01-10-10-06.png)
![](2023-10-01-10-15-32.png)
![](2023-10-01-10-18-41.png)
![](2023-10-01-10-18-49.png)
![](2023-10-01-10-21-04.png)
![](2023-10-01-10-24-13.png)