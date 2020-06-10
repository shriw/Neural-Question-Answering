import tensorflow as tf
tf.enable_eager_execution()

from sklearn.model_selection import train_test_split

import re
import numpy as np
import os
import io
import time
from num2words import num2words
from nltk.corpus import wordnet as wn

import pandas as pd


def preprocess_sentence(word):
    word = re.sub(r"([?.!,¿])", r" \1 ", word)
    word = re.sub(r'[" "]+', " ", word)
    word = re.sub(r'\d+',lambda x:num2words(x.group()), word)
    word = word.replace("point zero", " ")
    word = re.sub(r"[^a-zA-Z?.!,¿]+", " ", word)
    word = word.rstrip().strip()
    word = '<start> ' + word + ' <end>'

    return word

def create_dataset(path, num_examples):
    lines = io.open(path).read().strip().split('\n')
    word_pairs = [[preprocess_sentence(word) for word in l.split('\n')]  for l in lines[:None]]
    Qs = word_pairs[0::2]
    As = word_pairs[1::2]
    Qs = [item for sublist in Qs for item in sublist]
    As = [item for sublist in As for item in sublist]
    
    return Qs, As


def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(sent):
  tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  tokenizer.fit_on_texts(sent) 
  tensor = tokenizer.texts_to_sequences(sent)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')
  
  return tensor, tokenizer

def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    inpQs,targAs= create_dataset(path, num_examples)

    inputTensor, inpQsTokenized = tokenize(inpQs)
    targetTensor, targAsTokenized = tokenize(targAs)

    return inputTensor, targetTensor, inpQsTokenized, targAsTokenized


class Encoder(tf.keras.Model):
  def __init__(self, vocabSize, embedDim, encoderUnits, batchSize):
    super(Encoder, self).__init__()
    self.batchSize = batchSize
    self.encoderUnits = encoderUnits
    self.embedding = tf.keras.layers.Embedding(vocabSize, embedDim)
    self.gru = tf.keras.layers.GRU(self.encoderUnits, name='grutense',
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_initializer='glorot_uniform'
                                   )
    
     
  def call(self, x, hidden):
    x = self.embedding(x)
    
    output, state = self.gru(x, initial_state = hidden)
    
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batchSize, self.encoderUnits))




class Attention(tf.keras.Model):
  def __init__(self, units):
    super(Attention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
  
  def call(self, query, values):
   
    hidden_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
    attentionWeight = tf.nn.softmax(score, axis=1)
    contextVector = attentionWeight * values
    contextVector = tf.reduce_sum(contextVector, axis=1)
    
    return contextVector, attentionWeight




class Decoder(tf.keras.Model):
  def __init__(self, vocabSize, embedDim, decoderUnits, batchSize):
    super(Decoder, self).__init__()
    self.batchSize = batchSize
    self.decoderUnits = decoderUnits
    self.embedding = tf.keras.layers.Embedding(vocabSize, embedDim)
    self.gru = tf.keras.layers.GRU(self.decoderUnits, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocabSize)

    # used for attention
    self.attention = Attention(self.decoderUnits)

  def call(self, x, hidden, encoderOutputs):

    contextVector, attentionWeight = self.attention(hidden, encoderOutputs)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(contextVector, 1), x], axis=-1)
    output, state = self.gru(x)
    output = tf.reshape(output, (-1, output.shape[2]))
    x = self.fc(output)

    return x, state , attentionWeight



def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  lossOb = loss_object(real, pred)

  mask = tf.cast(mask, dtype=lossOb.dtype) 
  lossOb *= mask
  
  return tf.reduce_mean(lossOb)




def modelTraining(inp, targ, encoderHidden,MODE):
  loss = 0
  if MODE == "AutoEncoder":
      targTokenizer = inpQsTrain
  if MODE == "QuestionAnswering":
      targTokenizer = targAsTrain
        
  with tf.GradientTape() as tape:
    encoderOutput, encoderHidden = encoder(inp, encoderHidden)
    decoderHidden = encoderHidden
    decoderInput = tf.expand_dims([targTokenizer.word_index['<start>']] * BATCH_SIZE, 1)       

    for t in range(1, targ.shape[1]):
      predictions, decoderHidden,_ = decoder(decoderInput, decoderHidden, encoderOutput)
      loss += loss_function(targ[:, t], predictions)
      decoderInput = tf.expand_dims(targ[:, t], 1)

    batchLoss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients,variables))
    
  return batchLoss




def evaluate(question,ans,MODE):

    actualAnswer = ans
    question = preprocess_sentence(question)
    inputs = [inpQsTrain.word_index[i] for i in question.split(' ') if i!='']
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=maxLengInpTrain, padding='post')
    inputsTensor = tf.convert_to_tensor(inputs)
    result = ''
    predictSet = []

    hidden = [tf.zeros((1, units))]
    encoderOutput, encoderHidden = encoder(inputsTensor, hidden)

    decoderHidden = encoderHidden
   
    if MODE == "AutoEncoder":
        tokenzr = inpQsTrain
        decoderInput = tf.expand_dims([tokenzr.word_index['<start>']], 0)
        for t in range(maxLengthTargTrain):
            predictions, decoderHidden,_ = decoder(decoderInput, decoderHidden,  encoderOutput)
            predicted_id = tf.argmax(predictions[0]).numpy()
            predictSet.append(predicted_id)
            if tokenzr.index_word[predicted_id] != '<end>':
                result += tokenzr.index_word[predicted_id] + ' '
            else:
                break
            
            decoderInput = tf.expand_dims([predicted_id], 0)
        predictSet = tf.keras.preprocessing.sequence.pad_sequences([predictSet], maxlen=maxLengInpTrain, padding='post')
        predictSetTensor = tf.convert_to_tensor(predictSet)
        correct_prediction = tf.equal( tf.round( predictSetTensor ), tf.round( inputsTensor ) )
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
        return question, result, accuracy, None
    
            

    if MODE == "QuestionAnswering":
        tokenzr = targAsTrain
        decoderInput = tf.expand_dims([tokenzr.word_index['<start>']], 0)
        for t in range(maxLengthTargTrain):
            predictions, decoderHidden,_ = decoder(decoderInput, decoderHidden,  encoderOutput)
            predicted_id = tf.argmax(predictions[0]).numpy()
            predictSet.append(predicted_id)
            #print(predicted_id)
            if predicted_id!=0 and tokenzr.index_word[predicted_id] != '<end>':
                result += tokenzr.index_word[predicted_id] + ' '
            else:
                break
            decoderInput = tf.expand_dims([predicted_id], 0)
            
        predictSet = tf.keras.preprocessing.sequence.pad_sequences([predictSet], maxlen=maxLengthTargTrain, padding='post')
        predictSetTensor = tf.convert_to_tensor(predictSet)
        #print(actualAnswer)
        
        actualAnswer = re.split("_|,",actualAnswer)[0]
        result = result.split(",")[0].replace(" ","")

        actualAnswerWups = wn.synsets(actualAnswer)        
        resultWups = wn.synsets(result)
        if len(actualAnswerWups) >0 and len(resultWups) >0:
            WUPS = actualAnswerWups[0].wup_similarity(resultWups[0])
        else:
            WUPS = 0
            
        if ans.isdigit():
            ans=num2words(ans).replace("point zero","").replace(" ","")
        
        ans = re.split("_|,",ans)
        actualAnswer = [targAsTrain.word_index.get(key.replace(" ","")) for key in ans]
        actualAnswer = [x for x in actualAnswer if x is not None]
        actualAnswer = tf.keras.preprocessing.sequence.pad_sequences([actualAnswer], maxlen=maxLengthTargTrain, padding='post')
        actualAnswerTensor = tf.convert_to_tensor(actualAnswer)
        correct_prediction = tf.equal( tf.round( predictSetTensor ), tf.round( actualAnswerTensor ) )
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
        return question, result, accuracy , WUPS


    return question, result , accuracy, None


def solution(inp,outp,MODE):
    outputDF = pd.DataFrame(columns=["question","result","accuracy","WUPS"])
    for sent,ans in zip(inp, outp) :
        #print(sent,ans)
        question, result, accuracy, WUPS = evaluate(sent,ans,MODE)
        df=pd.DataFrame(data={"question":[question],"result":[result],"accuracy":accuracy, "WUPS":[WUPS]})
        outputDF=outputDF.append(df)
        
    totalAccuracy = outputDF['accuracy'].mean()    
    print("Mean Accuracy is: ",totalAccuracy)
    print("Mean WUPS is : ",outputDF['WUPS'].mean())
    return outputDF
    

def main():
    path_to_file = "D:/Sem 2/TBIR/VQA_Test.txt"
    MODE = "QuestionAnswering"  # Select between "AutoEncoder" and "QuestionAnswering"
    
    inputTensorTrain, targetTensorTrain, inpQsTrain, targAsTrain = load_dataset(path_to_file, None)
    maxLengthTargTrain, maxLengInpTrain = max_length(targetTensorTrain), max_length(inputTensorTrain)
    
    #### Parameters to adjust #####
    BUFFER_SIZE = len(inputTensorTrain)
    BATCH_SIZE = 150
    epochSteps = len(inputTensorTrain)//BATCH_SIZE
    embeddingDim = 256
    units = 555
    vocabSizeInp = len(inpQsTrain.word_index)+1
    vocabSizeTarg = len(targAsTrain.word_index)+1
    EPOCHS = 20
   
    encoder = Encoder(vocabSizeInp, embeddingDim, units, BATCH_SIZE)
    
    if MODE == "AutoEncoder":
        dataset = tf.data.Dataset.from_tensor_slices((inputTensorTrain, inputTensorTrain)).shuffle(BUFFER_SIZE)
        decoder = Decoder(vocabSizeInp, embeddingDim, units, BATCH_SIZE)

    if MODE == "QuestionAnswering":
        dataset = tf.data.Dataset.from_tensor_slices((inputTensorTrain, targetTensorTrain)).shuffle(BUFFER_SIZE)        
        decoder = Decoder(vocabSizeTarg, embeddingDim, units, BATCH_SIZE)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    
    
    
    optimizer = tf.train.AdamOptimizer()
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    checkpoint_dir = 'D:\\Sem 2\\TBIR\\training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
   

    for epoch in range(EPOCHS):
      start = time.time()
      encoderHidden = encoder.initialize_hidden_state()
      totalLoss = 0
      for (batch, (inp, targ)) in enumerate(dataset.take(epochSteps)):
        print(batch)
        batchLoss = modelTraining(inp, targ, encoderHidden,MODE)
        totalLoss +=  batchLoss
        if batch % 20 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch,batchLoss.numpy())) 
      
      checkpoint.save(file_prefix = checkpoint_prefix)
    
      print('Epoch {} Loss {:.4f}'.format(epoch + 1,totalLoss / epochSteps))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
      
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    #g = tf.get_default_graph()
    #w1 = g.get_tensor_by_name('encoder')
    
    path_to_file = "D:/Sem 2/TBIR/VQA_Test.txt"
    lines = io.open(path_to_file).read().strip().split('\n')
    Qs = lines[0::2]
    As = lines[1::2]
    inputTensorTrain, targetTensorTrain, inpQsTrain, targAsTrain = load_dataset(path_to_file, None)
    maxLengthTargTrain, maxLengInpTrain = max_length(targetTensorTrain), max_length(inputTensorTrain)

    # test for t
    outputDF=solution(Qs[0:10],As[0:10],MODE)
    