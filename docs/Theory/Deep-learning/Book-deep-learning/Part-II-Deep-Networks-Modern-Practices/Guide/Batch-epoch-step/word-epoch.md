
# [What is “epoch” in keras.models.Model.fit?](https://stackoverflow.com/questions/44907377/what-is-epoch-in-keras-models-model-fit)



## [A](https://stackoverflow.com/a/44907684)

Here is how Keras [documentation](https://keras.io/getting-started/faq/#what-does-sample-batch-epoch-mean) defines an epoch:

> **Epoch**: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.

So, in other words, a number of epochs means how many times you go through your training set.

The model is updated each time a batch is processed, which means that it can be updated multiple times during one epoch. If `batch_size` is set equal to the length of `x`, then the model will be updated once per epoch.







# [Meaning of an Epoch in Neural Networks Training](https://stackoverflow.com/questions/31155388/meaning-of-an-epoch-in-neural-networks-training)

while I'm reading in how to build ANN in [pybrain](http://pybrain.org/docs/tutorial/fnn.html), they say:

> Train the network for some epochs. Usually you would set something like 5 here,
>
> ```
> trainer.trainEpochs( 1 )
> ```

I looked for what is that mean , then I conclude that we use an epoch of data to update weights, If I choose to train the data with 5 epochs as pybrain advice, the dataset will be divided into 5 subsets, and the wights will update 5 times as maximum.

I'm familiar with online training where the wights are updated after each sample data or feature vector, My question is how to be sure that 5 epochs will be enough to build a model and setting the weights probably? what is the advantage of this way on online training? Also the term "epoch" is used on online training, does it mean one feature vector?



## [A](https://stackoverflow.com/a/31157729)

One epoch consists of *one* full training cycle on the training set. Once every sample in the set is seen, you start again - marking the beginning of the 2nd epoch.

This has nothing to do with batch or online training per se. Batch means that you update *once* at the end of the epoch (after **every** sample is seen, i.e. #epoch updates) and online that you update after **each** *sample* (#samples * #epoch updates).

You can't be sure if 5 epochs or 500 is enough for convergence（收敛） since it will vary from data to data. You can stop training when the error converges or gets lower than a certain threshold. This also goes into the territory of preventing overfitting. You can read up on [early stopping](https://en.wikipedia.org/wiki/Early_stopping#Early_stopping_based_on_cross-validation) and [cross-validation](http://artint.info/html/ArtInt_189.html) regarding that.



