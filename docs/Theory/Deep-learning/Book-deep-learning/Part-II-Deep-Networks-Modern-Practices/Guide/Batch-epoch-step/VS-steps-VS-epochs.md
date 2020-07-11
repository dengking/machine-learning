# [What is the difference between steps and epochs in TensorFlow?](https://stackoverflow.com/questions/38340311/what-is-the-difference-between-steps-and-epochs-in-tensorflow)



In most of the models, there is a *steps* parameter indicating the *number of steps to run over data*. But yet I see in most practical usage, we also execute the fit function N *epochs*.

What is the difference between running 1000 steps with 1 epoch and running 100 steps with 10 epoch? Which one is better in practice? Any logic changes between consecutive epochs? Data shuffling?

***COMMENTS*** : 

**Jason Brownlee** at machinelearningmastery.com has a very nice, [detailed answer](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/) to exactly that question. – [BmyGuest](https://stackoverflow.com/users/1302888/bmyguest) [Apr 16 at 20:12](https://stackoverflow.com/questions/38340311/what-is-the-difference-between-steps-and-epochs-in-tensorflow#comment98112303_38340311)



## [A](https://stackoverflow.com/a/38340420)

An epoch usually means one iteration over all of the training data. For instance if you have 20,000 images and a **batch size** of 100 then the epoch should contain 20,000 / 100 = 200 **steps**. However I usually just set a fixed number of steps like 1000 per epoch even though I have a much larger data set. At the end of the **epoch** I check the **average cost** and if it improved I save a checkpoint. There is no difference between steps from one epoch to another. I just treat them as checkpoints.

People often **shuffle** around the data set between epochs. I prefer to use the `random.sample` function to choose the data to process in my epochs. So say I want to do 1000 steps with a batch size of 32. I will just randomly pick 32,000 samples from the pool of training data.



## [A](https://stackoverflow.com/a/44416034)

A training step is one gradient update. In one step batch_size many examples are processed.

An epoch consists of one full cycle through the training data. This is usually many steps. As an example, if you have 2,000 images and use a batch size of 10 an epoch consists of 2,000 images / (10 images / step) = 200 steps.

If you choose our training image randomly (and independent) in each step, you normally do not call it epoch. [This is where my answer differs from the previous one. Also see my comment.]