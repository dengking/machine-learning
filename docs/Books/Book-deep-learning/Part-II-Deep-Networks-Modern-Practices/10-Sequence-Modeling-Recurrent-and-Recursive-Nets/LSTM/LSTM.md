# [Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory)



**Long short-term memory** (**LSTM**) is an artificial [recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network) (RNN) architecture[[1\]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-lstm1997-1) used in the field of [deep learning](https://en.wikipedia.org/wiki/Deep_learning). Unlike standard [feedforward neural networks](https://en.wikipedia.org/wiki/Feedforward_neural_network), LSTM has feedback connections. It can not only process single data points (such as images), but also entire **sequences of data** (such as speech or video). For example, LSTM is applicable to tasks such as unsegmented, connected [handwriting recognition](https://en.wikipedia.org/wiki/Handwriting_recognition)[[2\]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-2) or [speech recognition](https://en.wikipedia.org/wiki/Speech_recognition).[[3\]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-sak2014-3)[[4\]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-liwu2015-4) [Bloomberg Business Week](https://en.wikipedia.org/wiki/Bloomberg_Business_Week) wrote: "These powers make LSTM arguably the most commercial AI achievement, used for everything from predicting diseases to composing music."[[5\]](https://en.wikipedia.org/wiki/Long_short-term_memory#cite_note-bloomberg2018-5)

A common **LSTM unit** is composed of a **cell**, an **input gate**, an **output gate** and a **forget gate**. The cell remembers values over arbitrary time intervals and the three *gates* regulate（调节） the flow of information into and out of the cell.




LSTM networks are well-suited to [classifying](https://en.wikipedia.org/wiki/Classification_in_machine_learning), [processing](https://en.wikipedia.org/wiki/Computer_data_processing) and [making predictions](https://en.wikipedia.org/wiki/Predict) based on [time series](https://en.wikipedia.org/wiki/Time_series) data, since there can be lags（落后） of unknown duration between important events in a time series. LSTMs were developed to deal with the exploding and [vanishing](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) gradient problems that can be encountered when training traditional RNNs. Relative insensitivity to gap length is an advantage of LSTM over RNNs, [hidden Markov models](https://en.wikipedia.org/wiki/Hidden_Markov_models) and other sequence learning methods in numerous applications.[*citation needed*]

