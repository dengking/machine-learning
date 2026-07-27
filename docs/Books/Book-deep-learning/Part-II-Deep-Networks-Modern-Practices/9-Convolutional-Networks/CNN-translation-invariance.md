# [What is translation invariance in computer vision and convolutional neural network?](https://stats.stackexchange.com/questions/208936/what-is-translation-invariance-in-computer-vision-and-convolutional-neural-netwo)

翻译不变性

I don't have computer vision background, yet when I read some image processing and convolutional neural networks related articles and papers, I constantly face the term, `translation invariance`, or `translation invariant`.

Or I read a lot that the **convolution operation** provides `translation invariance`?!! what does this mean?
I myself always translated it to myself as if it means if we change an image in any shape, the actual concept of the image doesn't change.

For example if I rotate an image of a lets say tree, it's again a tree no matter what I do to that picture.
And I myself consider all operations that can happen to an image and transform it in a way (crop it, resize it, gray-scale it,color it etc...) to be this way. I have no idea if this is true so I would be grateful if anyone could explain this to me .



## [A](https://stats.stackexchange.com/a/208949)

You're on the right track.

**Invariance** means that you can recognize an object as an object, even when its appearance *varies* in some way. This is generally a good thing, because it preserves the object's identity, category, (etc) across changes in the specifics of the visual input, like relative positions of the viewer/camera and the object.

The image below contains many views of the same statue. You (and well-trained neural networks) can recognize that the same object appears in every picture, even though the actual pixel values are quite different.

[![Various kinds of invariance, demonstrated](https://i.stack.imgur.com/iY5n5.png)](https://i.stack.imgur.com/iY5n5.png)

Note that **translation** here has a [specific meaning](https://en.wikipedia.org/wiki/Translation_(geometry)) in vision, borrowed from geometry. It does not refer to any type of conversion, unlike say, a translation from French to English or between file formats. Instead, **it means that each point/pixel in the image has been moved the same amount in the same direction**. Alternately, you can think of the origin as having been shifted an equal amount in the opposite direction. For example, we can generate the 2nd and 3rd images in the first row from the first by moving each pixel 50 or 100 pixels to the right.



------

One can show that the convolution operator commutes with respect to translation. If you convolve $f$ with $g$, it doesn't matter if you translate the convolved output $f \star g$, or if you translate $f$ or $g$ first, then convolve them. Wikipedia has a [bit more](https://en.wikipedia.org/wiki/Convolution#Translation_invariance).

One approach to **translation-invariant object recognition** is to take a "template" of the object and convolve it with every possible location of the object in the image. If you get a large response at a location, it suggests that an object resembling the template is located at that location. This approach is often called **template-matching**.

------

### Invariance vs. Equivariance

Santanu_Pattanayak's answer ([here](https://stats.stackexchange.com/a/288102/7250)) points out that there is a difference between translation *invariance* and translation *equivariance*. Translation invariance means that the system produces exactly the same response, regardless of how its input is shifted. For example, a face-detector might report "FACE FOUND" for all three images in the top row. Equivariance means that the system works equally well across positions, but its response shifts with the position of the target. For example, a heat map of "face-iness" would have similar bumps at the left, center, and right when it processes the first row of images.

This is is sometimes an important distinction, but many people call both phenomena "invariance", especially since it is usually trivial to convert an equivariant response into an invariant one--just disregard all the position information).



- 2

- Glad I could help. This is one of my big research interests so if there's anything else that would be useful, I'll see what I can do. – [Matt Krause](https://stats.stackexchange.com/users/7250/matt-krause) [Apr 25 '16 at 10:18](https://stats.stackexchange.com/questions/208936/what-is-translation-invariance-in-computer-vision-and-convolutional-neural-netwo#comment397403_208949)

- 

  Could you clarify how **translation invariance** is achieved with CNN? The **activations** of a convolutional layer in a CNN are **not** invariant under **translations**: they move around as the image moves around (i.e., they are equivariant, rather than invarianct, to translations). Those **activations** are usually fed into a **pooling layer**, which also isn't invariant to translations. And pooling layer may feed into a **fully connected layer**. Do the weights in a **fully connected layer** somehow change **transalation equivariant** to translation invariant behavior? – [max](https://stats.stackexchange.com/users/10117/max) [Oct 16 '17 at 8:39](https://stats.stackexchange.com/questions/208936/what-is-translation-invariance-in-computer-vision-and-convolutional-neural-netwo#comment585559_208949)

- 

  @max, Pooling does increase **translation invariance**, especially max-pooling(!), which completely disregards（忽视） spatial（空间） information within the **pooling neighborhood**. See Chapter 9 of Deep Learning [deeplearningbook.org/contents/convnets.html](http://www.deeplearningbook.org/contents/convnets.html) (starting on p. 335). This idea is also popular in neuroscience--the HMAX model (e.g., here: [maxlab.neuro.georgetown.edu/docs/publications/nn99.pdf](http://maxlab.neuro.georgetown.edu/docs/publications/nn99.pdf)) uses a combination of averaging and max-pooling to generate translation (and other kinds of ) invariance. – [Matt Krause](https://stats.stackexchange.com/users/7250/matt-krause) [Oct 16 '17 at 17:16](https://stats.stackexchange.com/questions/208936/what-is-translation-invariance-in-computer-vision-and-convolutional-neural-netwo#comment585726_208949)

- 1

  Oh right, **pooling** provides invariance over small translations (I was thinking about larger shifts, but perhaps each successive layer of pooling can handle progressively larger shifts). But what about the [fully convolutional networks](https://arxiv.org/abs/1412.6806)? Without pooling, what provides (at least approximate) invariance? – [max](https://stats.stackexchange.com/users/10117/max) [Oct 16 '17 at 23:10](https://stats.stackexchange.com/questions/208936/what-is-translation-invariance-in-computer-vision-and-convolutional-neural-netwo#comment585805_208949)

- 

  How do one explain that "speech is translational invariant only along time-axis, but not frequency axis", what does it mean? – [Gene](https://stats.stackexchange.com/users/82085/gene) [Nov 14 '17 at 15:04](https://stats.stackexchange.com/questions/208936/what-is-translation-invariance-in-computer-vision-and-convolutional-neural-netwo#comment595465_208949)

- 1

  @Fredom, that might be better as a new question, but in brief--the audio signal sounds the same even when you shift it forwards in time (e.g., by adding a bunch of silence at the beginning). However, if you shift it in the frequency domain, it *sounds* different: not only is the spectrum shifted, but relationships between frequencies (e.g., harmonics) are also distorted. – [Matt Krause](https://stats.stackexchange.com/users/7250/matt-krause) [Nov 14 '17 at 17:24](https://stats.stackexchange.com/questions/208936/what-is-translation-invariance-in-computer-vision-and-convolutional-neural-netwo#comment595513_208949)





# [Ways of implementing Translation invariance](https://stats.stackexchange.com/questions/233030/ways-of-implementing-translation-invariance)



