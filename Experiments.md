# Writing down the process through which I've come:
===================================================
![alt Remember, kids, the only difference between screwing around and science is writing it down](https://goo.gl/dBsh1g)
## Features for the hatespeech classifier:
------------------------------------------
I started learning the model only on Amazon review dataset which resulted in features taken only from there.
In the very begining they were all the words from the reviews which were around 300k words(features).
Than I hugely reduced them by stemming and ended up with about 25k words(features).
After training a classifier and removing the unused features ended up with ~7.5k features from the review dataset.

I got a pretty good results on the Amazon review dataset:
|Test Accuracy   | 0.9057377049180327|
|Train Accuracy  | 0.9999761723217689|
|Test precision  | 0.9597017694837057|
|Train  precision| 0.999983824528485 |
|Test recall     | 0.9365570599613153|
|Train recall    | 1.0               |
|Test f1         | 0.9362552368675475|
|Train f1        | 0.9999838242668349|

I added hatespeech dataset from twitter and kept the same features. Yet I got good results again, however as the set has
around 4k positive and 4k negative examples from 25k examples in the whole dataset they were not as accurate as the numbers
tell. Whenever I tried some examples by hand I couldn't get any to give me positive result.
Results:
|Test Accuracy   | 0.8242095754290876|
|Train Accuracy  | 0.999232193667856 |
|Test precision  | 0.8710653526834068|
|Train  precision| 0.9995510829826029|
|Test recall     | 0.7953068592057762|
|Train recall    | 0.9987401007919366|
|Test f1         | 0.8191113589886596|
|Train f1        | 0.999234682393193 |

After adding features from the twitter hatespeech dataset:
coming soon