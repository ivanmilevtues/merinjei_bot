# Writing down the process through which I've gone:
<img src="http://i.imgur.com/Xedui4H.jpg" width="100%"></img>
## Features for the hatespeech classifier:
### All the Tables of results are on the RandomForestClassifier
I started learning the model only on Amazon review dataset which resulted in features taken only from there.
In the very begining they were all the words from the reviews which were around 300k words(features).
Than I hugely reduced them by stemming and ended up with about 25k words(features).
After training a classifier and removing the unused features ended up with ~7.5k features from the review dataset.

I got a pretty good results on the Amazon review dataset:
<table>
<tr><td>Test Accuracy   </td><td> 0.9057377049180327</td></tr>
<tr><td>Train Accuracy  </td><td> 0.9999761723217689</td></tr>
<tr><td>Test precision  </td><td> 0.9597017694837057</td></tr>
<tr><td>Train  precision</td><td> 0.999983824528485 </td></tr>
<tr><td>Test recall     </td><td> 0.9365570599613153</td></tr>
<tr><td>Train recall    </td><td> 1.0               </td></tr>
<tr><td>Test f1         </td><td> 0.9362552368675475</td></tr>
<tr><td>Train f1        </td><td> 0.9999838242668349</td></tr>
</table>

--------------------

I added hatespeech dataset from twitter and kept the same features. Yet I got good results again, however as the set has
around 4k positive and 4k negative examples from 25k examples in the whole dataset they were not as accurate as the numbers
tell. Whenever I tried some examples by hand I couldn't get any to give me positive result.
Results:
<table>
<tr><td>Test Accuracy   </td><td> 0.8242095754290876</td></tr>
<tr><td>Train Accuracy  </td><td> 0.999232193667856 </td></tr>
<tr><td>Test precision  </td><td> 0.8710653526834068</td></tr>
<tr><td>Train  precision</td><td> 0.9995510829826029</td></tr>
<tr><td>Test recall     </td><td> 0.7953068592057762</td></tr>
<tr><td>Train recall    </td><td> 0.9987401007919366</td></tr>
<tr><td>Test f1         </td><td> 0.8191113589886596</td></tr>
<tr><td>Train f1        </td><td> 0.999234682393193 </td></tr>
</table>

----------

After adding features from the twitter hatespeech dataset:
Firstly the features were 30K which is way too much and this was caused of the fact that user names were added.
After reducing them so that there are no features which are used only once ended up with 6.7K features.
After concatenating the hatespeech features and those from the reviews I ended up with 48K full features and
13K reduced.
I used only the features from the hate speech data to generate the data and learn the model on them.
It had good paper results on the RandomForestClassifier:
<table>
<tr><td>Test Accuracy   </td><td> 0.8586805555555556</td></tr>
<tr><td>Train Accuracy  </td><td> 0.9997222350817092</td></tr>
<tr><td>Test precision  </td><td> 0.8654183605544065</td></tr>
<tr><td>Train  precision</td><td> 0.9997895538721652</td></tr>
<tr><td>Test recall     </td><td> 0.854884340156432 </td></tr>
<tr><td>Train recall    </td><td> 0.9994564035659926</td></tr>
<tr><td>Test f1         </td><td> 0.8346738159070597</td></tr>
<tr><td>Train f1        </td><td> 0.9996737712048717</td></tr>
</table>
However it had bad time on recognizing on live examples tested by me. The result was that it mostly got 0 as result
which means it is hate/negative sentence. After examining the twitter hate speech dataset I ended up with the conclusion
that its "positive" examples are in fact not so positive they are just negative. So they don't give good features which
can be used in order to recognise positives.

------------------------------------------

Using the mixed features has good results overall however when live testing it had some problems with very short sentences
like:
```
Definitely one of the better products out there!
I am in love with your product!
```
Table of results:

<table>
<tr><td>Test Accuracy   </td><td> 0.8843803704410922</td></tr>
<tr><td>Train Accuracy  </td><td> 0.9997135855901728</td></tr>
<tr><td>Test precision  </td><td> 0.7495532905040647</td></tr>
<tr><td>Train  precision</td><td> 0.9992909245585498</td></tr>
<tr><td>Test recall     </td><td> 0.8343055555555555</td></tr>
<tr><td>Train recall    </td><td> 0.9996281836772635</td></tr>
<tr><td>Test f1         </td><td> 0.8322249930728733</td></tr>
<tr><td>Train f1        </td><td> 0.9995817260770554</td></tr>
</table>

As you can see the precision is a bit on the lower side and to me this means that it predicts with an ease the negatives
so I think that the labels should be changed in order to have better precision as the idea of the classifier is mostly to
get the negative examples right!

-----------------------------------------

Using TF IDF in order to reduce the impact of the size of the comment.
So after my last tests with the Bag of words features I saw the pattern that short comments tend to lead into being negative
even though that they were strongly positive. So I implemented the TF.IDF over the BagOfWords and it has improved the models accuracy on paper also in practice.

Table of results:
<table>
<tr><td>Test Accuracy   </td><td> 0.8423715867863281</td></tr>
<tr><td>Train Accuracy  </td><td> 0.9931976036470391</td></tr>
<tr><td>Test precision  </td><td> 0.8653890923746133</td></tr>
<tr><td>Train  precision</td><td> 0.9952123962300639</td></tr>
<tr><td>Test recall     </td><td> 0.8484093241460197</td></tr>
<tr><td>Train recall    </td><td> 0.9915920704526511</td></tr>
<tr><td>Test f1         </td><td> 0.8751606805293005</td></tr>
<tr><td>Train f1        </td><td> 0.9948188412383878</td></tr>
</table>

Therefore one of the examples lastly shown still doesn't pass:
```
I am in love with your product!
```