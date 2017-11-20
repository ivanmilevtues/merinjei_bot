# Writing down the process through which I've gone:
<img src="http://i.imgur.com/Xedui4H.jpg" width="100%"></img>
## Features for the hatespeech classifier:
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