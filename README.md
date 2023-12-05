
### For custom NER we need to label and annotate our text data. In this solution we will use a free open source tool called "NER Annotator for Spacy", which is a good alternative to Doccano and Prodigy.



##### NER Annotator - https://tecoholic.github.io/ner-annotator/
##### Model weights - https://drive.google.com/drive/folders/11Gay0NXj9IRf8Jb-Don_yp9UhWH46t8x

### Example 1


```python
import spacy
from spacy import displacy
nlp_ner = spacy.load("model-best")

doc = nlp_ner("""
Far from the well-trodden paths, Mount Fitz Roy in the remote Patagonian Andes silently stands as a testament to nature's raw beauty, with its jagged peaks, pristine glaciers, and remote wilderness, inviting those seeking solitude and untamed landscapes to embark on an off-the-beaten-path adventure
""")
displacy.render(doc, style="ent", jupyter=True)


```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr"><br>Far from the well-trodden paths, Mount 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Fitz Roy
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
 in the remote 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Patagonian
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Andes
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
 silently stands as a testament to nature's raw beauty, with its jagged peaks, pristine glaciers, and remote wilderness, inviting those seeking solitude and untamed landscapes to embark on an off-the-beaten-path adventure<br></div></span>


##### Although the training dataset was small, the model captures the context and correctly identifies the target class

### Example 2


```python
import spacy
from spacy import displacy
nlp_ner = spacy.load("model-best")

doc = nlp_ner("""
The towering peaks of the Himalayas, including Mount Everest, the world's highest mountain, attract adventurous mountaineers from around the globe. The Rocky Mountains in North America boast breathtaking landscapes, with rugged summits and picturesque alpine meadows. In the Andes of South America, iconic peaks like Aconcagua stand as testaments to the region's natural beauty. The Swiss Alps are renowned for their majestic snow-covered peaks and world-class ski resorts, making them a popular destination for winter sports enthusiasts.
""")
displacy.render(doc, style="ent", jupyter=True)


```


<span class="tex2jax_ignore"><div class="entities" style="line-height: 2.5; direction: ltr"><br>The towering peaks of the 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Himalayas
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
, including Mount 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Everest
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
, the world's highest mountain, attract adventurous mountaineers from around the globe. The 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Rocky
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
 Mountains in North America boast breathtaking landscapes, with rugged summits and picturesque alpine meadows. In the 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Andes
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
 of South America, iconic peaks like 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Aconcagua
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
 stand as testaments to the region's natural beauty. The Swiss Alps are renowned for their majestic snow-covered peaks and world-class ski resorts, making them a popular destination for winter sports enthusiasts.<br></div></span>


### Example 3


```python
import spacy
from spacy import displacy
nlp_ner = spacy.load("model-best")

doc = nlp_ner("""
Mont-Saint-Michel is a historic architectural structure, not a high mountain, but it can serve as an example to illustrate the use of a hyphen in a name.
""")
displacy.render(doc, style="ent", jupyter=True)


```



<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Mont-Saint
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
-
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Michel is
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">MOUNTAIN</span>
</mark>
 a historic architectural structure, not a high mountain, but it can serve as an example to illustrate the use of a hyphen in a name.<br></div></span>


##### In this example, you can see that the model does not perfectly classify mountains with hyphens in the name. To improve the quality of the model, the training and test dataset can be increased, paying attention to the names of mountains with hyphens.
