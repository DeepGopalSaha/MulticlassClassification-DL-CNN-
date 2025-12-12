# Understanding Training Outputs (Simple Explanation)

## 1. Training Setup Summary

- **Type of problem:** Multiclass Classification (10 classes)
- **Loss function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Learning rate scheduler:** Step function (StepLR)  
  - Every few epochs → LR is divided by 10  
- **Batch size:** 32  
- **Model output per image:** 10 class scores  

---

## 2. What is `x`?

`x` is the **input batch of images** after normalization.

Example pixel values:

[-2.03, -1.80, -0.55, 0.22, ...]


These are NOT predictions.  
They are just pixel intensities transformed into a normalized range.

### Shape of `x`:
x.shape = (32, 3, 224, 224)

Meaning:
- 32 images in a batch  
- each has 3 channels (RGB)  
- each is 224×224 pixels  

---

## 3. What is `y`?

`y` is the **actual class label** for each image.

Example: tensor([2, 1, 4, 3, 9, 0, 7, 6, 8, 1, ...])


### Shape of `y`:
y.shape = (32,)
- For 32 images.  
- Each label is a number between **0–9**, representing a class.

---

## 4. What is `out`?

`out` is the **model's raw output**, also called **logits**.

It contains 10 values per image (because there are 10 classes).

Example for one image:
[-0.2860, -1.4501, -1.8354, -0.7878, 0.0260, -1.6157, -2.5885, -2.4276, -0.2515, -0.9138] (10 values,each for one class)


### Shape of `out`:
out.shape = (32, 10)

Meaning:
- 32 rows = 32 images  
- 10 columns = score for each class  

The **highest value in each row is the predicted class**.

---

## 5. What is `preds`?

`preds` contains the **predicted class number** for each image.

Computed as:
preds = out.argmax(1)
Example: tensor([2, 1, 4, 3, 9, 0, 7, 6, 8, 1, ...])

Shape of preds:
preds.shape = (32,)
One predicted class per image.

## 6. What is loss?

loss is a single number showing how wrong the model is for the batch.

Example: tensor(1.4232)

- This comes from CrossEntropyLoss.
- Lower loss = better performance.

## 7. Counting Accuracy

In the training loop:

- correct += (preds == y).sum().item() [counts number of correct predictions]
- total += y.size(0) [returns first value i.e 32]

This means:

- Compare predicted labels (preds) with true labels (y)

- Count how many were correct

- Add batch size (32) to total images

- Accuracy = correct / total
## 8. Accuracy metric
Main Metric:
Final Validation Accuracy: 95%

The following table shows precision, recall, and F1-score for each class.

- Precision = how many classes were correctly predicted? (more precision less FALSE POSITIVES)
- Recall =  how many classes were correctly detected? (more recall less FALSE NEGATIVES)
- F1 Score= Balanced measure of precision + recall
 
| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| cane         | 0.93      | 0.98   | 0.95     | 973     |
| cavallo      | 0.92      | 0.92   | 0.92     | 525     |
| elefante     | 0.95      | 0.97   | 0.96     | 290     |
| farfalla     | 0.96      | 0.96   | 0.96     | 423     |
| gallina      | 0.99      | 0.96   | 0.97     | 620     |
| gatto        | 0.97      | 0.92   | 0.94     | 334     |
| mucca        | 0.93      | 0.84   | 0.88     | 374     |
| pecora       | 0.88      | 0.93   | 0.90     | 364     |
| ragno        | 0.98      | 0.98   | 0.98     | 965     |
| scoiattolo   | 0.98      | 0.96   | 0.97     | 373     |
|              |           |        |          |         |
| **accuracy** | -         | -      | **0.95** | 5241    |
| macro avg    | 0.95      | 0.94   | 0.94     | 5241    |
| weighted avg | 0.95      | 0.95   | 0.95     | 5241    |


##  Class-Wise Performance Summary

---

## Best and Worst by Precision

###  Best Precision → **gallina (0.99)**  
- When the model predicts **“gallina”**, it is correct **99% of the time**.  
- This means **almost zero false positives** for this class.

###  Worst Precision → **pecora (0.88)**  
- The model sometimes predicts **“pecora”** when the image belongs to another animal.  
- This indicates **more false positives** for the pecora class.

---

##  Best and Worst by Recall

###  Best Recall → **ragno (0.98)**  
- The model correctly identifies **almost all spider images**.  
- Very few spiders were missed.  
- Indicates **excellent sensitivity** for this class.

###  Worst Recall → **mucca (0.84)**  
- The model struggles to detect cow images accurately.  
- **16% of cow images were misclassified** as other animals.  
- This makes **mucca the hardest class** for the model.

---

## Best and Worst by F1-Score

### Best F1-Score → **ragno (0.98)**  
- High precision and high recall.  
- Very reliable predictions for spider images.

### Worst F1-Score → **mucca (0.88)**  
- Lower precision and recall combined.  
- Consistent difficulty in predicting cow images.  
- Indicates the model is **least confident and least accurate** for this class.

---

## Final Interpretation Summary

### Worst-Performing Class → **mucca (cow)**  
- Lowest recall (0.84)  -> hard to detect (more false negatives) than other classes
- Lowest F1-score (0.88)  
- Model frequently **confuses cows with other animals** 

### Best-Performing Classes → **gallina (hen)** & **ragno (spider)**  
- **gallina** has the **highest precision (0.99)** → extremely accurate predictions  
- **ragno** has the **highest F1-score (0.98)** and excellent recall → consistently strong performance  

---
