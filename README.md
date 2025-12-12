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
