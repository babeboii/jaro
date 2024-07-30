# Python3 implementation of above approach
from math import floor, ceil
 
# Function to calculate the
# Jaro Similarity of two s
def jaro_distance(s1, s2):
     
    # If the s are equal
    if (s1 == s2):
        return 1.0
 
    # Length of two s
    len1 = len(s1)
    len2 = len(s2)
 
    # Maximum distance upto which matching
    # is allowed
    max_dist = floor(max(len1, len2) / 2) - 1
 
    # Count of matches
    match = 0
 
    # Hash for matches
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first
    for i in range(len1):
 
        # Check if there is any matches
        for j in range(max(0, i - max_dist), 
                       min(len2, i + max_dist + 1)):
             
            # If there is a match
            if (s1[i] == s2[j] and hash_s2[j] == 0):
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break
 
    # If there is no match
    if (match == 0):
        return 0.0
 
    # Number of transpositions
    t = 0
    point = 0
 
    # Count number of occurrences
    # where two characters match but
    # there is a third matched character
    # in between the indices
    for i in range(len1):
        if (hash_s1[i]):
 
            # Find the next matched character
            # in second
            while (hash_s2[point] == 0):
                point += 1
 
            if (s1[i] != s2[point]):
                t += 1
            point += 1
    t = t//2
 
    # Return the Jaro Similarity
    return (match/ len1 + match / len2 +
            (match - t) / match)/ 3.0
 
# Driver code
s1 = """In the context of deep learning and Python, troubleshooting the scheduler for getting the learning rate (LR) using an optimizer can be a common task. Here are two paragraphs to help you with this process:

Paragraph 1:
The learning rate (LR) is a crucial hyperparameter in deep learning models that determines the step size of the optimization algorithm. The scheduler is responsible for adjusting the learning rate during training. In Python, you can use various optimizers such as Adam, SGD, or RMSprop, along with their respective schedulers. To troubleshoot the scheduler for getting the learning rate, first ensure that you have imported the necessary libraries, including the optimizer and scheduler classes. For example, if you are using the Adam optimizer with the ReduceLROnPlateau scheduler, you would import them as follows:

```python
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
```

Paragraph 2:
Next, create an instance of the optimizer and the scheduler. In this case, you would set up the Adam optimizer and the ReduceLROnPlateau scheduler as follows:

```python
optimizer = Adam(learning_rate=0.001)
scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001)
```

Here, the ReduceLROnPlateau scheduler is configured to monitor the validation loss, reduce the learning rate by a factor of 0.1 when the validation loss stops improving for 3 epochs, and set the minimum learning rate to 0.0001.

To troubleshoot the scheduler, you can check if the scheduler is being called during training and if the learning rate is being updated as expected. You can also verify that the monitoring metric (in this case, 'val_loss') is being tracked correctly. If you encounter issues, ensure that the scheduler is being added to the model's callbacks list and that the training process is running for a sufficient number of epochs to allow the scheduler to make adjustments.

By following these steps, you should be able to troubleshoot the scheduler for getting the learning rate using an optimizer in Python."""
s2 = """In the context of deep learning and Python, troubleshooting the scheduler for getting the learning rate (LR) using an optimizer can be a common task. Here are two paragraphs to help you with this process:

First, ensure that you have imported the necessary libraries and defined your model, optimizer, and scheduler. In Python, you can use libraries like TensorFlow, PyTorch, or Keras to create your model. For optimizers, you can use Adam, SGD, or RMSprop, and for schedulers, you can use ReduceLROnPlateau, StepLR, or CosineAnnealingLR.

Once you have defined your model, optimizer, and scheduler, check if the learning rate is being updated correctly. You can print the learning rate at different training steps to see if it is changing as expected. If the learning rate is not being updated, verify that you have set the correct parameters for your scheduler, such as the patience, step size, or minimum and maximum learning rates. Additionally, ensure that you have called the scheduler's `step()` or `lr_update()` method after each training step.

If the learning rate is still not being updated correctly, you may want to check your training code for any potential issues. Ensure that you are using the correct optimizer and scheduler methods, and that the learning rate is not being overwritten by other parts of your code. If you are still having trouble, consider seeking help from online forums or communities dedicated to deep learning in Python."""
 
# Prjaro Similarity of two s
print(round(jaro_distance(s1, s2),6))