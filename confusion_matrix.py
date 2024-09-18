import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('emotion_detection_model.h5')

# Define image paths and true labels

image_paths = ['dataset/test/angry/PrivateTest_10131363.jpg', 'dataset/test/disgust/PublicTest_8514439.jpg','dataset/test/fear/PrivateTest_11015881.jpg',
               'dataset/test/happy/PrivateTest_251881.jpg', 'dataset/test/sad/PublicTest_40686855.jpg',
               'dataset/test/neutral/PrivateTest_14704134.jpg','dataset/train/sad/Training_10022789.jpg', 'dataset/test/angry/PrivateTest_10131363.jpg',
               'dataset/test/disgust/PublicTest_8514439.jpg', 'dataset/test/angry/PrivateTest_10131363.jpg', 'dataset/test/disgust/PublicTest_8514439.jpg','dataset/test/fear/PrivateTest_11015881.jpg',
               'dataset/test/happy/PrivateTest_251881.jpg', 'dataset/test/sad/PublicTest_40686855.jpg',
               'dataset/test/neutral/PrivateTest_14704134.jpg','dataset/train/sad/Training_10022789.jpg', 'dataset/test/angry/PrivateTest_10131363.jpg',
               'dataset/test/disgust/PublicTest_8514439.jpg','dataset/test/angry/PrivateTest_10131363.jpg', 'dataset/test/disgust/PublicTest_8514439.jpg','dataset/test/fear/PrivateTest_11015881.jpg',
               'dataset/test/happy/PrivateTest_251881.jpg', 'dataset/test/sad/PublicTest_40686855.jpg',
               'dataset/test/neutral/PrivateTest_14704134.jpg', 'dataset/test/neutral/PrivateTest_10086748.jpg']
true_labels = ['angry', 'disgust', 'fear' ,'happy', 'sad', 'sad', 'surprise' ,'angry', 'disgust' ,'angry', 'disgust',
                'fear' ,'happy', 'sad', 'sad', 'surprise' ,'angry', 'disgust' ,'angry', 'disgust', 'fear' ,'happy', 
                'sad', 'sad','neutral']

predicted_labels = []
valid_true_labels = []  # New list to store true labels for successfully processed images
label_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

predicted_labels = []
valid_true_labels = []  # List to store true labels for successfully processed images

for img_path, true_label in zip(image_paths, true_labels):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Unable to load image at path: {img_path}")
        continue
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (48, 48))  # Assuming model expects 48x48 input
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)  # Add the channel dimension (1)

    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    predicted_labels.append(label_names[predicted_label])
    valid_true_labels.append(true_label)  # Append true label if the image was processed

# Ensure both lists are of the same length
if len(valid_true_labels) == len(predicted_labels):
    accuracy = accuracy_score(valid_true_labels, predicted_labels)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    cm = confusion_matrix(valid_true_labels, predicted_labels, labels=label_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Print classification report
    report = classification_report(valid_true_labels, predicted_labels, labels=label_names, target_names=label_names)
    print("Classification Report:")
    print(report)
else:
    print("Error: Inconsistent number of samples between true and predicted labels.")
