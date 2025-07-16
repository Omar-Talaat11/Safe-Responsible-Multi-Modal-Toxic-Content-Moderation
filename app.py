import warnings
warnings.filterwarnings("ignore")

import re
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
from transformers import AutoTokenizer, DistilBertForSequenceClassification, BlipProcessor, BlipForConditionalGeneration
import requests
import streamlit as st
from PIL import Image
import torch.nn.functional as F

classifier_model = DistilBertForSequenceClassification.from_pretrained("my_model", num_labels=4)
classifier_tokenizer = AutoTokenizer.from_pretrained("my_model")
encoder = joblib.load("my_model/label_encoder.pkl")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to('cuda')

def preprocess_query(new_query):
    wordnet=WordNetLemmatizer()
    query = re.sub('[^a-zA-Z]', ' ', new_query) # Remove regex
    query = query.lower() # Lower case all the text
    query = ' '.join(query.split()) # Remove extra whitespace
    query_tokens = query.split() # Split the query into words to be able to process them
    query_tokens = [word for word in query_tokens if not word in set(stopwords.words('english'))] # removing stop words
    query_tokens = [wordnet.lemmatize(word) for word in query_tokens] # Lemmatize the words
    query = ' '.join(query_tokens) # Join the words back into a string
    
    inputs = classifier_tokenizer(query, truncation=True, padding="max_length", return_tensors="pt")
    return inputs

def predict_label(processed_query):
    classifier_model.eval()
    with torch.no_grad():
        output = classifier_model(**processed_query)
        logits = output.logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()  # shape: (num_classes,)
        pred_index = torch.argmax(logits, dim=1).item()
        pred_class = encoder.inverse_transform([pred_index])[0]
        
        # Map probabilities to class labels
        label_names = encoder.inverse_transform(range(len(probs)))
        label_probs = dict(zip(label_names, probs))
        
        return pred_class, label_probs



headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

def classify_text(text):
    data = {
        "model": "meta-llama/llama-guard-3-8b",
        "max_tokens": 10,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict safety classifier. Respond with only one word: either 'safe' or 'unsafe'. "
                    "Do not add punctuation, explanation, or formatting."
                )
            },
            {"role": "user", "content": text}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    content = response.json()["choices"][0]["message"]["content"].strip().lower()

    # Only return the first keyword it says
    if "safe" in content:
        return "safe"
    else:
        return "unsafe"




st.title("📷📝 Image Captioning & Text Classification App")
option = st.radio("Choose input type:", ["Text", "Image"])

if option == "Text":
    user_input = st.text_area("Enter your text:")
    
    if st.button("Classify"):

        result_1 = classify_text(user_input)
        
        if(result_1 == "safe"):
            processed_text = preprocess_query(user_input)
            result_2, prob_dict = predict_label(processed_text)

            st.subheader(f"Classifier Prediction: {result_2}")
            st.write("Class probabilities:")
            for label, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
                st.write(f"- **{label}**: {prob:.2%}")
            
            if(result_2 == "Safe"):
                st.success(f"Text is classified as: {result_2}")
            
            else:
                st.success(f"Text is classified as unsafe by soft classifier: {result_2}")
        else:
                st.success(f"Text is classified as unsafe by hard classifier: {result_1}")
        
        
elif option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        inputs = processor(images=image, return_tensors="pt").to('cuda')
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        st.subheader(caption)
        # st.text(f"this is an image of : {caption}")
        # print("Generated Caption:", caption)
        
        if st.button("Classify"):

            result_1 = classify_text(caption)
            
            if(result_1 == "safe"):
                processed_text = preprocess_query(caption)
                result_2, prob_dict = predict_label(processed_text)

                st.subheader(f"Classifier Prediction: {result_2}")
                st.write("Class probabilities:")
                for label, prob in sorted(prob_dict.items(), key=lambda x: -x[1]):
                    st.write(f"- **{label}**: {prob:.2%}")
                
                if(result_2 == "Safe"):
                    st.success(f"Image is classified as: {result_2}")
                
                else:
                    st.success(f"Image is classified as unsafe by soft classifier: {result_2}")
            
            else:
                    st.success(f"Image is classified as unsafe by hard classifier:: {result_1}")
            
            # st.success(f"Generated Caption: {caption}")