🛡️ Multi-Modal Toxic Content Moderation
This project is focused on detecting and classifying toxic content from both text and images. It uses fine-tuned transformer models, image captioning, and a Streamlit interface to allow users to test and visualize predictions.

📌 Features
✅ Classify text into multiple toxic categories using DistilBERT fine-tuned with LoRA

🖼️ Classify images by generating captions using BLIP and feeding them to the text classifier

🧠 Includes a hard classifier (LLaMA Guard) to ensure safety before detailed classification

📊 Balanced dataset through data cleaning, class merging, and augmentation with BERT

🌐 Deployed as an interactive Streamlit app

🏁 How It Works
Text Input:

Cleaned and lemmatized

Passed to LLaMA Guard (hard classifier)

If safe, passed to DistilBERT (soft classifier) for toxic category classification

Image Input:

Caption generated using BLIP

Caption goes through same text classification pipeline

📊 Model Info
Model: DistilBERT fine-tuned with LoRA

Tokenizer: distilbert-base-uncased

Additional tools:

BLIP for image captioning

LLaMA Guard via OpenRouter for safety filtering

nlpaug for BERT-based data augmentation

📎 Notes
Handles class imbalance by merging rare classes and augmenting mid-range ones

Evaluation includes classification report and confusion matrix

Final model is saved for inference (my_model/ folder)
