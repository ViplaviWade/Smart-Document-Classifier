Title: AI-Powered Document Classification Web Application
This project is a web-based document classification system that enables users to upload documents and categorize them into predefined categories using a fine-tuned machine learning model. The system consists of a FastAPI backend, a React frontend, and integrates Hugging Face’s BART-large-MNLI model for classification. PostgreSQL is used for storing document metadata and classification results. The model has been fine-tuned for improved classification accuracy, and the system provides a user-friendly interface for document management.
•	Frontend: Built using React.js, providing an interface for document upload and classification result visualization.
•	Backend: Built with FastAPI, handling document processing, classification, and API requests.
•	Database: PostgreSQL stores document metadata, classification results, and timestamps.

Manually classifying documents is a tedious and error-prone task. This project aims to automate document classification by leveraging NLP techniques such as zero-shot classification and fine-tuned transformers. The application supports document upload in multiple formats (TXT, DOCX, PDF) and classifies them into the following categories:
•	Technical Documentation
•	Business Proposal
•	Legal Document
•	Academic Paper
•	General Article
•	Other

The system provides an intuitive user interface as shown in the screenshot below. We just have to choose a file and click the Upload button for the backend mechanism
 

When the user can upload any document and it is checked for the file type. If the file type is not (txt, docx, pdf) then the error handling of these types is done for user-friendliness
For example, I try to upload a file with ‘.mako’ extension. The system gives me a clear error message that this file type is not supported and also makes me aware that the supported file-type for this system is ‘.txt, .docx, .pdf’
 

Here, I have used zero-shot classification model and we can see that the confidence score is just 21% which clearly indicates that the classifier is not able to predict the document.


After changing the model from zero-shot classification to  Fine-tuned BART the accuracy increases from approximately 21% to 99% 
 

Overview of Document Classification

Document classification involves text feature extraction, vectorization, and supervised learning. Traditional methods like TF-IDF + Logistic Regression work for basic cases but lack deep contextual understanding. Modern transformer-based models such as BART, T5, and RoBERTa outperform traditional models in NLP tasks.
Hugging Face’s BART-large-MNLI Model

BART-large-MNLI is a zero-shot classification model, which enables categorization without requiring task-specific training data. However, fine-tuning on labelled datasets further enhances performance.
•	Frontend: Built using React.js, providing an interface for document upload and classification result visualization.
•	Backend: Built with FastAPI, handling document processing, classification, and API requests.
•	Database: PostgreSQL stores document metadata, classification results, and timestamps.

Document Processing Pipeline
1.	File Upload: Users upload documents through a React UI.
2.	Text Extraction: Extracts content from PDF, DOCX, or TXT files using python-docx, PyMuPDF.
3.	Classification: Uses the fine-tuned BART-large-MNLI model for document classification.
4.	Result Storage: Stores document metadata and classification results in PostgreSQL.
5.	Display Results: Users can view categorized documents and confidence scores in the UI.
   
Model Fine-Tuning Process
1.	Dataset Preparation: 
o	Curated domain-specific datasets for training.
o	Converted text into tokenized inputs using Hugging Face tokenizer.

2.	Training Configuration: 
o	Used AutoModelForSequenceClassification with 6 categories.
o	Fine-tuned using Trainer API with learning rate 2e-5, batch size 4, and 3 epochs.

3.	Model Evaluation: 
o	Evaluated on test data using precision, recall, and F1-score.
o	Improved classification accuracy from 68% (zero-shot) to 84% (fine-tuned model).



When the system tests the ‘zero-shot classification’  for the dataset then we have the accuracy of 68.42%
 

After fine-tuning the model accuracy improves from 68.42% to 80%

 
The fine-tuned model demonstrates a 17.65% improvement in accuracy compared to zero-shot classification.



System Performance
•	Document upload and classification latency: ~2-3 seconds per document.
•	Database query time: ~30-50ms for fetching classification history.

Future Scope
1.	Enhancing Model Accuracy: Further fine-tuning with additional labeled data.
2.	Batch Document Processing: Allowing bulk classification for enterprise use.
3.	Deployment to Cloud: Hosting backend (FastAPI) on AWS/GCP and frontend on Vercel/Netlify.
4.	Explainable AI (XAI): Implementing SHAP or LIME to provide insights into model predictions.
5.	Advanced UI Enhancements: Introducing better result visualization and filter options.
This project successfully integrates AI-powered document classification with an 84% accuracy rate after fine-tuning. The system leverages FastAPI, PostgreSQL, and React to provide a fully functional web application. Future improvements include further model fine-tuning, scalability enhancements, and deployment in a cloud environment to serve real-world applications.

Running the project:
Before running the backend, go to the backend folder and activate the virtual environment as all the dependencies are installed inside the virtual environment.

Backend: uvicorn app.main:app --reload 
starts a FastAPI application using the Uvicorn ASGI server. It looks inside the app/ directory, finds main.py, and runs the FastAPI instance named app. The --reload flag enables automatic server restarts whenever code changes, making development faster and easier.

Frontend: npm start
It runs the start script defined in the package.json file, launching a development server for React.


