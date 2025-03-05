
# Movie Genre Classification Using NLP

Author : [SyedFurqanAli](https://github.com/SyedFurqanAli540702)

## 📌 Project Description  
This project focuses on **Movie Genre Classification** using **Natural Language Processing (NLP)** techniques. The model automatically predicts the genres of movies based on their plot summaries from the IMDb dataset. It is a **multi-label classification task** where each movie can belong to one or more genres like Action, Comedy, Drama, Thriller, etc.

## 🔑 Features  
- Text Preprocessing (Cleaning, Stopword Removal)  
- Genre Label Encoding using **MultiLabelBinarizer**  
- Feature Extraction using **TF-IDF Vectorization**  
- Multi-Label Classification with **Logistic Regression**  
- Model Evaluation (Accuracy, Precision, Recall, F1-Score)  

## 🛠️ Technologies Used  
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Logistic Regression  

## 📄 Dataset  
The dataset is sourced from **Kaggle** and contains:  
- Movie Plot Summaries  
- Genres  
Each movie can have multiple genres.  

## 🔥 Project Structure  
```bash
├── dataset/
│   └── genre_classification.csv
├── movie_genre_classification.py
└── README.md
```

## ⚙️ How to Run  
1. Clone the repository:  
```bash
git clone https://github.com/YourUsername/Movie-Genre-Classification.git
cd Movie-Genre-Classification
```
2. Install Dependencies:  
```bash
pip install -r requirements.txt
```
3. Run the Project:  
```bash
python movie_genre_classification.py
```

## 🎯 Output Example  
```
Accuracy: 0.78  
Precision: 0.82  
Recall: 0.75  
```

## 📌 Future Improvements  
- Implement Deep Learning Models like **LSTM**  
- Use **BERT** for better text representations  
- Web App Deployment using Flask  

## 🤝 Contributing  
Contributions are welcome! Feel free to raise issues or submit pull requests.  

## 📝 License  
This project is licensed under the **MIT License**.  
