from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
  
  # Dados de exemplo
texts = [
  "O novo lançamento da Apple",
  "Resultado do jogo de ontem",
  "Eleições presidenciais",
  "Atualização no mundo da tecnologia",
  "Campeonato de futebol",
  "Política internacional"
]
categories = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política"]
  
# Convertendo textos em uma matriz de contagens de tokens
vectorizer = CountVectorizer()
document_matrix = vectorizer.fit_transform(texts)
  
# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(document_matrix, categories, test_size=0.5, random_state=42)
  
# Treinando o classificador
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
  
# Predição e Avaliação
y_prediction = classifier.predict(X_test)

for idx, prediction in enumerate(y_prediction):
    print(f"Texto {idx+1}: Predição: {prediction}")

for idx, (pred, actual) in enumerate(zip(y_prediction, y_test)):
    print(f"Texto {idx+1}: Predição: {pred}, Real: {actual}")