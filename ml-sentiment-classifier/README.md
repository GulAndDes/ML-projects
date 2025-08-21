классификация отзывов на **положительные** и **отрицательные**.  
Датасет: [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/).

## 🚀 Технологии
- Python 3
- scikit-learn
- nltk
- pandas, numpy

## 📊 Результаты
- Accuracy: ~0.87
- Модель: Logistic Regression + TF-IDF
- Примеры топ-слов:
  Top positive words: ['well' 'enjoyed' 'highly' 'amazing' 'favorite' 'wonderful' 'perfect'
 'best' 'excellent' 'great']
Top negative words: ['worst' 'bad' 'waste' 'awful' 'boring' 'poor' 'worse' 'terrible'
 'nothing' 'poorly']

## 📂 Запуск проекта
1. Склонировать репозиторий
2. Установить зависимости:
   pip install -r requirements.txt
3. Скачать и распаковать датасет aclImdb_v1.tar.gz в папку `data/aclImdb`.
   tar -xvzf aclImdb_v1.tar.gz
4. Запустить обучение:
   python src/train.py


