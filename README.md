# Курс по анализу данных и машинному обучению для школ

## Обзор

Курс по анализу данных и машинному обучению с нуля для старшеклассников Москвы. Материалы ориентированы на практику в Google Colab и построены вокруг методологии CRISP-DM: сбор и понимание данных, подготовка, моделирование, оценка и внедрение. Лекции представлены в формате `ipynb`. Приводится вспомогательный CSV `rub-to-usd-eur.csv` для занятия по временным рядам.

## Структура репозитория

| Файл | Тема |
| --- | --- |
| `[DA-1] Datasets and data mining.ipynb` / `.py` | Датасеты, поиск данных, Kaggle, pandas, CSV, API |
| `[DA-2] Pandas data processing.ipynb` / `.py` | Предобработка табличных данных и временных рядов |
| `[DA-3] Mathematical statistics basics.ipynb` / `.py` | Математическая статистика для анализа данных |
| `Synthetic atomic spectrum peaks dataset.ipynb` / `.py` | Генерация синтетического датасета спектров |
| `[ML-01]` … `[ML-12]` | Классическое и глубокое ML, от регрессии до PINN |
| `ipynb2py.py` | Конвертация `.ipynb` → `.py` |
| `rub-to-usd-eur.csv` | Курсы USD и EUR за октябрь 2025 |

## Блок анализа данных

### [DA-1] Датасеты. Поиск данных в интернете, сбор с открытых API, датасеты Kaggle. Обработка CSV с использованием Pandas

#### Обзор

Вводное занятие по этапу Data Understanding и сбору данных в CRISP-DM. Рассматриваются реальные и синтетические датасеты, каталоги открытых данных, продвинутый поиск, встроенные наборы `scikit-learn` и экосистема Kaggle. Базовая работа с `pandas.DataFrame` и CSV.

#### Источники данных

* [AWS Open Data Registry](https://registry.opendata.aws/)
* [Data.gov](https://catalog.data.gov/dataset)
* [WHO GHO](https://www.who.int/data/gho)
* [NASA Earthdata](https://www.earthdata.nasa.gov/)
* [UCI ML Repository](https://archive.ics.uci.edu/)
* [DataSetsAI](https://datasets.ai)
* [SNAP Stanford](http://snap.stanford.edu)
* [Stanford Common Data Set](https://irds.stanford.edu/cds)

Учебные датасеты `scikit-learn`: Iris, Breast Cancer, Diabetes. Обзор соревнований и наборов Kaggle: Titanic, House Prices, Credit Card Fraud, Plant Seedlings, NYC Taxi Trip Duration.

#### Практика

* Создание `DataFrame` из словарей и `numpy`, векторизованные операции.
* Скрейпинг таблицы планет Солнечной системы с [allplanets.ru](http://www.allplanets.ru/solar_sistem.htm), сохранение в `planets.csv`.
* [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata): разбиение по категориям, запись в отдельные CSV, объединение через `glob` и `pd.concat`.
* Semantic Scholar API: поиск статей по запросу `ai, physics` за 2025 год, экспорт в `1k-ai-physics-articles-2025.csv`.

### [DA-2] Обработка данных с использованием Pandas

#### Обзор

Занятие по этапу Data Preparation в CRISP-DM. Индексация `loc` и `iloc`, работа с пропусками, фильтрация, агрегация, feature engineering и визуализация временных рядов.

#### Локальный датасет

`rub-to-usd-eur.csv` — курсы доллара и евра к рублю за октябрь 2025 с пропусками. Практика `bfill`, `ffill`, заполнения средним, `inplace`, `where`, построение отношения EUR/USD.

#### Практика на Kaggle

[TED Talks](https://www.kaggle.com/datasets/ashishjangra27/ted-talks/) на 5 тысяч записей о выступлениях:

* очистка пропусков, `duplicated`, `groupby` по авторам
* сортировка, маски популярности, извлечение `slug` через `apply` и `.str`
* экспорт видео топ-10 авторов в отдельные CSV
* агрегация средних просмотров по годам, гистограмма распределения просмотров

### [DA-3] Основы математической статистики и её применение в анализе данных

#### Обзор

Теория вероятностей и описательная статистика в связке с `pandas`, `seaborn` и `matplotlib`. Очистка выбросов, корреляция и коррелограммы. Два полноценных кейса на Kaggle.

#### Базовые величины

Для выборки $X = \{x_1, \dots, x_N\}$:

* выборочное среднее: $\bar{x} = \frac{1}{N}\sum_{i=1}^N x_i$
* дисперсия и стандартное отклонение $\sigma$
* нормальное распределение, правило 68–95–99.7, критерий 5σ в физике

Межквартильное правило выбросов:

$$\text{outlier} < Q_1 - 1.5 \cdot \mathrm{IQR} \quad \text{или} \quad > Q_3 + 1.5 \cdot \mathrm{IQR}$$

Корреляция Пирсона:

$$\mathrm{corr}_{X,Y} = \frac{\mathrm{cov}(X,Y)}{\sigma_X \sigma_Y}$$

#### Практика на Kaggle

* [Global disaster response 2018–2024](https://www.kaggle.com/datasets/zubairdhuddi/global-daset) — около 50 тысяч событий, анализ `economic_loss_usd`, `boxplot`, очистка по IQR.
* [World University Rankings](https://www.kaggle.com/datasets/aritra100/world-university-ranking) — около 2.7 тысяч вузов, `heatmap` корреляций, `pairplot`, поиск записи МИФИ.

## Блок машинного обучения

### [ML-01] Линейная регрессия

#### Обзор

Формализация регрессии $f: \mathbb{R}^n \to \mathbb{R}$, константное, линейное и полиномиальное приближение, feature engineering. Метрики $MAE$, $MSE$, $RMSE$, $R^2$. Градиентный спуск и реализация регрессора на `numpy`. Регуляризация через штраф $\lambda \|w\|$.

#### Примеры

* синтетическая регрессия с двумя признаками
* баллистика: предсказание $v^2$ по давлению $P$ и глубине $h$ из уравнения Бернулли
* [Predict Calorie Expenditure](https://www.kaggle.com/datasets/adilshamim8/predict-calorie-expenditure) — синтетические данные спортсменов, полный цикл CRISP-DM, $R^2 \approx 0.98$ на тесте

### [ML-02] Логистическая регрессия

#### Обзор

Переход от регрессии к классификации. Перевес $\mathrm{odds} = p/(1-p)$, логит, сигмоида: 

$$
\sigma(z) = 1/(1+e^{-z}),
$$ 

правдоподобие Бернулли, бинарная кросс-энтропия. Градиентный спуск и ручная реализация.

#### Метрики классификации

Accuracy, precision, recall, F1, ROC-AUC, confusion matrix.

#### Практика

* баллистика: классификация снарядов по траектории
* [Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) — 30 признаков ядер клеток, `sklearn` logistic regression, accuracy около 97%
* многоклассовая постановка: softmax, One-vs-All, One-vs-One

### [ML-03] Дерево решений

#### Обзор

Решающие деревья, предикаты, энтропия Шеннона, информационный прирост. Классификация ирисов Фишера. Недостатки деревьев: переобучение, нестабильность.

#### Ансамбли

Бэггинг, бустинг, XGBoost, LightGBM, CatBoost.

#### Практика на Kaggle

[Stellar Classification SDSS17](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) — 100 000 объектов, 17 признаков, три класса: звезда, галактика, квазар. Сравнение `DecisionTreeClassifier`, `RandomForest`, `XGBoost`, `LightGBM`, `CatBoost`.

### [ML-04] K-means

#### Обзор

Обучение без учителя и кластеризация. Алгоритм K-means: инициализация центров, назначение точек, пересчёт средних. Реализация на `numpy`.

#### Выбор числа кластеров

Метод локтя и коэффициент силуэта на синтетических облаках точек.

#### Недостатки

Чувствительность к инициализации, форме кластеров, выбросам и разной плотности. Альтернатива: DBSCAN.

#### Практика на Kaggle

[Customer Transactions](https://www.kaggle.com/datasets/fares279/customers-transactions) — 10 000 клиентов e-commerce 2024–2025, кластеризация на две группы, PCA для визуализации.

### [ML-05] Рекомендательные системы

#### Обзор

Кандидаты, скоринг, ранжирование, reranking. Explicit и implicit feedback, базовая модель $\hat r_{ui} = \mu + b_u + b_i + \text{персонализация}$.

#### Методы

* коллаборативная фильтрация: User-to-User, Item-to-Item
* контентные рекомендации, TF-IDF
* гибридная схема
* метрики: Precision@K, Recall@K, полнота, новизна, разнообразие, serendipity
* меры схожести: косинус, Жаккар, нормализованное расстояние

#### Практика на Kaggle

[Movies and Ratings](https://www.kaggle.com/datasets/nicoletacilibiu/movies-and-ratings-for-recommendation-system) — 10 000 фильмов, 100 000 оценок. Коллаборативная, контентная и гибридная выдача.

### [ML-06] Нейронные сети и PyTorch

#### Обзор

Искусственный нейрон, перцептрон Розенблатта, MLP, прямой и обратный проход. Функции активации: Sigmoid, ReLU, LeakyReLU, ELU, Softplus, GELU. Инициализации Xavier и He, регуляризация, dropout.

#### PyTorch

Тензоры, `CUDA`, `autograd`, обучение MLP на нелинейно разделимых данных.

#### Детектор пиков в спектрах

Практика на [Synthetic Atomic Spectrum Peaks](https://www.kaggle.com/datasets/alexzyukov/discrete-spectrum-peaks):

* `PeakNet`: `Linear` → `Tanh` → `Linear`, `BCEWithLogitsLoss`
* оптимизатор `Adam`, `lr = 10^{-5}`, 200 эпох
* сохранение весов `peaknet.pt`, кривые train/val loss

Подробнее про задачу ниже.

### Synthetic atomic spectrum peaks dataset

#### Обзор

Вспомогательный ноутбук для генерации данных для детекции пиков в одномерных спектрах и публикации на Kaggle.

#### Датасет

* 2000 сэмплов, длина спектра 1000 частотных отсчётов
* от 3 до 10 пиков на сэмпл, асимметричные профили Гаусса, дрейф и шум
* нормировка интенсивности на $[0, 1]$
* бинарная разметка: метка `1` в бине, ближайшем к центру пика

Артефакты:

* `spectra_peaks.npz` — массивы `X`, `Y`
* `samples_metadata.csv` — координаты центров пиков
* `dataset-metadata.json` — метаданные для Kaggle

### [ML-07] Глубокое обучение. Свёрточные нейронные сети

#### Обзор

Ограничения MLP на изображениях, свёртки, многоканальные тензоры, рецептивное поле. Pooling, stride, dilation, Global Average Pooling, residual connections.

#### Датасеты и архитектуры

MNIST, CIFAR-10/100, ImageNet, JFT-300M. LeNet, AlexNet, VGG, Inception, ResNet, MobileNet, EfficientNet. Аугментация, transfer learning и fine-tuning.

#### Практика

[Imagenette2-160](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz) — 10 классов, поднабор ImageNet. ResNet18 с замороженным backbone, обучение классификатора, `torchvision.transforms` и `DataLoader`.

### [ML-08] Генеративный ИИ

#### Обзор

* AE: энкодер–декодер, MSE на изображениях кружков
* GAN: состязание генератора и дискриминатора, риск коллапса моды
* VAE: ELBO, репараметризация, семплирование из латентного пространства
* Диффузия: прямой и обратный процесс шумения, связь с DALLE 2/3, Midjourney, Stable Diffusion

Даны строгие математические постановки задач. Архитектуры протестированы на игрушечной задаче: генерация белых кружочков на темном фоне. Рассмотрены современные подходы к генеративному ИИ.

### [ML-09] Детекция и сегментация на изображениях

#### Обзор

Детекция: sliding window, YOLO, SSD, R-CNN, Selective Search. Семантическая сегментация: encoder–decoder, U-Net, DeepLab. Инстансная сегментация: Mask R-CNN.

#### Практика

[Cat Dataset](https://www.kaggle.com/datasets/samuelayman/cat-dataset) — 1000 изображений, разметка YOLO `bbox`. Hold-out 80/20, класс кошки переиндексирован в `0`, обучение YOLOv8 через Ultralytics.

### [ML-10] Рекуррентные нейронные сети

#### Обзор

Последовательные данные, базовая RNN, затухание градиентов. LSTM: forget, input, output gates и cell state $c_t$. GRU: reset и update gates. Bidirectional RNN для задач с полным контекстом. Формула скрытого состояния RNN:

$$
h_t = \tanh\left(\sum_i w_{hi} h_{t-1,i} + \sum_j w_{xj} x_{t,j} + b\right)
$$

#### Практика

[Mastercard Stock Data](https://www.kaggle.com/datasets/kalilurrahman/mastercard-stock-data-latest-and-updated) — цены с 25.05.2006 по 11.10.2021, 4733 дня. Окно 60 дней, предсказание `High`, сравнение RNN, LSTM, GRU, BiLSTM. RMSE на валидации порядка 7–15 долларов после `MinMaxScaler`.

### [ML-11] Трансформеры

#### Обзор

Архитектура классического трансформера рассмотрена с нуля на основе оригинальной статьи Attention Is All You Need. Токенизация, эмбеддинги, scaled dot-product attention:

$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Multi-Head Attention, энкодер и декодер, позиционные кодировки. Семейства: GPT, BERT, encoder–decoder. Бенчмарки: perplexity, BLEU, ROUGE, GLUE, SuperGLUE, SQuAD. Кратко ViT и Whisper для не-текстовых последовательностей.

#### Inference на Hugging Face

| Задача | Модель |
| --- | --- |
| Токенизация | `DeepPavlov/rubert-base-cased` |
| Sentiment | `blanchefort/rubert-base-cased-sentiment` |
| Zero-shot | `DeepPavlov/xlm-roberta-large-en-ru-mnli` |
| Перевод ru→en | `Helsinki-NLP/opus-mt-ru-en` |
| Суммаризация | `RussianNLP/FRED-T5-Summarizer` |

### [ML-12] Машинное обучение для физических процессов. PINN

#### Обзор

Производные в механике: $v = \dot x$, $a = \ddot x$. Дифференциальные уравнения, начальные и граничные условия. Физически-информированная нейросеть — MLP из 4–8 слоёв по 50–100 нейронов, активации ReLU, GELU или Tanh.

#### Loss PINN

$$\mathrm{Loss} = \lambda_{\mathrm{PDE}} \mathrm{Loss}_{\mathrm{PDE}} + \lambda_{\mathrm{BC}} \mathrm{Loss}_{\mathrm{BC}} + \lambda_{\mathrm{IC}} \mathrm{Loss}_{\mathrm{IC}}$$

Для равноускоренного падения $y_{tt} = -g$ невязка PDE и штраф за начальные $y_0$, $v_0$. Обучение нестабильно: $lr \approx 10^{-3}$, тысячи эпох, периодическое обновление весов $\lambda$.

#### Практика

Сравнение аналитической траектории $y = y_0 + v_0 t - gt^2/2$ и предсказания PINN на $t \in [0, 3]$ с автоматическим дифференцированием в `torch`.
