# New ML Practice Categories (10-20)

This document describes **51 new machine learning practice examples** added to the repository, organized into 11 comprehensive categories covering advanced ML topics.

## 📊 Overview

**Total new examples**: 51 Python scripts
**Categories added**: 11 (Categories 10-20)
**Focus areas**: Time Series, Computer Vision, NLP, Recommender Systems, Anomaly Detection, Interpretability, AutoML, Reinforcement Learning, Multi-Modal, MLOps, and **Ensemble Methods**

### 📂 Directory Structure

All examples are located in: `/code_folder/`

```
ml_problem_practice/
└── code_folder/
    ├── 10_*.py  # Time Series Forecasting (4 files)
    ├── 11_*.py  # Computer Vision (4 files)
    ├── 12_*.py  # NLP (5 files)
    ├── 13_*.py  # Recommender Systems (4 files)
    ├── 14_*.py  # Anomaly Detection (4 files)
    ├── 15_*.py  # Model Interpretability (4 files)
    ├── 16_*.py  # AutoML (4 files)
    ├── 17_*.py  # Reinforcement Learning (4 files)
    ├── 18_*.py  # Multi-Modal Learning (4 files)
    ├── 19_*.py  # MLOps & Production (8 files) ⭐ +3 NEW
    ├── 20_*.py  # Ensemble Methods (6 files)
    └── README_NEW_CATEGORIES.md  # This file
```

### 🎯 Naming Convention

Files follow the pattern: `<category_number>_<framework>_<topic>.py`

**Examples**:
- `10_sklearn_time_series_arima.py` - Category 10, scikit-learn, ARIMA
- `11_torch_object_detection_yolo.py` - Category 11, PyTorch, YOLO
- `20_xgboost_gradient_boosting.py` - Category 20, XGBoost, gradient boosting

---

## Category 10: Time Series Forecasting (4 files)

### 📖 Background

Time series forecasting is the task of predicting future values based on previously observed values ordered in time. Unlike traditional ML problems, time series data has temporal dependencies where order matters. Applications include stock price prediction, weather forecasting, sales forecasting, and demand planning.

**Key Concepts**: Stationarity, seasonality, trends, autocorrelation, ARIMA models, recurrent networks, attention mechanisms

**Prerequisites**: Basic statistics, understanding of sequences, pandas for time series manipulation

**Problem Type**: Regression (continuous values over time)

**Directory**: `code_folder/10_*.py`

### 📈 Files:
1. **10_sklearn_time_series_arima.py**
   - ARIMA and SARIMA models
   - Seasonal decomposition
   - Automatic order selection
   - **Use cases**: Stock prices, sales forecasting, demand planning

2. **10_torch_lstm_time_series.py**
   - LSTM networks for sequences
   - Bidirectional LSTM
   - Multi-step forecasting
   - **Use cases**: Energy prediction, traffic flow, sensor data

3. **10_prophet_forecasting.py**
   - Facebook Prophet
   - Holiday effects, custom seasonality
   - Cross-validation
   - **Use cases**: Business forecasting, capacity planning

4. **10_tensorflow_temporal_fusion_transformer.py**
   - Attention-based forecasting
   - Multi-horizon predictions
   - Variable selection
   - **Use cases**: Retail demand, financial markets

---

## Category 11: Computer Vision (4 files)

### 📖 Background

Computer Vision enables machines to interpret and understand visual information from images and videos. This category covers core CV tasks including object detection (finding objects in images), image segmentation (pixel-level classification), transfer learning (leveraging pretrained models), and image preprocessing pipelines.

**Key Concepts**: Convolutional Neural Networks (CNNs), bounding boxes, segmentation masks, feature extraction, data augmentation, transfer learning

**Prerequisites**: Basic deep learning, NumPy arrays, understanding of images as tensors

**Problem Types**: Object detection, semantic segmentation, image classification

**Directory**: `code_folder/11_*.py`

### 👁️ Files:
1. **11_torch_object_detection_yolo.py**
   - YOLO architecture
   - Bounding box prediction
   - Non-Maximum Suppression (NMS)
   - **Use cases**: Autonomous vehicles, surveillance

2. **11_tensorflow_image_segmentation_unet.py**
   - U-Net encoder-decoder
   - Skip connections
   - Dice coefficient, IoU metrics
   - **Use cases**: Medical imaging, satellite imagery

3. **11_torch_transfer_learning_resnet.py**
   - ResNet transfer learning
   - Feature extraction vs fine-tuning
   - Data augmentation
   - **Use cases**: Custom classifiers with limited data

4. **11_opencv_image_preprocessing_pipeline.py**
   - Color space conversions
   - Geometric transformations
   - Filtering and edge detection
   - **Use cases**: Data preprocessing, image quality enhancement

---

## Category 12: Natural Language Processing (5 files)

### 📖 Background

Natural Language Processing (NLP) enables machines to understand, interpret, and generate human language. This category covers text classification, sentiment analysis, named entity recognition, and transformer-based models. NLP is foundational for chatbots, translation, content analysis, and information extraction.

**Key Concepts**: Tokenization, embeddings, TF-IDF, recurrent networks (LSTM), transformers, attention mechanisms, BERT, sequence-to-sequence models

**Prerequisites**: Basic text processing, understanding of sequences, familiarity with word vectors

**Problem Types**: Classification, sequence labeling (NER), language modeling

**Directory**: `code_folder/12_*.py`

### 📝 Files:
1. **12_sklearn_text_classification_tfidf.py**
   - TF-IDF vectorization
   - N-grams, traditional ML classifiers
   - Feature importance analysis
   - **Use cases**: Spam detection, document categorization

2. **12_torch_sentiment_analysis_lstm.py**
   - LSTM/BiLSTM for sentiment
   - Word embeddings
   - Sequence classification
   - **Use cases**: Customer reviews, social media monitoring

3. **12_transformers_named_entity_recognition.py**
   - NER with BiLSTM-CRF
   - BIO tagging scheme
   - Entity extraction
   - **Use cases**: Information extraction, knowledge graphs

4. **12_huggingface_text_classification_finetuning.py**
   - Transformer fine-tuning
   - Self-attention mechanisms
   - Transfer learning for NLP
   - **Use cases**: Sentiment analysis, intent detection

5. **12_spacy_nlp_pipeline.py**
   - Production NLP pipelines
   - Tokenization, POS tagging
   - Custom components
   - **Use cases**: Text preprocessing, linguistic analysis

---

## Category 13: Recommender Systems (4 files)

### 📖 Background

Recommender Systems predict user preferences and suggest items they might like. This category covers collaborative filtering (finding similar users/items), matrix factorization (decomposing user-item interactions), and deep learning approaches. Recommenders power Netflix, Amazon, Spotify, and YouTube.

**Key Concepts**: Collaborative filtering, matrix factorization (SVD), implicit/explicit feedback, cold-start problem, embedding layers, alternating least squares

**Prerequisites**: Linear algebra, understanding of sparse matrices, basic neural networks

**Problem Types**: Rating prediction, ranking, recommendation

**Directory**: `code_folder/13_*.py`

### 🎯 Files:
1. **13_sklearn_collaborative_filtering.py**
   - User-based & item-based CF
   - Cosine similarity
   - Rating prediction
   - **Use cases**: Movie/product recommendations

2. **13_surprise_matrix_factorization.py**
   - SVD matrix factorization
   - Gradient descent optimization
   - Latent factor models
   - **Use cases**: Netflix-style recommendations

3. **13_torch_neural_collaborative_filtering.py**
   - Deep learning for recommendations
   - Neural CF architecture
   - Non-linear interactions
   - **Use cases**: Complex personalization

4. **13_implicit_als_recommendation.py**
   - ALS for implicit feedback
   - Confidence weighting
   - Scalable recommendations
   - **Use cases**: YouTube views, Spotify plays

---

## Category 14: Anomaly Detection (4 files)

### 📖 Background

Anomaly Detection identifies unusual patterns that don't conform to expected behavior. Unlike supervised learning, anomaly detection often works with unlabeled data, learning what "normal" looks like and flagging deviations. Critical for fraud detection, network security, and quality control.

**Key Concepts**: Unsupervised learning, outlier detection, reconstruction error, isolation, density estimation, one-class classification

**Prerequisites**: Statistics, understanding of distributions, basic ML concepts

**Problem Types**: Unsupervised anomaly detection, novelty detection

**Directory**: `code_folder/14_*.py`

### 🔍 Files:
1. **14_sklearn_isolation_forest.py**
   - Tree-based anomaly detection
   - Unsupervised learning
   - Contamination parameter
   - **Use cases**: Fraud detection, quality control

2. **14_torch_autoencoder_anomaly_detection.py**
   - Reconstruction error method
   - Deep autoencoders
   - Threshold selection
   - **Use cases**: Manufacturing defects, cybersecurity

3. **14_pyod_outlier_detection.py**
   - Multiple detection methods
   - LOF, One-Class SVM, Elliptic Envelope
   - Algorithm comparison
   - **Use cases**: Comprehensive anomaly analysis

4. **14_sklearn_one_class_svm.py**
   - Novelty detection
   - Decision boundary learning
   - RBF kernel
   - **Use cases**: One-class classification

---

## Category 15: Model Interpretability (4 files)


### 📖 Background

Model Interpretability explains how ML models make decisions. As models become more complex (deep learning, ensembles), understanding their predictions becomes crucial for trust, debugging, and regulatory compliance. This category covers tools like SHAP and LIME for black-box interpretation.

**Key Concepts**: SHAP values, LIME, feature importance, partial dependence plots, glass-box vs black-box models, model-agnostic methods

**Prerequisites**: Understanding of ML models, basic statistics

**Problem Types**: Model explanation, feature attribution, interpretability

**Directory**: `code_folder/15_*.py`

### 🔬 Files:
1. **15_shap_model_interpretation.py**
   - SHAP values (Shapley values)
   - Feature contributions
   - Model-agnostic explanations
   - **Use cases**: Model debugging, regulatory compliance

2. **15_lime_local_interpretation.py**
   - Local interpretable models
   - Instance-level explanations
   - Black-box model interpretation
   - **Use cases**: Understanding specific predictions

3. **15_sklearn_feature_importance.py**
   - Tree-based importance
   - Random Forest & Gradient Boosting
   - Feature selection
   - **Use cases**: Feature engineering, model understanding

4. **15_interpret_ml_glass_box.py**
   - Inherently interpretable models
   - Logistic regression coefficients
   - Decision tree rules
   - **Use cases**: Regulated industries, high-stakes decisions

---

## Category 16: AutoML (4 files)


### 📖 Background

AutoML (Automated Machine Learning) automates the process of model selection, hyperparameter tuning, and feature engineering. Instead of manually trying different algorithms, AutoML systematically searches for the best approach. Ideal for rapid prototyping and non-expert users.

**Key Concepts**: Hyperparameter optimization, neural architecture search (NAS), Bayesian optimization, genetic algorithms, model selection

**Prerequisites**: Understanding of ML pipelines, basic ML algorithms

**Problem Types**: Meta-learning, optimization, pipeline automation

**Directory**: `code_folder/16_*.py`

### 🤖 Files:
1. **16_autosklearn_automated_ml.py**
   - Automatic model selection
   - Hyperparameter tuning
   - Pipeline automation
   - **Use cases**: Rapid prototyping, baselines

2. **16_tpot_genetic_programming.py**
   - Genetic algorithms for ML
   - Pipeline evolution
   - Novel configurations
   - **Use cases**: Complex pipeline optimization

3. **16_optuna_neural_architecture_search.py**
   - Neural architecture search
   - Bayesian optimization
   - Hyperparameter optimization
   - **Use cases**: Finding optimal architectures

4. **16_h2o_automl.py**
   - Enterprise AutoML
   - Model leaderboard
   - Distributed computing
   - **Use cases**: Production AutoML

---

## Category 17: Reinforcement Learning (4 files)


### 📖 Background

Reinforcement Learning (RL) trains agents to make sequential decisions by rewarding desired behaviors. Unlike supervised learning, there's no labeled data—agents learn through trial and error. RL powers game AI (AlphaGo), robotics, and autonomous systems.

**Key Concepts**: Agent-environment interaction, rewards, Q-learning, policy gradients, value functions, exploration vs exploitation

**Prerequisites**: Basic probability, understanding of Markov Decision Processes (helpful but not required)

**Problem Types**: Sequential decision making, control, game playing

**Directory**: `code_folder/17_*.py`

### 🎮 Files:
1. **17_gym_q_learning.py**
   - Q-learning algorithm
   - Epsilon-greedy exploration
   - Gridworld environment
   - **Use cases**: Game AI, control systems

2. **17_torch_deep_q_network.py**
   - DQN with experience replay
   - Target networks
   - Deep RL
   - **Use cases**: Atari games, robotics

3. **17_stable_baselines3_ppo.py**
   - Proximal Policy Optimization
   - Policy gradient methods
   - Clipped objective
   - **Use cases**: Continuous control, OpenAI Five

4. **17_ray_rllib_distributed_rl.py**
   - Distributed RL training
   - Parallel data collection
   - Scalable RL
   - **Use cases**: Large-scale RL experiments

---

## Category 18: Multi-Modal Learning (4 files)


### 📖 Background

Multi-Modal Learning combines different types of data (images, text, audio, graphs) to create richer representations. CLIP connects vision and language, Whisper handles speech, Stable Diffusion generates images from text, and GNNs process graph-structured data like social networks.

**Key Concepts**: Joint embeddings, cross-modal learning, graph neural networks, diffusion models, attention across modalities

**Prerequisites**: Understanding of both vision and NLP, basic graph theory for GNNs

**Problem Types**: Cross-modal retrieval, text-to-image, graph classification

**Directory**: `code_folder/18_*.py`

### 🌐 Files:
1. **18_clip_image_text_retrieval.py**
   - Vision-language models
   - Joint embeddings
   - Zero-shot classification
   - **Use cases**: Image search, content retrieval

2. **18_whisper_speech_recognition.py**
   - Speech-to-text transcription
   - Multilingual recognition
   - Robust to noise
   - **Use cases**: Transcription, translation, accessibility

3. **18_stable_diffusion_image_generation.py**
   - Text-to-image generation
   - Latent diffusion models
   - Iterative denoising
   - **Use cases**: Art generation, design, content creation

4. **18_graph_neural_network_pytorch_geometric.py**
   - GNNs for graph data
   - Message passing
   - Node/graph classification
   - **Use cases**: Social networks, molecules, knowledge graphs

---

## Category 19: MLOps & Production (8 files)


### 📖 Background

MLOps brings DevOps practices to machine learning, focusing on deploying, monitoring, and maintaining ML systems in production. This includes model serving, A/B testing, monitoring for drift, and ensuring models remain accurate over time. Also covers experiment tracking, model versioning, data validation, and cloud deployment. Essential for production ML systems.

**Key Concepts**: Model serving, API design, monitoring, A/B testing, data drift, feature stores, CI/CD for ML, experiment tracking, model registry, data validation, cloud deployment (SageMaker)

**Prerequisites**: Basic software engineering, REST APIs, understanding of production systems, Docker, cloud platforms (AWS)

**Problem Types**: Deployment, monitoring, operations, model lifecycle management, data quality

**Directory**: `code_folder/19_*.py`

**Infrastructure**: See [MLOPS_INFRASTRUCTURE.md](../MLOPS_INFRASTRUCTURE.md) for local (Docker Compose + MLflow) and cloud (Terraform + SageMaker) setup.

### 🚀 Files:

#### Model Serving & APIs
1. **19_model_serving_fastapi.py**
   - REST API for models
   - Request/response handling
   - Health checks
   - **Use cases**: Production model serving

#### Monitoring & Testing
2. **19_model_monitoring_prometheus.py**
   - Metrics collection
   - Latency tracking
   - Performance monitoring
   - **Use cases**: Production monitoring, alerting

3. **19_ab_testing_ml_models.py**
   - A/B testing framework
   - Traffic splitting
   - Statistical analysis
   - **Use cases**: Model comparison, gradual rollouts

4. **19_data_drift_detection.py**
   - Distribution shift detection
   - Statistical tests (KS test)
   - Trigger retraining
   - **Use cases**: Data quality monitoring

5. **19_great_expectations_data_validation.py** ⭐ NEW
   - Data validation framework
   - Expectation suites
   - Quality reports
   - CI/CD integration
   - **Use cases**: Data testing, pipeline validation

#### Model Lifecycle Management
6. **19_mlflow_model_registry.py** ⭐ NEW
   - Experiment tracking
   - Model versioning
   - Stage transitions (Staging → Production)
   - Model registry
   - **Use cases**: Model lifecycle, version control
   - **Requires**: MLflow server (run `make up` to start)

7. **19_bentoml_model_packaging.py** ⭐ NEW
   - Model packaging
   - API generation
   - Docker containerization
   - Multi-cloud deployment
   - **Use cases**: Model packaging, deployment

#### Cloud Deployment
8. **19_sagemaker_training_deployment.py** ⭐ NEW
   - AWS SageMaker workflow
   - Training jobs at scale
   - Model deployment
   - Endpoint creation
   - **Use cases**: Cloud ML, scalable training
   - **Requires**: AWS account, credentials

#### Feature Engineering
9. **19_feature_engineering_pipeline.py**
   - Production feature engineering
   - Consistent transformations
   - Pipeline serialization
   - **Use cases**: Data pipelines, feature stores

---

## Category 20: Ensemble Methods (6 files)


### 📖 Background

Ensemble Methods combine multiple models to produce better predictions than any individual model. By aggregating predictions from diverse models, ensembles reduce overfitting and improve generalization. Random Forests, XGBoost, and Stacking dominate Kaggle competitions and production systems.

**Key Concepts**: Bagging (bootstrap aggregating), boosting (sequential learning), stacking (meta-learning), voting, weak learners, variance reduction

**Prerequisites**: Understanding of decision trees, basic ML algorithms, overfitting vs underfitting

**Problem Types**: Classification, regression (applies to most supervised tasks)

**Directory**: `code_folder/20_*.py`

### 🎭 Files:
1. **20_sklearn_random_forest_bagging.py**
   - Random Forest classification & regression
   - Bagging with decision trees
   - Feature importance from ensembles
   - Out-of-bag error estimation
   - **Use cases**: General-purpose ML, feature selection, robust predictions

2. **20_xgboost_gradient_boosting.py**
   - XGBoost for classification & regression
   - Learning rate and regularization
   - Early stopping
   - Handling imbalanced data
   - **Use cases**: Kaggle competitions, structured data, high performance

3. **20_lightgbm_advanced_boosting.py**
   - LightGBM leaf-wise tree growth
   - Categorical feature handling
   - DART boosting with dropout
   - Speed optimization
   - **Use cases**: Large-scale datasets, production systems, fast training

4. **20_sklearn_adaboost.py**
   - AdaBoost adaptive boosting
   - Weak learner concept
   - Sample weight adaptation
   - Convergence analysis
   - **Use cases**: Binary classification, face detection, interpretable models

5. **20_sklearn_stacking_blending.py**
   - Stacking with cross-validation
   - Blending with hold-out set
   - Multi-level stacking
   - Meta-learner selection
   - **Use cases**: Kaggle competitions, combining diverse models

6. **20_sklearn_voting_ensembles.py**
   - Hard voting (majority vote)
   - Soft voting (probability averaging)
   - Weighted voting
   - Decision boundary visualization
   - **Use cases**: Simple ensembling, variance reduction

---

## 🎯 Quick Start

### Run any example:
```bash
python code_folder/10_sklearn_time_series_arima.py
python code_folder/11_torch_object_detection_yolo.py
python code_folder/12_sklearn_text_classification_tfidf.py
# ... etc
```

### Install dependencies:
```bash
pip install torch torchvision scikit-learn tensorflow pandas numpy matplotlib seaborn
pip install statsmodels prophet scipy opencv-python
pip install xgboost lightgbm  # For ensemble methods (Category 20)
```

---

## 📚 Learning Path

### Beginner → Intermediate → Advanced

**Beginner**:
- Start with Categories 10, 12 (classical ML approaches)
- Move to Category 14 (anomaly detection)

**Intermediate**:
- Categories 11, 13, 20 (deep learning applications, ensemble methods)
- Category 15 (interpretability)

**Advanced**:
- Categories 16, 17, 18 (AutoML, RL, Multi-Modal)
- Category 19 (MLOps & Production)

---

## 🔑 Key Concepts Covered

- **Time Series**: ARIMA, LSTM, Prophet, Transformers
- **Computer Vision**: YOLO, U-Net, Transfer Learning, OpenCV
- **NLP**: TF-IDF, LSTM, Transformers, spaCy
- **Recommenders**: Collaborative Filtering, Matrix Factorization, Neural CF
- **Anomaly Detection**: Isolation Forest, Autoencoders, One-Class SVM
- **Interpretability**: SHAP, LIME, Feature Importance
- **AutoML**: Model Selection, Hyperparameter Tuning, NAS
- **Reinforcement Learning**: Q-Learning, DQN, PPO
- **Multi-Modal**: CLIP, Whisper, Stable Diffusion, GNNs
- **MLOps**: Serving, Monitoring, A/B Testing, Drift Detection
- **Ensemble Methods**: Random Forest, XGBoost, LightGBM, AdaBoost, Stacking, Voting

---

## 🏆 Best Practices

Each example demonstrates:
- ✅ Clean, readable code
- ✅ Comprehensive documentation
- ✅ Practical use cases
- ✅ Key takeaways and learning points
- ✅ Production-ready patterns
- ✅ Error handling
- ✅ Metrics and evaluation

---

## 📞 Next Steps

1. **Explore examples** in areas of interest
2. **Modify hyperparameters** and experiment
3. **Apply to real datasets** from Kaggle, UCI ML Repository
4. **Combine techniques** for production pipelines
5. **Build end-to-end projects** using multiple categories

---

## 🤝 Contributing

To add more examples:
1. Follow the naming convention: `<category>_<framework>_<topic>.py`
2. Include docstring with description, use cases
3. Add comprehensive comments
4. Include "Key Takeaways" section
5. Test on sample data

---

**Happy Learning! 🚀**

For questions or issues, refer to the main README.md or create an issue in the repository.
