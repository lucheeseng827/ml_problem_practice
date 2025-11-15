# New ML Practice Categories (10-20)

This document describes **48 new machine learning practice examples** added to the repository, organized into 11 comprehensive categories covering advanced ML topics.

## 📊 Overview

Total new examples: **48 Python scripts**
Categories added: **11 (Categories 10-20)**
Focus areas: Time Series, Computer Vision, NLP, Recommender Systems, Anomaly Detection, Interpretability, AutoML, Reinforcement Learning, Multi-Modal, MLOps, and **Ensemble Methods**

---

## Category 10: Time Series Forecasting (4 files)

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

## Category 19: MLOps & Production (5 files)

### 🚀 Files:
1. **19_model_serving_fastapi.py**
   - REST API for models
   - Request/response handling
   - Health checks
   - **Use cases**: Production model serving

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

4. **19_feature_engineering_pipeline.py**
   - Production feature engineering
   - Consistent transformations
   - Pipeline serialization
   - **Use cases**: Data pipelines, feature stores

5. **19_data_drift_detection.py**
   - Distribution shift detection
   - Statistical tests (KS test)
   - Trigger retraining
   - **Use cases**: Data quality monitoring

---

## Category 20: Ensemble Methods (6 files)

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
