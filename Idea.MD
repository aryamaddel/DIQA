- ## Introduction & Problem Statement
	- The proliferation of digital imaging devices and online media platforms has led to an exponential increase in image consumption and transmission. However, images often undergo various distortions during acquisition, compression, transmission, or processing, leading to a degradation in perceived visual quality. Traditional Image Quality Assessment (IQA) methods often require a pristine reference image (Full-Reference IQA), which is frequently unavailable in real-world scenarios.
	- **No-Reference Image Quality Assessment (NR-IQA)**, also known as Blind IQA, aims to predict the perceptual quality of an image without any reference to its original, undistorted version. This project focuses on developing and evaluating NR-IQA models using machine learning techniques to address this critical need. The ability to accurately estimate image quality blindly has significant implications for media optimization, allowing for dynamic adjustments in encoding, transmission, and display parameters to enhance user experience and optimize resource utilization.
	- **Problem:** Existing NR-IQA methods often struggle with generalization across diverse distortion types and content, and there's a continuous need for robust and perceptually aligned quality predictors, especially for applications like adaptive streaming and real-time image processing.
	    
---
- ## Why this idea works (Reinforced & Expanded)
  collapsed:: true
	- **Niche & High Impact:** NR-IQA is a highly specialized and challenging sub-field of [[Image Processing]] and [[Computer Vision]]. Unlike common classification or detection tasks, it delves into **subjective quality estimation**, which is inherently more complex and requires a nuanced understanding of [[Human Visual Perception]]. This demonstrates a deeper engagement with the subject matter.
	- **Direct Real-world Applications:** The practical utility of NR-IQA is immense.
	  collapsed:: true
		- **Video Streaming (Netflix, YouTube):** Optimizing bitrates for adaptive streaming based on predicted quality, ensuring a consistent user experience even with varying network conditions.
		- **Surveillance Systems:** Automatically flagging low-quality frames for improved analysis or re-acquisition.
		- **Mobile Camera Applications:** Providing real-time feedback for image capture settings or post-processing enhancement.
		- **Image Compression & Storage:** Guiding optimal compression levels to maintain perceptual quality while minimizing file size.
	-
	- **Hybrid Approach (ML + Image Processing):** The proposed methodology leverages both **handcrafted feature engineering** (e.g., BRISQUE, NIQE, PIQE) and potentially **deep feature extraction** (using pre-trained CNNs). This showcases a comprehensive understanding of both traditional signal processing techniques and modern [[Deep Learning]] paradigms, demonstrating versatility and a balanced perspective.
	- **Scalability & Research Trajectory:** The project offers excellent scalability. You can start with well-established statistical NR-IQA methods and then progressively incorporate more complex deep learning architectures, allowing for a phased development and potential for future research extensions.
	- **Emphasis on Perceptual Quality:** The core objective is to align predicted scores with **Mean Opinion Scores (MOS)**, which are derived from human subjective judgments. This highlights the project's focus on human perception, a crucial aspect often overlooked in purely objective metrics.
	      
	    
---
- ## Project Objectives
  collapsed:: true
	- Literature Review
	  collapsed:: true
		- Conduct a comprehensive review of existing NR-IQA techniques
			- Traditional (e.g., statistical natural scene statistics)
			- Deep learning-based approaches
	- Dataset Acquisition & Preprocessing
	  collapsed:: true
		- Identify and acquire at least two diverse, publicly available image quality datasets with associated MOS values
			- [[KonIQ-10k]]
			- [[LIVE Image Quality Database]]
			- [[TID2013]]
		- Prepare the data for feature extraction and model training
	- Feature Engineering
	  collapsed:: true
		- Handcrafted Features
			- Implement and extract features using established NR-IQA algorithms
				- [[BRISQUE]]
				- [[NIQE]]
				- [[PIQE]]
			- Analyse their strengths and limitations
		- Optionally (Advanced): Deep Features
			- Explore extracting features from pre-trained [[Convolutional Neural Networks (CNNs)]]
				- [[VGG16]]
				- [[ResNet]]
			- Use average pooled activations from a mid-level convolutional layer
			- Investigate the perceptual relevance of these features
	- Model Development & Training
	  collapsed:: true
		- Train and evaluate various regression models to predict MOS
			- [[Random Forest Regressor]]
			- [[Support Vector Regressor (SVR)]]
			- [[Feed-forward Neural Network]]
		- Implement appropriate cross-validation strategies to ensure model generalization
	- Performance Evaluation
	  collapsed:: true
		- Evaluate the developed models using standard IQA evaluation metrics
			- **Mean Squared Error (MSE)**
			- **Mean Absolute Error (MAE)**
				- Quantify prediction accuracy
			- **Pearson Linear Correlation Coefficient (PLCC)**
			- **Spearman Rank-Order Correlation Coefficient (SRCC)**
				- Assess monotonic relationship between predicted and true MOS values
				- Crucial for perceptual agreement
	- Analysis & Discussion
	  collapsed:: true
		- Analyze the performance of different feature sets and regression models
		- Discuss the strengths and weaknesses of each approach
		- Mention limitations
		- Suggest potential avenues for future improvement
- ## Methodology (Detailed Breakdown)
  collapsed:: true
	- Data Collection & Pre-processing
	  collapsed:: true
		- Datasets
		  collapsed:: true
			- [[KonIQ-10k]]: Large-scale, diverse dataset with real-world distorted images and MOS
			- [[LIVE Image Quality Database]]: Contains images with synthetic distortions (e.g., JPEG, JPEG2000, Gaussian blur, white noise, fast fading)
			- [[TID2013]]: Comprehensive dataset with various types and levels of distortions
		- Preprocessing
		  collapsed:: true
			- Resize images if necessary
			- Normalize pixel values
			- Align MOS scores with corresponding image IDs
	- Feature Extraction
	  collapsed:: true
		- Traditional NR-IQA Metrics
		  collapsed:: true
			- [[BRISQUE]] (Blind/Referenceless Image Spatial Quality Evaluator): Based on [[Natural Scene Statistics (NSS)]] models of locally normalized luminance coefficients
			- [[NIQE]] (Natural Image Quality Evaluator): NSS-based metric that models the statistical regularities of natural, pristine images and computes the distance of a given image from this "natural" model
			- [[PIQE]] (Perceptual Image Quality Evaluator): A newer metric that may offer advantages in certain scenarios
			- Implementation: Utilize existing libraries (e.g., `skimage`, `OpenCV` bindings if available, or dedicated IQA libraries/MATLAB conversions for specific metrics)
		- Deep Features (Optional but Recommended for Depth)
		  collapsed:: true
			- Pre-trained CNN: Load a pre-trained model such as [[VGG16]], [[ResNet]], or [[MobileNet]] (trained on [[ImageNet]])
			- Feature Extraction Layer: Select a mid-level convolutional layer (e.g., `block3_conv3` or `block4_conv3` for VGG16) to capture rich semantic features
			- Pooling: Apply global average pooling to the feature maps to obtain a fixed-size feature vector for each image
	- Model Training
	  collapsed:: true
		- Regression Models
		  collapsed:: true
			- [[Random Forest Regressor]]: Ensemble learning method, robust to overfitting, good for non-linear relationships
			- [[Support Vector Regressor (SVR)]]: Effective for high-dimensional data, good generalization
			- Simple [[Feed-forward Neural Network]]: Multi-layer perceptron (MLP) with a few dense layers and ReLU activations, outputting a single regression value
		- Training Protocol
		  collapsed:: true
			- Split data into training and testing sets (e.g., 80/20 split)
			- Implement k-fold cross-validation for robust evaluation
			- Perform hyperparameter tuning with [[GridSearchCV]] or [[RandomizedSearchCV]]
	- Evaluation Metrics
		- Prediction Accuracy
		  collapsed:: true
			- $MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$
			- $MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|$
			- Where $y_i$ are the true MOS and $\hat{y}_i$ are the predicted MOS
		- Perceptual Correlation
		  collapsed:: true
			- [[PLCC (Pearson Linear Correlation Coefficient)]]: Measures linear correlation between predicted and true scores. Values closer to 1 indicate strong linear agreement
			- [[SRCC (Spearman Rank-Order Correlation Coefficient)]]: Measures monotonic relationship, robust to outliers, important for rank preservation. Values closer to 1 indicate strong monotonic agreement
- ## Tools & Technologies
  collapsed:: true
	- **Programming Language:** [[Python]]
	- **Libraries:**
	      
	    - **Data Handling:** `pandas`, `numpy`  
	          
	    - **Image Processing:** `OpenCV`, `scikit-image`, `Pillow`  
	          
	    - **Machine Learning:** `scikit-learn` (for Random Forest, SVR), [[TensorFlow]]`/`[[Keras]] or [[PyTorch]] (for Neural Networks and deep feature extraction)  
	          
	    - **Visualization:** `matplotlib`, `seaborn`  
	- **Development Environment:** [[Jupyter Notebooks]] or a suitable IDE (e.g., [[VS Code]])
- ## Expected Outcomes & Deliverables
  collapsed:: true
	- A functional NR-IQA system capable of predicting perceptual image quality.
	- Comparative analysis of different handcrafted features (BRISQUE, NIQE, PIQE) in terms of their performance and limitations.
	- Evaluation of the effectiveness of deep features extracted from pre-trained CNNs for NR-IQA.
	- Performance comparison of various regression models (Random Forest, SVR, NN) on the selected datasets.
	- A well-documented codebase and a comprehensive project report detailing the methodology, experiments, results, and conclusions.
- ## Future Enhancements (Beyond Project Scope, for later consideration)
  collapsed:: true
	- **Hybrid Feature Fusion:** Investigate methods to combine handcrafted and deep features for improved performance.
	- **End-to-End Deep Learning Models:** Explore training an end-to-end deep learning model specifically for NR-IQA (e.g., deep convolutional neural networks designed for quality prediction).
	- **Attention Mechanisms:** Incorporate attention mechanisms to focus on perceptually significant regions of an image.
	- **Cross-Dataset Generalization:** Test the trained models on datasets with different types of distortions or content to assess their generalization capabilities.
	- **Real-time Implementation:** Explore optimizing models for real-time inference in applications like video streaming.
- ## Papers:
	- Foundational & Classical Methods
	  collapsed:: true
		- BRISQUE
		  collapsed:: true
			- Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012). "No-Reference Image Quality Assessment in the Spatial Domain." *IEEE Transactions on Image Processing, 21*(12), 4695-4708
			- DOI: 10.1109/TIP.2012.2214050
			- https://ieeexplore.ieee.org/abstract/document/6272356
			- link to read : https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

		- NIQE
		  collapsed:: true
			- Mittal, A., Soundararajan, R., & Bovik, A. C. (2013). "Making a 'Completely Blind' Image Quality Analyzer." *IEEE Signal Processing Letters, 20*(3), 209-212
			- DOI: 10.1109/LSP.2012.2227726
			-https://ieeexplore.ieee.org/abstract/document/6353522
			- Link to read: https://live.ece.utexas.edu/publications/2013/mittal2013.pdf


		- CORNIA
		  collapsed:: true
			- Ye, P., Kumar, J., Kang, L., & Doermann, D. (2012). "Unsupervised feature learning framework for no-reference image quality assessment." *2012 IEEE Conference on Computer Vision and Pattern Recognition*, 1098-1105
			- DOI: 10.1109/CVPR.2012.6247790
			- https://ieeexplore.ieee.org/document/6247789
			- link to read: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6247789

		- DB-CNN
		  collapsed:: truev
			- Zhang, W., Ma, K., Yan, J., Deng, D., & Wang, Z. (2018). "Blind Image Quality Assessment Using a Deep Bilinear Convolutional Neural Network." *IEEE Transactions on Circuits and Systems for Video Technology, 30*(1), 36-47
			- DOI: 10.1109/TCSVT.2018.2886771
			- https://ieeexplore.ieee.org/document/8576582
			- Link to read: https://ece.uwaterloo.ca/~z70wang/publications/TCSVT_BIQA.pdf


			


	- State-of-the-Art & Deep Learning Methods
	  collapsed:: true
		- HyperIQA
		  collapsed:: true
			- Su, S., Yan, Q., Zhu, Y., Zhang, C., Ge, S., Sun, W., & Zhang, J. (2020). "Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper-Network." *2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 13730-13739

			
			- https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf

			

			
		- MANIQA
		  collapsed:: true
			- Yang, S., Lin, T., Wang, H., Wu, J., Zhang, W., & Cao, X. (2022). "MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment." *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 1191-1200
		
			- https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Yang_MANIQA_Multi-Dimension_Attention_Network_for_No-Reference_Image_Quality_Assessment_CVPRW_2022_paper.pdf
			

		- TOPIQ
		  collapsed:: true
			- Cui, Z., Chen, T., Xu, S., & Li, C. (2023). "From Patches to Pictures: A Coarse-to-Fine Network for Blind Image Quality Assessment." *arXiv preprint arXiv:2308.03060*
			- DOI: 10.48550/arXiv.2308.03060
			
			- https://ieeexplore.ieee.org/abstract/document/10478301

	- Survey & Review Papers
	  collapsed:: true
		- "No-Reference Image Quality Assessment: Past, Present, and Future" (2025)
		  collapsed:: true
			
		- "A Survey of Learning-Based No-Reference Image Quality Assessment"
		  collapsed:: true
			

		- "Progress in Blind Image Quality Assessment: A Brief Review"
		  collapsed:: true
			

	- Image Smoothing & Noise Reduction (Filters)
	  collapsed:: true
		- Gaussian Filter
		  collapsed:: true
			- Smooths images by averaging pixels with a Gaussian-weighted kernel
			- Useful for reducing Gaussian noise
		- Median Filter
		  collapsed:: true
			- Replaces each pixel with the median of its neighborhood
			- Very effective for removing salt-and-pepper noise
		- Box Filter (Mean Filter)
		  collapsed:: true
			- Replaces each pixel with the average of its neighborhood
			- Simple but can overly blur edges
		- Weighted Average Filter
		  collapsed:: true
			- Similar to box filter but assigns different weights to neighbors
			- Helps preserve edges while reducing noise