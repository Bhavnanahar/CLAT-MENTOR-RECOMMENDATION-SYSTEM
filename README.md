# ✨ Personalized Mentor Recommendation System for CLAT Aspirants (KNN Model) 🧑‍🎓👩‍🎓

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%230057a3.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=matplotlib&logoColor=black)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-%234C76A3.svg?style=flat&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)

## 🚀 Overview

Empowering CLAT (Common Law Admission Test) aspirants to connect with the right mentors! This project utilizes the power of the K-Nearest Neighbors (KNN) algorithm to recommend CLAT toppers (mentors) based on the unique profiles of aspiring students. By finding aspirants with similar preferences and learning styles, we can suggest mentors who are likely to provide the most relevant guidance and support.

**🌟 Key Features:**

* **🎯 Profile-Driven Matching:** Recommends mentors based on an aspirant's preferred subjects, target colleges, preparation level, and learning style.
* **🤖 KNN Magic:** Employs the KNN model to identify aspirants with the most similar profiles in the feature space.
* **🤝 Mentor Alignment:** Suggests mentors who resonate with the collective characteristics of an aspirant's closest peers.
* **📊 Insightful Visualizations:** Includes clear visualizations of aspirant similarity using Principal Component Analysis (PCA) for a deeper understanding of the data.
* **⚙️ Simple Setup:** Easy-to-follow instructions to get the system up and running quickly.

## 🛠️ Setup Instructions

1.  **📥 Clone the Repository (if you have it on GitHub):**
    ```bash
    git clone <repository_url>
    cd mentor_recommendation_knn
    ```

2.  **🐍 Install Dependencies:**
    Make sure you have Python 3.x installed. Then, install the required libraries using pip:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **🏃 Run the Script:**
    Execute the main Python script to witness the magic!
    ```bash
    python mentor_recommendation.py
    ```
    Or, if you prefer Jupyter Notebook:
    ```bash
    jupyter notebook mentor_recommendation.ipynb
    ```

## 🧠 Summary of Our Approach

Here's a step-by-step breakdown of how our recommendation engine works:

1.  **🌱 Data Generation:** We start by creating sample (or you can use your anonymized) data for CLAT aspirants and mentors, capturing key attributes:
    * **🧑‍🎓 Aspirants:** Preferred Subjects, Target Colleges, Preparation Level, Learning Style.
    * **👨‍🏫 Mentors:** Expertise Subjects, Graduated College, Mentoring Style, Mock CLAT Rank.

2.  **⚙️ Feature Engineering & Preprocessing:** We transform the raw data into a numerical format that our KNN model can understand:
    * **`MultiLabelBinarizer`:** For handling multiple selections like subjects and colleges.
    * **One-Hot Encoding (`pd.get_dummies`)**: For single-value categorical features.
    * **📏 Feature Alignment:** Ensuring consistency in the features used for both aspirants and mentors.

3.  **🏋️‍♂️ KNN Model Training:** We train a `NearestNeighbors` model on the processed aspirant profiles, using cosine similarity to measure how alike aspirants are.

4.  **✨ Mentor Recommendation Logic:** When an aspirant seeks a mentor:
    * We find their 'k' most similar peers using our trained KNN model.
    * We calculate the average profile of these similar aspirants.
    * We then compare this average profile with the profiles of all available mentors using cosine similarity.
    * Finally, we present the top 'n' mentors who have the highest similarity scores.

5.  **📊 Visualizing Aspirant Similarity:** We use Principal Component Analysis (PCA) to reduce the complexity of aspirant profiles, allowing us to visualize their similarities in a 2D scatter plot. This helps understand how aspirants cluster based on their characteristics.

## 📂 Code Structure

The main Python script (`mentor_recommendation.py` or `mentor_recommendation.ipynb`) is organized as follows:

* **📚 Imports:** Loading necessary libraries.
* **🌱 Data Generation:** Creating the initial datasets.
* **🛠️ Feature Processing:** Preparing the data for the KNN model.
* **🤖 KNN Model:** Setting up and training the `NearestNeighbors` model.
* **✨ `recommend_mentors_knn`:** The core function for generating mentor recommendations.
* **🚀 Recommendation Execution:** Showing how to get recommendations.
* **📊 Aspirant Similarity Visualization:** Using PCA and `seaborn` for plotting.
* **(Optional) 👨‍🏫 Mentor Distribution Visualization:** Visualizing mentor profiles.

## 💡 Potential Enhancements

* **📈 Real-World Data Integration:** Using actual (anonymized) CLAT aspirant and topper data would significantly improve recommendation accuracy.
* **⚖️ Feature Weighting:** Assigning different importance levels to various features could lead to more personalized recommendations.
* **ibrid Approaches:** Combining KNN with other recommendation techniques (e.g., content-based filtering on mentor expertise) might provide a more robust system.
* **👂 Feedback Mechanism:** Incorporating user feedback on mentor quality would enable continuous learning and refinement of the recommendations.
* **☁️ Scalability Solutions:** For larger datasets, exploring more efficient data structures and algorithms would be crucial.
* **🧪 Evaluation Metrics:** Implementing metrics to quantitatively assess the performance of the recommendation system.
