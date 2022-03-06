# A combined system metrics approach to cloud service reliability using artificial intelligence

> Identifying and anticipating potential failures in the cloud is an effective method for increasing cloud reliability and proactive failure management. Many studies have been conducted to predict potential failure, but none have combined SMART (Self-Monitoring, Analysis, and Reporting Technology) hard drive metrics with other system metrics such as CPU utilisation. Therefore, we propose a combined metrics approach for failure prediction based on Artificial Intelligence to improve reliability. We tested over 100 cloud servers' data and four AI algorithms: Random Forest, Gradient Boosting, Long-Short-Term Memory, and Gated Recurrent Unit. Our experimental result shows the benefits of combining metrics, outperforming state-of-the-art.

[![Build Status](https://travis-ci.org/badges/badgerbadgerbadger.svg?branch=master)](https://travisci.org/badges/badgerbadgerbadger)

---
### Prerequisites

You need to install the following inorder to get started.

* Scikit-learn
    * A machine learning library in python.
    * Follow the installation instructions described here [https://scikit-learn.org/stable/install.html](https://scikit-learn.org/stable/install.html)
* Tensorflow2 
    * An end-to-end open source machine learning platform [More](https://www.tensorflow.org/)
    * Installation instructions - [https://www.tensorflow.org/install](https://www.tensorflow.org/install)
    * Install with GPU support.
* Pandas 
    * Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language. [More](https://pandas.pydata.org/)
    * Installation instructions - [https://pandas.pydata.org/getting_started.html](https://pandas.pydata.org/getting_started.html)   



---

## Citing

If you find this code useful, consider citing:

```

@Article{bdcc6010026,
AUTHOR = {Chhetri, Tek Raj and Dehury, Chinmaya Kumar and Lind, Artjom and Srirama, Satish Narayana and Fensel, Anna},
TITLE = {A Combined System Metrics Approach to Cloud Service Reliability Using Artificial Intelligence},
JOURNAL = {Big Data and Cognitive Computing},
VOLUME = {6},
YEAR = {2022},
NUMBER = {1},
ARTICLE-NUMBER = {26},
URL = {https://www.mdpi.com/2504-2289/6/1/26},
ISSN = {2504-2289},
ABSTRACT = {Identifying and anticipating potential failures in the cloud is an effective method for increasing cloud reliability and proactive failure management. Many studies have been conducted to predict potential failure, but none have combined SMART (self-monitoring, analysis, and reporting technology) hard drive metrics with other system metrics, such as central processing unit (CPU) utilisation. Therefore, we propose a combined system metrics approach for failure prediction based on artificial intelligence to improve reliability. We tested over 100 cloud servers’ data and four artificial intelligence algorithms: random forest, gradient boosting, long short-term memory, and gated recurrent unit, and also performed correlation analysis. Our correlation analysis sheds light on the relationships that exist between system metrics and failure, and the experimental results demonstrate the advantages of combining system metrics, outperforming the state-of-the-art.},
DOI = {10.3390/bdcc6010026}
}

```
