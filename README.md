# Caterina Gallo
Hello! Welcome to my personal portfolio! 

## Projects
In the following you can find a list of projects developed during my MS in Applied Artificial Intelligence at San Diego University. 

### Study_Of_Obesity - [Github Link]()

Final Project of the course "Probability and Statistics for Artificial Intelligence"

We analyzed an [existing dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) including 2111 records and 15 attributes possibly linked to different weight levels (underweight, normal, overweight I, overweight II, obesity I, obesity II and obesity III). 

We first identified the most important factors related to overweight and obesity by evaluating the relationships between the status of being overweight or obese (weight status: 0 for underweight or normal weight people and 1 for overweight or obese people) and each of the other available variables in our database (consumption of high caloric food, vegetables, food between meals, water and alcohol, number of main meals, calories monitoring, physical activity frequency, time using technology, transportation used, gender age, height, overweight family history and smoke). For categorical variables we relied on the chi-squared test of independence, while for numerical variables we used logistic regression. Assuming a significance level of 0.05, we recognized the variables having the strongest association with the weight status: number of meals per day, daily water consumption, physical activity frequency per week, time using technology per day, age and height. 

We then trained a decision tree classifier to predict if one is overweight/obese or not based on the variables with the highest impact on the weight status. The final model exhibited an accuracy of 85% and the values for precision, recall and f1-score were 0.9/0.72, 0.89/0.74 and 0.89/0.73 for people suffering/not suffering from overweight or obesity, respectively. Overall, the model performed reasonably well alghough a few limitations should be addressed. In fact, the data were self-reported, which could have led to biases and inaccuracies, and other potential factors potentially correlated with obesity such as genetics were neglected in this project.

### COVID_Detection - [Github Link](https://github.com/CatGallo/COVID_Detection.git)

Final Project of the course "Introduction to Artificial Intelligence"

We tested three different Convolutional Neural Network models, VGG16, ResNet50 and InceptionV3, to classify chest X-Ray images into normal or Covid-19 infected. 

To train and test these models we used the [Covid X-Ray Dataset](https://www.kaggle.com/datasets/ahemateja19bec1025/covid-xray-dataset) including 1301 images without Covid-19 (see Figure 1) and 1790 images with Covid-19 (see Figure 2). While image standardization and normalization were applied to the entire dataset, image augmentation was applied to the training set only through rotation and by filling in any empty pixel after rotation with the nearest pixel value. 

<figure>
<figcaption>Figure 1 - Examples of images without Covid</figcaption>
<img src="assets/no_covid.jpg" width=400>
</figure>

<figure>
<figcaption>Figure 2 - Examples of images with Covid</figcaption>
<img src="assets/covid.jpg" width=400>
</figure>

VGG16 showed the highest performance with 94.2% accuracy, 98.5% sensitivity, and 91.1% specificity. InceptionV3 trailed closely with 93.4% accuracy, 93.5% sensitivity and 93.3% specificity, while ResNet50 had 84.7% accuracy, 91.6% sensitivity and 79.6% specificity. Despite the high scores associated with the three models, additional adjustments would be necessary in future model developments. The ResNet50 model would in fact require a greater number of epochs, which we reduced due to memory limitations. Also, the head layer structure and parameters like the learning rate should be individually optimized for each model instead of keeping them consistent across models,  as we did here to get a direct comparison. Finally, additional patient-specific information, e.g. the number of days from the first symptoms or the presence of other clinical conditions, could further improve the capability of our models to early detect Covid-19 and similar respiratory diseases like pneumonia. 

### Text_To_Image - [Github Link](https://github.com/CatGallo/Text_To_Image.git) - [Application Deployment](https://huggingface.co/sglasher/van-gogh-stable-diffusion)

Final Project of the course of "Introduction to Computer Vision" 

We developed a text-to-image model able to generate Van Gogh style pictures based on text prompts.

We exploited a [pre-trained model](https://huggingface.co/CompVis/stable-diffusion-v1-4) of Stable Diffusion available in Hugging Face, which hosts the [WikiArt dataset](https://huggingface.co/datasets/huggan/wikiart/viewer/default/train) comprising 103.250 pieces of art (paintings, drawings and sculptures). The pre-trained model was fine-tuned on a portion of the WikiArt dataset including images and descriptions of more than 400 Van Gogh's masterpieces. Figure 3 shows four images produced by the model for each of the following categories: cities, landscapes, sky flowers, professions and situations. 

<figure>
<figcaption>Figure 3 - Examples of images returned by the model in response to the prompts written above the panels</figcaption>
<img src="assets/vg_images.png" width=600>
</figure>

The model was quantitatively evaluated through a couple of metrics normally used for diffusion models: the [CLIP score](https://huggingface.co/docs/diffusers/conceptual/evaluation) - to determine the similarity level between one or more label and the corresponding images returned by the model - and the [Fréchet Inception Distance](https://huggingface.co/docs/diffusers/conceptual/evaluation) - to determine the similarity level between one or more sets of images, real and fake. We got an average CLIP score of 28.89 with the couples label-image reported in Figure 3, while a FID of 677.19 was found by comparing 16 real Van Gogh paintings with the corresponding images produced by the model based on the titles of the 16 real masterpieces. Also, in order to assess the quality of the output images, 16 users were also asked to evaluate the output images by assigning a score from 1 to 5 to five different criteria: relevance to the initial prompt (4.56 ± 0.7), Van Gogh style (3.94 ± 1.34), complexity (4.0 ± 0.87), creativity (4.0 ± 1.12) and general satisfaction (4.38 ± 0.78). 

Based on the results described above, one can easily guess the weakest point of our model: its inability to always generate Van Gogh style images. If the model is asked to show typical subjects of Van Gogh's pictures, like sunflowers and farmers, the images generated by the model will probably embody the style of the painter. Otherwise the output images might appear very distant from the works of the painter we want to imitate. To enhance our model performance, we could increase the size of the dataset used for training and testing, maybe including the paintings of other authors sharing the same artistic period. Finally, the model should be judged by a greater number of users to make this kind of evaluation significant. 

### Human_Activity_Classification_Heart_Rate_Prediction - [Github Link]()

Final Project of the course "Data Analytics and Internet of Things"

We proposed a physical activity tracking system that exploits machine learning algorithms to predict the physical activity of the user (among a list of 12 different activities: standing still, sitting relaxing, lying down, walking, cycling, jogging, running, climbing stairs, arm elevation, knee bend, waist bends and front back jump) and the average user's heart rate (HR) for the next 5-10 heartbeats. 

The IoT system includes a smartwatch, an ankle sensor and a chest sensor. While the chest sensor consists of an accelerometer to measure position and 2 electrocardiogram sensors, the smartwatch and the ankle sensors contain an accelerometer, a gyroscope to detect motion and a magnetometer to recognize changes in orientation. The chest and ankle sensors are also equipped with a Bluetooth connectivity module to transfer data to the smartwatch. The latter then runs the machine learning algorithms we developed in the form of edge processing and shows the final data in the form of key summary statistics for the user to see. An example of tableau dashboard for this IoT device is provided below.

<img src="assets/Dashboard.png" alt="img-verification" width=500>

Our machine learning algorithms were built based on the [mHealth Dataset](https://archive.ics.uci.edu/dataset/319/mhealth+dataset), which reports the data recorded by the three sensors described above (a total of 24 features) for 10 different subjects. We combined the data files of the 10 subjects thereby obtaining a data frame of over 1.2 million entries. 

To classify the activity performed by the user we tested 3 different models, a K Nearest Neighbors (KNN) n = 6 model, a Support Vector Machine (SVM) model and a Random Forest Classifier (RFC) model with the scikit-learn default number of estimators set at 100. Input data were extracted from the time series recorded by all the sensors for the first 6 subjects, by calculating a rolling average and standard deviation over a set period of 4 seconds for each of the available features. The RFC model showed the best overall performance with a 92.9% validation accuracy and a 97.9% accuracy on the testing set. We have in fact verified that our RFC model can misclassify climbing stairs with walking and knee bends, thereby leading to sporadic errors. 

To predict the average user's HR for the next 5-10 heartbeats, we adopted an LSTM model. This time, considering that ECGs can drammatically change from one subject to another, we trained one subject-specific LSTM model starting from one ECG signal of one subject only. The training set is nothing more than a sequence of average HR values, with each HR value representing the average of 5 sequential HRs in a portion of the original HR sequence extracted from the ECG signal of the chosen subject. HRs can be calculated as 60/RRs, with RRs the distances between siccessive peaks in an ECG signal. By comparing the real and predicted average seguence of HR values for the chosen subject, we got the following errors: mean+/-std: -0.94+/-13.37%, 25th percentile: -9.77%, 50th percentile: -0.14%, 75th percentile: 8.80%.

## Contacts
You can reach me at [LinkedIn](https://www.linkedin.com/in/caterina-gallo) or [GitHub](https://github.com/CatGallo).
