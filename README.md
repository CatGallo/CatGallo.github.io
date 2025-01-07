# Caterina Gallo
Hello! Welcome to my personal portfolio! 

## Projects
In the following you can find a list of projects developed during my MS in Applied Artificial Intelligence at San Diego University. 

### Study_Of_Obesity - [Github Link]()

Final Project of the course "Probability and Statistics for Artificial Intelligence"

We analyzed an [existing dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) including 2111 records and 17 attributes possibly linked to different weight levels (underweight, normal, overweight I, overweight II, obesity I, obesity II and obesity III). We first identified the most important factors related to overweight and obesity (number of meals per day, daily water consumption, physical activity frequency per week, time using technology per day, age and height) and then trained a decision tree classifier to predict if one is overweight/obese or not. The model exhibits an accuracy of 85% and the values for precision, recall and f1-score are 0.9/0.72, 0.89/0.74 and 0.89/0.73 for people suffering/not suffering from overweight or obesity, respectively. 

### COVID_Detection - [Github Link](https://github.com/CatGallo/COVID_Detection.git)

Final Project of the course "Introduction to Artificial Intelligence"

We tested three different Convolutional Neural Network models, VGG16, ResNet50 and InceptionV3, to classify chest X-Ray images into normal or Covid-19 infected. To train and test these models we used the [Covid X-Ray Dataset](https://www.kaggle.com/datasets/ahemateja19bec1025/covid-xray-dataset) including 1301 images without Covid-19 and 1790 images with Covid-19. We applied image standardization and normalization to the entire dataset and image augmentation to the training set. VGG16 shows the highest performance with 94.2% accuracy, 98.5% sensitivity, and 91.1% specificity. InceptionV3 trails closely with 93.4% accuracy, 93.5% sensitivity and 93.3% specificity, while ResNet50 has 84.7% accuracy, 91.6% sensitivity and 79.6% specificity. 

### Text_To_Image - [Github Link](https://github.com/CatGallo/Text_To_Image.git) - [Application Deployment](https://huggingface.co/sglasher/van-gogh-stable-diffusion)

Final Project of the course of "Introduction to Computer Vision" 

We developed a text-to-image model able to generate Van Gogh style pictures based on text prompts. We exploited a [pre-trained model](https://huggingface.co/CompVis/stable-diffusion-v1-4) of Stable Diffusion available in Hugging Face, which hosts the [WikiArt dataset](https://huggingface.co/datasets/huggan/wikiart/viewer/default/train) comprising 103.250 pieces of art (paintings, drawings and sculptures). The pre-trained model was fine-tuned on a portion of the WikiArt dataset including including images and descriptions of more than 400 Van Gogh's masterpieces. The model was quantitatively evaluated through a couple of metrics normally used for diffusion models: the [CLIP score](https://huggingface.co/docs/diffusers/conceptual/evaluation) (28.89) and the [Fréchet Inception Distance](https://huggingface.co/docs/diffusers/conceptual/evaluation) (677.19). In order to assess the quality of the output images, different users were also asked to evaluate the output images by assigning a score from 1 to 5 to five different criteria: relevance to the initial prompt (4.56 ± 0.7), Van Gogh style (3.94 ± 1.34), complexity (4.0 ± 0.87), creativity (4.0 ± 1.12), and general satisfaction (4.38 ± 0.78). 

### Human_Activity_Classification_Heart_Rate_Prediction - [Github Link]()

Final Project of the course "Data Analytics and Internet of Things"



## Contacts
You can reach me at [LinkedIn](https://www.linkedin.com/in/caterina-gallo) or [GitHub](https://github.com/CatGallo).
