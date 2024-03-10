## Placebo style lyrics generator

### Project description
Pre-trained GPT-2 was fine-tuned on the dataset made from Placebo songs lyrics. Model generates lyrics by prompt. Training process is documented in the notebook lsml-final. 
To provide user-friendly interface for lyrics generation I developed web application using the Streamlit framework. Folder streamlit-gpt contains files needed to run web application on a local machine. 
I used following tutorial to fine-tune the model: [Towards Data Science](https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272). 

###
Installation and running the app
1. Dowload folder streamlit-gpt to your local machine;
2. Navigate to the foler and run command docker build -t lsml_final;
3. Whe building is completed run command docker-compose up.

These commands will build and run three docker containers two of them contain web application and one contains load balancer. Access web application at http://localhost:80.



