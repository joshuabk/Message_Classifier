# Disaster Response Pipeline Project

Introduction
This application helps organizations during a disater, it categorizes incoming messages so the message can be sent to the appropriate people and make disaster responses more efficient.  The data is presented in a web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
5. 

Files in the repository

app
| - template
| |- master.html  # main webpage
| |- go.html  # classification date page of web app
|- run.py  # Flask file that runs app and presents the webpage with the data
data
|- disaster_categories.csv  # message category data to process and train on
|- disaster_messages.csv  # message data to process and train on 
|- process_data.py  # this file process the data and puts it into a database for learning
|- DisasterResponse.db  # database to save clean data to
models
|- train_classifier.py  # this file builds an ML pipline and trains the model to make the predictions
|- classifier.pkl  # this is the saved ML model
README.md
