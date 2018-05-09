# Merinjei Bot
Merinjei bot is system for facebook page maintenance.

The idea behind him is to make facebook page maintenance easier than ever. If you start using the bot your only job will be to post new content in your page.Everything else will be covered by your fellow Merinjei.

He will control the hate comments by deleting them so that your page stays clean and positive.
Also he has a chat bot which you can add to your facebook page so that not only you will have `100%` responding rate but also your clients will be able to communicate with the page whenever they want to and get imidiate response.
The chatbot is using online services for question answering for the moment the bot uses only [StackOverflow](https://stackoverflow.com).

The application is **hosted** on https://merinjei-bot.herokuapp.com.
However yet it is not usable from non-testers as Facebook's policy states code review for the project before it gets released so that they are sure the application is not malicious in any way.

-----------------------------

## Setup MerinjeiBot localy
### Get yourself https connection
Firstly you should find yourself an https comminication and domain for it. I would suggest you to use [ngrok](https://ngrok.com/download)
it is very easy to setup and use.
#### How to use ngrok
To use ngrok simply after installing it type:
On linux:
```
ngrok http 8000
```
On Windows:
```
ngrok.exe http 8000
```
### Create Facebook application
Secondly you will need to create yourself an application in [Facebook](https://developers.facebook.com).
After creating one you should add to it some of their products:
- Facebook login
- Messenger
- Webhooks

_Note that you should add to the Facebook login app your domain in the Valid OAuth Redirect URIs_


### Install project dependencies
The next step is to install the dependencies for the project. There are two ways for the moment:

#### Install dependencies using pip
1. Install pip:
```Bash
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```
2. Install the dependencies: 
```
pip install pip_requirements.txt
python -m nltk.downloader all
```
#### Install dependencies using conda
1. Install [conda](https://conda.io/docs/user-guide/install/index.html)
2. Run the environment:
```
conda env  create --file conda_environment.yml
```
### Put domain settings
Open the merinjei_bot\merinjeiweb\CONSTANTS.py and put your data according to your application in the certain variables. Where
```Python
APP_ID = '' # The APP Id which facebook has given for your application.
DOMAIN = '' # The domain which you are using and is added to the Valid OAuth Redirect URIs in FB
VERIFY_TOKEN = '' # A token which facebook will use for authenticate your application. You pick it.
APP_SECRET = '' # The secret code which facebook has given for you application. It can be taken from the dashboard of your application from developers.facebook.com.
```

### Run the application
Navigate to the merinjeiweb folder and run the application:
```
cd merinjeiweb
python manage.py
```