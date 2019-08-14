# Smile Meter

This repo contains the project of smile detection and to give a score for the degree of smile. 

## Modules:

1.  Smile detection- Uses CNN to predict the emotion
2.  Smile meter    - Uses facial landmarks to to the score  

### Installation and running the server

Installing dependencies

```sh
$ git clone -b feature-mcdonalds https://github.com/shubham-2817/mcdonalds.git
$ cd mcdonalds/
$ virtualenv env_mcd
$ source env_mcd/bin/activate
$ pip3 install -r requirements.txt
```

Training the model
```sh
$ python3 keras7.py --train=yes
$ python3 keras8.py --train=yes
$ python3 keras9.py --train=yes
```


Get the flask-api running
```sh
$ python3 api_call.py
```

