#!/bin/bash

sudo yum -y update
sudo yum -y install epel-release
sudo yum -y install python-pip
sudo pip install pip --upgrade
sudo pip install tensorflow

