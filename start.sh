#!/usr/bin/zsh
sudo touch ./output.log
sudo nohup python3 ./server.py > ./output.log &
