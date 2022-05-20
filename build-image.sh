#!/bin/bash

docker build -t remlabuild:latest -f Dockerfile_build .
#&& docker run -v "$(pwd)/exports:/exports" remlabuild:latest
