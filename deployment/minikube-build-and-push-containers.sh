#!/usr/bin/env sh

# which path syntax to use, depends on the OS used to run this script
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SHARED_FOLDER_HOSTPATH=$(pwd)/../services-shared-folder/
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    SHARED_FOLDER_HOSTPATH=$(pwd | sed -e 's!/!//!g' | sed -e 's!^//c!C:!g')//..//services-shared-folder//
fi

MOUNTING_PERSISTENT_STORAGE_MINIKUBE=/data/shared/

echo "$SHARED_FOLDER_HOSTPATH:$MOUNTING_PERSISTENT_STORAGE_MINIKUBE"

minikube status || minikube start --cpus 4 --memory 8192

eval $(minikube docker-env --shell bash)
#eval $(minikube docker-env)

# NOTE: use docker-compose to create the images beforehand, since minikube uses those images for deployment
minikube image load inference-service:latest
minikube image load scraping-service:latest
minikube image load scraping-controller:latest
minikube image load training-service:latest

#docker-compose -f docker-compose/docker-compose.yml build --no-cache

minikube mount $SHARED_FOLDER_HOSTPATH:$MOUNTING_PERSISTENT_STORAGE_MINIKUBE

