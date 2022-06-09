#!/usr/bin/env sh

# which path syntax to use, depends on the OS used to run this script
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PROJECT_ROOT_PATH=$(pwd)/../
    # DVC_CONFIG_PATH=$PROJECT_ROOT_PATH/.dvc
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    # PROJECT_ROOT_PATH=$(pwd | sed -e 's!/!//!g' | sed -e 's!^//c!C:!g')//..//.dvc//
    PROJECT_ROOT_PATH=$(pwd | sed -e 's!/!//!g' | sed -e 's!^//c!C:!g')//..//dvc-cache
    # DVC_CONFIG_PATH=$PROJECT_ROOT_PATH//.dvc
fi

# MOUNTING_PERSISTENT_STORAGE_MINIKUBE=/data/shared/.dvc/
MOUNTING_PERSISTENT_STORAGE_MINIKUBE=/data/shared/dvc-cache/
# MOUNTING__CONF_PATH=$MOUNTING_PERSISTENT_STORAGE_MINIKUBE/.dvc

echo "Starting Minikube and mounting $PROJECT_ROOT_PATH:$MOUNTING_PERSISTENT_STORAGE_MINIKUBE"
# echo "DVC conf path set: $MOUNTING_DVC_CONF_PATH\n"
echo "$PROJECT_ROOT_PATH:$MOUNTING_PERSISTENT_STORAGE_MINIKUBE"

minikube status || minikube start

eval $(minikube docker-env --shell bash)

# NOTE: use docker-compose to create the images beforehand, since minikube uses those images for deployment
# minikube image load inference-service:latest --overwrite
# minikube image load test-service:latest --overwrite
docker-compose -f docker-compose/docker-compose.yml build --no-cache

minikube mount $PROJECT_ROOT_PATH:$MOUNTING_PERSISTENT_STORAGE_MINIKUBE
