# which path syntax to use, depends on the OS used to run this script
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SHARED_FOLDER_HOSTPATH=$(pwd)/../services-shared-folder/
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    SHARED_FOLDER_HOSTPATH=$(pwd | sed -e 's!/!//!g' | sed -e 's!^//c!C:!g')//..//services-shared-folder//
fi

MOUNTING_PERSISTENT_STORAGE_MINIKUBE=/data/shared/

echo "$SHARED_FOLDER_HOSTPATH:$MOUNTING_PERSISTENT_STORAGE_MINIKUBE"

minikube mount $SHARED_FOLDER_HOSTPATH:$MOUNTING_PERSISTENT_STORAGE_MINIKUBE
