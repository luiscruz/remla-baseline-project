# Release engeneering of Multilabel classification on Stack Overflow tags

This project designs a Release pipeline for a multilabel classifying ML applications. The application detials can be found on [the forked project](https://github.com/luiscruz/remla-baseline-project/blob/main/README.md). The team working on this project has applied the following changes to improve the development of the ML applications:

- Modularise code in separate code files
- PyTest
- MLLint
- Docker
- DVC
- Kubernetes

# Testing: PyTest

Pytests can be found in the "tests" directory. Newly added test classes should end with "test\_\*.py" and newly added test functions should start with "test"

# Installation

## Local (with `venv` as virtual environemnt)

```
python -m venv ./venv
. ./venv/bin/activate
pip install -r requirements.txt
dvc pull
python src/serve_model.py
```

## Docker

```
export GIT_COMMIT=$(git rev-parse HEAD)
export PORT=5000
docker build --build-arg GIT_COMMIT=$GIT_COMMIT --build-arg PORT=$PORT -t group7/remla:$GIT_COMMIT .
docker run --rm -p $PORT:$PORT group7/remla:${GIT_COMMIT}
```
alternatively with powershell you can run the following from the root:
```
docker build -t remla7 ./
docker run --rm -p 5000:5000 -it remla7
```

## Server

The application is now served at 127.0.0.1:5000.

# DVC

DVC is used to manage the ML pipeline version control artifacts. The artifacts are pushed to a project google drive repo which, the first time you connect to it, needs authentication.

To use DVC,
Run the pipeline by:

```console
dvc repro
```

commit completed artifacts by:

```console
dvc commit -am "<message>"
```

and push the commit with:

```console
dvc push
```

To use pushed artifacts you can simply use:

```console
dvc pull
```

# Kubernetes

First [install minikube](https://minikube.sigs.k8s.io/docs/start/) to run a local cluster (only applicable for this project as you normally don't want to run a local cluster).
Once installed, start the cluster by:

```console
minikube start
```

Next, apply the Kubernetes deployment to the cluster by running:

```console
kubectl apply -f kubernetes-deploy.yml
```

Then to get the Prometheus real-time database for monitoring and Grafana for visualizations set up, we first need to [install Helm](https://helm.sh/docs/intro/install/)

Once Helm is installed, perform the following commands (based on the [ArtifactHub information](https://artifacthub.io/packages/helm/prometheus-community/kube-prometheus-stack)):

```console
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install promstack prometheus-community/kube-prometheus-stack
```