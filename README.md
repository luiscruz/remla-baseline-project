# Release engeneering of Multilabel classification on Stack Overflow tags

This project designs a Release pipeline for a multilabel classifying ML applications. The application detials can be found on [the forked project](https://github.com/luiscruz/remla-baseline-project/blob/main/README.md). The team working on this project has applied the following changes to improve the development of the ML applications:

- Modularise code in separate code files
- PyTest
- MLLint
- Docker
- DVC
- Kubernetes

# Table of Content

1. [Installation](#installation)
2. [DVC](#dvc)
3. [Testing with PyTest](#testing-with-pytest)
4. [Kubernetes](#kubernetes)
5. [Run release pipeline](#run-release-pipeline)

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

Alternatively with powershell you can run the following from the root:

```
docker build -t remla7 ./
docker run --rm -p 5000:5000 -it remla7
```

The application is now served at 127.0.0.1:5000

# DVC

DVC is used to manage the ML pipeline version control artifacts.
The artifacts are pushed to a project google drive repo which,
the first time you connect to it, it needs authentication.

There is a data folder in a Google Drive folder.
This folder has read-only access to a gmail account of which the credentials
are available in the `gdrive-creds.json` file.
You can move this file to the dvc location to get read-only access.

```
mkdir .dvc/tmp
mv gdrive-creds.json .dvc/tmp/gdrive-user-credentials.json
```

This is the same account used in the CI/CD pipelines.

As this is just a regular account it might get eventually blocked by Google
because of suspicious activity. For more robust usage create a service account
using a Google Cloud (GC) account:

https://github.com/iterative/dvc.org/blob/master/content/docs/user-guide/setup-google-drive-remote.md#using-service-accounts

To use get the data from DVC, first pull the artifacts by:

```console
dvc pull
```

The pipeline can be run by:

```console
dvc repro
```

Commit completed artifacts by:

```console
dvc commit -am "<message>"
```

Push the commit with:

```console
dvc push
```

# Testing with PyTest

Pytests can be found in the "tests" directory. Newly added test classes should end with "test\_\*.py" and newly added test functions should start with "test"

# Kubernetes

First [install minikube](https://minikube.sigs.k8s.io/docs/start/) to run a local cluster (only applicable for this project as you normally don't want to run a local cluster).
Once installed, start the cluster by:

```console
minikube start
minikube addons enable ingress
```

Next, to get the Prometheus real-time database for monitoring and Grafana for visualizations set up, we first need to [install Helm](https://helm.sh/docs/intro/install/)

Once Helm is installed, perform the following commands (based on the [ArtifactHub information](https://artifacthub.io/packages/helm/prometheus-community/kube-prometheus-stack)):

```console
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install promstack prometheus-community/kube-prometheus-stack
```

Then, apply the Kubernetes deployment to the cluster by running:

```console
kubectl apply -f kubernetes-deploy.yml
```

This will run the Machine Learning model application of group 7 on the IP address of minikube on port 5000. This IP can be found via `minikube ip`)

You can visit the Prometheus or Grafana applications by running the following commands:

```console
kubectl get pods # Find pod names
kubectl port-forward prometheus-promstack-kube-prometheus-prometheus-0 9090 # --> Prometheus on localhost:9090
kubectl port-forward promstack-grafana-[hash] 3000 # --> Grafana on localhost:3000
```

Log in on Grafana with the following credentials:
Username: `admin`
Password: `prom-operator`

# Run release pipeline

To add a semantic version tag to a commit, please do the following on the `dev` branch.

```console
# Add release tag locally and on the remote
git tag release
git push origin release
```

This will then trigger the "Deploy - Release tag" pipeline, which will increment the version in the VERSION.txt file.
It will also create and push a commit with a v?.?.? tag as a release on `dev`.

If you need to rerun the "Deploy - Release tag" pipeline, you'll most likely need to delete the `release` tag first. This can be done by running the following commands.

```console
# Remove release tag locally and on the remote
git tag -d release
git push origin :release
```

Then just re-add the `release` tag again by running the first commands specified in this section.
