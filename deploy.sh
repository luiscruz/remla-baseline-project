#!/bin/bash

# This script can be run as-is or used as a reference
# to deploy the remla app on a fully-functioning minikube cluster.

# Install requirements: helm v3.9, minikube v1.25


# Create a new minikube cluster with ingress enabled:
minikube start
minikube addons enable ingress


# Show cluster info to verify that cluster is running:
kubectl cluster-info
# Run "minikube dashboard" for an in-browser dashboard


# Install kube-prometheus-stack used for monitoring
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install -f k8s/values.yml promstack prometheus-community/kube-prometheus-stack


# Create pods, services, and ingress:
kubectl apply -f k8s/deployment.yml


# Quick way to get the ip:port / urls you need to access the API
minikube service list

# To access the /predict endpoint, simply access the IP of the ingress
# which is displayed by "minikube service list" with the endpoint appended, 
# i.e. <ingress-ip>/predict
# Available endpoints: /predict, /metrics, /dashboard
