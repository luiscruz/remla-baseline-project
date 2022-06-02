#!/bin/bash

# This script can be run as-is or used as a reference
# for minikube / kubectl commands

# Create new minikube cluster with ingress enabled
minikube start
minikube addons enable ingress

# Show cluster info to verify that cluster is running
kubectl cluster-info
# Run "minikube dashboard" for an in-browser dashboard

# Create pod, service, and ingress
kubectl apply -f deployment.yml
# Run "kubectl delete -f deployment.yml" to delete all defined components

# Verify that it went correctly, or use minikube dashboard
kubectl get pods -o wide
kubectl get services -o wide
kubectl get ingress -o wide
kubectl get nodes -o wide

# Quick way to get the ip:port / urls you need to access the API
minikube service list

# To access the /predict endpoint, simply access the IP of the ingress
# which is displayed by "minikube service list" with the endpoint appended, 
# i.e. <ingress-ip>/predict
