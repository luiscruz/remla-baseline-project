#!/bin/bash

minikube start

minikube enable addons ingress

kubectl cluster-info

kubectl apply -f deployment.yml

kubectl get pods -o wide
kubectl get services -o wide
kubectl get ingress -o wide
