# Deploying to minikube
From the deployment folder:
1. Run `minikube-build-and-push-containers.sh` to start minikube and build the images
2. Run `deploy-charts-minikube.sh` to deploy the prometheus stack
3. Run `kubectl apply -f deployment/k8s` to deploy the ingress and inference services
4. If you now run `kubectl get pods` everything should be either starting/running