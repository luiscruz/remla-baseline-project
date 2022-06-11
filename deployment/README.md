# Deploying to minikube
From the deployment folder:
1. Run `minikube-build-and-push-containers.sh` to start minikube and build the images
2. Run `deploy-charts-minikube.sh` to deploy the prometheus stack
3. Create a file `secrets.yaml` in `./k8s/` with contents:
   ```yaml
    apiVersion: v1
    kind: Secret
    metadata:
      name: api-keys
    type: Opaque
    data:
      key: <api-keys delimited by ','>
    ```
   Make sure to replace `<api-keys delimited b ','` by the actual api keys.
4. Run `kubectl apply -f deployment/k8s` to deploy the ingress and inference services
5. If you now run `kubectl get pods` everything should be either starting/running