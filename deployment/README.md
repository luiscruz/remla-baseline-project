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
   stringData:
     API_KEY: <Stackexchange API key>
   ---
   apiVersion: v1
   kind: Secret
   metadata:
     name: drive-secrets
   type: Opaque
   stringData:
     DRIVE_SVC_JSON: >
       <contents of gdrive secret json file>
    ```
   Make sure to replace `<api-keys delimited b ','` by the actual api keys.
4. Run `kubectl apply -f deployment/k8s` to deploy the ingress and inference services
5. If you now run `kubectl get pods` everything should be either starting/running