helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install promstack prometheus-community/kube-prometheus-stack