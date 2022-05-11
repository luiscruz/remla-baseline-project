data "google_client_config" "provider" {}

provider "kubernetes" {
  host    = google_container_cluster.gke_cluster.endpoint
  token   = data.google_client_config.provider.access_token
  client_certificate = base64decode(
    google_container_cluster.gke_cluster.master_auth[0].client_certificate,
  )
  cluster_ca_certificate = base64decode(
    google_container_cluster.gke_cluster.master_auth[0].cluster_ca_certificate,
  )
  client_key = base64decode(google_container_cluster.gke_cluster.master_auth[0].client_key)
}
