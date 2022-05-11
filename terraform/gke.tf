resource "google_container_cluster" "gke_cluster" {
  name               = "nidhogg-gke-cluster"
  location           = local.region // Making it regional
  project            = local.project_id
  initial_node_count = 1
  network            = google_compute_network.gke_vpc.name
  subnetwork         = google_compute_subnetwork.gke_subnet.name
}