resource "google_compute_network" "gke_vpc" {
  project                 = local.project_id
  name                    = local.vpc_name
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "gke_subnet" {
  project                  = local.project_id
  name                     = "${local.vpc_name}-subnet"
  ip_cidr_range            = "10.10.0.0/20"
  network                  = google_compute_network.gke_vpc.self_link
  region                   = local.region
  private_ip_google_access = true
}

resource "google_compute_address" "nidhogg-external-address" {
  project = local.project_id
  name    = local.vpc_name
  region  = local.region
}