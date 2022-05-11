output "public_url" {
  value = "https://${google_compute_address.nidhogg-external-address.address}"
}