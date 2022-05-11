terraform {
  required_providers {
    kubernetes = {
      source = "hashicorp/kubernetes"
      version = "2.11.0"
    }
    google = {
      source = "hashicorp/google"
      version = "4.20.0"
    }
  }
}