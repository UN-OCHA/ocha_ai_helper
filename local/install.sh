#!/usr/bin/env bash

set -e -u

# Usage.
usage() {
  echo "Usage: ./local/install.sh [OPTIONS]" >&2
  echo "-h                    : Display usage" >&2
  echo "-m                    : Create local image" >&2
  echo "-b                    : Rebuild service images" >&2
  echo "-u                    : Pull latest service and base images and recreate containers" >&2
  echo "-s                    : Stop the site containers" >&2
  echo "-x                    : Shutdown and remove the site containers" >&2
  echo "-v                    : Also remove the volumes when shutting down the containers" >&2
  exit 1
}

create_image="no"
rebuild_service_images="no"
update="no"
stop="no"
shutdown="no"
shutdown_options=""
start_options=""

# Parse options.
while getopts "hmbdusxv" opt; do
  case $opt in
    h)
      usage
      ;;
    m)
      create_image="yes"
      ;;
    b)
      start_options="$start_options --build"
      ;;
    u)
      update="yes"
      ;;
    s)
      stop="yes"
      ;;
    x)
      shutdown="yes"
      ;;
    v)
      shutdown_options="$shutdown_options -v"
      ;;
    *)
      usage
      ;;
  esac
done

function docker_compose {
  docker compose -f local/docker-compose.yml "$@"
}

# Load the environment variables.
# They are only available in this script as we don't export them.
source local/.env

# Stop the containers.
if [ "$stop" = "yes" ]; then
  echo "Stop the containers."
  docker_compose stop || true
  exit 0
fi

# Stop and remove the containers.
if [ "$shutdown" = "yes" ]; then
  echo "Stop and remove the containers."
  docker_compose down $shutdown_options || true
  exit 0
fi

# Update the image.
if [ "$update" = "yes" ]; then
  echo "Pull service images."
  docker_compose pull --ignore-pull-failures
  echo "Pull base site image."
  docker pull "$(grep -E -o "FROM ([^ ]+)$" docker/Dockerfile | awk '{print $2}')"
  create_image="yes"
fi;

# Build local image.
if [ "$create_image" = "yes" ]; then
  echo "Build local image."
  make IMAGE_NAME=$IMAGE_NAME IMAGE_TAG=$IMAGE_TAG
fi;

# Create the site, memcache and mysql containers.
echo "Create the site and elasticsearch containers."
docker_compose up -d --remove-orphans $start_options

# Dump some information about the created containers.
echo "Dump some information about the created containers."
docker_compose ps -a
