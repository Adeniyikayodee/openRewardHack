#!/usr/bin/env bash

# Stop if any command fails.
set -e

# Change to the directory of this script so that relative paths resolve relative to this script.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Challenge victim images (agent sandbox already exists on DockerHub).
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-avatar-victim:latest ./challenges/avatar/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-back_to_the_past-victim:latest ./challenges/back_to_the_past/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-chunky-blog:latest ./challenges/chunky/images/blog
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-chunky-nginx:latest ./challenges/chunky/images/nginx
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-chunky-rcache:latest ./challenges/chunky/images/rcache
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-delulu-victim:latest ./challenges/delulu/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-diffecient-victim:latest ./challenges/diffecient/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-ezmaze-victim:latest ./challenges/ezmaze/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-failproof-victim:latest ./challenges/failproof/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-frog_waf-victim:latest ./challenges/frog_waf/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-flag_command-victim:latest ./challenges/flag_command/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-glacier_exchange-victim:latest ./challenges/glacier_exchange/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-just_another_pickle_jail-victim:latest ./challenges/just_another_pickle_jail/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-labyrinth_linguist-victim:latest ./challenges/labyrinth_linguist/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-locktalk-victim:latest ./challenges/locktalk/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-motp-victim:latest ./challenges/motp/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-network_tools-victim:latest ./challenges/network_tools/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-noisier_crc-victim:latest ./challenges/noisier_crc/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-noisy_crc-victim:latest ./challenges/noisy_crc/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-randsubware-victim:latest ./challenges/randsubware/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-robust_cbc-victim:latest ./challenges/robust_cbc/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-shuffled_aes-victim:latest ./challenges/shuffled_aes/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-were_pickle_phreaks_revenge-victim:latest ./challenges/were_pickle_phreaks_revenge/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-walking_to_the_sea_side-victim:latest ./challenges/walking_to_the_sea_side/images/walking-to-the-sea-side
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-unbreakable-victim:latest ./challenges/unbreakable/images/victim
docker build --platform linux/amd64 -t generalreasoning/evals-cybench-skilift-victim:latest ./challenges/skilift/images/victim
