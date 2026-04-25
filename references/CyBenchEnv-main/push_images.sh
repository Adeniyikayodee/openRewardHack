#!/usr/bin/env bash

# Stop if any command fails.
set -e

# Push challenge victim images to DockerHub under generalreasoning/.
# Agent sandbox image (generalreasoning/evals-cybench-agent-sandbox) already exists.

docker push generalreasoning/evals-cybench-avatar-victim:latest
docker push generalreasoning/evals-cybench-back_to_the_past-victim:latest
docker push generalreasoning/evals-cybench-chunky-blog:latest
docker push generalreasoning/evals-cybench-chunky-nginx:latest
docker push generalreasoning/evals-cybench-chunky-rcache:latest
docker push generalreasoning/evals-cybench-delulu-victim:latest
docker push generalreasoning/evals-cybench-diffecient-victim:latest
docker push generalreasoning/evals-cybench-ezmaze-victim:latest
docker push generalreasoning/evals-cybench-failproof-victim:latest
docker push generalreasoning/evals-cybench-frog_waf-victim:latest
docker push generalreasoning/evals-cybench-flag_command-victim:latest
docker push generalreasoning/evals-cybench-glacier_exchange-victim:latest
docker push generalreasoning/evals-cybench-just_another_pickle_jail-victim:latest
docker push generalreasoning/evals-cybench-labyrinth_linguist-victim:latest
docker push generalreasoning/evals-cybench-locktalk-victim:latest
docker push generalreasoning/evals-cybench-motp-victim:latest
docker push generalreasoning/evals-cybench-network_tools-victim:latest
docker push generalreasoning/evals-cybench-noisier_crc-victim:latest
docker push generalreasoning/evals-cybench-noisy_crc-victim:latest
docker push generalreasoning/evals-cybench-randsubware-victim:latest
docker push generalreasoning/evals-cybench-robust_cbc-victim:latest
docker push generalreasoning/evals-cybench-shuffled_aes-victim:latest
docker push generalreasoning/evals-cybench-were_pickle_phreaks_revenge-victim:latest
docker push generalreasoning/evals-cybench-walking_to_the_sea_side-victim:latest
docker push generalreasoning/evals-cybench-unbreakable-victim:latest
docker push generalreasoning/evals-cybench-skilift-victim:latest
