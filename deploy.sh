#!/usr/bin/env bash
# Deploy the chatbot to a GPU machine, where it runs as two systemd user services:
# $SERVICE-ollama (the model server) and $SERVICE-chat (chat_server.py).
# Usage: DEPLOY_HOST=user@host DATASET=<name> ./deploy.sh [push|log]
#   push (default): copy code + the dataset's reviewed chunks, rebuild the vector store there, restart the chat
#   log:            copy the live chat log (questions, answers, ratings) back here for review
# The live chat log is never overwritten by a push.
set -euo pipefail
HOST=${DEPLOY_HOST:?set DEPLOY_HOST=user@host}
DATASET=${DATASET:?set DATASET=<dataset name>}
DIR=${DEPLOY_DIR:-slm_chatbot}
SERVICE=${SERVICE:-chatbot}
PORT=${PORT:-8010}
cd "$(dirname "$0")"

case "${1:-push}" in
  push)
    rsync -a --relative ./*.py ./static "./datasets/$DATASET/chunks.md" "./datasets/$DATASET/config.json" \
      "./datasets/$DATASET/test_questions.csv" "$HOST:$DIR/"
    ssh "$HOST" "cd $DIR && FASTEMBED_CACHE_PATH=\$HOME/$DIR/.cache/fastembed .venv/bin/python build.py $DATASET \
      && systemctl --user restart $SERVICE-chat \
      && for i in \$(seq 60); do curl -sk -o /dev/null https://localhost:$PORT/ && break; sleep 0.5; done \
      && echo 'chat restarted and ready on port $PORT'"
    ;;
  log)
    rsync -a "$HOST:$DIR/datasets/$DATASET/chat_log.jsonl" "datasets/$DATASET/chat_log.jsonl"
    echo "live log copied to datasets/$DATASET/chat_log.jsonl"
    ;;
  *)
    echo "usage: $0 [push|log]" >&2; exit 1
    ;;
esac
