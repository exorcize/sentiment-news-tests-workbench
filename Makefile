.PHONY: help build up down logs test clean

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker image"
	@echo "  make up       - Start services"
	@echo "  make down     - Stop services"
	@echo "  make logs     - View logs"
	@echo "  make test     - Run test client"
	@echo "  make clean    - Remove containers and volumes"

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	docker compose --profile test up test-client

clean:
	docker compose down -v
