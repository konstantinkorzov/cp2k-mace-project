CACHE_DIR_CPU = .buildx-cache-cpu
CACHE_DIR_GPU = .buildx-cache-gpu

# Имя вашего логина на GitHub
GITHUB_USER = konstantinkorzov
# Имя образа (можно любое, например cp2k-mace)
IMAGE_NAME = cp2k-mace
# Реестр
REGISTRY = ghcr.io

# Версия (можно менять вручную или брать из git)
TAG ?= latest

# Пути к образам
CPU_IMAGE = $(REGISTRY)/$(GITHUB_USER)/$(IMAGE_NAME):$(TAG)-cpu
GPU_IMAGE = $(REGISTRY)/$(GITHUB_USER)/$(IMAGE_NAME):$(TAG)-gpu

.PHONY: build build-cpu build-gpu push push-cpu push-gpu release release-cpu release-gpu clean

# ==========================================
# Сборка (локально)
# ==========================================

# Сборка обоих образов (используйте, только если меняли Dockerfile)
build: build-cpu build-gpu

# Сборка только CPU-версии (для локальной проверки)
build-cpu:
	docker buildx build \
		--build-arg CP2K_IMAGE=cp2k/cp2k:2024.3_mpich_generic_psmp@sha256:979e011c2ea15a2a56eae16b5f3a2d214142bb74fe644bfb302d366dddc2a443 \
		--load \
		--cache-to type=local,dest=$(CACHE_DIR_CPU),mode=max \
		--cache-from type=local,src=$(CACHE_DIR_CPU) \
		-t $(CPU_IMAGE) .

# Сборка только GPU-версии (для сервера)
build-gpu:
	docker buildx build \
		--build-arg CP2K_IMAGE=cp2k/cp2k:2024.3_mpich_generic_cuda_P100_psmp@sha256:e18b06ef11df6593b88f4cc35e52678b80e9edebfc56de70e7bc5e3215625d3d \
		--load \
		--cache-to type=local,dest=$(CACHE_DIR_GPU),mode=max \
		--cache-from type=local,src=$(CACHE_DIR_GPU) \
		-t $(GPU_IMAGE) .
# ==========================================
# Отправка в GHCR
# ==========================================

# Отправка обоих образов
push: push-cpu push-gpu

# Отправка только CPU-версии
push-cpu:
	docker push $(CPU_IMAGE)

# Отправка только GPU-версии
push-gpu:
	docker push $(GPU_IMAGE)

# ==========================================
# Комбинированные команды (сборка + отправка)
# ==========================================

# Собрать и отправить всё (используйте, только если меняли Dockerfile)
release: build push
	@echo "✅ Все образы собраны и отправлены в GHCR!"

# Собрать и отправить только CPU (самая частая команда для локальной работы)
release-cpu: build-cpu push-cpu
	@echo "✅ CPU-образ собран и отправлен в GHCR!"

# Собрать и отправить только GPU (когда готовы к деплою на сервер)
release-gpu: build-gpu push-gpu
	@echo "✅ GPU-образ собран и отправлен в GHCR!"

# ==========================================
# Очистка
# ==========================================

# Удалить локальные образы (осторожно: если не отправили в GHCR, они пропадут)
clean:
	-docker rmi $(CPU_IMAGE) $(GPU_IMAGE) 2>/dev/null || true
	@echo "🧹 Локальные образы удалены (если существовали)."