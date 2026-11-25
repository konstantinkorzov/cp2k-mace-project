# Берем официальный образ CP2K
ARG CP2K_IMAGE=cp2k/cp2k:2024.3_mpich_generic_psmp
FROM ${CP2K_IMAGE}

# Устанавливаем Python и pip (без mc, чтобы не ломать кэш)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
	
# Обновляем pip
RUN pip3 install --upgrade pip

# Устанавливаем PyTorch (тяжелый слой, кэшируется)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Устанавливаем MACE, ASE, FastAPI и Uvicorn (тяжелый слой, кэшируется)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install mace-torch ase fastapi uvicorn numpy requests
	
# Устанавливаем Node.js (LTS 20.x) через официальный скрипт
RUN apt-get update && apt-get install -y curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*
	
# Устанавливаем Midnight Commander (и другие утилиты) в самом конце
RUN apt-get update && apt-get install -y mc git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /workspace

# Метки OCI: связывают пакет в GHCR с репозиторием GitHub.
# LABEL стоит в конце, чтобы не сбрасывать кэш тяжёлых слоёв выше.
LABEL org.opencontainers.image.source="https://github.com/konstantinkorzov/cp2k-mace-project" \
      org.opencontainers.image.title="cp2k-mace" \
      org.opencontainers.image.description="CP2K + MACE pipeline: zero-trust master-worker DFT, blockchain ledger, web panel" \
      org.opencontainers.image.licenses="MIT"