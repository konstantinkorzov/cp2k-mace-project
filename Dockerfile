# Базовая сборка CP2K — пин по digest, а не по плавающему тегу: тег могут
# пересобрать, и тогда у узлов сети окажутся разные бинарники (другая
# арифметика и другой формат лога). Digest меняется только вместе с версией
# протокола: см. CP2K_BUILD_EXPECTED в src/cp2k_runner.py.
ARG CP2K_IMAGE=cp2k/cp2k:2024.3_mpich_generic_psmp@sha256:979e011c2ea15a2a56eae16b5f3a2d214142bb74fe644bfb302d366dddc2a443
FROM ${CP2K_IMAGE}

# ARG не переживает FROM — объявляем снова, чтобы подставить в ENV.
# CP2K_BUILD уезжает в header.json каждого блока: по нему видно, какой
# именно сборкой посчитан кадр.
ARG CP2K_IMAGE
ENV CP2K_BUILD=${CP2K_IMAGE}

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
    pip3 install mace-torch ase fastapi uvicorn numpy requests cryptography
	
# Устанавливаем Node.js (LTS 20.x) через официальный скрипт
RUN apt-get update && apt-get install -y curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*
	
# Устанавливаем Midnight Commander (и другие утилиты) в самом конце
RUN apt-get update && apt-get install -y mc git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Предобученная MACE-MP-0 (small) — универсальная модель на все элементы
# (Materials Project, ~89 элементов), 32.6 МБ. Нужна офлайн: без неё MACE при
# первом запуске лезет в интернет и падает. Путь по умолчанию —
# mace_pipeline/config.json → foundation_model.
ENV MACE_MODEL=/opt/mace-models/mace-mp-0-small.model
RUN mkdir -p /opt/mace-models \
    && curl -fsSL --retry 3 --max-time 600 -o "$MACE_MODEL" \
       "https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model" \
    && python3 -c "import os; print(\"MACE model:\", os.path.getsize(os.environ['MACE_MODEL']), \"bytes\")"

# Создаем рабочую директорию
WORKDIR /workspace

# Метки OCI: связывают пакет в GHCR с репозиторием GitHub.
# LABEL стоит в конце, чтобы не сбрасывать кэш тяжёлых слоёв выше.
LABEL org.opencontainers.image.source="https://github.com/konstantinkorzov/cp2k-mace-project" \
      org.opencontainers.image.title="cp2k-mace" \
      org.opencontainers.image.description="CP2K + MACE pipeline: zero-trust master-worker DFT, blockchain ledger, web panel" \
      org.opencontainers.image.licenses="MIT"