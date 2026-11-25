# Публикация: Docker-образы и репозиторий

> Назад: [README](../README.md) · [Whitepaper](WHITEPAPER.md) · [Измерения](MEASUREMENTS.md)

Два независимых артефакта: **образ в GHCR** и **код на GitHub**. Можно публиковать
их по отдельности.

---

## 1. Docker-образ в GHCR

### Что публикуется

| Тег | База | Размер (несжатый) |
|---|---|---|
| `latest-cpu` | `cp2k/cp2k:2024.3_mpich_generic_psmp` | ~12.7 ГБ |
| `latest-gpu` | `cp2k/cp2k:2024.3_mpich_generic_cuda_P100_psmp` | ~21.5 ГБ |

Образ содержит всё окружение (CP2K, Python, PyTorch, MACE, Node) — **код в него не
копируется**, он монтируется из репозитория при запуске.

### Вход в реестр

Нужен classic PAT с правом **`write:packages`** (тот же токен подойдёт для `git push`,
если добавить `repo`):

```bash
docker login ghcr.io -u konstantinkorzov
# Password: ghp_…   (ввод не отображается)
```

`Login Succeeded` ещё не значит, что можно пушить: права проверяются при `push`.

### Сборка и отправка

```bash
make release-cpu     # собрать и отправить CPU
make release-gpu     # собрать и отправить GPU
make release         # обе версии
```

Полезные варианты:

```bash
make build-cpu       # только собрать локально
make push-cpu        # только отправить уже собранное
make release-cpu TAG=v1.0.0   # тег v1.0.0-cpu
```

Если `buildx` ещё не настроен:

```bash
docker buildx create --name cp2k-builder --use
docker buildx inspect --bootstrap
```

### Сделать пакет публичным

Приватный пакет считается в вашу квоту (а образы большие), и `docker pull` без логина
не работает.

<https://github.com/users/konstantinkorzov/packages/container/cp2k-mace/settings>
→ **Danger Zone** → **Change visibility** → **Public**.

Проверка без авторизации:

```bash
docker logout ghcr.io
docker pull ghcr.io/konstantinkorzov/cp2k-mace:latest-cpu
```

### Ошибки

| Ошибка | Причина |
|---|---|
| `denied: requested access to the resource is denied` | у токена нет `write:packages` |
| `password is empty` | `$GHCR_TOKEN` не задан — вводите токен интерактивно |
| `Cache export is not supported for the docker driver` | создайте `docker-container` builder (см. выше) |
| `image not found` при `make push` | сборка шла через `docker-container` без `--load` (в Makefile уже есть `--load`) |

---

## 2. Код на GitHub

### Первый пуш

Репозиторий уже инициализирован (`git init`, ветка `main`), логин `konstantinkorzov`
прописан в `Makefile`, `run.sh`, `package.json`, README.

```bash
cd cp2k-mace-project
gh repo create konstantinkorzov/cp2k-mace-project --public --source=. --remote=origin --push
# или вручную:
git remote add origin https://github.com/konstantinkorzov/cp2k-mace-project.git
git push -u origin main
```

GitHub не принимает пароль аккаунта — нужен **PAT** (scope `repo`):

```
Username: konstantinkorzov
Password: ghp_…
```

Чтобы не вводить каждый раз: `git config --global credential.helper store`.

### Что не попадает в репозиторий

| Путь | Почему |
|---|---|
| `results/`, `results2/`, `results-*/` | результаты расчётов, блоки, `keys.db` с (зашифрованными) ключами |
| `.webui/` | логи процессов панели |
| `node_modules/`, `__pycache__/` | зависимости и кэш |
| `*.wfn`, `*.out`, `*.xyz`, `*.zip` | артефакты CP2K и данные |
| `src/webserver/*.pem` | приватный ключ TLS; панель создаёт сертификат сама |

### Если логин или имя репозитория изменятся

| Где | Что править |
|---|---|
| `Makefile` | `GITHUB_USER`, `IMAGE_NAME` |
| `run.sh` | `IMAGE_CPU`, `IMAGE_GPU` |
| `package.json` | `repository`, `bugs`, `homepage` |
| `README.md` | ссылки `git clone` и `docker pull` |
| `Dockerfile` | `org.opencontainers.image.source` |

Быстрая замена:

```bash
grep -rl 'konstantinkorzov' --exclude-dir=node_modules --exclude-dir=.git . \
  | xargs sed -i 's/konstantinkorzov/НОВЫЙ_ЛОГИН/g'
```

### Полезные команды

```bash
git status                       # что не закоммичено, насколько впереди origin
git log --oneline origin/main..HEAD   # что ещё не на GitHub
git add . && git commit -m "…"   # закоммитить
git push                         # отправить
```

---

## 3. Обновление образов у пользователей

```bash
docker pull ghcr.io/konstantinkorzov/cp2k-mace:latest-cpu
docker pull ghcr.io/konstantinkorzov/cp2k-mace:latest-gpu
./run.sh --cpu -d --no-pull      # перезапуск на локальном образе
```

---

## 4. Шпаргалка Docker

| Задача | Команда |
|---|---|
| активные контейнеры | `docker ps` |
| все, включая остановленные | `docker ps -a` |
| логи | `docker logs -f cp2k-mace-cpu` |
| внутрь | `docker exec -it cp2k-mace-cpu bash` |
| остановить | `docker stop cp2k-mace-cpu` |
| удалить | `docker rm -f cp2k-mace-cpu` |
