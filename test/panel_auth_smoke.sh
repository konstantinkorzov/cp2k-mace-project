#!/usr/bin/env bash
B=https://127.0.0.1:8443
PW=$(cat /workspace/results/webui_admin_password.txt)
code() { curl -sk -o /dev/null -w '%{http_code}' "$@"; echo; }

echo -n "GET / без входа:            "; code "$B/"
echo -n "GET /api/overview без входа: "; code "$B/api/overview"
echo -n "POST /login неверный пароль: "; code -c /tmp/cj0 -d 'password=wrong' "$B/login"
echo -n "POST /login верный пароль:   "; code -c /tmp/cj -d "password=$PW" "$B/login"
echo -n "GET /api/overview с сессией: "; code -b /tmp/cj "$B/api/overview"
echo -n "POST /password смена:        "; code -b /tmp/cj -d "old_password=$PW&new_password=newpass123&confirm_password=newpass123" "$B/password"
echo -n "вход старым паролем:         "; code -c /tmp/cj2 -d "password=$PW" "$B/login"
echo -n "вход новым паролем:          "; code -c /tmp/cj3 -d 'password=newpass123' "$B/login"
echo -n "вернуть исходный пароль:     "; code -b /tmp/cj3 -d "old_password=newpass123&new_password=$PW&confirm_password=$PW" "$B/password"
echo -n "вход исходным паролем:       "; code -c /tmp/cj4 -d "password=$PW" "$B/login"
echo -n "GET / с сессией:             "; code -b /tmp/cj4 "$B/"
echo -n "GET /login с сессией (302):  "; code -b /tmp/cj4 "$B/login"
echo -n "GET /logout:                 "; code -b /tmp/cj4 "$B/logout"
echo -n "GET /api/overview после logout: "; code -b /tmp/cj4 "$B/api/overview"
