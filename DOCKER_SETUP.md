# 🐳 Docker Setup — VN Weather Hub

## 1. Tạo MongoDB Replica Set (chạy 1 lần đầu)

```bash
# Tạo network
docker network create mongoNet

# Tạo 3 container (r4 = PRIMARY, r5 r6 = SECONDARY)
docker run -d --name r4 --net mongoNet -p 27108:27017 mongo:latest mongod --replSet mongoRepSet --bind_ip_all --port 27017
docker run -d --name r5 --net mongoNet -p 27109:27017 mongo:latest mongod --replSet mongoRepSet --bind_ip_all --port 27017
docker run -d --name r6 --net mongoNet -p 27110:27017 mongo:latest mongod --replSet mongoRepSet --bind_ip_all --port 27017

# Khởi tạo replica set
docker exec r4 mongosh --eval "rs.initiate({ _id: 'mongoRepSet', members: [ { _id: 0, host: 'r4:27017' }, { _id: 1, host: 'r5:27017' }, { _id: 2, host: 'r6:27017' } ] })"

# Kiểm tra trạng thái
docker exec r4 mongosh --eval "rs.status().members.map(m=>({name:m.name, state:m.stateStr}))"
```

## 2. Chạy hàng ngày

```bash
# Start MongoDB (nếu tắt máy rồi mở lại)
docker start r4 r5 r6

# Build & chạy web + airflow
docker compose up -d
```

## 3. Dừng

```bash
docker compose down        # Dừng web + airflow
docker stop r4 r5 r6      # Dừng MongoDB
```

## 4. Truy cập

| Service | URL |
|---------|-----|
| Web App | http://localhost:8000 |
| Airflow | http://localhost:8080 (admin / admin) |
| MongoDB Compass | `mongodb://localhost:27108/Login?directConnection=true` |

## 5. Lệnh hữu ích

```bash
# Xem logs
docker compose logs -f web

# Vào shell web container
docker compose exec web bash

# Chạy lệnh Django trong container
docker compose exec web python manage.py insert_first_data

# Xem trạng thái replica set
docker exec r4 mongosh --eval "rs.status().members.map(m=>({name:m.name, state:m.stateStr}))"

# Build lại (khi đổi requirements.txt)
docker compose build --no-cache
```

## 6. Xoá sạch & làm lại (⚠️ mất dữ liệu)

```bash
docker rm -f r4 r5 r6
docker network rm mongoNet
docker compose down -v
# Rồi chạy lại từ Bước 1
```
