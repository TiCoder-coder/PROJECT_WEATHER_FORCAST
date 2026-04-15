# Docker Setup - Weather Forecast Application

## Prerequisites
- Docker
- Docker Compose

## Quick Start
```bash
# Build và start containers
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop containers
docker-compose down

# Remove volumes (reset database)
docker-compose down -v
```

## Useful Commands
```bash
# Access Django shell
docker-compose exec app python manage.py shell

# Run migrations
docker-compose exec app python manage.py migrate

# Create superuser
docker-compose exec app python manage.py createsuperuser

# Train model
docker-compose exec app python manage.py train_model

# Access MongoDB shell
docker-compose exec mongodb mongosh -u admin --authenticationDatabase admin
```

## Environment Variables
Default values are defined in `docker-compose.yml`:
- `MONGO_ROOT_PASSWORD=weatherpass123`
- `MONGODB_URI=mongodb://admin:${MONGO_ROOT_PASSWORD}@mongodb:27017/weather_db?authSource=admin`
- `DJANGO_SECRET_KEY=your-secret-key-here`
- `DEBUG=True`

If needed, create an `.env` file in project root to override values for production, for example:

```env
MONGO_ROOT_PASSWORD=change-this-password
DJANGO_SECRET_KEY=change-this-secret-key
```

## First Run Notes
After first startup:
1. Run migrations: `docker-compose exec app python manage.py migrate`
2. Create admin account: `docker-compose exec app python manage.py createsuperuser`
3. (Optional) Train model: `docker-compose exec app python manage.py train_model`

## Troubleshooting
- **App cannot connect to MongoDB**: ensure `mongodb` container is healthy and `MONGODB_URI` is correct.
- **Port conflict (8000/27017)**: change host port mapping in `docker-compose.yml`.
- **Dependency build errors**: rebuild image without cache: `docker-compose build --no-cache`.
- **Permission issues on mounted folders**: verify host folder permissions for `data/` and ML artifacts path.


## Security Notes
- Default values are for local development only.
- For production, set strong values for `MONGO_ROOT_PASSWORD` and `DJANGO_SECRET_KEY`, and set `DEBUG=False`.
