FROM python:3.9-slim

# Instalar dependencias necesarias para compilar dlib y psycopg2
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libpq-dev \
    && apt-get clean

# Crear directorio de trabajo
WORKDIR /code

# Copiar requirements.txt al contenedor
COPY requirements.txt .

# Instalar dependencias Python
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Descargar e instalar dlib precompilado para Linux
RUN python -m pip install dlib

# Copy the shape predictor file
COPY shape_predictor_68_face_landmarks.dat /code/

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
