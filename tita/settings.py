"""
Django settings for tita project.

Generated by 'django-admin startproject' using Django 5.0.6.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.0/ref/settings/
"""

from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


import os


# Obtener las variables de entorno
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')





# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-^c^!fa0*84@ui#z5+!rlgy9+d-4!$^5pfyxqr$)l-a1csjt8b9'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['.vercel.app','*']

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'  # Usa tu proveedor de email
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'beidenti2024@gmail.com'
EMAIL_HOST_PASSWORD = 'bdlwavjqddlvypgd'
DEFAULT_FROM_EMAIL = EMAIL_HOST_USER

# Asegurarse de que Django maneje correos en UTF-8
DEFAULT_CHARSET = 'utf-8'




# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'odontologo',
    'login',
    'paciente',
    'servicio',
    'inventario',
    'category',
    'cita',
    'car',
    'radiografia',
    'HistoriaClinica',
    'facecam',
    'cha',
    'diente',
    
    
    
    
   
    
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'tita.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'category.context_processors.menu_links',
                
            ],
        },
    },
]

WSGI_APPLICATION = 'tita.wsgi.application'

AUTH_USER_MODEL = 'login.Account'

# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'p2',  # Nombre de la base de datos que ya tienes creada
        'USER': 'postgres',  # Nombre de usuario de tu PostgreSQL
        'PASSWORD': '12345',  # Contraseña de tu usuario PostgreSQL
        'HOST': 'localhost',  # Host donde se está ejecutando PostgreSQL
        'PORT': '5432',  # Puerto donde se está ejecutando PostgreSQL (generalmente 5432)
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]
AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',  # Backend por defecto
    'login.backends.EmailBackend',  # Tu backend personalizado
)

LOGIN_REDIRECT_URL = 'home'
LOGOUT_REDIRECT_URL = 'login/role_selecion'

# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR /'static'
STATICFILES_DIRS = [
    'tita/static'
]

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR /'media'

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
