# Generated by Django 4.2.13 on 2024-07-08 02:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('cita', '0006_remove_cita_servic'),
    ]

    operations = [
        migrations.AddField(
            model_name='cita',
            name='visualizada',
            field=models.BooleanField(default=False),
        ),
    ]
