# Generated by Django 4.2.13 on 2024-06-27 20:33

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('servicio', '0001_initial'),
        ('car', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='cartitem',
            name='servicio',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='servicio.servicio'),
        ),
    ]
