# Generated by Django 4.2.13 on 2024-07-01 23:13

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('servicio', '0001_initial'),
        ('cita', '0004_remove_cita_servic_cita_servicio_opcion_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='cita',
            name='servicio_opcion',
        ),
        migrations.AddField(
            model_name='cita',
            name='servic',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='servicio.servicio'),
        ),
        migrations.AlterField(
            model_name='cita',
            name='servicio',
            field=models.CharField(choices=[('sin servicio', 'Sin Servicio'), ('con servicio', 'Con Servicio')], default='sin servicio', max_length=20),
        ),
    ]
