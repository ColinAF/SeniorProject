# Generated by Django 4.0.3 on 2022-04-19 09:16

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('produce_detector', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='produceimage',
            name='image',
        ),
    ]