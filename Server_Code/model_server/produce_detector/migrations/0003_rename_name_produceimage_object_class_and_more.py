# Generated by Django 4.0.3 on 2022-04-19 09:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('produce_detector', '0002_remove_produceimage_image'),
    ]

    operations = [
        migrations.RenameField(
            model_name='produceimage',
            old_name='name',
            new_name='object_class',
        ),
        migrations.AddField(
            model_name='produceimage',
            name='qty',
            field=models.CharField(default=1, max_length=50),
            preserve_default=False,
        ),
    ]
