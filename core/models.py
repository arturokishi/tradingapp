from django.db import models

# Create your models here.



class UserWatchlist(models.Model):
    ticker = models.CharField(max_length=10, unique=True)

    def __str__(self):
        return self.ticker
