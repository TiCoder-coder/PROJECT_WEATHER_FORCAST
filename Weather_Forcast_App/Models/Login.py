# ----------------------------- DINH NGHIA CAC THUOC TINH CHO DATABASE -----------------------------------------------------------


from django.db import models
from Weather_Forcast_App.Enums.Enums import Role
import uuid
class LoginModel(models.Model):
    
    # id se tu tao cho manager trong mongodb
    name = models.CharField(max_length = 100, unique=True)
    userName = models.CharField(max_length = 100, unique=True)
    password = models.CharField(max_length = 255)
    email = models.CharField(max_length = 100, unique=True)
    role = models.CharField(max_length=100, choices=[(choice.name, choice.value) for choice in Role])
    is_active = models.BooleanField(default=True)
    
    
    
    # Dinh nghia cac security fields --- DE SU DUNG TRONG TANG SERVICES
    failed_attempts = models.IntegerField(default=0)                # Dem so lan dang nhap that bai lien tiep
    lock_until = models.DateTimeField(null=True, blank=True)        # Neu dang nhap that bai qua so lan quy dinh thi cho dang nhap lai sau mot khoang thoi gian quy dinh
    
    
    # Dinh nghia cac field dung de tracking (theo doi) --- DE SU DUNG TRONG TANG SERVICES
    last_login = models.DateTimeField(null=True, blank=True)        # Luu lai thoi diem dang nhap thanh cong gan nhat
    createdAt = models.DateTimeField(auto_now_add=True)             # Tu dong luu ngay khi tao ra mot manager moi
    updatedAt = models.DateTimeField(auto_now=True)                 # Tu dong cap nhap khi mot manager moi duoc update
    
    
    # Dat ten cho table trong database la managers
    class Meta:
        db_table = "managers"                                       # Chi dinh ten database
        ordering = ["userName"]                                     # Sap xep cac userName theo thu tu bang chu cai
    
    
    # Dung de log thong tin khi can
    def __str__(self):
        return f"{self.name} - ({self.userName}): {self.role}"