from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Kết quả training model
@dataclass
class TrainingResult:
    success: bool = True
    message: str = ""
    metrics: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None

# Kết quả dự đoán model
@dataclass
class PredictionResult:
    predictions: Any = None
    probabilities: Any = None
    message: str = ""
    timestamp: Optional[float] = None
# ----------------------------- BASE MODEL - LỚP CƠ SỞ CHO TẤT CẢ MODELS -----------------------------------------------------------
"""
Base_model.py - Định nghĩa lớp cơ sở (Base Model) cho tất cả các Model trong hệ thống

Mục đích:
    - Cung cấp các trường (fields) chung cho tất cả các models
    - Cung cấp các phương thức (methods) tiện ích chung
    - Đảm bảo tính nhất quán trong toàn bộ hệ thống
    - Giảm code trùng lặp (DRY - Don't Repeat Yourself)

Cách sử dụng:
    class MyModel(BaseModel):
        # Các trường riêng của model
        name = models.CharField(max_length=100)
        ...
        
        class Meta(BaseModel.Meta):
            db_table = "my_table"
"""

from django.db import models
from django.utils import timezone
from datetime import datetime, timedelta
import uuid


class BaseModel(models.Model):
    """
    Lớp cơ sở (Abstract Base Model) cho tất cả các Model trong hệ thống.
    
    Cung cấp:
        - Các trường tracking thời gian (createdAt, updatedAt)
        - Các trường trạng thái (is_active, is_deleted)
        - Các phương thức tiện ích chung
    
    Attributes:
        createdAt (DateTimeField): Thời điểm tạo record
        updatedAt (DateTimeField): Thời điểm cập nhật gần nhất
        is_active (BooleanField): Trạng thái hoạt động của record
        is_deleted (BooleanField): Đánh dấu xóa mềm (soft delete)
    """
    
    # ============================= TRACKING FIELDS =============================
    # Các trường dùng để theo dõi thời gian và trạng thái
    
    createdAt = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Ngày tạo",
        help_text="Tự động lưu thời điểm tạo record"
    )
    
    updatedAt = models.DateTimeField(
        auto_now=True,
        verbose_name="Ngày cập nhật",
        help_text="Tự động cập nhật khi record được sửa đổi"
    )
    
    # ============================= STATUS FIELDS =============================
    # Các trường trạng thái
    
    is_active = models.BooleanField(
        default=True,
        verbose_name="Đang hoạt động",
        help_text="True nếu record đang hoạt động, False nếu bị vô hiệu hóa"
    )
    
    is_deleted = models.BooleanField(
        default=False,
        verbose_name="Đã xóa",
        help_text="True nếu record bị xóa mềm (soft delete)"
    )
    
    # ============================= META CLASS =============================
    
    class Meta:
        """
        Meta class cho BaseModel.
        
        abstract = True: Đánh dấu đây là abstract model,
        Django sẽ không tạo table cho class này trong database.
        """
        abstract = True
        ordering = ['-createdAt']  # Mặc định sắp xếp theo thời gian tạo mới nhất
    
    # ============================= INSTANCE METHODS =============================
    
    def soft_delete(self):
        """
        Xóa mềm (soft delete) - đánh dấu record đã bị xóa mà không xóa khỏi database.
        
        Ưu điểm của soft delete:
            - Có thể khôi phục dữ liệu
            - Bảo toàn lịch sử
            - Tránh mất dữ liệu do lỗi người dùng
        
        Example:
            >>> user = User.objects.get(id=1)
            >>> user.soft_delete()
            >>> user.is_deleted
            True
        """
        self.is_deleted = True
        self.is_active = False
        self.save(update_fields=['is_deleted', 'is_active', 'updatedAt'])
    
    def restore(self):
        """
        Khôi phục record đã bị xóa mềm.
        
        Example:
            >>> user = User.objects.get(id=1)
            >>> user.restore()
            >>> user.is_deleted
            False
        """
        self.is_deleted = False
        self.is_active = True
        self.save(update_fields=['is_deleted', 'is_active', 'updatedAt'])
    
    def deactivate(self):
        """
        Vô hiệu hóa record (không xóa, chỉ tắt hoạt động).
        
        Sử dụng khi muốn tạm dừng hoạt động của một record
        mà không cần xóa mềm.
        """
        self.is_active = False
        self.save(update_fields=['is_active', 'updatedAt'])
    
    def activate(self):
        """
        Kích hoạt lại record đã bị vô hiệu hóa.
        """
        self.is_active = True
        self.save(update_fields=['is_active', 'updatedAt'])
    
    def get_created_time_display(self):
        """
        Trả về thời gian tạo dưới dạng chuỗi đọc được.
        
        Returns:
            str: Chuỗi thời gian format 'dd/mm/yyyy HH:MM:SS'
        """
        if self.createdAt:
            return self.createdAt.strftime('%d/%m/%Y %H:%M:%S')
        return None
    
    def get_updated_time_display(self):
        """
        Trả về thời gian cập nhật dưới dạng chuỗi đọc được.
        
        Returns:
            str: Chuỗi thời gian format 'dd/mm/yyyy HH:MM:SS'
        """
        if self.updatedAt:
            return self.updatedAt.strftime('%d/%m/%Y %H:%M:%S')
        return None
    
    def get_time_since_created(self):
        """
        Tính thời gian đã trôi qua kể từ khi tạo record.
        
        Returns:
            timedelta: Khoảng thời gian từ lúc tạo đến hiện tại
        """
        if self.createdAt:
            return timezone.now() - self.createdAt
        return None
    
    def get_time_since_updated(self):
        """
        Tính thời gian đã trôi qua kể từ lần cập nhật cuối.
        
        Returns:
            timedelta: Khoảng thời gian từ lần cập nhật cuối đến hiện tại
        """
        if self.updatedAt:
            return timezone.now() - self.updatedAt
        return None
    
    def to_dict(self, exclude_fields=None):
        """
        Chuyển đổi model instance thành dictionary.
        
        Args:
            exclude_fields (list, optional): Danh sách các field không muốn include
            
        Returns:
            dict: Dictionary chứa dữ liệu của record
            
        Example:
            >>> user = User.objects.get(id=1)
            >>> user.to_dict(exclude_fields=['password'])
            {'id': 1, 'name': 'John', 'email': 'john@example.com', ...}
        """
        exclude_fields = exclude_fields or []
        data = {}
        
        for field in self._meta.fields:
            field_name = field.name
            if field_name not in exclude_fields:
                value = getattr(self, field_name)
                
                # Xử lý các kiểu dữ liệu đặc biệt
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    value = str(value)
                
                data[field_name] = value
        
        return data
    
    # ============================= CLASS METHODS =============================
    
    @classmethod
    def get_active(cls):
        """
        Lấy tất cả records đang hoạt động (is_active=True, is_deleted=False).
        
        Returns:
            QuerySet: Các records đang hoạt động
            
        Example:
            >>> active_users = User.get_active()
        """
        return cls.objects.filter(is_active=True, is_deleted=False)
    
    @classmethod
    def get_deleted(cls):
        """
        Lấy tất cả records đã bị xóa mềm.
        
        Returns:
            QuerySet: Các records đã xóa mềm
        """
        return cls.objects.filter(is_deleted=True)
    
    @classmethod
    def get_inactive(cls):
        """
        Lấy tất cả records không hoạt động.
        
        Returns:
            QuerySet: Các records không hoạt động
        """
        return cls.objects.filter(is_active=False)
    
    @classmethod
    def get_recent(cls, days=7):
        """
        Lấy các records được tạo trong N ngày gần đây.
        
        Args:
            days (int): Số ngày muốn lấy (mặc định 7)
            
        Returns:
            QuerySet: Các records được tạo trong khoảng thời gian
            
        Example:
            >>> recent_users = User.get_recent(days=30)  # Users trong 30 ngày
        """
        threshold = timezone.now() - timedelta(days=days)
        return cls.objects.filter(createdAt__gte=threshold, is_deleted=False)
    
    @classmethod
    def count_active(cls):
        """
        Đếm số lượng records đang hoạt động.
        
        Returns:
            int: Số lượng records active
        """
        return cls.get_active().count()
    
    @classmethod
    def bulk_soft_delete(cls, queryset):
        """
        Xóa mềm nhiều records cùng lúc.
        
        Args:
            queryset: QuerySet chứa các records cần xóa
            
        Returns:
            int: Số lượng records đã xóa mềm
            
        Example:
            >>> users_to_delete = User.objects.filter(is_active=False)
            >>> deleted_count = User.bulk_soft_delete(users_to_delete)
        """
        return queryset.update(
            is_deleted=True,
            is_active=False,
            updatedAt=timezone.now()
        )
    
    @classmethod
    def bulk_restore(cls, queryset):
        """
        Khôi phục nhiều records đã xóa mềm cùng lúc.
        
        Args:
            queryset: QuerySet chứa các records cần khôi phục
            
        Returns:
            int: Số lượng records đã khôi phục
        """
        return queryset.update(
            is_deleted=False,
            is_active=True,
            updatedAt=timezone.now()
        )


class UUIDBaseModel(BaseModel):
    """
    Base Model sử dụng UUID làm primary key thay vì auto-increment integer.
    
    Ưu điểm của UUID:
        - Unique trên toàn hệ thống phân tán
        - Khó đoán (bảo mật hơn)
        - Có thể generate offline (trước khi insert vào DB)
    
    Nhược điểm:
        - Tốn bộ nhớ hơn (36 ký tự vs 4-8 bytes cho integer)
        - Index chậm hơn một chút
    """
    
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        verbose_name="ID",
        help_text="UUID tự động generate"
    )
    
    class Meta(BaseModel.Meta):
        abstract = True


class TimestampMixin(models.Model):
    """
    Mixin class chỉ cung cấp các trường timestamp.
    
    Sử dụng khi chỉ cần tracking thời gian mà không cần
    các tính năng khác của BaseModel.
    
    Example:
        class MyModel(TimestampMixin, models.Model):
            name = models.CharField(max_length=100)
    """
    
    createdAt = models.DateTimeField(
        auto_now_add=True,
        verbose_name="Ngày tạo"
    )
    
    updatedAt = models.DateTimeField(
        auto_now=True,
        verbose_name="Ngày cập nhật"
    )
    
    class Meta:
        abstract = True


class SoftDeleteMixin(models.Model):
    """
    Mixin class chỉ cung cấp tính năng soft delete.
    
    Sử dụng khi chỉ cần soft delete mà không cần
    các tính năng khác của BaseModel.
    """
    
    is_deleted = models.BooleanField(
        default=False,
        verbose_name="Đã xóa"
    )
    
    deleted_at = models.DateTimeField(
        null=True,
        blank=True,
        verbose_name="Thời điểm xóa"
    )
    
    class Meta:
        abstract = True
    
    def soft_delete(self):
        """Xóa mềm record."""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save(update_fields=['is_deleted', 'deleted_at'])
    
    def restore(self):
        """Khôi phục record."""
        self.is_deleted = False
        self.deleted_at = None
        self.save(update_fields=['is_deleted', 'deleted_at'])


# ============================= CUSTOM MANAGERS =============================

class ActiveManager(models.Manager):
    """
    Custom Manager chỉ trả về các records đang hoạt động.
    
    Sử dụng:
        class MyModel(BaseModel):
            objects = models.Manager()  # Manager mặc định
            active_objects = ActiveManager()  # Chỉ lấy active records
        
        # Sử dụng
        MyModel.active_objects.all()  # Chỉ trả về active records
    """
    
    def get_queryset(self):
        return super().get_queryset().filter(is_active=True, is_deleted=False)


class DeletedManager(models.Manager):
    """
    Custom Manager chỉ trả về các records đã xóa mềm.
    """
    
    def get_queryset(self):
        return super().get_queryset().filter(is_deleted=True)