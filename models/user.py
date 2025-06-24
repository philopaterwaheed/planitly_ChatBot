from mongoengine import Document, StringField, EmailField, BooleanField, DateTimeField, IntField, ListField , DictField
import datetime
import re


class User(Document):
    id = StringField(primary_key=True, auto_generate=True)
    firebase_uid = StringField(required=True)  # firebase_id
    username = StringField(required=True, unique=True)
    email = EmailField(required=True, unique=True)
    email_verified = BooleanField(required=True, default=False)
    admin = BooleanField(default=False, required=False)
    devices = ListField(StringField(), default=[], max_length=5)
    invalid_attempts = IntField(default=0)  # Count of invalid login attempts
    last_reset = DateTimeField(default=datetime.datetime.utcnow)  
    firstname = StringField(required=True)
    lastname = StringField(required=True)
    phone_number = DictField(default=lambda: {"country_code": "", "number": ""})
    birthday = DateTimeField(required=True)
    profile_image = StringField(required=False)  # Cloudinary URL for profile image
    default_subjects = DictField(default={})
    settings = DictField(default=lambda: {"ai_accessible": []})

    meta = {'collection': 'users'}
