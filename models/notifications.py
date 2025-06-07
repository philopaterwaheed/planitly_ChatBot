from mongoengine import Document, StringField, BooleanField, DateTimeField
from datetime import datetime

class NotificationCount(Document):
    user_id = StringField(required=True)
    count = StringField(default='0')
    meta = {'collection': 'notification_counts'
            }


class Notification_db(Document):
    # ID of the user receiving the notification
    user_id = StringField(required=True)
    title = StringField(required=True)  # Title of the notification
    message = StringField(required=True)  # Notification message
    # Whether the notification has been read
    is_read = BooleanField(default=False)
    created_at = DateTimeField(
        default=datetime.utcnow)  # Timestamp of creation
    meta = {'collection': 'notifications'}

    def to_dict(self):
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat()
        }
